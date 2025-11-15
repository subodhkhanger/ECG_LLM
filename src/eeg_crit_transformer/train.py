from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from eeg_crit_transformer.models.crit_transformer import build_model
from eeg_crit_transformer.data.chbmit import CHBMITWindows, WindowingConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EEG Criticality Transformer (baseline)")
    p.add_argument("--data-dir", type=str, default=None, help="Path to CHB-MIT root (contains EDFs)")
    p.add_argument("--annotations", type=str, default=None, help="Path to annotations CSV (file,start,end,label)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--sample-rate", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--ff", type=int, default=256, help="Transformer feedforward dim")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--channels", type=int, default=18, help="Model input channels (synthetic or selected from data)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--save-history", action="store_true", help="Save training history to JSON")
    return p.parse_args()


def compute_grad_norm(model: torch.nn.Module) -> Dict[str, float]:
    """Compute gradient norms for monitoring training dynamics."""
    total_norm = 0.0
    param_norms = []

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            param_norms.append(param_norm)
            total_norm += param_norm ** 2

    total_norm = total_norm ** 0.5

    return {
        "grad_norm_total": total_norm,
        "grad_norm_mean": np.mean(param_norms) if param_norms else 0.0,
        "grad_norm_max": np.max(param_norms) if param_norms else 0.0,
        "grad_norm_min": np.min(param_norms) if param_norms else 0.0,
    }


def compute_weight_stats(model: torch.nn.Module) -> Dict[str, float]:
    """Compute weight statistics for monitoring."""
    weights = []
    for p in model.parameters():
        weights.append(p.data.cpu().numpy().flatten())

    if weights:
        weights = np.concatenate(weights)
        return {
            "weight_mean": float(np.mean(weights)),
            "weight_std": float(np.std(weights)),
            "weight_max": float(np.max(np.abs(weights))),
        }
    return {"weight_mean": 0.0, "weight_std": 0.0, "weight_max": 0.0}


def make_dataloaders(args: argparse.Namespace):
    """Load real CHB-MIT data - no synthetic fallback."""

    # Check if data directory and annotations are provided
    if not args.data_dir:
        print("\n" + "=" * 70)
        print("ERROR: --data-dir is required!")
        print("=" * 70)
        print("\nUsage:")
        print("  python -m eeg_crit_transformer.train \\")
        print("    --data-dir data/chbmit \\")
        print("    --annotations data/chbmit/annotations.csv \\")
        print("    --epochs 10")
        print("\nSteps to prepare data:")
        print("  1. Download CHB-MIT data:")
        print("     export PHYSIONET_USER='your_username'")
        print("     export PHYSIONET_PASS='your_password'")
        print("     bash scripts/download_chbmit.sh data/chbmit chb01 chb02")
        print("\n  2. Create annotations CSV:")
        print("     python scripts/create_annotations.py --data-dir data/chbmit")
        print("\n  3. Train:")
        print("     python -m eeg_crit_transformer.train \\")
        print("       --data-dir data/chbmit \\")
        print("       --annotations data/chbmit/annotations.csv \\")
        print("       --epochs 10")
        print("=" * 70 + "\n")
        exit(1)

    if not args.annotations:
        print("\n" + "=" * 70)
        print("ERROR: --annotations is required!")
        print("=" * 70)
        print("\nCreate annotations CSV with:")
        print("  python scripts/create_annotations.py --data-dir data/chbmit")
        print("=" * 70 + "\n")
        exit(1)

    # Check if annotations file exists
    if not os.path.exists(args.annotations):
        print("\n" + "=" * 70)
        print(f"ERROR: Annotations file not found: {args.annotations}")
        print("=" * 70)
        print("\nCreate it with:")
        print(f"  python scripts/create_annotations.py --data-dir {args.data_dir}")
        print("=" * 70 + "\n")
        exit(1)

    # Load real CHB-MIT data
    print(f"\n📊 Loading CHB-MIT data from: {args.data_dir}")
    print(f"📋 Using annotations: {args.annotations}")

    cfg = WindowingConfig(sample_rate=args.sample_rate, window_sec=args.window_sec)
    ds = CHBMITWindows(
        root=args.data_dir,
        annotations_csv=args.annotations,
        config=cfg,
        max_files=args.max_files,
    )

    if len(ds) == 0:
        print("\n" + "=" * 70)
        print("ERROR: No windows loaded from data!")
        print("=" * 70)
        print("\nPossible reasons:")
        print("  1. EDF files not found in data directory")
        print("  2. Annotations CSV is empty or incorrectly formatted")
        print("  3. Summary files (.txt) missing")
        print("\nCheck:")
        print(f"  ls {args.data_dir}/chb01/")
        print(f"  head {args.annotations}")
        print("=" * 70 + "\n")
        exit(1)

    in_channels = ds[0][0].shape[0]

    print(f"✅ Loaded {len(ds)} windows from {len(ds)} samples")
    print(f"📈 Input channels: {in_channels}")

    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    print(f"🔄 Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train_loader, val_loader, in_channels


def train_one_epoch(model, loader, opt, device) -> Dict[str, Any]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    grad_norms = []
    batch_losses = []
    all_preds = []
    all_labels = []
    all_probs = []

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        # pad time to multiple of patch size
        T = x.shape[-1]
        pad = (-T) % model.cfg.patch_size
        if pad:
            x = F.pad(x, (0, pad))

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Compute gradient norm before optimizer step
        grad_stats = compute_grad_norm(model)
        grad_norms.append(grad_stats["grad_norm_total"])

        opt.step()

        # Track metrics
        batch_losses.append(loss.item())
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)

        correct += int((pred == y).sum().item())
        total += x.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    # Compute aggregate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Per-class accuracy
    per_class_acc = {}
    for cls in np.unique(all_labels):
        mask = all_labels == cls
        if mask.sum() > 0:
            per_class_acc[f"class_{cls}_acc"] = (all_preds[mask] == all_labels[mask]).mean()

    # Confidence statistics
    max_probs = all_probs.max(axis=1)
    conf_stats = {
        "confidence_mean": float(max_probs.mean()),
        "confidence_std": float(max_probs.std()),
        "confidence_min": float(max_probs.min()),
        "confidence_max": float(max_probs.max()),
    }

    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_std": float(np.std(grad_norms)),
        "grad_norm_max": float(np.max(grad_norms)),
        "batch_loss_std": float(np.std(batch_losses)),
        **per_class_acc,
        **conf_stats,
    }


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, Any]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    batch_losses = []

    for x, y in tqdm(loader, desc="valid", leave=False):
        x, y = x.to(device), y.to(device)
        T = x.shape[-1]
        pad = (-T) % model.cfg.patch_size
        if pad:
            x = F.pad(x, (0, pad))

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        batch_losses.append(loss.item())
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)

        correct += int((pred == y).sum().item())
        total += x.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    # Compute aggregate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Per-class accuracy and precision/recall
    per_class_metrics = {}
    for cls in np.unique(all_labels):
        mask_true = all_labels == cls
        mask_pred = all_preds == cls

        if mask_true.sum() > 0:
            per_class_metrics[f"class_{cls}_acc"] = (all_preds[mask_true] == all_labels[mask_true]).mean()
            per_class_metrics[f"class_{cls}_recall"] = (all_preds[mask_true] == cls).mean()

        if mask_pred.sum() > 0:
            per_class_metrics[f"class_{cls}_precision"] = (all_labels[mask_pred] == cls).mean()

    # Confidence statistics
    max_probs = all_probs.max(axis=1)
    conf_stats = {
        "confidence_mean": float(max_probs.mean()),
        "confidence_std": float(max_probs.std()),
        "confidence_min": float(max_probs.min()),
        "confidence_max": float(max_probs.max()),
    }

    # Confusion matrix elements for binary classification
    if len(np.unique(all_labels)) == 2:
        tn = ((all_preds == 0) & (all_labels == 0)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        tp = ((all_preds == 1) & (all_labels == 1)).sum()

        conf_matrix = {
            "true_neg": int(tn),
            "false_pos": int(fp),
            "false_neg": int(fn),
            "true_pos": int(tp),
        }
    else:
        conf_matrix = {}

    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "batch_loss_std": float(np.std(batch_losses)),
        **per_class_metrics,
        **conf_stats,
        **conf_matrix,
    }


def print_metrics(prefix: str, metrics: Dict[str, Any], width: int = 80) -> None:
    """Pretty print metrics dictionary."""
    print("=" * width)
    print(f"{prefix:^{width}}")
    print("=" * width)

    # Main metrics first
    main_keys = ["loss", "acc"]
    for key in main_keys:
        if key in metrics:
            print(f"  {key:20s}: {metrics[key]:.6f}")

    # Gradient norms (training only)
    grad_keys = [k for k in metrics if k.startswith("grad_")]
    if grad_keys:
        print(f"\n  {'Gradient Norms':20s}:")
        for key in sorted(grad_keys):
            print(f"    {key:18s}: {metrics[key]:.6f}")

    # Per-class metrics
    class_keys = [k for k in metrics if k.startswith("class_")]
    if class_keys:
        print(f"\n  {'Per-Class Metrics':20s}:")
        for key in sorted(class_keys):
            print(f"    {key:18s}: {metrics[key]:.6f}")

    # Confidence stats
    conf_keys = [k for k in metrics if k.startswith("confidence_")]
    if conf_keys:
        print(f"\n  {'Confidence Stats':20s}:")
        for key in sorted(conf_keys):
            print(f"    {key:18s}: {metrics[key]:.6f}")

    # Confusion matrix (validation only)
    conf_matrix_keys = ["true_neg", "false_pos", "false_neg", "true_pos"]
    if any(k in metrics for k in conf_matrix_keys):
        print(f"\n  {'Confusion Matrix':20s}:")
        print(f"    TN: {metrics.get('true_neg', 0):4d}  FP: {metrics.get('false_pos', 0):4d}")
        print(f"    FN: {metrics.get('false_neg', 0):4d}  TP: {metrics.get('true_pos', 0):4d}")

    # Other metrics
    other_keys = [k for k in metrics if k not in main_keys + grad_keys + class_keys + conf_keys + conf_matrix_keys]
    if other_keys:
        print(f"\n  {'Other Metrics':20s}:")
        for key in sorted(other_keys):
            val = metrics[key]
            if isinstance(val, float):
                print(f"    {key:18s}: {val:.6f}")
            else:
                print(f"    {key:18s}: {val}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    train_loader, val_loader, in_channels = make_dataloaders(args)

    model = build_model(
        in_channels=in_channels,
        num_classes=2,
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 80)
    print("Training Configuration".center(80))
    print("=" * 80)
    print(f"  {'Device':<20s}: {device}")
    print(f"  {'Total Parameters':<20s}: {n_params:,}")
    print(f"  {'Trainable Parameters':<20s}: {n_trainable:,}")
    print(f"  {'Input Channels':<20s}: {in_channels}")
    print(f"  {'Epochs':<20s}: {args.epochs}")
    print(f"  {'Batch Size':<20s}: {args.batch_size}")
    print(f"  {'Learning Rate':<20s}: {args.lr}")
    print(f"  {'Weight Decay':<20s}: {args.weight_decay}")
    print(f"  {'Window (sec)':<20s}: {args.window_sec}")
    print(f"  {'Sample Rate':<20s}: {args.sample_rate}")
    print(f"  {'Patch Size':<20s}: {args.patch_size}")
    print(f"  {'Model Dim':<20s}: {args.d_model}")
    print(f"  {'Num Heads':<20s}: {args.nhead}")
    print(f"  {'Num Layers':<20s}: {args.num_layers}")
    print(f"  {'Feedforward Dim':<20s}: {args.ff}")
    print(f"  {'Dropout':<20s}: {args.dropout}")
    print("=" * 80)

    best_val_acc = 0.0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Get weight stats before training
        weight_stats_before = compute_weight_stats(model)

        # Train
        train_metrics = train_one_epoch(model, train_loader, opt, device)
        train_metrics["lr"] = opt.param_groups[0]["lr"]
        train_metrics["epoch"] = epoch

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_metrics["epoch"] = epoch

        # Weight stats after training
        weight_stats_after = compute_weight_stats(model)
        train_metrics.update({f"weight_{k}_after": v for k, v in weight_stats_after.items()})
        train_metrics["weight_change"] = abs(weight_stats_after["weight_mean"] - weight_stats_before["weight_mean"])

        # Store history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Print metrics
        print_metrics(f"Training Metrics - Epoch {epoch}", train_metrics)
        print()
        print_metrics(f"Validation Metrics - Epoch {epoch}", val_metrics)

        # Save best model
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": {
                    "in_channels": in_channels,
                    "patch_size": args.patch_size,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "ff": args.ff,
                    "dropout": args.dropout,
                },
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_val_acc": best_val_acc,
            }, "checkpoints/best.pt")
            print(f"\n{'':>40}⭐ New best model saved! (val_acc: {best_val_acc:.4f})")

        # Quick summary
        print(f"\n{'Summary':^80}")
        print(f"  Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} grad_norm={train_metrics['grad_norm_mean']:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} best={best_val_acc:.4f}")

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete".center(80))
    print("=" * 80)
    print(f"  {'Best Validation Acc':<30s}: {best_val_acc:.6f}")
    print(f"  {'Final Train Acc':<30s}: {history['train'][-1]['acc']:.6f}")
    print(f"  {'Final Val Acc':<30s}: {history['val'][-1]['acc']:.6f}")
    print("=" * 80)

    # Save history if requested
    if args.save_history:
        import json
        os.makedirs("checkpoints", exist_ok=True)
        history_path = "checkpoints/training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n📊 Training history saved to: {history_path}")


if __name__ == "__main__":
    main()

