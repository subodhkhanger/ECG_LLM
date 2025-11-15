#!/usr/bin/env python3
"""
Test the trained model on validation/test data.
Shows how to evaluate performance and make predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from torch.utils.data import DataLoader, random_split

from eeg_crit_transformer.models.crit_transformer import build_model
from eeg_crit_transformer.data.chbmit import CHBMITWindows, WindowingConfig


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    # Recreate model
    model = build_model(
        in_channels=18,
        patch_size=64,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    )

    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model


def evaluate_on_test_data(model, test_loader, device):
    """Evaluate model on test data and compute metrics."""

    all_preds = []
    all_labels = []
    all_probs = []
    all_losses = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # Pad to multiple of patch size
            T = x.shape[-1]
            pad = (-T) % 64  # patch_size=64
            if pad:
                x = F.pad(x, (0, pad))

            # Forward pass
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            # Get predictions
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            # Store results
            all_losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels, all_probs, all_losses)

    return metrics, all_preds, all_labels, all_probs


def compute_metrics(preds, labels, probs, losses):
    """Compute evaluation metrics."""

    metrics = {
        'loss': np.mean(losses),
        'accuracy': (preds == labels).mean(),
    }

    # Per-class metrics
    for cls in np.unique(labels):
        mask = labels == cls
        class_name = 'Seizure' if cls == 1 else 'Non-Seizure'

        if mask.sum() > 0:
            # Accuracy
            acc = (preds[mask] == labels[mask]).mean()
            metrics[f'{class_name}_accuracy'] = acc

            # Recall (Sensitivity)
            recall = (preds[mask] == cls).sum() / mask.sum()
            metrics[f'{class_name}_recall'] = recall

            # Precision
            pred_mask = preds == cls
            if pred_mask.sum() > 0:
                precision = (labels[pred_mask] == cls).sum() / pred_mask.sum()
                metrics[f'{class_name}_precision'] = precision

                # F1 score
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                metrics[f'{class_name}_f1'] = f1

    # Confusion matrix
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tp = ((preds == 1) & (labels == 1)).sum()

    metrics['TP'] = int(tp)
    metrics['FP'] = int(fp)
    metrics['FN'] = int(fn)
    metrics['TN'] = int(tn)

    # Sensitivity and Specificity
    if (tp + fn) > 0:
        metrics['Sensitivity'] = tp / (tp + fn)  # Catch seizures
    if (tn + fp) > 0:
        metrics['Specificity'] = tn / (tn + fp)  # Avoid false alarms

    return metrics


def print_metrics(metrics):
    """Pretty print metrics."""
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"\nOverall Performance:")
    print(f"  Accuracy:            {metrics['accuracy']:.2%}")
    print(f"  Loss:                {metrics['loss']:.6f}")

    print(f"\nSeizure Detection (Class 1):")
    print(f"  Recall/Sensitivity:  {metrics.get('Seizure_recall', 0):.2%}  ← Catch seizures")
    print(f"  Precision:           {metrics.get('Seizure_precision', 0):.2%}")
    print(f"  F1 Score:            {metrics.get('Seizure_f1', 0):.4f}")

    print(f"\nNon-Seizure Detection (Class 0):")
    print(f"  Accuracy:            {metrics.get('Non-Seizure_accuracy', 0):.2%}")
    print(f"  Recall/Specificity:  {metrics.get('Specificity', 0):.2%}  ← Avoid false alarms")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {metrics['TP']:4d}  (correctly detected seizures)")
    print(f"  False Positives (FP): {metrics['FP']:4d}  (false alarms)")
    print(f"  False Negatives (FN): {metrics['FN']:4d}  (missed seizures)")
    print(f"  True Negatives (TN):  {metrics['TN']:4d}  (correct non-seizure)")

    print("\n" + "=" * 70)

    # Medical relevance
    print("\nMedical Interpretation:")
    if metrics.get('Seizure_recall', 0) > 0.90:
        print("   EXCELLENT: Model catches >90% of seizures (safe for clinical use)")
    elif metrics.get('Seizure_recall', 0) > 0.80:
        print("  GOOD: Model catches 80-90% of seizures (acceptable but can improve)")
    else:
        print(" POOR: Model misses too many seizures (needs improvement)")

    if metrics['FP'] / max(1, metrics['FP'] + metrics['TN']) < 0.2:
        print(" Low false alarm rate (<20%)")
    else:
        print("  High false alarm rate (>20%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test trained model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to CHB-MIT data")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\n ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"\nTrain the model first with:")
        print(f"  python -m eeg_crit_transformer.train \\")
        print(f"    --data-dir {args.data_dir} \\")
        print(f"    --annotations {args.annotations} \\")
        print(f"    --epochs 10")
        return

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    # Load data
    cfg = WindowingConfig()
    ds = CHBMITWindows(
        root=args.data_dir,
        annotations_csv=args.annotations,
        config=cfg,
    )

    # Create test split
    n_test = max(1, int(len(ds) * 0.2))  # 20% for testing
    n_train = len(ds) - n_test
    train_ds, test_ds = random_split(ds, [n_train, n_test])

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(test_ds)} test samples")

    # Load model
    print(f"\n Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device=args.device)
    print(f" Model loaded (device: {args.device})")

    # Evaluate
    print(f"\n Evaluating on {len(test_ds)} test samples...")
    metrics, preds, labels, probs = evaluate_on_test_data(model, test_loader, args.device)

    # Print results
    print_metrics(metrics)

    # Save results
    results_file = "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n Results saved to: {results_file}")


if __name__ == "__main__":
    main()
