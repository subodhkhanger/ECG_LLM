#!/usr/bin/env python3
"""
Make predictions on individual EEG samples.
Shows how to use the trained model for inference.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from eeg_crit_transformer.models.crit_transformer import build_model


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

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


def predict_sample(model, eeg_data, device='cuda'):
    """
    Make prediction on a single EEG sample.

    Args:
        model: Trained model
        eeg_data: (channels, time_steps) numpy array or torch tensor
        device: Device to run on

    Returns:
        prediction: 0 (non-seizure) or 1 (seizure)
        confidence: Probability score (0-1)
        probs: Class probabilities
    """

    # Convert to tensor if needed
    if isinstance(eeg_data, np.ndarray):
        x = torch.from_numpy(eeg_data).float()
    else:
        x = eeg_data.float()

    # Add batch dimension
    if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, channels, time)

    # Move to device
    x = x.to(device)

    # Pad to multiple of patch size
    T = x.shape[-1]
    pad = (-T) % 64  # patch_size=64
    if pad:
        x = F.pad(x, (0, pad))

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
        confidence = probs.max().item()

    return pred, confidence, probs.cpu().numpy()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Make predictions on EEG samples")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\n ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"\nTrain a model first with:")
        print(f"  python -m eeg_crit_transformer.train \\")
        print(f"    --data-dir data/chbmit \\")
        print(f"    --annotations data/chbmit/annotations.csv")
        return

    print(f"\n Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device=args.device)

    # Example 1: Synthetic test signal (non-seizure)
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Testing on Synthetic Non-Seizure Signal")
    print("=" * 70)

    # Generate random "non-seizure" EEG
    non_seizure_eeg = np.random.randn(18, 1280).astype(np.float32) * 0.5

    pred, conf, probs = predict_sample(model, non_seizure_eeg, device=args.device)

    print(f"\n Prediction: {'SEIZURE ' if pred == 1 else 'NON-SEIZURE '}")
    print(f" Confidence: {conf:.2%}")
    print(f" Probabilities:")
    print(f"   Non-Seizure: {probs[0][0]:.2%}")
    print(f"   Seizure:     {probs[0][1]:.2%}")

    # Example 2: Synthetic test signal (seizure-like)
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Testing on Synthetic Seizure-Like Signal")
    print("=" * 70)

    # Generate "seizure-like" EEG (higher amplitude oscillations)
    t = np.arange(1280)
    seizure_eeg = np.zeros((18, 1280), dtype=np.float32)
    for ch in range(18):
        # Add high-frequency oscillations
        seizure_eeg[ch] = (
            2.0 * np.sin(2 * np.pi * 10 * t / 256) +  # 10 Hz
            1.5 * np.sin(2 * np.pi * 20 * t / 256) +  # 20 Hz
            0.5 * np.random.randn(1280)  # noise
        )

    pred, conf, probs = predict_sample(model, seizure_eeg, device=args.device)

    print(f"\n Prediction: {'SEIZURE ' if pred == 1 else 'NON-SEIZURE '}")
    print(f" Confidence: {conf:.2%}")
    print(f" Probabilities:")
    print(f"   Non-Seizure: {probs[0][0]:.2%}")
    print(f"   Seizure:     {probs[0][1]:.2%}")

    # Example 3: From real data
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Testing on Real EEG Data")
    print("=" * 70)

    try:
        from eeg_crit_transformer.data.chbmit import CHBMITWindows, WindowingConfig

        cfg = WindowingConfig()
        ds = CHBMITWindows(
            root="data/chbmit",
            annotations_csv="data/chbmit/annotations.csv",
            config=cfg,
        )

        if len(ds) > 0:
            # Get first few samples
            print(f"\n Found {len(ds)} samples in dataset")
            print(f"\nTesting on first 5 samples:")
            print("-" * 70)

            for i in range(min(5, len(ds))):
                x, y = ds[i]
                pred, conf, probs = predict_sample(model, x.numpy(), device=args.device)

                true_label = "SEIZURE" if y.item() == 1 else "NON-SEIZURE"
                pred_label = "SEIZURE" if pred == 1 else "NON-SEIZURE"
                is_correct = "" if pred == y.item() else ""

                print(f"\nSample {i+1}: {is_correct}")
                print(f"  True:      {true_label}")
                print(f"  Predicted: {pred_label}")
                print(f"  Confidence: {conf:.2%}")

    except Exception as e:
        print(f"\n  Could not load real data: {e}")
        print(f"  Make sure data is downloaded and annotations exist")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
