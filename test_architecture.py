#!/usr/bin/env python3
"""Quick architecture test with tiny synthetic dataset."""
from __future__ import annotations

import torch
from src.eeg_crit_transformer.models.crit_transformer import build_model

def test_forward_pass():
    """Test model forward pass with minimal synthetic data."""
    # Tiny test configuration
    batch_size = 4
    in_channels = 18
    time_length = 256  # 1 second at 256 Hz, or 4 patches of size 64
    patch_size = 64
    num_classes = 2

    # Build minimal model
    model = build_model(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        d_model=64,      # Smaller than default 128
        nhead=2,         # Smaller than default 4
        num_layers=2,    # Smaller than default 4
        dim_feedforward=128,  # Smaller than default 256
        dropout=0.0,     # No dropout for deterministic testing
    )

    # Create tiny synthetic batch
    x = torch.randn(batch_size, in_channels, time_length)

    # Forward pass
    print(f"Input shape: {x.shape}")
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {num_classes})")

    # Verify output shape
    assert logits.shape == (batch_size, num_classes), f"Shape mismatch!"

    # Test predictions
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    print(f"\nPredictions: {preds}")
    print(f"Probabilities:\n{probs}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    print("\nArchitecture test passed!")


def test_with_dataloader():
    """Test model with DataLoader and tiny synthetic dataset."""
    from torch.utils.data import DataLoader
    from src.eeg_crit_transformer.data.test_datasets import SyntheticEEG

    # Create tiny dataset
    tiny_ds = SyntheticEEG(
        n=16,           # Only 16 samples
        channels=18,    # 18 EEG channels
        length=256,     # 1 second of data
        seed=42,
    )

    loader = DataLoader(tiny_ds, batch_size=4, shuffle=True)

    # Build model
    model = build_model(
        in_channels=18,
        patch_size=64,
        d_model=64,
        nhead=2,
        num_layers=2,
    )

    # Test one batch
    for x, y in loader:
        print(f"\nBatch shapes: x={x.shape}, y={y.shape}")
        logits = model(x)
        print(f"Logits shape: {logits.shape}")

        # Compute loss
        loss = torch.nn.functional.cross_entropy(logits, y)
        print(f"Loss: {loss.item():.4f}")

        # Check accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        print(f"Accuracy: {acc.item():.2%}")
        print(f"Labels: {y}")
        print(f"Predictions: {preds}")
        break  # Only test first batch

    print("\nDataLoader test passed!")


def test_training_step():
    """Test a single training step."""
    from torch.optim import AdamW
    from src.eeg_crit_transformer.data.test_datasets import SyntheticEEG
    from torch.utils.data import DataLoader

    # Tiny dataset
    ds = SyntheticEEG(n=8, channels=18, length=256, seed=42)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    # Model
    model = build_model(in_channels=18, patch_size=64, d_model=64, nhead=2, num_layers=2)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Training step
    model.train()
    x, y = next(iter(loader))

    print(f"\nBefore training:")
    logits = model(x)
    loss_before = torch.nn.functional.cross_entropy(logits, y)
    print(f"Loss: {loss_before.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    # Check loss changed
    logits = model(x)
    loss_after = torch.nn.functional.cross_entropy(logits, y)
    print(f"\nAfter training step:")
    print(f"Loss: {loss_after.item():.4f}")
    print(f"Loss changed: {abs(loss_after.item() - loss_before.item()):.6f}")

    print("\nTraining step test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing EEG Crit Transformer Architecture")
    print("=" * 60)

    print("\n[1/3] Testing forward pass...")
    test_forward_pass()

    print("\n[2/3] Testing with DataLoader...")
    test_with_dataloader()

    print("\n[3/3] Testing training step...")
    test_training_step()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
