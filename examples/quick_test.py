#!/usr/bin/env python3
"""Quick examples of using tiny datasets for architecture testing."""

from eeg_crit_transformer.data.test_datasets import TinyEEG, SyntheticEEG
from eeg_crit_transformer.models.crit_transformer import build_model
import torch

# Example 1: Ultra-fast with TinyEEG (8 samples, minimal overhead)
print("=" * 60)
print("Example 1: TinyEEG - Ultra-fast testing (8 samples)")
print("=" * 60)

tiny_ds = TinyEEG(n=8, channels=18, length=256)
print(f"Dataset size: {len(tiny_ds)}")
x, y = tiny_ds[0]
print(f"Sample shape: x={x.shape}, y={y.shape}")
print(f"Label: {y.item()}")

# Quick forward pass test
model = build_model(in_channels=18, patch_size=64, d_model=64, nhead=2, num_layers=2)
x_batch = torch.stack([tiny_ds[i][0] for i in range(4)])
y_batch = torch.stack([tiny_ds[i][1] for i in range(4)])
logits = model(x_batch)
print(f"Batch input: {x_batch.shape}")
print(f"Batch output: {logits.shape}")
print()

# Example 2: SyntheticEEG for more realistic testing
print("=" * 60)
print("Example 2: SyntheticEEG - More realistic (16 samples)")
print("=" * 60)

synthetic_ds = SyntheticEEG(n=16, channels=18, length=256, seed=42)
print(f"Dataset size: {len(synthetic_ds)}")
x, y = synthetic_ds[0]
print(f"Sample shape: x={x.shape}, y={y.shape}")
print(f"Data stats: mean={x.mean():.4f}, std={x.std():.4f}")
print()

# Example 3: Full training loop with TinyEEG
print("=" * 60)
print("Example 3: Quick training loop (1 epoch)")
print("=" * 60)

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

train_ds = TinyEEG(n=12, channels=18, length=256)
val_ds = TinyEEG(n=4, channels=18, length=256, seed=99)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

model = build_model(in_channels=18, patch_size=64, d_model=64, nhead=2, num_layers=2)
optimizer = AdamW(model.parameters(), lr=1e-3)

# Train
model.train()
for x, y in train_loader:
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Train loss: {loss.item():.4f}")

# Validate
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        print(f"Val loss: {loss.item():.4f}, acc: {acc.item():.2%}")

print("\n" + "=" * 60)
print("All examples completed!")
print("=" * 60)
