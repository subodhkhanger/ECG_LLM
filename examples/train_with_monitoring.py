#!/usr/bin/env python3
"""Demo: Training with full monitoring enabled."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Run training with synthetic data and monitoring
if __name__ == "__main__":
    import subprocess

    cmd = [
        "python3", "-m", "src.eeg_crit_transformer.train",
        "--epochs", "3",
        "--batch-size", "8",
        "--lr", "1e-3",
        "--d-model", "64",
        "--nhead", "2",
        "--num-layers", "2",
        "--ff", "128",
        "--channels", "18",
        "--save-history",
        "--workers", "0",  # Avoid multiprocessing issues
    ]

    print("Running training with full monitoring...")
    print(f"Command: {' '.join(cmd)}")
    print()

    subprocess.run(cmd)
