#!/usr/bin/env python3
"""Visualize training history from saved JSON file."""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def load_history(path: str):
    """Load training history from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_training_history(history, save_path=None):
    """Create comprehensive training visualization."""
    train_metrics = history['train']
    val_metrics = history['val']

    epochs = [m['epoch'] for m in train_metrics]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Training Monitoring Dashboard', fontsize=16, fontweight='bold')

    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, [m['loss'] for m in train_metrics], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, [m['acc'] for m in train_metrics], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [m['acc'] for m in val_metrics], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Gradient norms
    ax = axes[0, 2]
    ax.plot(epochs, [m['grad_norm_mean'] for m in train_metrics], 'g-', label='Mean', linewidth=2)
    ax.plot(epochs, [m['grad_norm_max'] for m in train_metrics], 'r--', label='Max', linewidth=1, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 4. Per-class accuracy (train)
    ax = axes[1, 0]
    class_keys = [k for k in train_metrics[0].keys() if k.startswith('class_') and k.endswith('_acc')]
    for key in class_keys:
        ax.plot(epochs, [m.get(key, 0) for m in train_metrics], marker='o', label=key, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy (Train)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Confidence statistics (validation)
    ax = axes[1, 1]
    ax.plot(epochs, [m['confidence_mean'] for m in val_metrics], 'b-', label='Mean', linewidth=2)
    ax.fill_between(epochs,
                     [m['confidence_mean'] - m['confidence_std'] for m in val_metrics],
                     [m['confidence_mean'] + m['confidence_std'] for m in val_metrics],
                     alpha=0.3, label='±1 std')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Confidence')
    ax.set_title('Model Confidence (Val)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Precision/Recall (validation)
    ax = axes[1, 2]
    for cls in [0, 1]:
        prec_key = f'class_{cls}_precision'
        rec_key = f'class_{cls}_recall'
        if prec_key in val_metrics[0]:
            ax.plot(epochs, [m.get(prec_key, 0) for m in val_metrics],
                   marker='o', label=f'Class {cls} Precision', linewidth=2)
        if rec_key in val_metrics[0]:
            ax.plot(epochs, [m.get(rec_key, 0) for m in val_metrics],
                   marker='s', linestyle='--', label=f'Class {cls} Recall', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall (Val)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7. Confusion matrix evolution (validation)
    ax = axes[2, 0]
    if 'true_pos' in val_metrics[0]:
        width = 0.2
        x = np.array(epochs)
        ax.bar(x - 1.5*width, [m.get('true_neg', 0) for m in val_metrics], width, label='TN', alpha=0.8)
        ax.bar(x - 0.5*width, [m.get('false_pos', 0) for m in val_metrics], width, label='FP', alpha=0.8)
        ax.bar(x + 0.5*width, [m.get('false_neg', 0) for m in val_metrics], width, label='FN', alpha=0.8)
        ax.bar(x + 1.5*width, [m.get('true_pos', 0) for m in val_metrics], width, label='TP', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Count')
        ax.set_title('Confusion Matrix Evolution (Val)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 8. Weight statistics
    ax = axes[2, 1]
    weight_keys = [k for k in train_metrics[0].keys() if k.startswith('weight_') and k.endswith('_after')]
    for key in weight_keys:
        display_name = key.replace('weight_', '').replace('_after', '')
        ax.plot(epochs, [m.get(key, 0) for m in train_metrics], marker='o', label=display_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Weight Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Learning rate and weight change
    ax = axes[2, 2]
    ax_twin = ax.twinx()
    ax.plot(epochs, [m['lr'] for m in train_metrics], 'b-', marker='o', label='Learning Rate', linewidth=2)
    ax_twin.plot(epochs, [m['weight_change'] for m in train_metrics], 'r-', marker='s', label='Weight Change', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate', color='b')
    ax_twin.set_ylabel('Weight Change', color='r')
    ax.set_title('LR & Weight Change')
    ax.tick_params(axis='y', labelcolor='b')
    ax_twin.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def print_summary(history):
    """Print summary statistics."""
    train_metrics = history['train']
    val_metrics = history['val']

    print("\n" + "="*80)
    print("Training Summary".center(80))
    print("="*80)

    print(f"\nFinal Epoch Metrics:")
    print(f"  Train Loss: {train_metrics[-1]['loss']:.6f}")
    print(f"  Train Acc:  {train_metrics[-1]['acc']:.6f}")
    print(f"  Val Loss:   {val_metrics[-1]['loss']:.6f}")
    print(f"  Val Acc:    {val_metrics[-1]['acc']:.6f}")

    best_val_idx = max(range(len(val_metrics)), key=lambda i: val_metrics[i]['acc'])
    print(f"\nBest Validation Performance (Epoch {best_val_idx + 1}):")
    print(f"  Val Loss: {val_metrics[best_val_idx]['loss']:.6f}")
    print(f"  Val Acc:  {val_metrics[best_val_idx]['acc']:.6f}")

    print(f"\nGradient Norm Statistics:")
    print(f"  Mean:   {train_metrics[-1]['grad_norm_mean']:.6f}")
    print(f"  Max:    {train_metrics[-1]['grad_norm_max']:.6f}")

    if 'true_pos' in val_metrics[-1]:
        print(f"\nFinal Confusion Matrix (Validation):")
        print(f"  TN: {val_metrics[-1]['true_neg']:4d}  FP: {val_metrics[-1]['false_pos']:4d}")
        print(f"  FN: {val_metrics[-1]['false_neg']:4d}  TP: {val_metrics[-1]['true_pos']:4d}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Visualize training history")
    parser.add_argument("--history", type=str, default="checkpoints/training_history.json",
                       help="Path to training history JSON file")
    parser.add_argument("--save", type=str, default=None,
                       help="Save visualization to file instead of showing")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only print summary without visualization")
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        print(f"Error: History file not found: {history_path}")
        print("Run training with --save-history flag first")
        sys.exit(1)

    print(f"Loading training history from: {history_path}")
    history = load_history(history_path)

    print_summary(history)

    if not args.summary_only:
        plot_training_history(history, save_path=args.save)


if __name__ == "__main__":
    main()
