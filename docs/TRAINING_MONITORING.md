# Training Monitoring Guide

This guide explains the comprehensive monitoring features added to the training pipeline.

## Overview

The training script now tracks extensive metrics at each epoch to help you understand your model's training dynamics:

- **Gradient norms** - Monitor gradient flow and detect vanishing/exploding gradients
- **Per-class metrics** - Track accuracy, precision, and recall for each class
- **Confidence statistics** - Monitor model confidence and calibration
- **Weight statistics** - Track weight changes throughout training
- **Confusion matrix** - See classification errors over time
- **Loss variance** - Detect training instability

## Monitored Metrics

### Training Metrics

| Metric | Description |
|--------|-------------|
| `loss` | Cross-entropy loss |
| `acc` | Overall accuracy |
| `grad_norm_mean` | Mean gradient norm across all parameters |
| `grad_norm_std` | Standard deviation of gradient norms |
| `grad_norm_max` | Maximum gradient norm |
| `batch_loss_std` | Variance in batch losses (training stability) |
| `class_X_acc` | Accuracy for class X |
| `confidence_mean` | Average prediction confidence |
| `confidence_std` | Variance in prediction confidence |
| `weight_mean_after` | Mean weight value after update |
| `weight_std_after` | Weight standard deviation after update |
| `weight_max_after` | Maximum absolute weight after update |
| `weight_change` | Change in mean weight value |
| `lr` | Current learning rate |

### Validation Metrics

| Metric | Description |
|--------|-------------|
| `loss` | Cross-entropy loss |
| `acc` | Overall accuracy |
| `class_X_acc` | Accuracy for class X |
| `class_X_precision` | Precision for class X |
| `class_X_recall` | Recall for class X |
| `confidence_mean` | Average prediction confidence |
| `confidence_std` | Variance in prediction confidence |
| `true_neg` | True negatives (binary) |
| `false_pos` | False positives (binary) |
| `false_neg` | False negatives (binary) |
| `true_pos` | True positives (binary) |

## Usage

### Basic Training with Monitoring

```bash
python -m src.eeg_crit_transformer.train \
    --epochs 10 \
    --batch-size 16 \
    --save-history
```

The monitoring is automatic - all metrics are computed and displayed at each epoch.

### Save Training History

Add `--save-history` flag to save metrics to JSON:

```bash
python -m src.eeg_crit_transformer.train \
    --epochs 10 \
    --save-history
```

This saves to `checkpoints/training_history.json`.

### Quick Test with Tiny Dataset

```bash
python -m src.eeg_crit_transformer.train \
    --epochs 3 \
    --batch-size 8 \
    --d-model 64 \
    --nhead 2 \
    --num-layers 2 \
    --save-history
```

### Using the Demo Script

```bash
python examples/train_with_monitoring.py
```

## Visualizing Results

### View Training Curves

After training with `--save-history`:

```bash
python scripts/visualize_training.py
```

This creates a comprehensive dashboard with:
- Loss and accuracy curves
- Gradient norm evolution
- Per-class metrics
- Confidence statistics
- Confusion matrix evolution
- Weight statistics
- Learning rate schedule

### Save Visualization

```bash
python scripts/visualize_training.py --save training_plots.png
```

### Print Summary Only

```bash
python scripts/visualize_training.py --summary-only
```

## Interpreting Metrics

### Gradient Norms

**Good signs:**
- Gradient norms stay relatively stable (1e-3 to 1e1)
- Gradual decrease over time

**Warning signs:**
- Very large norms (>100): May indicate exploding gradients
- Very small norms (<1e-6): May indicate vanishing gradients
- Sudden spikes: Training instability

### Confidence Statistics

**Good signs:**
- `confidence_mean` increases over epochs
- High confidence on correct predictions
- Lower `confidence_std` on validation

**Warning signs:**
- High confidence on wrong predictions (overconfidence)
- Very low confidence throughout training (underconfident model)

### Weight Changes

**Good signs:**
- Weights change significantly in early epochs
- Changes decrease and stabilize in later epochs

**Warning signs:**
- Weights barely change: Learning rate too low
- Weights change erratically: Learning rate too high or instability

### Per-Class Metrics

**Good signs:**
- Similar accuracy across classes (balanced learning)
- High precision AND recall

**Warning signs:**
- One class dominates (high acc, low recall on other class)
- Large gap between precision and recall

## Example Output

```
================================================================================
                         Training Metrics - Epoch 1
================================================================================
  loss                : 0.693147
  acc                 : 0.500000

  Gradient Norms      :
    grad_norm_max     : 12.345678
    grad_norm_mean    : 2.345678
    grad_norm_std     : 1.234567

  Per-Class Metrics   :
    class_0_acc       : 0.480000
    class_1_acc       : 0.520000

  Confidence Stats    :
    confidence_max    : 0.876543
    confidence_mean   : 0.543210
    confidence_min    : 0.234567
    confidence_std    : 0.123456

  Other Metrics       :
    batch_loss_std    : 0.012345
    epoch             : 1
    lr                : 0.000300
    weight_change     : 0.001234
```

## Advanced Usage

### Custom Metrics

To add custom metrics, modify the `train_one_epoch` and `evaluate` functions in [train.py](../src/eeg_crit_transformer/train.py):

```python
def train_one_epoch(model, loader, opt, device):
    # ... existing code ...

    # Add your custom metric
    custom_metric = compute_my_metric(all_preds, all_labels)

    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        # ... existing metrics ...
        "my_custom_metric": custom_metric,  # Add here
    }
```

### Export Metrics for TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(epochs):
    train_metrics = train_one_epoch(...)

    for key, value in train_metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f'train/{key}', value, epoch)

writer.close()
```

## Troubleshooting

### High Memory Usage

The monitoring stores predictions and labels for all batches. For very large datasets:

1. Reduce batch size
2. Reduce number of workers
3. Use gradient checkpointing (requires model modification)

### Slow Training

Monitoring adds ~5-10% overhead. To speed up:

1. Use `--workers 0` for small datasets (avoids multiprocessing overhead)
2. Comment out unused metrics in the code
3. Disable history saving if not needed

## Files Reference

| File | Purpose |
|------|---------|
| [train.py](../src/eeg_crit_transformer/train.py) | Main training script with monitoring |
| [visualize_training.py](../scripts/visualize_training.py) | Visualization script |
| [train_with_monitoring.py](../examples/train_with_monitoring.py) | Quick demo |
| `checkpoints/best.pt` | Best model checkpoint |
| `checkpoints/training_history.json` | Saved metrics history |

## See Also

- [Main README](../README.md)
- [Model Architecture](../src/eeg_crit_transformer/models/crit_transformer.py)
- [Dataset Documentation](../src/eeg_crit_transformer/data/chbmit.py)
