# Quick Start Guide - CHB-MIT Training

This guide shows you how to train on the real CHB-MIT EEG dataset.

## Prerequisites

1. **GPU with CUDA** (Recommended for fast training)
   - Check GPU availability:
     ```bash
     python scripts/check_gpu.py
     ```
   - If no GPU available, training will use CPU (much slower)
   - To install PyTorch with CUDA: https://pytorch.org/get-started/locally/

2. **PhysioNet Account**: Register at https://physionet.org/register/

3. **Python Environment**: Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Option 1: Automatic (Recommended)

Use the all-in-one script that downloads data, creates annotations, and trains:

```bash
# Set your PhysioNet credentials
export PHYSIONET_USER="your_username"
export PHYSIONET_PASS="your_password"

# Run the complete pipeline
bash scripts/train_chbmit.sh
```

That's it! This will:
- Download CHB-MIT data (chb01, chb02 by default)
- Create annotations CSV
- Start training with monitoring
- Save best model and training history

### Customize the automatic script:

```bash
# Train on more patients
bash scripts/train_chbmit.sh --patients "chb01 chb02 chb03 chb04"

# Run for more epochs
bash scripts/train_chbmit.sh --epochs 20 --batch-size 32

# Use custom data directory
bash scripts/train_chbmit.sh --data-dir /path/to/data
```

## Option 2: Step-by-Step

### Step 1: Download CHB-MIT Data

```bash
export PHYSIONET_USER="your_username"
export PHYSIONET_PASS="your_password"

# Download specific patients (start small!)
bash scripts/download_chbmit.sh data/chbmit chb01 chb02
```

**Note**: The full dataset is ~50GB. Start with 2-3 patients (~4GB) to test.

### Step 2: Create Annotations CSV

```bash
# Create annotations from downloaded data
python scripts/create_annotations.py \
    --data-dir data/chbmit \
    --output data/chbmit/annotations.csv \
    --patients chb01 chb02
```

This parses the CHB-MIT summary files and creates a CSV with seizure times.

**Output format** (`data/chbmit/annotations.csv`):
```csv
file,start,end,label
chb01/chb01_03.edf,2996,3036,1
chb01/chb01_04.edf,1467,1494,1
chb02/chb02_16.edf,2972,3053,1
```

### Step 3: Start Training

```bash
python -m eeg_crit_transformer.train \
    --data-dir data/chbmit \
    --annotations data/chbmit/annotations.csv \
    --epochs 10 \
    --batch-size 16 \
    --save-history \
    --workers 0
```

### Step 4: Visualize Results

```bash
python scripts/visualize_training.py
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | None | Path to CHB-MIT directory |
| `--annotations` | None | Path to annotations CSV |
| `--epochs` | 5 | Number of training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--weight-decay` | 1e-2 | Weight decay |
| `--window-sec` | 5.0 | EEG window length (seconds) |
| `--sample-rate` | 256 | Target sample rate (Hz) |
| `--patch-size` | 64 | Transformer patch size |
| `--d-model` | 128 | Model dimension |
| `--nhead` | 4 | Number of attention heads |
| `--num-layers` | 4 | Number of transformer layers |
| `--dropout` | 0.1 | Dropout rate |
| `--save-history` | flag | Save training history to JSON |
| `--workers` | 2 | DataLoader workers (use 0 if issues) |
| `--max-files` | None | Limit number of files (for testing) |

## Example Training Commands

### Quick Test (2 patients, 5 epochs)
```bash
python -m eeg_crit_transformer.train \
    --data-dir data/chbmit \
    --annotations data/chbmit/annotations.csv \
    --epochs 5 \
    --batch-size 8 \
    --max-files 10 \
    --save-history
```

### Full Training (Multiple patients)
```bash
python -m eeg_crit_transformer.train \
    --data-dir data/chbmit \
    --annotations data/chbmit/annotations.csv \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --save-history
```

### Large Model Configuration
```bash
python -m eeg_crit_transformer.train \
    --data-dir data/chbmit \
    --annotations data/chbmit/annotations.csv \
    --epochs 30 \
    --batch-size 16 \
    --d-model 256 \
    --nhead 8 \
    --num-layers 6 \
    --ff 512 \
    --save-history
```

## Expected Output

During training you'll see:

```
================================================================================
                          Training Configuration
================================================================================
  Device              : gpu
  Total Parameters    : 234,562
  Trainable Parameters: 234,562
  Input Channels      : 23
  Epochs              : 10
  Batch Size          : 16
  ...

Epoch 1/10
================================================================================
                         Training Metrics - Epoch 1
================================================================================
  loss                : 0.693147
  acc                 : 0.500000

  Gradient Norms      :
    grad_norm_max     : 12.345678
    grad_norm_mean    : 2.345678
    ...

                                Summary
  Train: loss=0.6931 acc=0.5000 grad_norm=2.3457
  Val:   loss=0.6931 acc=0.5000 best=0.5000

                         New best model saved! (val_acc: 0.5000)
```

## Output Files

After training:

```
checkpoints/
├── best.pt                      # Best model checkpoint
└── training_history.json        # All metrics per epoch
```

**Best checkpoint contains**:
- Model state dict
- Optimizer state
- Model configuration
- Training and validation metrics
- Best validation accuracy

## Troubleshooting

### GPU / CUDA Issues (Most Important!)

#### Check GPU status first
```bash
python scripts/check_gpu.py
```

#### Training on CPU instead of GPU
**This is the #1 performance issue!**

Force GPU usage:
```bash
python -m eeg_crit_transformer.train --device cuda ...
```

Or use the automatic script (already includes `--device cuda`):
```bash
bash scripts/train_chbmit.sh
```

#### GPU not detected
1. **Install CUDA drivers** from NVIDIA
2. **Install PyTorch with CUDA**:
   ```bash
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Verify**: `python -c "import torch; print(torch.cuda.is_available())"`

#### CUDA out of memory
- Reduce `--batch-size` (try 8, 4, or 2)
- Reduce model size (`--d-model 64 --num-layers 2`)
- Use specific GPU: `CUDA_VISIBLE_DEVICES=0 python -m eeg_crit_transformer.train ...`

### Python Environment Issues

#### "No module named 'numpy'"
```bash
pip install -r requirements.txt
```

#### "mne is required to read EDF files"
```bash
pip install mne
```

### Data Loading Issues

#### Slow data loading
- Set `--workers 0` (disables multiprocessing)
- Use fewer patients initially

#### Out of memory (RAM)
- Reduce `--workers` to 0
- Use `--max-files` to limit dataset size

### Training Issues

#### Low accuracy
- **Ensure GPU is being used** (CPU training converges slowly!)
- Train for more epochs (`--epochs 50`)
- Increase model capacity (`--d-model 256 --num-layers 6`)
- Tune learning rate (`--lr 1e-4`)
- Check gradient norms in monitoring output

#### Training very slow
- **Most common**: Training on CPU instead of GPU
  - Check: `python scripts/check_gpu.py`
  - Fix: Add `--device cuda` or use `scripts/train_chbmit.sh`
- Use larger batch size if GPU memory allows (`--batch-size 32`)

## Next Steps

After training:

1. **Visualize metrics**:
   ```bash
   python scripts/visualize_training.py
   ```

2. **Load and use the model**:
   ```python
   import torch
   from eeg_crit_transformer.models.crit_transformer import CritTransformer

   checkpoint = torch.load('checkpoints/best.pt')
   config = checkpoint['config']

   model = CritTransformer(config)
   model.load_state_dict(checkpoint['model'])
   model.eval()
   ```

3. **Experiment with hyperparameters** - See [TRAINING_MONITORING.md](TRAINING_MONITORING.md)

## Testing Without Real Data

If you don't have PhysioNet access yet, test with synthetic data:

```bash
# Uses synthetic EEG data automatically
python -m eeg_crit_transformer.train \
    --epochs 3 \
    --batch-size 16 \
    --save-history
```

This is perfect for validating your setup before downloading real data.

## See Also

- [README.md](../README.md) - Main documentation
- [TRAINING_MONITORING.md](TRAINING_MONITORING.md) - Detailed monitoring guide
- [DATASET_ORGANIZATION.md](DATASET_ORGANIZATION.md) - Dataset structure
