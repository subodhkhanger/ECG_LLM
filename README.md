EEG Criticality Transformer (project scaffold)

Overview
- This repo scaffolds a minimal, reproducible pipeline for seizure detection on the CHB-MIT Scalp EEG Database using a custom Transformer architecture.
- It includes: a lightweight Transformer for multichannel EEG, a CHB-MIT dataset loader (EDF + CSV annotations), a training script with comprehensive monitoring, and a download helper that you can use with your PhysioNet credentials.

What you can run right now
- With no data present, the training script falls back to a synthetic dataset so you can validate the environment and pipeline quickly.
- Comprehensive training monitoring including gradient norms, per-class metrics, confidence statistics, and confusion matrices.
- Multiple test datasets: TinyEEG (8 samples, ultra-fast) and SyntheticEEG (256 samples, realistic).

Quick start

**Test with synthetic data (no download needed):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Check GPU availability (training will use GPU automatically if available)
python scripts/check_gpu.py

# Train
python -m eeg_crit_transformer.train --epochs 3 --batch-size 8
```

**Train on real CHB-MIT data:**
```bash
# Set PhysioNet credentials
export PHYSIONET_USER="your_username"
export PHYSIONET_PASS="your_password"

# One-command training (downloads data, creates annotations, trains)
bash scripts/train_chbmit.sh
```

**Or step-by-step:**
```bash
# 1. Download data
bash scripts/download_chbmit.sh data/chbmit chb01 chb02

# 2. Create annotations CSV
python scripts/create_annotations.py --data-dir data/chbmit

# 3. Train
python -m eeg_crit_transformer.train \
  --data-dir data/chbmit \
  --annotations data/chbmit/annotations.csv \
  --epochs 10 \
  --save-history

# 4. Visualize
python scripts/visualize_training.py
```

📖 **See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed instructions**

Training monitoring
- The training script automatically tracks 20+ metrics per epoch:
  - Gradient norms (mean, std, max) for detecting vanishing/exploding gradients
  - Per-class accuracy, precision, and recall
  - Model confidence statistics
  - Confusion matrix (binary classification)
  - Weight statistics and changes
  - Batch loss variance (training stability)
- Use `--save-history` to save metrics to JSON for later analysis
- See [docs/TRAINING_MONITORING.md](docs/TRAINING_MONITORING.md) for detailed documentation

Project layout
- `requirements.txt` – dependencies
- `src/eeg_crit_transformer/models/crit_transformer.py` – EEG Transformer model
- `src/eeg_crit_transformer/data/chbmit.py` – CHB-MIT dataset + windowing
- `src/eeg_crit_transformer/data/test_datasets.py` – TinyEEG/SyntheticEEG for testing
- `src/eeg_crit_transformer/train.py` – training loop with comprehensive monitoring
- `scripts/download_chbmit.sh` – download CHB-MIT data from PhysioNet
- `scripts/create_annotations.py` – generate annotations CSV from CHB-MIT summaries
- `scripts/train_chbmit.sh` – one-command training pipeline
- `scripts/visualize_training.py` – training metrics visualization tool
- `test_architecture.py` – quick architecture validation tests
- `examples/quick_test.py` – examples using test datasets
- `docs/QUICKSTART.md` – complete training guide
- `docs/TRAINING_MONITORING.md` – detailed monitoring documentation
- `docs/DATASET_ORGANIZATION.md` – dataset structure reference

Notes
- CHB-MIT is hosted on PhysioNet and may require login/credential prompts. Do not commit credentials.
- The provided model is a clean, minimal baseline to get you training quickly; refine hyperparameters and preprocessing as needed for your use case.

