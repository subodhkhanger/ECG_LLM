EEG Criticality Transformer (project scaffold)

Overview
- This repo scaffolds a minimal, reproducible pipeline for seizure detection on the CHB-MIT Scalp EEG Database using a custom Transformer architecture.
- It includes: a lightweight Transformer for multichannel EEG, a CHB-MIT dataset loader (EDF + CSV annotations), a training script with comprehensive monitoring, and a download helper that you can use with your PhysioNet credentials.

What you can run right now
- With no data present, the training script falls back to a synthetic dataset so you can validate the environment and pipeline quickly.
- Comprehensive training monitoring including gradient norms, per-class metrics, confidence statistics, and confusion matrices.
- Multiple test datasets: TinyEEG (8 samples, ultra-fast) and SyntheticEEG (256 samples, realistic).

Quick start
1) Create and activate an environment (Python 3.10+ recommended):
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`

2) Fast dry-run on synthetic data:
   - `python -m eeg_crit_transformer.train --epochs 1 --batch-size 8`

3) Download CHB-MIT (requires PhysioNet account):
   - Export your PhysioNet credentials, then run the helper script. This dataset is large; download selectively first.
   - `export PHYSIONET_USER="your_username"`
   - `export PHYSIONET_PASS="your_password"`
   - `bash scripts/download_chbmit.sh data/chbmit chb01 chb02`

4) Prepare annotations
- Place a CSV at `data/chbmit/annotations.csv` with columns:
  - `file` (EDF filename relative to the dataset root)
  - `start` (seconds from start of that EDF file)
  - `end` (seconds)
  - `label` (0 or 1; 1 indicates seizure segment)

Example row:
```
chb01/chb01_03.edf,2996,3036,1
```
You can list multiple seizure intervals per file. Non-seizure windows are sampled automatically from the remaining time.

5) Train on CHB-MIT
```
python -m eeg_crit_transformer.train \
  --data-dir data/chbmit \
  --epochs 10 \
  --batch-size 16 \
  --window-sec 5 \
  --sample-rate 256 \
  --save-history
```

6) Visualize training metrics
```
python scripts/visualize_training.py
```

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
- `scripts/download_chbmit.sh` – dataset downloader using PhysioNet credentials
- `scripts/visualize_training.py` – training metrics visualization tool
- `test_architecture.py` – quick architecture validation tests
- `examples/quick_test.py` – examples using test datasets
- `docs/TRAINING_MONITORING.md` – detailed monitoring documentation

Notes
- CHB-MIT is hosted on PhysioNet and may require login/credential prompts. Do not commit credentials.
- The provided model is a clean, minimal baseline to get you training quickly; refine hyperparameters and preprocessing as needed for your use case.

