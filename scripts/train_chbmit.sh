#!/bin/bash
# Quick start script for training with CHB-MIT dataset

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DATA_DIR="data/chbmit"
PATIENTS="chb01 chb02"
EPOCHS=10
BATCH_SIZE=16

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --patients)
            PATIENTS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/train_chbmit.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR       Data directory (default: data/chbmit)"
            echo "  --patients \"P1 P2\"  Patient IDs to download (default: \"chb01 chb02\")"
            echo "  --epochs N           Number of epochs (default: 10)"
            echo "  --batch-size N       Batch size (default: 16)"
            echo ""
            echo "Environment variables:"
            echo "  PHYSIONET_USER       PhysioNet username (required for download)"
            echo "  PHYSIONET_PASS       PhysioNet password (required for download)"
            echo ""
            echo "Example:"
            echo "  export PHYSIONET_USER=your_username"
            echo "  export PHYSIONET_PASS=your_password"
            echo "  bash scripts/train_chbmit.sh --patients \"chb01 chb02 chb03\" --epochs 20"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "================================================================"
echo "         CHB-MIT EEG Seizure Detection Training"
echo "================================================================"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Data directory not found. Will download CHB-MIT data...${NC}"
    echo ""

    # Check credentials
    if [ -z "$PHYSIONET_USER" ] || [ -z "$PHYSIONET_PASS" ]; then
        echo -e "${RED}Error: PhysioNet credentials not set!${NC}"
        echo ""
        echo "Please set your credentials:"
        echo "  export PHYSIONET_USER=your_username"
        echo "  export PHYSIONET_PASS=your_password"
        echo ""
        echo "Register at: https://physionet.org/register/"
        exit 1
    fi

    # Download data
    echo "Downloading CHB-MIT data for: $PATIENTS"
    bash scripts/download_chbmit.sh "$DATA_DIR" $PATIENTS

    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download data${NC}"
        exit 1
    fi
    echo ""
fi

# Create annotations CSV
ANNOTATIONS_CSV="$DATA_DIR/annotations.csv"

if [ ! -f "$ANNOTATIONS_CSV" ]; then
    echo -e "${YELLOW}Creating annotations CSV...${NC}"
    python3 scripts/create_annotations.py \
        --data-dir "$DATA_DIR" \
        --output "$ANNOTATIONS_CSV" \
        --patients $PATIENTS

    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create annotations${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${GREEN}✓ Annotations already exist: $ANNOTATIONS_CSV${NC}"
    echo ""
fi

# Check if annotations file has content
if [ ! -s "$ANNOTATIONS_CSV" ]; then
    echo -e "${RED}Error: Annotations file is empty!${NC}"
    exit 1
fi

# Count number of seizure events
NUM_SEIZURES=$(tail -n +2 "$ANNOTATIONS_CSV" | wc -l)
echo "Found $NUM_SEIZURES seizure events in annotations"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'Device count: {torch.cuda.device_count()}')"
echo ""

# Start training
echo "================================================================"
echo "                    Starting Training"
echo "================================================================"
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Annotations: $ANNOTATIONS_CSV"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Patients: $PATIENTS"
echo "  Device: CUDA (if available)"
echo "================================================================"
echo ""

python3 -m eeg_crit_transformer.train \
    --data-dir "$DATA_DIR" \
    --annotations "$ANNOTATIONS_CSV" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device cuda \
    --save-history \
    --workers 0

# Check training status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo "================================================================"
    echo ""
    echo "Outputs:"
    echo "  Best model: checkpoints/best.pt"
    echo "  Training history: checkpoints/training_history.json"
    echo ""
    echo "Visualize results:"
    echo "  python scripts/visualize_training.py"
else
    echo ""
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi
