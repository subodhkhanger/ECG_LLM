#!/usr/bin/env bash
set -euo pipefail

# Download CHB-MIT summary files only
# Usage:
#   export PHYSIONET_USER=your_user
#   export PHYSIONET_PASS=your_pass
#   bash scripts/download_summaries.sh

if [[ -z "${PHYSIONET_USER:-}" || -z "${PHYSIONET_PASS:-}" ]]; then
  echo "Please export PHYSIONET_USER and PHYSIONET_PASS" >&2
  exit 1
fi

BASE="https://physionet.org/files/chbmit/1.0.0"
ROOT="data/chbmit"

cd "$(dirname "$0")/.."  # Go to project root

echo "Downloading summary files for CHB-MIT dataset..."
echo ""

for patient in chb01 chb02 chb03 chb04; do
  if [ ! -d "$ROOT/$patient" ]; then
    echo "Warning: Directory $ROOT/$patient not found, skipping..."
    continue
  fi

  summary_file="${patient}-summary.txt"
  echo "Downloading $patient/$summary_file ..."

  curl -fL --user "$PHYSIONET_USER:$PHYSIONET_PASS" \
    -o "$ROOT/$patient/$summary_file" \
    "$BASE/$patient/$summary_file"

  if [ $? -eq 0 ]; then
    echo "  ✓ Downloaded"
  else
    echo "  ✗ Failed"
  fi
done

echo ""
echo "Done! You can now run:"
echo "  python scripts/create_annotations.py --data-dir data/chbmit"
