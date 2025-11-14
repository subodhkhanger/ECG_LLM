#!/usr/bin/env bash
set -euo pipefail

# CHB-MIT download helper (PhysioNet)
# Usage:
#   export PHYSIONET_USER=your_user
#   export PHYSIONET_PASS=your_pass
#   bash scripts/download_chbmit.sh data/chbmit chb01 chb02

ROOT=${1:-data/chbmit}
shift || true
SUBJECTS=("$@")

if [[ -z "${PHYSIONET_USER:-}" || -z "${PHYSIONET_PASS:-}" ]]; then
  echo "Please export PHYSIONET_USER and PHYSIONET_PASS" >&2
  exit 1
fi

BASE="https://physionet.org/files/chbmit/1.0.0"

mkdir -p "$ROOT"
cd "$ROOT"

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
  SUBJECTS=(chb01)
fi

for s in "${SUBJECTS[@]}"; do
  mkdir -p "$s"
  echo "Downloading $s ..."
  # Fetch directory listing and selectively download EDF files
  # Note: Access may prompt for authentication; we use curl with basic auth.
  curl -fL --user "$PHYSIONET_USER:$PHYSIONET_PASS" "$BASE/$s/" | \
    grep -Eo 'href=\"[^\"]+\.edf\"' | sed -E 's/href=\"(.*)\"/\1/' | while read -r f; do
      if [[ ! -f "$s/$f" ]]; then
        echo "  -> $f"
        curl -fL --user "$PHYSIONET_USER:$PHYSIONET_PASS" -o "$s/$f" "$BASE/$s/$f"
      fi
    done
done

echo "Done. Place annotations at $ROOT/annotations.csv (see README)."

