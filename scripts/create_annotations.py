#!/usr/bin/env python3
"""
Parse CHB-MIT dataset and create annotations CSV for training.

The CHB-MIT dataset includes RECORDS-WITH-SEIZURES files that list seizure events.
This script parses those and creates the required CSV format.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


def parse_summary_file(summary_path: str) -> List[Dict[str, any]]:
    """Parse a CHB-MIT patient summary file to extract seizure information.

    Args:
        summary_path: Path to the summary file (e.g., chb01-summary.txt)

    Returns:
        List of dictionaries with file, start, end, label information
    """
    annotations = []

    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return annotations

    with open(summary_path, 'r') as f:
        content = f.read()

    # Parse file records
    # Format example:
    # File Name: chb01_03.edf
    # ...
    # Number of Seizures in File: 1
    # Seizure Start Time: 2996 seconds
    # Seizure End Time: 3036 seconds

    file_blocks = re.split(r'File Name:', content)[1:]  # Skip header

    for block in file_blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue

        # Extract filename
        filename_match = re.search(r'(\S+\.edf)', lines[0])
        if not filename_match:
            continue
        filename = filename_match.group(1)

        # Extract number of seizures
        num_seizures = 0
        for line in lines:
            if 'Number of Seizures' in line:
                match = re.search(r'(\d+)', line)
                if match:
                    num_seizures = int(match.group(1))
                break

        if num_seizures == 0:
            continue

        # Extract seizure times
        seizure_starts = []
        seizure_ends = []

        for line in lines:
            if 'Seizure Start Time' in line or 'Seizure 1 Start Time' in line:
                match = re.search(r'(\d+)\s*seconds', line)
                if match:
                    seizure_starts.append(int(match.group(1)))
            elif 'Seizure End Time' in line or 'Seizure 1 End Time' in line:
                match = re.search(r'(\d+)\s*seconds', line)
                if match:
                    seizure_ends.append(int(match.group(1)))
            # Handle multiple seizures
            elif 'Seizure 2 Start Time' in line:
                match = re.search(r'(\d+)\s*seconds', line)
                if match:
                    seizure_starts.append(int(match.group(1)))
            elif 'Seizure 2 End Time' in line:
                match = re.search(r'(\d+)\s*seconds', line)
                if match:
                    seizure_ends.append(int(match.group(1)))

        # Create annotations for each seizure
        for start, end in zip(seizure_starts, seizure_ends):
            annotations.append({
                'file': filename,
                'start': start,
                'end': end,
                'label': 1
            })

    return annotations


def create_annotations_csv(data_dir: str, output_csv: str, patients: List[str] = None):
    """Create annotations CSV from CHB-MIT dataset.

    Args:
        data_dir: Root directory of CHB-MIT dataset
        output_csv: Output CSV file path
        patients: List of patient IDs to process (e.g., ['chb01', 'chb02']).
                 If None, processes all patients.
    """
    data_path = Path(data_dir)
    all_annotations = []

    # If no patients specified, scan all patient directories
    if patients is None:
        patients = []
        for item in data_path.iterdir():
            if item.is_dir() and item.name.startswith('chb'):
                patients.append(item.name)
        patients.sort()

    print(f"Processing {len(patients)} patients: {', '.join(patients)}")

    for patient_id in patients:
        patient_dir = data_path / patient_id

        if not patient_dir.exists():
            print(f"Warning: Patient directory not found: {patient_dir}")
            continue

        # Look for summary file
        summary_file = patient_dir / f"{patient_id}-summary.txt"

        if not summary_file.exists():
            print(f"Warning: No summary file for {patient_id}")
            continue

        print(f"  Parsing {patient_id}...")
        patient_annotations = parse_summary_file(str(summary_file))

        # Add patient directory to file paths
        for ann in patient_annotations:
            ann['file'] = f"{patient_id}/{ann['file']}"

        all_annotations.extend(patient_annotations)
        print(f"    Found {len(patient_annotations)} seizure events")

    # Write CSV
    if all_annotations:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv, 'w') as f:
            f.write("file,start,end,label\n")
            for ann in all_annotations:
                f.write(f"{ann['file']},{ann['start']},{ann['end']},{ann['label']}\n")

        print(f"\n✓ Created annotations CSV: {output_csv}")
        print(f"  Total seizure events: {len(all_annotations)}")
    else:
        print("\n✗ No annotations found!")


def main():
    parser = argparse.ArgumentParser(
        description="Create annotations CSV from CHB-MIT dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all patients in the dataset
  python scripts/create_annotations.py --data-dir data/chbmit

  # Process specific patients
  python scripts/create_annotations.py --data-dir data/chbmit --patients chb01 chb02 chb03

  # Specify custom output location
  python scripts/create_annotations.py --data-dir data/chbmit --output my_annotations.csv
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to CHB-MIT root directory (contains chb01/, chb02/, etc.)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: <data-dir>/annotations.csv)'
    )

    parser.add_argument(
        '--patients',
        nargs='+',
        default=None,
        help='Specific patient IDs to process (e.g., chb01 chb02). If not provided, processes all.'
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        args.output = os.path.join(args.data_dir, 'annotations.csv')

    # Validate data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        print("\nPlease download the CHB-MIT dataset first:")
        print("  bash scripts/download_chbmit.sh data/chbmit chb01 chb02")
        return 1

    print("=" * 60)
    print("CHB-MIT Annotations CSV Generator")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output CSV: {args.output}")
    print()

    create_annotations_csv(args.data_dir, args.output, args.patients)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("Train with the generated annotations:")
    print(f"  python -m eeg_crit_transformer.train \\")
    print(f"    --data-dir {args.data_dir} \\")
    print(f"    --annotations {args.output} \\")
    print(f"    --epochs 10 \\")
    print(f"    --batch-size 16 \\")
    print(f"    --save-history")

    return 0


if __name__ == '__main__':
    exit(main())
