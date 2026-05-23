"""
Download the PanNuke dataset from the official Warwick University source.

PanNuke is split into 3 folds, each containing:
  - images/fold{i}/images.npy  (N, 256, 256, 3) uint8
  - images/fold{i}/types.npy   (N,) tissue type strings
  - masks/fold{i}/masks.npy    (N, 256, 256, 6) uint16 — 5 nuclei classes + background

Usage:
    python src/download_data.py

Downloads go to data/raw/fold{1,2,3}/
"""

import os
import zipfile
import urllib.request
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

FOLD_URLS = {
    "fold_1": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip",
    "fold_2": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip",
    "fold_3": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip",
}


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    print(f"Downloading {url}")
    print(f"  -> {dest_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct}% ({mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print()


def extract_zip(zip_path, extract_to):
    """Extract a zip file and remove it afterward."""
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)
    print(f"  Removed {zip_path}")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    for fold_name, url in FOLD_URLS.items():
        fold_dir = os.path.join(RAW_DIR, fold_name)
        if os.path.isdir(fold_dir):
            print(f"Skipping {fold_name} (already exists at {fold_dir})")
            continue

        zip_path = os.path.join(RAW_DIR, f"{fold_name}.zip")
        download_file(url, zip_path)
        extract_zip(zip_path, RAW_DIR)

    print("\nDone. Raw data is in:", os.path.abspath(RAW_DIR))
    print("Structure:")
    print("  data/raw/fold_{1,2,3}/images/fold{i}/images.npy")
    print("  data/raw/fold_{1,2,3}/images/fold{i}/types.npy")
    print("  data/raw/fold_{1,2,3}/masks/fold{i}/masks.npy")


if __name__ == "__main__":
    main()
