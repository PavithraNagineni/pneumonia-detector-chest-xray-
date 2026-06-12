"""
download_data.py — Download Chest X-Ray Dataset from Kaggle
============================================================
Requires Kaggle API credentials: ~/.kaggle/kaggle.json

Usage:
    pip install kaggle
    python download_data.py
"""

import os
import subprocess
import zipfile

DATASET = "paultimothymooney/chest-xray-pneumonia"
DATA_DIR = "./data"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading dataset: {DATASET}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR],
        check=True,
    )
    zip_path = os.path.join(DATA_DIR, "chest-xray-pneumonia.zip")
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"Dataset ready at {DATA_DIR}/chest_xray/")
    print("\nExpected structure:")
    print("  data/train/NORMAL/  + data/train/PNEUMONIA/")
    print("  data/val/NORMAL/    + data/val/PNEUMONIA/")
    print("  data/test/NORMAL/   + data/test/PNEUMONIA/")


if __name__ == "__main__":
    download()
