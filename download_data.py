"""
download_data.py — Download Chest X-Ray Dataset from Kaggle
============================================================
Requires Kaggle API credentials: ~/.kaggle/kaggle.json

Usage:
    pip install kaggle
    python download_data.py
"""

import os
import shutil
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

    # Kaggle zip extracts to chest_xray/ — move splits to data/train, data/val, data/test
    src = os.path.join(DATA_DIR, "chest_xray")
    if os.path.isdir(src):
        for split in ("train", "val", "test"):
            split_src = os.path.join(src, split)
            split_dst = os.path.join(DATA_DIR, split)
            if os.path.isdir(split_src):
                if os.path.exists(split_dst):
                    shutil.rmtree(split_dst)
                os.rename(split_src, split_dst)
        try:
            os.rmdir(src)
        except OSError:
            pass

    print(f"Dataset ready at {DATA_DIR}/")
    print("\nExpected structure:")
    print("  data/train/NORMAL/  + data/train/PNEUMONIA/")
    print("  data/val/NORMAL/    + data/val/PNEUMONIA/")
    print("  data/test/NORMAL/   + data/test/PNEUMONIA/")


if __name__ == "__main__":
    download()
