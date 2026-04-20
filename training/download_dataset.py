"""Download TeleAntiFraud-28k from HuggingFace and cache locally.

Usage: python training/download_dataset.py

The dataset ID must be confirmed from the TeleAntiFraud-28k paper / repo.
If the hub name differs, set TELE_DATASET_ID env var.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset


DATASET_ID = os.environ.get("TELE_DATASET_ID", "TeleAntiFraud/TeleAntiFraud-28k")
CACHE_DIR = Path("data/tele_antifraud")


def download():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_ID} -> {CACHE_DIR}...")
    ds = load_dataset(DATASET_ID, cache_dir=str(CACHE_DIR))
    print("Splits:", {k: len(v) for k, v in ds.items()})

    manifest = {
        "dataset_id": DATASET_ID,
        "splits": {split: len(ds[split]) for split in ds},
        "fields": list(ds[list(ds.keys())[0]].features.keys()),
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {CACHE_DIR / 'manifest.json'}")
    return ds


if __name__ == "__main__":
    download()
