"""Run Layer 1 + Layer 2 over the TeleAntiFraud-28k dataset and cache ProcessedAudio JSONs.

This is the slow step (ASR). Run on Colab with GPU if possible; resume-safe via file-exists check.

Usage: python training/prepare_data.py [--split train|validation|test] [--limit N]
"""

import argparse
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.layer2_processor import Layer2Processor


DATASET_ID = os.environ.get("TELE_DATASET_ID", "TeleAntiFraud/TeleAntiFraud-28k")
CACHE_DIR = Path("data/tele_antifraud")
PROCESSED_DIR = CACHE_DIR / "processed"


def prepare(split: str, limit=None):
    ds = load_dataset(DATASET_ID, split=split, cache_dir=str(CACHE_DIR))
    processor = Layer2Processor()
    out_dir = PROCESSED_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = {}
    iterable = ds if limit is None else ds.select(range(min(limit, len(ds))))
    for i, row in enumerate(tqdm(iterable, desc=f"prep/{split}")):
        sample_id = row.get("id") or f"{split}_{i:06d}"
        out_path = out_dir / f"{sample_id}.json"
        if out_path.exists():
            label_map[sample_id] = row["label"]
            continue
        audio_path = row["audio"]["path"] if isinstance(row.get("audio"), dict) else row.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            print(f"skip {sample_id}: no audio at {audio_path}")
            continue
        try:
            pa = processor.process(str(audio_path), file_id=sample_id)
            pa.save_to_file(str(out_path))
            label_map[sample_id] = row["label"]
        except Exception as e:
            print(f"fail {sample_id}: {e}")

    (out_dir / "_labels.json").write_text(json.dumps(label_map, indent=2))
    print(f"Wrote {len(label_map)} samples + _labels.json to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    prepare(args.split, args.limit)
