"""Load cached ProcessedAudio JSONs + labels into (texts, features, labels) tuples."""

import json
from pathlib import Path
from typing import Tuple, List
from src.layer2_models import AudioFeatures


def load_split(split_dir: Path) -> Tuple[List[str], List[AudioFeatures], List[str]]:
    labels_path = split_dir / "_labels.json"
    label_map = json.loads(labels_path.read_text())
    texts, features, labels = [], [], []
    for sample_id, label in label_map.items():
        p = split_dir / f"{sample_id}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        texts.append(d["transcript"]["full_text"])
        f = d["features"]
        features.append(AudioFeatures(
            mfcc_mean=f["mfcc_mean"], mfcc_std=f["mfcc_std"],
            pitch_mean=f["pitch_mean"], pitch_std=f["pitch_std"], pitch_variance=f["pitch_variance"],
            energy_mean=f["energy_mean"], energy_std=f["energy_std"],
            zero_crossing_rate=f["zero_crossing_rate"],
            spectral_centroid_mean=f["spectral_centroid_mean"],
            spectral_rolloff_mean=f["spectral_rolloff_mean"],
            pause_count=f["pause_count"], pause_duration_mean=f["pause_duration_mean"],
            speech_rate=f["speech_rate"],
            total_speech_duration=f["total_speech_duration"],
            total_pause_duration=f["total_pause_duration"],
        ))
        labels.append(label)
    return texts, features, labels
