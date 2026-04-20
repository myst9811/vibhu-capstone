"""Train AcousticClassifier on the train split, save to models/acoustic_classifier.joblib."""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.acoustic_classifier import AcousticClassifier
from training.load_processed import load_split


SPLIT_DIR = Path("data/tele_antifraud/processed/train")
MODEL_PATH = Path("models/acoustic_classifier.joblib")
IMPORTANCE_PATH = Path("models/acoustic_importance.json")


def main():
    _, features, labels = load_split(SPLIT_DIR)
    print(f"Loaded {len(features)} train samples")

    clf = AcousticClassifier()
    clf.train(features, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(MODEL_PATH))
    IMPORTANCE_PATH.write_text(json.dumps(clf.feature_importances(), indent=2))
    print(f"Saved -> {MODEL_PATH}")
    print(f"Feature importances -> {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
