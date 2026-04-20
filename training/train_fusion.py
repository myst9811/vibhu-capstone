"""Train LateFusion on the validation split, save to models/fusion_classifier.joblib."""

import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_classifier import TextClassifier
from src.acoustic_classifier import AcousticClassifier
from src.fusion import LateFusion
from training.load_processed import load_split


VAL_DIR = Path("data/tele_antifraud/processed/validation")
TEXT_PATH = Path("models/text_classifier.joblib")
AC_PATH = Path("models/acoustic_classifier.joblib")
FUSION_PATH = Path("models/fusion_classifier.joblib")


def main():
    texts, features, labels = load_split(VAL_DIR)
    print(f"Loaded {len(labels)} validation samples")

    text_clf = TextClassifier.load(str(TEXT_PATH))
    ac_clf = AcousticClassifier.load(str(AC_PATH))

    text_probs = [text_clf.predict_proba(t) for t in tqdm(texts, desc="text/proba")]
    ac_probs = [ac_clf.predict_proba(f) for f in tqdm(features, desc="ac/proba")]

    fusion = LateFusion()
    fusion.fit(text_probs, ac_probs, labels)
    FUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fusion.save(str(FUSION_PATH))
    print(f"Saved -> {FUSION_PATH}")


if __name__ == "__main__":
    main()
