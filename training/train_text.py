"""Train TextClassifier on the train split, save to models/text_classifier.joblib."""

import sys
import os
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_classifier import TextClassifier
from src.layer3_constants import LABEL_TO_IDX
from training.load_processed import load_split


SPLIT_DIR = Path("data/tele_antifraud/processed/train")
MODEL_PATH = Path("models/text_classifier.joblib")
VOCAB_PATH = Path("models/text_vocab.json")


def main():
    texts, _, labels = load_split(SPLIT_DIR)
    print(f"Loaded {len(texts)} train samples")

    clf = TextClassifier()
    clf.train(texts, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(MODEL_PATH))

    vec = TfidfVectorizer(max_features=2000, stop_words="english")
    X_tfidf = vec.fit_transform(texts)
    y = np.array([LABEL_TO_IDX[l] for l in labels])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_tfidf, y)
    payload = {
        "vocabulary": vec.get_feature_names_out().tolist(),
        "coef": lr.coef_.tolist(),
        "classes": lr.classes_.tolist(),
    }
    VOCAB_PATH.write_text(json.dumps(payload))
    print(f"Saved text classifier -> {MODEL_PATH}")
    print(f"Saved TF-IDF vocab -> {VOCAB_PATH}")


if __name__ == "__main__":
    main()
