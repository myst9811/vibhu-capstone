from typing import List, Dict, Optional, Any
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from .layer3_constants import CLASS_LABELS, LABEL_TO_IDX, IDX_TO_LABEL


DEFAULT_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_default_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(DEFAULT_ENCODER_NAME)


class TextClassifier:
    def __init__(self, encoder: Optional[Any] = None, model: Optional[LogisticRegression] = None):
        self._encoder = encoder
        self.model = model

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = _load_default_encoder()
        return self._encoder

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def train(self, texts: List[str], labels: List[str]) -> None:
        X = self._embed(texts)
        y = np.array([LABEL_TO_IDX[l] for l in labels])
        self.model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
        )
        self.model.fit(X, y)

    def predict_proba(self, text: str) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("TextClassifier is not trained")
        X = self._embed([text])
        probs = self.model.predict_proba(X)[0]
        out = {label: 0.0 for label in CLASS_LABELS}
        for cls_idx, p in zip(self.model.classes_, probs):
            out[IDX_TO_LABEL[int(cls_idx)]] = float(p)
        return out

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model}, path)

    @classmethod
    def load(cls, path: str, encoder: Optional[Any] = None) -> "TextClassifier":
        payload = joblib.load(path)
        return cls(encoder=encoder, model=payload["model"])
