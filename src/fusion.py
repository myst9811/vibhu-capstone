from typing import List, Dict, Optional
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from .layer3_constants import CLASS_LABELS, LABEL_TO_IDX, IDX_TO_LABEL


def _probs_to_vec(probs: Dict[str, float]) -> np.ndarray:
    return np.array([probs[label] for label in CLASS_LABELS], dtype=np.float64)


class LateFusion:
    def __init__(self, model: Optional[LogisticRegression] = None):
        self.model = model

    def fit(
        self,
        text_probs: List[Dict[str, float]],
        acoustic_probs: List[Dict[str, float]],
        labels: List[str],
    ) -> None:
        X = np.vstack([
            np.concatenate([_probs_to_vec(t), _probs_to_vec(a)])
            for t, a in zip(text_probs, acoustic_probs)
        ])
        y = np.array([LABEL_TO_IDX[l] for l in labels])
        self.model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
        )
        self.model.fit(X, y)

    def predict_proba(
        self,
        text_probs: Dict[str, float],
        acoustic_probs: Dict[str, float],
    ) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("LateFusion is not trained")
        x = np.concatenate([_probs_to_vec(text_probs), _probs_to_vec(acoustic_probs)]).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        out = {label: 0.0 for label in CLASS_LABELS}
        for cls_idx, p in zip(self.model.classes_, probs):
            out[IDX_TO_LABEL[int(cls_idx)]] = float(p)
        return out

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model}, path)

    @classmethod
    def load(cls, path: str) -> "LateFusion":
        payload = joblib.load(path)
        return cls(model=payload["model"])
