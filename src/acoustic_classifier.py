from typing import List, Dict, Optional
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from .layer2_models import AudioFeatures
from .feature_vector import flatten_features, FEATURE_NAMES
from .layer3_constants import CLASS_LABELS, LABEL_TO_IDX, IDX_TO_LABEL, N_CLASSES


class AcousticClassifier:
    def __init__(self, model: Optional[XGBClassifier] = None):
        self.model = model

    def train(self, features: List[AudioFeatures], labels: List[str]) -> None:
        X = np.vstack([flatten_features(f) for f in features])
        y = np.array([LABEL_TO_IDX[l] for l in labels])
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=N_CLASSES,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            tree_method="hist",
        )
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, features: AudioFeatures) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("AcousticClassifier is not trained")
        X = flatten_features(features).reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        out = {label: 0.0 for label in CLASS_LABELS}
        for cls_idx, p in zip(self.model.classes_, probs):
            out[IDX_TO_LABEL[int(cls_idx)]] = float(p)
        return out

    def feature_importances(self) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("AcousticClassifier is not trained")
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)}

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model}, path)

    @classmethod
    def load(cls, path: str) -> "AcousticClassifier":
        payload = joblib.load(path)
        return cls(model=payload["model"])
