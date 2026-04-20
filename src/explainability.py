from typing import List, Dict
import numpy as np
from .layer2_models import AudioFeatures
from .feature_vector import flatten_features, FEATURE_NAMES


def top_acoustic_drivers(
    features: AudioFeatures,
    feature_importances: Dict[str, float],
    k: int = 5,
) -> List[str]:
    """Return the top-k feature names whose (importance * |value|) is largest for this sample."""
    values = flatten_features(features)
    scored = []
    for name, val in zip(FEATURE_NAMES, values):
        imp = feature_importances.get(name, 0.0)
        scored.append((name, imp * abs(val)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:k]]


def top_keywords_from_tfidf(
    text: str,
    vocabulary: List[str],
    class_coefficients: np.ndarray,
    k: int = 5,
) -> List[str]:
    """Rough-but-honest keyword attribution: match tokens in text against vocab, rank by class coef."""
    tokens = set(text.lower().split())
    scored = []
    for term, coef in zip(vocabulary, class_coefficients):
        if term in tokens:
            scored.append((term, float(coef)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in scored[:k]]
