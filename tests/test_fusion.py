import numpy as np
from src.fusion import LateFusion
from src.layer3_constants import CLASS_LABELS, N_CLASSES


def _probs_vec(peak_idx: int, noise: float, rng) -> dict:
    v = np.abs(rng.randn(N_CLASSES)) * noise
    v[peak_idx] += 2.0
    v = v / v.sum()
    return {label: float(v[i]) for i, label in enumerate(CLASS_LABELS)}


def test_fit_and_predict():
    rng = np.random.RandomState(0)
    text_probs = []
    acoustic_probs = []
    labels = []
    for i in range(160):
        cls = i % N_CLASSES
        text_probs.append(_probs_vec(cls, 0.3, rng))
        acoustic_probs.append(_probs_vec(cls, 0.5, rng))
        labels.append(CLASS_LABELS[cls])

    fusion = LateFusion()
    fusion.fit(text_probs, acoustic_probs, labels)

    fused = fusion.predict_proba(
        text_probs={label: (1.0 if label == "phishing" else 0.0) for label in CLASS_LABELS},
        acoustic_probs={label: (1.0 if label == "phishing" else 0.0) for label in CLASS_LABELS},
    )
    assert set(fused.keys()) == set(CLASS_LABELS)
    assert abs(sum(fused.values()) - 1.0) < 1e-5
    assert max(fused, key=fused.get) == "phishing"


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.RandomState(1)
    text_probs, acoustic_probs, labels = [], [], []
    for i in range(160):
        cls = i % N_CLASSES
        text_probs.append(_probs_vec(cls, 0.3, rng))
        acoustic_probs.append(_probs_vec(cls, 0.5, rng))
        labels.append(CLASS_LABELS[cls])
    fusion = LateFusion()
    fusion.fit(text_probs, acoustic_probs, labels)
    path = tmp_path / "f.joblib"
    fusion.save(str(path))
    loaded = LateFusion.load(str(path))
    t = {label: 0.125 for label in CLASS_LABELS}
    a = {label: 0.125 for label in CLASS_LABELS}
    p1 = fusion.predict_proba(t, a)
    p2 = loaded.predict_proba(t, a)
    for k in CLASS_LABELS:
        assert abs(p1[k] - p2[k]) < 1e-6
