import numpy as np
from src.text_classifier import TextClassifier
from src.layer3_constants import CLASS_LABELS, N_CLASSES


class _FakeEncoder:
    """Deterministic fake encoder — returns hashed vectors so tests don't hit the network."""

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 16).astype(np.float32)


def test_train_and_predict_returns_probs_for_all_classes():
    clf = TextClassifier(encoder=_FakeEncoder())
    texts = [f"fake transcript {i}" for i in range(80)]
    labels = [CLASS_LABELS[i % N_CLASSES] for i in range(80)]
    clf.train(texts, labels)
    probs = clf.predict_proba("we need to verify your account right now")
    assert set(probs.keys()) == set(CLASS_LABELS)
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_save_and_load_roundtrip(tmp_path):
    clf = TextClassifier(encoder=_FakeEncoder())
    texts = [f"x{i}" for i in range(80)]
    labels = [CLASS_LABELS[i % N_CLASSES] for i in range(80)]
    clf.train(texts, labels)
    path = tmp_path / "txt.joblib"
    clf.save(str(path))
    loaded = TextClassifier.load(str(path), encoder=_FakeEncoder())
    p1 = clf.predict_proba("hello")
    p2 = loaded.predict_proba("hello")
    for k in CLASS_LABELS:
        assert abs(p1[k] - p2[k]) < 1e-6
