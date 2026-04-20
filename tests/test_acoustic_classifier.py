import numpy as np
from src.layer2_models import AudioFeatures
from src.acoustic_classifier import AcousticClassifier
from src.layer3_constants import CLASS_LABELS, N_CLASSES


def _feat(seed: int) -> AudioFeatures:
    rng = np.random.RandomState(seed)
    return AudioFeatures(
        mfcc_mean=rng.randn(13).tolist(), mfcc_std=rng.rand(13).tolist(),
        pitch_mean=float(100 + rng.randn() * 30), pitch_std=float(abs(rng.randn() * 5)),
        pitch_variance=float(abs(rng.randn() * 50)),
        energy_mean=float(abs(rng.rand())), energy_std=float(abs(rng.rand()) * 0.1),
        zero_crossing_rate=float(rng.rand() * 0.3),
        spectral_centroid_mean=float(1500 + rng.randn() * 500),
        spectral_rolloff_mean=float(3500 + rng.randn() * 500),
        pause_count=int(rng.randint(0, 10)), pause_duration_mean=float(rng.rand()),
        speech_rate=float(2 + rng.rand() * 3),
        total_speech_duration=float(10 + rng.rand() * 20),
        total_pause_duration=float(rng.rand() * 3),
    )


def test_train_and_predict_returns_probs_for_all_classes():
    clf = AcousticClassifier()
    feats = [_feat(i) for i in range(80)]
    labels = [CLASS_LABELS[i % N_CLASSES] for i in range(80)]
    clf.train(feats, labels)
    probs = clf.predict_proba(_feat(999))
    assert set(probs.keys()) == set(CLASS_LABELS)
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_feature_importances_returns_39_values():
    clf = AcousticClassifier()
    feats = [_feat(i) for i in range(80)]
    labels = [CLASS_LABELS[i % N_CLASSES] for i in range(80)]
    clf.train(feats, labels)
    imp = clf.feature_importances()
    assert len(imp) == 39


def test_save_and_load_roundtrip(tmp_path):
    clf = AcousticClassifier()
    feats = [_feat(i) for i in range(80)]
    labels = [CLASS_LABELS[i % N_CLASSES] for i in range(80)]
    clf.train(feats, labels)
    path = tmp_path / "a.joblib"
    clf.save(str(path))
    loaded = AcousticClassifier.load(str(path))
    f = _feat(999)
    p1 = clf.predict_proba(f)
    p2 = loaded.predict_proba(f)
    for k in CLASS_LABELS:
        assert abs(p1[k] - p2[k]) < 1e-6
