import numpy as np
from src.explainability import top_acoustic_drivers, top_keywords_from_tfidf
from src.layer2_models import AudioFeatures
from src.feature_vector import FEATURE_NAMES


def _feat():
    return AudioFeatures(
        mfcc_mean=[0.0] * 13, mfcc_std=[0.0] * 13,
        pitch_mean=150.0, pitch_std=10.0, pitch_variance=100.0,
        energy_mean=0.05, energy_std=0.01, zero_crossing_rate=0.12,
        spectral_centroid_mean=2000.0, spectral_rolloff_mean=4000.0,
        pause_count=5, pause_duration_mean=0.3,
        speech_rate=4.2, total_speech_duration=20.0, total_pause_duration=1.5,
    )


def test_top_acoustic_drivers_respects_k():
    importances = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    top = top_acoustic_drivers(_feat(), importances, k=5)
    assert len(top) == 5
    assert "total_pause_duration" in top or "total_speech_duration" in top


def test_top_keywords_from_tfidf_basic():
    text = "please verify your account urgent wire transfer now"
    vocabulary = ["verify", "account", "urgent", "wire", "transfer", "hello", "weather"]
    coefs = np.array([3.0, 2.5, 4.0, 3.5, 3.8, -1.0, -2.0])
    top = top_keywords_from_tfidf(text, vocabulary, coefs, k=5)
    assert len(top) == 5
    assert "urgent" in top
    assert "weather" not in top
