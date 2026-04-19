# Layer 3 Multimodal Fraud Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Layer 3 — a multimodal (text + acoustic) late-fusion classifier that consumes `ProcessedAudio` from Layer 2 and emits a `DetectionResult` with one of 8 labels (`legitimate` + 7 fraud categories), plus light explainability.

**Architecture:** Two independent classifiers — text (sentence embeddings + LogReg) and acoustic (XGBoost on flattened `AudioFeatures`) — trained on TeleAntiFraud-28k. A learned meta-classifier fuses their class probabilities. Training runs on Colab free tier; inference runs locally via saved `joblib` artifacts.

**Tech Stack:** Python 3.11, scikit-learn, xgboost, sentence-transformers (all-MiniLM-L6-v2), joblib, pytest, HuggingFace datasets, matplotlib (confusion matrix).

---

## Task 0: Test infrastructure + new dependencies

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `requirements_layer3.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Add Layer 3 requirements file**

Create `requirements_layer3.txt`:

```
scikit-learn>=1.3.0
xgboost>=2.0.0
sentence-transformers>=2.7.0
joblib>=1.2.0
pytest>=8.0.0
matplotlib>=3.8.0
datasets>=2.18.0
tqdm>=4.66.0
```

- [ ] **Step 2: Update .gitignore**

Append to `.gitignore`:

```
# Layer 3
data/tele_antifraud/
models/text_classifier.joblib
models/acoustic_classifier.joblib
models/fusion_classifier.joblib
models/sentence_encoder/
reports/
__pycache__/
.pytest_cache/
```

- [ ] **Step 3: Create tests directory scaffolding**

Create `tests/__init__.py` (empty) and `tests/conftest.py`:

```python
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 4: Install and sanity-check**

Run: `pip install -r requirements_layer3.txt && pytest --version`
Expected: pytest prints its version cleanly.

- [ ] **Step 5: Commit**

```bash
git add requirements_layer3.txt .gitignore tests/
git commit -m "chore: set up layer 3 deps and test scaffolding"
```

---

## Task 1: DetectionResult dataclass

**Files:**
- Create: `src/layer3_models.py`
- Create: `tests/test_layer3_models.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_layer3_models.py`:

```python
import json
from src.layer3_models import DetectionResult


def test_detection_result_to_dict_has_all_fields():
    result = DetectionResult(
        file_id="call_001",
        label="phishing",
        confidence=0.87,
        class_probabilities={"legitimate": 0.05, "phishing": 0.87, "banking": 0.03,
                             "investment": 0.01, "kidnapping": 0.01, "lottery": 0.01,
                             "customer_service": 0.01, "identity_theft": 0.01},
        per_modality={
            "text": {"legitimate": 0.1, "phishing": 0.85, "banking": 0.01, "investment": 0.01,
                     "kidnapping": 0.01, "lottery": 0.01, "customer_service": 0.01, "identity_theft": 0.0},
            "acoustic": {"legitimate": 0.0, "phishing": 0.9, "banking": 0.05, "investment": 0.01,
                         "kidnapping": 0.01, "lottery": 0.01, "customer_service": 0.01, "identity_theft": 0.01},
            "fused": {"legitimate": 0.05, "phishing": 0.87, "banking": 0.03, "investment": 0.01,
                      "kidnapping": 0.01, "lottery": 0.01, "customer_service": 0.01, "identity_theft": 0.01},
        },
        top_keywords=["verify", "account", "urgent", "wire", "transfer"],
        top_acoustic_drivers=["speech_rate", "pitch_variance", "pause_count", "energy_mean", "mfcc_3"],
    )
    d = result.to_dict()
    assert d["file_id"] == "call_001"
    assert d["label"] == "phishing"
    assert d["version"] == "3.0"
    assert len(d["top_keywords"]) == 5
    assert "fused" in d["per_modality"]


def test_detection_result_json_roundtrip(tmp_path):
    result = DetectionResult(
        file_id="x", label="legitimate", confidence=0.99,
        class_probabilities={k: 0.125 for k in
            ["legitimate","phishing","banking","investment","kidnapping","lottery","customer_service","identity_theft"]},
        per_modality={"text": {}, "acoustic": {}, "fused": {}},
        top_keywords=[], top_acoustic_drivers=[],
    )
    out = tmp_path / "r.json"
    result.save_to_file(str(out))
    loaded = json.loads(out.read_text())
    assert loaded["label"] == "legitimate"
    assert loaded["confidence"] == 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer3_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.layer3_models'` — wait, the module exists. Expected: FAIL with `ImportError: cannot import name 'DetectionResult' from 'src.layer3_models'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/layer3_models.py` (file already exists for Layer 2 models — add below existing content):

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any
import json


@dataclass
class DetectionResult:
    file_id: str
    label: str
    confidence: float
    class_probabilities: Dict[str, float]
    per_modality: Dict[str, Dict[str, float]]
    top_keywords: List[str] = field(default_factory=list)
    top_acoustic_drivers: List[str] = field(default_factory=list)
    version: str = "3.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "label": self.label,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "per_modality": self.per_modality,
            "top_keywords": self.top_keywords,
            "top_acoustic_drivers": self.top_acoustic_drivers,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, output_path: str) -> None:
        with open(output_path, "w") as f:
            f.write(self.to_json())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layer3_models.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/layer3_models.py tests/test_layer3_models.py
git commit -m "feat: DetectionResult dataclass — the shape of a verdict"
```

---

## Task 2: Class label constants + feature flattening helper

**Files:**
- Create: `src/layer3_constants.py`
- Create: `src/feature_vector.py`
- Create: `tests/test_feature_vector.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_feature_vector.py`:

```python
import numpy as np
from src.layer2_models import AudioFeatures
from src.feature_vector import flatten_features, FEATURE_NAMES


def _make_features():
    return AudioFeatures(
        mfcc_mean=[0.1] * 13, mfcc_std=[0.2] * 13,
        pitch_mean=150.0, pitch_std=10.0, pitch_variance=100.0,
        energy_mean=0.05, energy_std=0.01,
        zero_crossing_rate=0.12,
        spectral_centroid_mean=2000.0, spectral_rolloff_mean=4000.0,
        pause_count=5, pause_duration_mean=0.3,
        speech_rate=4.2, total_speech_duration=20.0, total_pause_duration=1.5,
    )


def test_flatten_features_produces_39_element_vector():
    v = flatten_features(_make_features())
    assert isinstance(v, np.ndarray)
    assert v.shape == (39,)


def test_feature_names_match_vector_length():
    assert len(FEATURE_NAMES) == 39
    assert "pitch_mean" in FEATURE_NAMES
    assert "mfcc_mean_0" in FEATURE_NAMES
    assert "mfcc_std_12" in FEATURE_NAMES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_feature_vector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.feature_vector'`.

- [ ] **Step 3: Create class constants**

Create `src/layer3_constants.py`:

```python
CLASS_LABELS = [
    "legitimate",
    "phishing",
    "banking",
    "investment",
    "kidnapping",
    "lottery",
    "customer_service",
    "identity_theft",
]

N_CLASSES = len(CLASS_LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(CLASS_LABELS)}
IDX_TO_LABEL = {i: label for i, label in enumerate(CLASS_LABELS)}
```

- [ ] **Step 4: Create feature vector module**

Create `src/feature_vector.py`:

```python
import numpy as np
from typing import List
from .layer2_models import AudioFeatures


FEATURE_NAMES: List[str] = (
    [f"mfcc_mean_{i}" for i in range(13)]
    + [f"mfcc_std_{i}" for i in range(13)]
    + ["pitch_mean", "pitch_std", "pitch_variance"]
    + ["energy_mean", "energy_std"]
    + ["zero_crossing_rate"]
    + ["spectral_centroid_mean", "spectral_rolloff_mean"]
    + ["pause_count", "pause_duration_mean"]
    + ["speech_rate", "total_speech_duration", "total_pause_duration"]
)


def flatten_features(f: AudioFeatures) -> np.ndarray:
    parts = (
        list(f.mfcc_mean)
        + list(f.mfcc_std)
        + [f.pitch_mean, f.pitch_std, f.pitch_variance]
        + [f.energy_mean, f.energy_std]
        + [f.zero_crossing_rate]
        + [f.spectral_centroid_mean, f.spectral_rolloff_mean]
        + [float(f.pause_count), f.pause_duration_mean]
        + [f.speech_rate, f.total_speech_duration, f.total_pause_duration]
    )
    return np.asarray(parts, dtype=np.float64)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_feature_vector.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/layer3_constants.py src/feature_vector.py tests/test_feature_vector.py
git commit -m "feat: 39-dim acoustic feature vector + the 8 sacred class labels"
```

---

## Task 3: TextClassifier (embeddings + LogReg wrapper)

**Files:**
- Create: `src/text_classifier.py`
- Create: `tests/test_text_classifier.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_text_classifier.py`:

```python
import numpy as np
import pytest
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
    # Balanced synthetic labels: 10 per class
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_text_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.text_classifier'`.

- [ ] **Step 3: Write implementation**

Create `src/text_classifier.py`:

```python
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
            multi_class="multinomial",
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
        # model.classes_ contains the integer class indices actually seen during training
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_text_classifier.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/text_classifier.py tests/test_text_classifier.py
git commit -m "feat: TextClassifier — embeds transcripts, guesses vibes"
```

---

## Task 4: AcousticClassifier (XGBoost on AudioFeatures)

**Files:**
- Create: `src/acoustic_classifier.py`
- Create: `tests/test_acoustic_classifier.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_acoustic_classifier.py`:

```python
import numpy as np
import pytest
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_acoustic_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.acoustic_classifier'`.

- [ ] **Step 3: Write implementation**

Create `src/acoustic_classifier.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_acoustic_classifier.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/acoustic_classifier.py tests/test_acoustic_classifier.py
git commit -m "feat: AcousticClassifier — xgboost reads the vibes of the voice"
```

---

## Task 5: LateFusion meta-classifier

**Files:**
- Create: `src/fusion.py`
- Create: `tests/test_fusion.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_fusion.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.fusion'`.

- [ ] **Step 3: Write implementation**

Create `src/fusion.py`:

```python
from typing import List, Dict, Optional
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from .layer3_constants import CLASS_LABELS, LABEL_TO_IDX, IDX_TO_LABEL, N_CLASSES


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
            multi_class="multinomial",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fusion.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fusion.py tests/test_fusion.py
git commit -m "feat: late-fusion meta-classifier — referee between text and audio"
```

---

## Task 6: Explainability helpers (top keywords + top acoustic drivers)

**Files:**
- Create: `src/explainability.py`
- Create: `tests/test_explainability.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_explainability.py`:

```python
import numpy as np
from src.explainability import top_acoustic_drivers, top_keywords_from_tfidf
from src.layer2_models import AudioFeatures


def _feat():
    return AudioFeatures(
        mfcc_mean=[0.0]*13, mfcc_std=[0.0]*13,
        pitch_mean=150.0, pitch_std=10.0, pitch_variance=100.0,
        energy_mean=0.05, energy_std=0.01, zero_crossing_rate=0.12,
        spectral_centroid_mean=2000.0, spectral_rolloff_mean=4000.0,
        pause_count=5, pause_duration_mean=0.3,
        speech_rate=4.2, total_speech_duration=20.0, total_pause_duration=1.5,
    )


def test_top_acoustic_drivers_respects_k():
    importances = {f"feat_{i}": float(i) for i in range(20)}
    # Use real feature names to validate filtering path
    from src.feature_vector import FEATURE_NAMES
    importances = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    top = top_acoustic_drivers(_feat(), importances, k=5)
    assert len(top) == 5
    # Highest importance names should appear
    assert "total_pause_duration" in top or "total_speech_duration" in top


def test_top_keywords_from_tfidf_basic():
    text = "please verify your account urgent wire transfer now"
    vocabulary = ["verify", "account", "urgent", "wire", "transfer", "hello", "weather"]
    # Fake logreg coefficients: higher = more fraud-indicative
    coefs = np.array([3.0, 2.5, 4.0, 3.5, 3.8, -1.0, -2.0])
    top = top_keywords_from_tfidf(text, vocabulary, coefs, k=5)
    assert len(top) == 5
    assert "urgent" in top
    assert "weather" not in top
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explainability.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.explainability'`.

- [ ] **Step 3: Write implementation**

Create `src/explainability.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_explainability.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/explainability.py tests/test_explainability.py
git commit -m "feat: light explainability — top keywords + top acoustic drivers"
```

---

## Task 7: Layer3Detector (inference wrapper)

**Files:**
- Create: `src/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_detector.py`:

```python
import numpy as np
from src.detector import Layer3Detector
from src.text_classifier import TextClassifier
from src.acoustic_classifier import AcousticClassifier
from src.fusion import LateFusion
from src.layer2_models import AudioFeatures, Transcript, ProcessedAudio, TranscriptSegment
from src.layer3_constants import CLASS_LABELS, N_CLASSES


class _FakeEncoder:
    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 16).astype(np.float32)


def _feat(seed: int):
    rng = np.random.RandomState(seed)
    return AudioFeatures(
        mfcc_mean=rng.randn(13).tolist(), mfcc_std=rng.rand(13).tolist(),
        pitch_mean=150.0, pitch_std=10.0, pitch_variance=100.0,
        energy_mean=0.05, energy_std=0.01, zero_crossing_rate=0.12,
        spectral_centroid_mean=2000.0, spectral_rolloff_mean=4000.0,
        pause_count=5, pause_duration_mean=0.3,
        speech_rate=4.2, total_speech_duration=20.0, total_pause_duration=1.5,
    )


def _train_fake_detector():
    text_clf = TextClassifier(encoder=_FakeEncoder())
    ac_clf = AcousticClassifier()
    texts = [f"fake transcript {i}" for i in range(80)]
    feats = [_feat(i) for i in range(80)]
    labels = [CLASS_LABELS[i % N_CLASSES] for i in range(80)]
    text_clf.train(texts, labels)
    ac_clf.train(feats, labels)

    text_probs = [text_clf.predict_proba(t) for t in texts[:160 - 80] + texts]
    ac_probs = [ac_clf.predict_proba(f) for f in feats[:160 - 80] + feats]
    fused_labels = [CLASS_LABELS[i % N_CLASSES] for i in range(len(text_probs))]
    fusion = LateFusion()
    fusion.fit(text_probs, ac_probs, fused_labels)

    vocab = ["verify", "account", "wire", "urgent", "hello"]
    coefs = np.array([3.0, 2.0, 4.0, 3.5, -2.0])
    return Layer3Detector(
        text_classifier=text_clf,
        acoustic_classifier=ac_clf,
        fusion=fusion,
        text_vocabulary=vocab,
        text_class_coefficients=coefs,
    )


def test_predict_returns_detection_result():
    detector = _train_fake_detector()
    pa = ProcessedAudio(
        file_id="test_001",
        audio_path="/dev/null",
        features=_feat(999),
        transcript=Transcript(
            full_text="verify your account wire urgent",
            segments=[], word_count=5, language="en-US",
        ),
        processing_timestamp="2026-04-20T00:00:00",
    )
    result = detector.predict(pa)
    assert result.file_id == "test_001"
    assert result.label in CLASS_LABELS
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.class_probabilities.keys()) == set(CLASS_LABELS)
    assert "text" in result.per_modality
    assert "acoustic" in result.per_modality
    assert "fused" in result.per_modality
    assert len(result.top_keywords) <= 5
    assert len(result.top_acoustic_drivers) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.detector'`.

- [ ] **Step 3: Write implementation**

Create `src/detector.py`:

```python
from typing import List, Optional
import numpy as np
from .text_classifier import TextClassifier
from .acoustic_classifier import AcousticClassifier
from .fusion import LateFusion
from .layer2_models import ProcessedAudio
from .layer3_models import DetectionResult
from .explainability import top_acoustic_drivers, top_keywords_from_tfidf


class Layer3Detector:
    def __init__(
        self,
        text_classifier: TextClassifier,
        acoustic_classifier: AcousticClassifier,
        fusion: LateFusion,
        text_vocabulary: Optional[List[str]] = None,
        text_class_coefficients: Optional[np.ndarray] = None,
    ):
        self.text_classifier = text_classifier
        self.acoustic_classifier = acoustic_classifier
        self.fusion = fusion
        self.text_vocabulary = text_vocabulary or []
        self.text_class_coefficients = text_class_coefficients

    def predict(self, processed_audio: ProcessedAudio) -> DetectionResult:
        text = processed_audio.transcript.full_text
        features = processed_audio.features

        text_probs = self.text_classifier.predict_proba(text)
        acoustic_probs = self.acoustic_classifier.predict_proba(features)
        fused_probs = self.fusion.predict_proba(text_probs, acoustic_probs)

        label = max(fused_probs, key=fused_probs.get)
        confidence = fused_probs[label]

        keywords = []
        if self.text_vocabulary and self.text_class_coefficients is not None:
            keywords = top_keywords_from_tfidf(
                text, self.text_vocabulary, self.text_class_coefficients, k=5,
            )

        drivers = []
        try:
            importances = self.acoustic_classifier.feature_importances()
            drivers = top_acoustic_drivers(features, importances, k=5)
        except RuntimeError:
            drivers = []

        return DetectionResult(
            file_id=processed_audio.file_id,
            label=label,
            confidence=confidence,
            class_probabilities=fused_probs,
            per_modality={
                "text": text_probs,
                "acoustic": acoustic_probs,
                "fused": fused_probs,
            },
            top_keywords=keywords,
            top_acoustic_drivers=drivers,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detector.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/detector.py tests/test_detector.py
git commit -m "feat: Layer3Detector — one predict() to rule them all"
```

---

## Task 8: Dataset download + prep scripts

**Files:**
- Create: `training/__init__.py`
- Create: `training/download_dataset.py`
- Create: `training/prepare_data.py`

- [ ] **Step 1: Create training package**

Create `training/__init__.py` (empty).

- [ ] **Step 2: Write download script**

Create `training/download_dataset.py`:

```python
"""Download TeleAntiFraud-28k from HuggingFace and cache locally.

Usage: python training/download_dataset.py

The dataset ID must be confirmed from the TeleAntiFraud-28k paper / repo.
If the hub name differs, set TELE_DATASET_ID env var.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset


DATASET_ID = os.environ.get("TELE_DATASET_ID", "TeleAntiFraud/TeleAntiFraud-28k")
CACHE_DIR = Path("data/tele_antifraud")


def download():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_ID} → {CACHE_DIR}...")
    ds = load_dataset(DATASET_ID, cache_dir=str(CACHE_DIR))
    print("Splits:", {k: len(v) for k, v in ds.items()})

    manifest = {
        "dataset_id": DATASET_ID,
        "splits": {split: len(ds[split]) for split in ds},
        "fields": list(ds[list(ds.keys())[0]].features.keys()),
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {CACHE_DIR / 'manifest.json'}")
    return ds


if __name__ == "__main__":
    download()
```

- [ ] **Step 3: Write prep script**

Create `training/prepare_data.py`:

```python
"""Run Layer 1 + Layer 2 over the TeleAntiFraud-28k dataset and cache ProcessedAudio JSONs.

This is the slow step (ASR). Run on Colab with GPU if possible; resume-safe via file-exists check.

Usage: python training/prepare_data.py [--split train|validation|test] [--limit N]
"""

import argparse
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.layer2_processor import Layer2Processor


DATASET_ID = os.environ.get("TELE_DATASET_ID", "TeleAntiFraud/TeleAntiFraud-28k")
CACHE_DIR = Path("data/tele_antifraud")
PROCESSED_DIR = CACHE_DIR / "processed"


def prepare(split: str, limit: int | None = None):
    ds = load_dataset(DATASET_ID, split=split, cache_dir=str(CACHE_DIR))
    processor = Layer2Processor()
    out_dir = PROCESSED_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = {}
    iterable = ds if limit is None else ds.select(range(min(limit, len(ds))))
    for i, row in enumerate(tqdm(iterable, desc=f"prep/{split}")):
        sample_id = row.get("id") or f"{split}_{i:06d}"
        out_path = out_dir / f"{sample_id}.json"
        if out_path.exists():
            label_map[sample_id] = row["label"]
            continue
        audio_path = row["audio"]["path"] if isinstance(row.get("audio"), dict) else row.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            print(f"skip {sample_id}: no audio at {audio_path}")
            continue
        try:
            pa = processor.process(str(audio_path), file_id=sample_id)
            pa.save_to_file(str(out_path))
            label_map[sample_id] = row["label"]
        except Exception as e:
            print(f"fail {sample_id}: {e}")

    (out_dir / "_labels.json").write_text(json.dumps(label_map, indent=2))
    print(f"Wrote {len(label_map)} samples + _labels.json to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    prepare(args.split, args.limit)
```

- [ ] **Step 4: Smoke test download script with a dry-run on manifest print**

Run: `python -c "from training.download_dataset import DATASET_ID; print(DATASET_ID)"`
Expected: prints `TeleAntiFraud/TeleAntiFraud-28k` (or the env-var override).

- [ ] **Step 5: Commit**

```bash
git add training/__init__.py training/download_dataset.py training/prepare_data.py
git commit -m "feat: dataset download + L1/L2 prep — resumable because ASR is slow"
```

---

## Task 9: Training scripts (text + acoustic + fusion)

**Files:**
- Create: `training/load_processed.py`
- Create: `training/train_text.py`
- Create: `training/train_acoustic.py`
- Create: `training/train_fusion.py`

- [ ] **Step 1: Shared loader**

Create `training/load_processed.py`:

```python
"""Load cached ProcessedAudio JSONs + labels into (texts, features, labels) tuples."""

import json
from pathlib import Path
from typing import Tuple, List
from src.layer2_models import AudioFeatures, Transcript, TranscriptSegment


def load_split(split_dir: Path) -> Tuple[List[str], List[AudioFeatures], List[str]]:
    labels_path = split_dir / "_labels.json"
    label_map = json.loads(labels_path.read_text())
    texts, features, labels = [], [], []
    for sample_id, label in label_map.items():
        p = split_dir / f"{sample_id}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        texts.append(d["transcript"]["full_text"])
        f = d["features"]
        features.append(AudioFeatures(
            mfcc_mean=f["mfcc_mean"], mfcc_std=f["mfcc_std"],
            pitch_mean=f["pitch_mean"], pitch_std=f["pitch_std"], pitch_variance=f["pitch_variance"],
            energy_mean=f["energy_mean"], energy_std=f["energy_std"],
            zero_crossing_rate=f["zero_crossing_rate"],
            spectral_centroid_mean=f["spectral_centroid_mean"],
            spectral_rolloff_mean=f["spectral_rolloff_mean"],
            pause_count=f["pause_count"], pause_duration_mean=f["pause_duration_mean"],
            speech_rate=f["speech_rate"],
            total_speech_duration=f["total_speech_duration"],
            total_pause_duration=f["total_pause_duration"],
        ))
        labels.append(label)
    return texts, features, labels
```

- [ ] **Step 2: Train text classifier**

Create `training/train_text.py`:

```python
"""Train TextClassifier on the train split, save to models/text_classifier.joblib."""

import sys, os, json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_classifier import TextClassifier
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

    # Fit a TF-IDF vocabulary + per-class coefficient matrix for explainability
    vec = TfidfVectorizer(max_features=2000, stop_words="english")
    X_tfidf = vec.fit_transform(texts)
    from sklearn.linear_model import LogisticRegression
    from src.layer3_constants import LABEL_TO_IDX
    y = np.array([LABEL_TO_IDX[l] for l in labels])
    lr = LogisticRegression(multi_class="multinomial", max_iter=1000, class_weight="balanced")
    lr.fit(X_tfidf, y)
    payload = {
        "vocabulary": vec.get_feature_names_out().tolist(),
        "coef": lr.coef_.tolist(),
        "classes": lr.classes_.tolist(),
    }
    VOCAB_PATH.write_text(json.dumps(payload))
    print(f"Saved text classifier → {MODEL_PATH}")
    print(f"Saved TF-IDF vocab → {VOCAB_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Train acoustic classifier**

Create `training/train_acoustic.py`:

```python
"""Train AcousticClassifier on the train split, save to models/acoustic_classifier.joblib."""

import sys, os, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.acoustic_classifier import AcousticClassifier
from training.load_processed import load_split


SPLIT_DIR = Path("data/tele_antifraud/processed/train")
MODEL_PATH = Path("models/acoustic_classifier.joblib")
IMPORTANCE_PATH = Path("models/acoustic_importance.json")


def main():
    _, features, labels = load_split(SPLIT_DIR)
    print(f"Loaded {len(features)} train samples")

    clf = AcousticClassifier()
    clf.train(features, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(MODEL_PATH))
    IMPORTANCE_PATH.write_text(json.dumps(clf.feature_importances(), indent=2))
    print(f"Saved → {MODEL_PATH}")
    print(f"Feature importances → {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Train fusion meta-classifier**

Create `training/train_fusion.py`:

```python
"""Train LateFusion on the validation split, save to models/fusion_classifier.joblib."""

import sys, os
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
    print(f"Saved → {FUSION_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Smoke test imports**

Run: `python -c "import training.train_text, training.train_acoustic, training.train_fusion; print('ok')"`
Expected: prints `ok` (imports succeed without executing `main()`).

- [ ] **Step 6: Commit**

```bash
git add training/load_processed.py training/train_text.py training/train_acoustic.py training/train_fusion.py
git commit -m "feat: training scripts — text, acoustic, and the fusion referee"
```

---

## Task 10: Evaluation harness with ablation

**Files:**
- Create: `training/evaluate.py`

- [ ] **Step 1: Write evaluate script**

Create `training/evaluate.py`:

```python
"""Evaluate Layer 3 on the test split. Produces:
  - reports/classification_report.txt
  - reports/confusion_matrix.png
  - reports/ablation.json  (text-only / acoustic-only / fused macro-F1)
"""

import sys, os, json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_classifier import TextClassifier
from src.acoustic_classifier import AcousticClassifier
from src.fusion import LateFusion
from src.layer3_constants import CLASS_LABELS
from training.load_processed import load_split


TEST_DIR = Path("data/tele_antifraud/processed/test")
REPORT_DIR = Path("reports")


def _argmax_label(probs):
    return max(probs, key=probs.get)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    texts, features, labels = load_split(TEST_DIR)
    print(f"Loaded {len(labels)} test samples")

    text_clf = TextClassifier.load("models/text_classifier.joblib")
    ac_clf = AcousticClassifier.load("models/acoustic_classifier.joblib")
    fusion = LateFusion.load("models/fusion_classifier.joblib")

    text_probs = [text_clf.predict_proba(t) for t in tqdm(texts, desc="text")]
    ac_probs = [ac_clf.predict_proba(f) for f in tqdm(features, desc="acoustic")]
    fused_probs = [fusion.predict_proba(t, a) for t, a in zip(text_probs, ac_probs)]

    text_pred = [_argmax_label(p) for p in text_probs]
    ac_pred = [_argmax_label(p) for p in ac_probs]
    fused_pred = [_argmax_label(p) for p in fused_probs]

    ablation = {
        "text_only_macro_f1": float(f1_score(labels, text_pred, average="macro", labels=CLASS_LABELS, zero_division=0)),
        "acoustic_only_macro_f1": float(f1_score(labels, ac_pred, average="macro", labels=CLASS_LABELS, zero_division=0)),
        "fused_macro_f1": float(f1_score(labels, fused_pred, average="macro", labels=CLASS_LABELS, zero_division=0)),
    }
    (REPORT_DIR / "ablation.json").write_text(json.dumps(ablation, indent=2))
    print("Ablation:", json.dumps(ablation, indent=2))

    report = classification_report(labels, fused_pred, labels=CLASS_LABELS, zero_division=0)
    (REPORT_DIR / "classification_report.txt").write_text(report)
    print(report)

    cm = confusion_matrix(labels, fused_pred, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_yticks(range(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Layer 3 Confusion Matrix (Fused)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=120)
    print(f"Wrote {REPORT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test import**

Run: `python -c "import training.evaluate; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add training/evaluate.py
git commit -m "feat: eval harness — confusion matrix + the money ablation table"
```

---

## Task 11: CLI — layer3_main.py

**Files:**
- Create: `layer3_main.py`
- Create: `src/detector_loader.py`

- [ ] **Step 1: Detector loader**

Create `src/detector_loader.py`:

```python
"""Load a fully-trained Layer3Detector from models/ and models/text_vocab.json."""

import json
from pathlib import Path
import numpy as np
from .detector import Layer3Detector
from .text_classifier import TextClassifier
from .acoustic_classifier import AcousticClassifier
from .fusion import LateFusion
from .layer3_constants import IDX_TO_LABEL


def load_trained_detector(models_dir: str = "models") -> Layer3Detector:
    mdir = Path(models_dir)
    text_clf = TextClassifier.load(str(mdir / "text_classifier.joblib"))
    ac_clf = AcousticClassifier.load(str(mdir / "acoustic_classifier.joblib"))
    fusion = LateFusion.load(str(mdir / "fusion_classifier.joblib"))

    vocab_path = mdir / "text_vocab.json"
    vocabulary, coefs = [], None
    if vocab_path.exists():
        payload = json.loads(vocab_path.read_text())
        vocabulary = payload["vocabulary"]
        # Use the "phishing" row of coefs for keyword attribution (most-salient class for capstone demo)
        classes = payload["classes"]
        coef_matrix = np.array(payload["coef"])
        phishing_idx_in_model = None
        for i, cls_idx in enumerate(classes):
            if IDX_TO_LABEL[int(cls_idx)] == "phishing":
                phishing_idx_in_model = i
                break
        coefs = coef_matrix[phishing_idx_in_model] if phishing_idx_in_model is not None else coef_matrix.mean(axis=0)

    return Layer3Detector(
        text_classifier=text_clf,
        acoustic_classifier=ac_clf,
        fusion=fusion,
        text_vocabulary=vocabulary,
        text_class_coefficients=coefs,
    )
```

- [ ] **Step 2: CLI script**

Create `layer3_main.py`:

```python
#!/usr/bin/env python3
"""Run Layer 1 + Layer 2 + Layer 3 on a single audio file, print DetectionResult JSON.

Usage: python layer3_main.py --input data/input/call.wav
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import AudioProcessor
from src.layer2_processor import Layer2Processor
from src.detector_loader import load_trained_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to 16kHz mono 16-bit WAV")
    ap.add_argument("--output", default=None, help="optional path to write DetectionResult JSON")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    l1 = AudioProcessor().process_file(args.input)
    if l1.status != "ready_for_processing":
        print(f"❌ Layer 1 failed: {l1.error_message}")
        sys.exit(1)

    l2 = Layer2Processor().process(args.input, file_id=l1.file_id)
    detector = load_trained_detector()
    result = detector.predict(l2)

    print(result.to_json())
    if args.output:
        result.save_to_file(args.output)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test imports**

Run: `python -c "import layer3_main; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add layer3_main.py src/detector_loader.py
git commit -m "feat: layer3_main.py CLI — one audio file in, a verdict out"
```

---

## Task 12: Wire Layer 3 into pipeline.py

**Files:**
- Modify: `pipeline.py`

- [ ] **Step 1: Read current pipeline.py**

Run: `cat pipeline.py`
Note the existing structure — L1 runs, then L2, then combined JSON is written.

- [ ] **Step 2: Update pipeline.py**

Replace the entire contents of `pipeline.py` with:

```python
#!/usr/bin/env python3

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import AudioProcessor
from src.layer2_processor import Layer2Processor
from src.detector_loader import load_trained_detector


def process_full_pipeline(audio_path: str):
    print("\n" + "=" * 60)
    print("🎯 FULL PIPELINE: Layer 1 + Layer 2 + Layer 3")
    print("=" * 60)

    print("\n📍 LAYER 1: Data Acquisition")
    print("-" * 60)
    l1 = AudioProcessor().process_file(audio_path)
    if l1.status != "ready_for_processing":
        print(f"❌ Layer 1 failed: {l1.error_message}")
        return
    print(f"✅ Metadata: duration={l1.metadata.duration_seconds}s, sr={l1.metadata.sample_rate_hz}Hz, ch={l1.metadata.channels}")
    l1_out = f"data/output/{l1.file_id}.json"
    os.makedirs("data/output", exist_ok=True)
    l1.save_to_file(l1_out)
    print(f"   Saved: {l1_out}")

    print("\n📍 LAYER 2: Signal & Text Processing")
    print("-" * 60)
    l2 = Layer2Processor().process(audio_path, file_id=l1.file_id)
    print(f"✅ Pitch {l2.features.pitch_mean:.1f}Hz | Rate {l2.features.speech_rate:.2f} syl/s | Pauses {l2.features.pause_count}")
    print(f"✅ Transcript: {l2.transcript.word_count} words — {l2.transcript.full_text[:80]}")
    l2_out = f"data/layer2_output/{l2.file_id}.json"
    os.makedirs("data/layer2_output", exist_ok=True)
    l2.save_to_file(l2_out)
    print(f"   Saved: {l2_out}")

    print("\n📍 LAYER 3: Fraud Detection")
    print("-" * 60)
    try:
        detector = load_trained_detector()
        l3 = detector.predict(l2)
        print(f"✅ Verdict: {l3.label}  (confidence {l3.confidence:.2%})")
        if l3.top_keywords:
            print(f"   Keywords: {', '.join(l3.top_keywords)}")
        if l3.top_acoustic_drivers:
            print(f"   Acoustic drivers: {', '.join(l3.top_acoustic_drivers)}")
        l3_out = f"data/layer3_output/{l1.file_id}.json"
        os.makedirs("data/layer3_output", exist_ok=True)
        l3.save_to_file(l3_out)
        print(f"   Saved: {l3_out}")
        l3_dict = l3.to_dict()
    except FileNotFoundError as e:
        print(f"⚠️  Layer 3 models not found — skipping. ({e})")
        print("   Train with: python training/train_text.py && python training/train_acoustic.py && python training/train_fusion.py")
        l3_dict = None

    combined = {"layer1": l1.to_dict(), "layer2": l2.to_dict(), "layer3": l3_dict}
    combined_path = f"data/combined/{l1.file_id}_combined.json"
    os.makedirs("data/combined", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n📦 Combined: {combined_path}")

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    try:
        process_full_pipeline(audio_file)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

- [ ] **Step 3: Smoke test import**

Run: `python -c "import pipeline; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Verify graceful fallback when models are missing**

If `models/text_classifier.joblib` does NOT exist yet: running `python pipeline.py nonexistent.wav` should exit cleanly at the file-not-found check — that's fine, the fallback only matters once Layer 2 succeeds. Skip full e2e test until models are trained in Task 13.

- [ ] **Step 5: Commit**

```bash
git add pipeline.py
git commit -m "feat: pipeline — now with Layer 3, fails soft if models aren't trained yet"
```

---

## Task 13: Colab training notebook

**Files:**
- Create: `notebooks/train_colab.ipynb`

- [ ] **Step 1: Create Colab notebook**

Create `notebooks/train_colab.ipynb` with this exact content (a valid Jupyter notebook JSON):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Layer 3 Training — Colab\n",
              "Runs dataset download, L1+L2 prep, trains text/acoustic/fusion, evaluates.\n",
              "Clone the repo, then run cells top-to-bottom. T4 GPU recommended."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["!git clone https://github.com/REPLACE_ME/vibhu-capstone.git\n",
              "%cd vibhu-capstone\n",
              "!pip install -q -r requirements.txt\n",
              "!pip install -q -r requirements_layer2.txt\n",
              "!pip install -q -r requirements_layer3.txt"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["# Download Vosk model for ASR\n",
              "!mkdir -p models && cd models && wget -q https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && unzip -q vosk-model-small-en-us-0.15.zip"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["# 1. Download dataset\n",
              "!python training/download_dataset.py"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["# 2. Prepare data (slow — ASR over thousands of files)\n",
              "!python training/prepare_data.py --split train\n",
              "!python training/prepare_data.py --split validation\n",
              "!python training/prepare_data.py --split test"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["# 3. Train\n",
              "!python training/train_text.py\n",
              "!python training/train_acoustic.py\n",
              "!python training/train_fusion.py"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["# 4. Evaluate\n",
              "!python training/evaluate.py\n",
              "from IPython.display import Image\n",
              "Image('reports/confusion_matrix.png')"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["# 5. Zip trained artifacts for download\n",
              "!zip -r trained_models.zip models/text_classifier.joblib models/text_vocab.json models/acoustic_classifier.joblib models/fusion_classifier.joblib models/acoustic_importance.json reports/\n",
              "from google.colab import files\n",
              "files.download('trained_models.zip')"]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Validate notebook JSON**

Run: `python -c "import json; json.load(open('notebooks/train_colab.ipynb')); print('valid')"`
Expected: prints `valid`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/train_colab.ipynb
git commit -m "feat: colab notebook — click play, get trained models"
```

---

## Task 14: README update + final acceptance checks

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README.md contents**

Overwrite `README.md` with:

```markdown
# Voice Phishing Detection System

Multi-layer voice phishing detection. Takes in a call recording, emits a fraud verdict with explanation.

## Quick Start

```bash
source venv/bin/activate
python pipeline.py data/input/yourfile_16k.wav
```

## Layers

### Layer 1 — Data Acquisition
```bash
python main.py --input data/input/call.wav
```
Output: metadata (duration, sample rate, channels, checksum).

### Layer 2 — Signal & Text Processing
```bash
python layer2_main.py --input data/input/call.wav
```
Output: 39-dim acoustic features + Vosk transcript.

### Layer 3 — Multimodal Fraud Detection
```bash
python layer3_main.py --input data/input/call.wav
```
Output: `DetectionResult` — one of 8 classes (legitimate / phishing / banking / investment / kidnapping / lottery / customer_service / identity_theft) with confidence, per-modality breakdown, top keywords, top acoustic drivers.

### Full Pipeline
```bash
python pipeline.py data/input/call.wav
```

## Training Layer 3

Use Colab (free tier is enough). Open `notebooks/train_colab.ipynb`, run all cells. Download `trained_models.zip`, unzip into the repo root. Models land in `models/`.

Dataset: [TeleAntiFraud-28k](https://arxiv.org/abs/2503.24115).

## Input Format

Audio must be 16kHz, mono, 16-bit WAV. Use `python resample.py input.wav output_16k.wav` to convert.

## Tests

```bash
pytest tests/ -v
```
```

- [ ] **Step 2: Run all unit tests**

Run: `pytest tests/ -v`
Expected: all tests pass (11+ tests across Tasks 1–7).

- [ ] **Step 3: Verify acceptance criteria from spec**

Run these checks (once models are trained via Colab):

```bash
# AC-1: layer3_main produces DetectionResult JSON
python layer3_main.py --input data/input/sample_16k.wav

# AC-2: pipeline produces combined JSON with layer3 key
python pipeline.py data/input/sample_16k.wav
python -c "import json; d=json.load(open('data/combined/sample_16k_combined.json' if False else sorted(__import__('glob').glob('data/combined/*_combined.json'))[-1])); assert 'layer3' in d; print('layer3 key present ✓')"

# AC-3: evaluate produces ablation + confusion matrix
python training/evaluate.py
ls reports/confusion_matrix.png reports/ablation.json reports/classification_report.txt
```

Expected: all three succeed, print the outputs described in the spec.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README — layer 3 is alive, here's how to run it"
```
