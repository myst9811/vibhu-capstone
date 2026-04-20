# Handoff — Voice Phishing Detection (Layer 3)

Hey. The code is done. Your job is to train the models on Colab, drop the artifacts back in, and run the full pipeline end-to-end. Should take a day, maybe two.

## What's already done

- **Layer 1** (data acquisition) and **Layer 2** (signal features + Vosk transcript) — untouched, already working.
- **Layer 3** (fraud detection) — all code written and unit-tested (14 tests, all green). Still needs to be **trained**.
- Design spec: `docs/superpowers/specs/2026-04-20-layer3-detection-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-20-layer3-detection.md`
- Architecture: late-fusion of two classifiers (text: sentence-embeddings + LogReg; acoustic: XGBoost on 39-dim features). 8 output classes: `legitimate`, `phishing`, `banking`, `investment`, `kidnapping`, `lottery`, `customer_service`, `identity_theft`.

## What you need to do

### 1. Local environment setup (one-time)

```bash
git clone <this-repo>
cd vibhu-capstone
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements_layer2.txt -r requirements_layer3.txt
```

**Mac gotcha:** xgboost needs OpenMP. If you hit a `libomp.dylib` error, run `brew install libomp`.

Verify:
```bash
pytest tests/ -v    # should show 28 passed
```

### 2. Download the Vosk ASR model (one-time)

```bash
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..
```

### 3. Train the models on Colab

Open `notebooks/train_colab.ipynb` in Google Colab. Runtime → Change runtime type → **T4 GPU**.

In **cell 2**, replace `REPLACE_WITH_YOUR_REPO_URL` with your GitHub repo URL.

Run cells top-to-bottom. Expect:
- **Cells 1–3** (Drive mount + env setup): ~5 min
- **Cell 4** (dataset download): ~5 min
- **Cell 5** (prep — runs ASR on 28k clips): **several hours**. Sleep on it. It's resume-safe — Colab disconnects don't erase progress. Re-run cells 1–3 to restore the environment after reconnect, then re-run cell 5 and it picks up where it stopped.
- **Cell 6** (train all 3 classifiers): ~10–15 min
- **Cell 7** (evaluate): ~2 min — produces the confusion matrix and ablation table.
- **Cell 8**: prints the ablation table inline.
- **Cell 9**: zips artifacts and triggers a browser download of `trained_models.zip`.

**Dataset location:**
- HuggingFace: [`JimmyMa99/TeleAntiFraud`](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud)
- GitHub (paper + code): https://github.com/JimmyMa99/TeleAntiFraud
- Paper: https://arxiv.org/abs/2503.24115

28,511 speech-text pairs, 307 hours of audio. The download script already points here. If the upstream ID ever changes, override with `%env TELE_DATASET_ID=<new/path>` in Colab before cell 3.

### 4. Drop the trained models back in

On your local machine:
```bash
cd vibhu-capstone
unzip ~/Downloads/trained_models.zip   # creates models/*.joblib + reports/
```

You should now have:
- `models/text_classifier.joblib`
- `models/text_vocab.json`
- `models/acoustic_classifier.joblib`
- `models/fusion_classifier.joblib`
- `models/acoustic_importance.json`
- `reports/confusion_matrix.png`
- `reports/classification_report.txt`
- `reports/ablation.json`

### 5. Run end-to-end

Any 16kHz mono 16-bit WAV file:
```bash
python layer3_main.py --input data/input/your_call.wav
```

Or the full three-layer pipeline:
```bash
python pipeline.py data/input/your_call.wav
```

If your audio isn't 16kHz mono, convert it first:
```bash
python resample.py input.wav output_16k.wav
```

## What "working" looks like

`python layer3_main.py --input <wav>` prints a JSON like:
```json
{
  "file_id": "...",
  "label": "phishing",
  "confidence": 0.87,
  "class_probabilities": { ... },
  "per_modality": { "text": {...}, "acoustic": {...}, "fused": {...} },
  "top_keywords": ["verify", "account", "urgent", "wire", "transfer"],
  "top_acoustic_drivers": ["speech_rate", "pitch_variance", ...]
}
```

## For the defense presentation

The **ablation table** in `reports/ablation.json` is your money slide. It compares:
- Text-only macro-F1
- Acoustic-only macro-F1
- Fused macro-F1

If `fused > max(text, acoustic)`, you've demonstrated that multimodal fusion actually helps. If not, that's a finding worth discussing (one modality dominates).

The **confusion matrix** (`reports/confusion_matrix.png`) shows which classes get confused with which — useful for pointing out model weaknesses honestly.

The **top keywords** and **top acoustic drivers** on a sample prediction give you a human-readable explanation for each verdict — that's the explainability angle.

## If something breaks

- **Tests fail locally:** `pytest tests/ -v` — if any of the 28 break, something in the environment drifted. Reinstall deps.
- **`python pipeline.py` says "Layer 3 models not found":** you haven't unzipped `trained_models.zip` into the repo root yet. That's Step 4.
- **Vosk errors:** the model isn't in `models/vosk-model-small-en-us-0.15/`. Re-run Step 2.
- **Audio errors:** input wasn't 16kHz mono 16-bit WAV. Run it through `resample.py` first.

## File map (where stuff lives)

```
src/
  layer3_models.py          DetectionResult dataclass
  layer3_constants.py       8 class labels
  feature_vector.py         Flattens AudioFeatures -> 39-dim vector
  text_classifier.py        Sentence embeddings + LogReg
  acoustic_classifier.py    XGBoost on acoustic features
  fusion.py                 Late-fusion meta-classifier
  explainability.py         Top keywords + top acoustic drivers
  detector.py               Layer3Detector wrapper
  detector_loader.py        Loads trained artifacts from models/

training/
  download_dataset.py       HuggingFace fetch
  prepare_data.py           Runs L1+L2 over dataset, caches JSONs
  load_processed.py         Reads cached JSONs back for training
  train_text.py             Trains text classifier + TF-IDF vocab
  train_acoustic.py         Trains XGBoost acoustic classifier
  train_fusion.py           Trains fusion meta-classifier
  evaluate.py               Confusion matrix + ablation table

tests/                      28 unit tests, all green
notebooks/train_colab.ipynb Colab training notebook
layer3_main.py              Single-file CLI
pipeline.py                 Full L1+L2+L3 pipeline
```

That's it. Text me if something blows up.
