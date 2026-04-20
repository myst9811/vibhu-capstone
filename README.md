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
