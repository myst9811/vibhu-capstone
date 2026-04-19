# Graph Report - .  (2026-04-20)

## Corpus Check
- Corpus is ~3,546 words - fits in a single context window. You may not need a graph.

## Summary
- 120 nodes · 177 edges · 12 communities detected
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Layer 2 Signal & Text Processing Dependencies` - 15 edges
2. `SignalProcessor` - 13 edges
3. `AudioProcessor` - 11 edges
4. `MetadataExtractor` - 9 edges
5. `SpeechRecognizer` - 7 edges
6. `Layer2Processor` - 7 edges
7. `AudioValidator` - 6 edges
8. `print_header()` - 5 edges
9. `print_section()` - 5 edges
10. `demonstrate_layer1()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `main.py` --shares_data_with--> `Layer 1 Dependencies`  [INFERRED]
  README.md → requirements.txt
- `Layer 2 Output: audio features + transcript` --shares_data_with--> `librosa (>=0.10.0)`  [INFERRED]
  README.md → requirements_layer2.txt
- `layer2_main.py` --shares_data_with--> `vosk (==0.3.44)`  [INFERRED]
  README.md → requirements_layer2.txt
- `Layer 2 Output: audio features + transcript` --shares_data_with--> `vosk (==0.3.44)`  [INFERRED]
  README.md → requirements_layer2.txt
- `soxr (>=0.3.0)` --shares_data_with--> `resample.py`  [INFERRED]
  requirements_layer2.txt → README.md

## Hyperedges (group relationships)
- **Full pipeline combines Layer 1 and Layer 2** — readme_pipeline_py, readme_main_py, readme_layer2_main_py, readme_full_pipeline [EXTRACTED 0.90]
- **Layer 2 audio processing stack** — requirements_layer2_librosa, requirements_layer2_soundfile, requirements_layer2_audioread, requirements_layer2_soxr [INFERRED 0.80]
- **Layer 2 ML and speech recognition stack** — requirements_layer2_numpy, requirements_layer2_scipy, requirements_layer2_sklearn, requirements_layer2_vosk, requirements_layer2_joblib [INFERRED 0.80]

## Communities

### Community 0 - "Layer 2 Data Models"
Cohesion: 0.13
Nodes (6): AudioFeatures, ProcessedAudio, Transcript, TranscriptSegment, Layer2Processor, SpeechRecognizer

### Community 1 - "Layer 2 Pipeline & Rationale"
Cohesion: 0.12
Nodes (21): Audio format: 16kHz mono 16-bit WAV, layer2_main.py, Layer 2 Output: audio features + transcript, Layer 2: Signal & Text Processing, Rationale: Use resample.py if audio not 16kHz mono 16-bit, resample.py, audioread (>=3.0.0), decorator (>=5.0.0) (+13 more)

### Community 2 - "Demo & Results Showcase"
Cohesion: 0.27
Nodes (12): demonstrate_combined(), demonstrate_layer1(), demonstrate_layer2(), main(), print_header(), print_section(), Print a formatted header, Show how layers work together (+4 more)

### Community 3 - "Audio Processing Core"
Cohesion: 0.23
Nodes (3): AudioProcessor, ValidationResult, AudioValidator

### Community 4 - "Layer 1 Modules & Metadata"
Cohesion: 0.29
Nodes (2): AudioMetadata, ProcessingResult

### Community 5 - "Signal Feature Extraction"
Cohesion: 0.33
Nodes (1): SignalProcessor

### Community 6 - "Pipeline Orchestration"
Cohesion: 0.28
Nodes (9): Full Pipeline, Layer 1: Data Acquisition, Layer 1 Output: duration, sample rate, channels, checksum, main.py, pipeline.py, Voice Phishing Detection System, Layer 1 Dependencies, mutagen (>=1.47.0) (+1 more)

### Community 7 - "Metadata Extraction"
Cohesion: 0.43
Nodes (1): MetadataExtractor

### Community 8 - "Layer 1 Entrypoint"
Cohesion: 0.7
Nodes (4): load_config(), main(), process_batch(), process_single_file()

### Community 9 - "Audio Resampling"
Cohesion: 1.0
Nodes (0): 

### Community 10 - "Layer 2 Entrypoint"
Cohesion: 1.0
Nodes (0): 

### Community 11 - "Pipeline Entry"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **18 isolated node(s):** `Print a formatted header`, `Print a formatted section`, `Demonstrate Layer 1: Data Acquisition results`, `Demonstrate Layer 2: Signal & Text Processing results`, `Show how layers work together` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Audio Resampling`** (2 nodes): `resample.py`, `resample_to_16khz()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Layer 2 Entrypoint`** (2 nodes): `layer2_main.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pipeline Entry`** (2 nodes): `pipeline.py`, `process_full_pipeline()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SignalProcessor` connect `Signal Feature Extraction` to `Layer 2 Data Models`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `AudioProcessor` connect `Audio Processing Core` to `Layer 1 Modules & Metadata`, `Metadata Extraction`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Layer 2 Signal & Text Processing Dependencies` (e.g. with `numpy (>=2.0.0)` and `scipy (>=1.14.0)`) actually correct?**
  _`Layer 2 Signal & Text Processing Dependencies` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `SignalProcessor` (e.g. with `AudioFeatures` and `Layer2Processor`) actually correct?**
  _`SignalProcessor` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `AudioProcessor` (e.g. with `AudioValidator` and `MetadataExtractor`) actually correct?**
  _`AudioProcessor` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `MetadataExtractor` (e.g. with `AudioProcessor` and `AudioMetadata`) actually correct?**
  _`MetadataExtractor` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `SpeechRecognizer` (e.g. with `Transcript` and `TranscriptSegment`) actually correct?**
  _`SpeechRecognizer` has 3 INFERRED edges - model-reasoned connections that need verification._