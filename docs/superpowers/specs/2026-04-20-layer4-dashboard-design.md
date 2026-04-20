# Layer 4 — Analytics Dashboard Design

**Date:** 2026-04-20
**Status:** Draft — awaiting user review
**Context:** Capstone extension to voice phishing detection pipeline (Layers 1-3). Layers 1-3 are feature-complete and produce `DetectionResult` JSON.

## Goal

A Streamlit dashboard that accepts a WAV upload, runs the existing L1+L2+L3 pipeline, and displays the fraud verdict alongside a colour-coded highlighted transcript. Finishable in one evening. Suitable for defense presentations and portfolio screenshots.

## Non-goals (explicit YAGNI)

- No real-time/streaming audio (deferred stretch goal per user)
- No history panel, no per-modality drill-down (user chose "Standard" over "Complete" scope)
- No authentication, no multi-file batch upload
- No async processing or progress polling — sync with spinner is sufficient for ~30s demo calls
- No separate FastAPI backend — Streamlit imports pipeline modules directly

## Architecture

Single-process Streamlit app. Imports existing pipeline modules directly; no HTTP layer, no subprocess.

```
streamlit_app.py
    │
    ├─ st.file_uploader(type=["wav"])  →  tempfile path
    │                                           │
    │                                           ▼
    │                              resample.py (if sr≠16k / ch≠1)
    │                                           │
    │                                           ▼
    ├─ pipeline_runner.run(path)
    │       AudioProcessor().process_file(path)         ── Layer 1
    │       Layer2Processor().process(path, file_id)    ── Layer 2
    │       load_trained_detector().predict(l2)         ── Layer 3
    │                                           │
    │                                           ▼
    │                   (l1: AudioFile, l2: ProcessedAudio, l3: DetectionResult)
    │                                           │
    │                                           ▼
    └─ render_verdict_banner · render_class_probs · render_audio
       render_transcript_highlighted · render_keyword_chips · render_driver_chips
```

## Components & File Structure

### New files

| Path | Responsibility | ~LOC |
|---|---|---|
| `streamlit_app.py` | Entry point. Upload widget, orchestration, page layout. Calls `dashboard/*` helpers. | 150 |
| `src/phishing_vocabulary.py` | Module-level constant: `PHISHING_VOCABULARY: frozenset[str]` — ~40 curated phrases. | 50 |
| `src/dashboard/__init__.py` | Empty package marker. | 1 |
| `src/dashboard/pipeline_runner.py` | `run_pipeline(audio_path: str) -> PipelineResult` — wraps L1+L2+L3 with a single try/except, returns a dataclass combining all three outputs (or a typed error). | 80 |
| `src/dashboard/highlighting.py` | `render_highlighted_transcript(transcript_segments, top_keywords, vocabulary) -> str` — pure function returning HTML. Three colour classes: `kw-both` (red, keyword AND in vocab), `kw-model` (yellow, model-only), `kw-vocab` (blue underline, vocab-only). | 80 |
| `src/dashboard/components.py` | Streamlit render helpers: `render_verdict_banner(l3)`, `render_class_probs_chart(l3)`, `render_keyword_chips(l3)`, `render_driver_chips(l3)`, `render_audio_player(path)`. | 100 |
| `tests/test_highlighting.py` | Unit tests for `render_highlighted_transcript` across all three colour categories + empty inputs. | 80 |
| `tests/test_pipeline_runner.py` | Smoke test with monkeypatched L1/L2/L3 — verifies error paths (non-WAV, models-missing) return typed errors rather than raising. | 60 |
| `tests/test_phishing_vocabulary.py` | Assertions that curated list contains canonical phrases (bank account, social security, wire transfer, etc.) and is normalised (lowercase, trimmed). | 30 |

### Modified files

| Path | Change |
|---|---|
| `README.md` | Add "Layer 4 — Dashboard" section with `streamlit run streamlit_app.py` instructions and a link to a screenshot. |
| `requirements.txt` | Append `streamlit>=1.32,<2.0` and `plotly>=5.18,<6.0`. |

### Phishing vocabulary (curated list)

Must include all phrases the user's friend flagged plus adjacent high-signal terms. Final list:

```
bank account, social security, ssn, credit card, debit card, routing number,
wire transfer, gift card, bitcoin, cryptocurrency, verify, confirm, urgent,
immediately, suspended, locked, arrest, warrant, lawsuit, police, irs,
refund, prize, won, lottery, inheritance, investment opportunity, guaranteed return,
microsoft, apple support, tech support, remote access, password, pin,
one time password, otp, authentication code, kidnapped, hostage, ransom
```

Stored as `frozenset[str]` — lowercase, single- and multi-word phrases. Highlighting does case-insensitive match on whole words/phrases.

## Data Flow

1. User drops a WAV into `st.file_uploader` → Streamlit writes it to a temp path.
2. `streamlit_app.py` reads sample rate with `soundfile.info()`. If sr≠16000 or channels≠1, call `resample.resample_to_16k_mono(src, dst)` → get a 16kHz mono copy.
3. Pass the (possibly resampled) path to `pipeline_runner.run(path)`.
4. Runner calls L1, L2, L3 in sequence. Returns `PipelineResult` — a dataclass with `layer1`, `layer2`, `layer3`, and `error: Optional[str]`.
5. On success: render six panels (banner, class-prob chart, audio player, highlighted transcript, keyword chips, driver chips).
6. On error: render `st.error(result.error)` with a help message.

### UI layout (single page, scroll)

```
┌─────────────────────────────────────────────────────┐
│ Voice Phishing Detection — Layer 4 Dashboard       │
├─────────────────────────────────────────────────────┤
│ [ Upload WAV file ]  (drop zone)                   │
├─────────────────────────────────────────────────────┤
│ ⚠ PHISHING  |  Confidence: 87%                     │  ← red/green/amber verdict banner
├─────────────────────────────────────────────────────┤
│ [ Class probabilities bar chart (Plotly) ]         │
│ [ 🔊 Audio player ]                                │
├─────────────────────────────────────────────────────┤
│ Transcript:                                        │
│ hello sir, I'm calling from your [bank]red         │
│ about a charge on your [account]red. Please        │
│ [verify]yellow your [social security]blue number.  │
│                                                    │
│ Legend: ■ model+curated  ■ model-only  ▁ curated   │
├─────────────────────────────────────────────────────┤
│ Top keywords:  [verify] [account] [urgent] [bank]  │  ← chip row
│ Top drivers:   [speech_rate] [pitch_variance]      │  ← chip row
└─────────────────────────────────────────────────────┘
```

Verdict banner colour: `legitimate` → green, `customer_service` → grey, everything else → red. Confidence shown as percent.

### Highlighting algorithm

Input: `transcript_segments: list[dict]` (each has `text`, `start_time`, `end_time`), `top_keywords: list[str]`, `vocabulary: frozenset[str]`.

1. Build a set of keyword strings (lowercase) from `top_keywords`.
2. Walk the segments in order, concatenating words into a running string with single spaces.
3. For each word-or-phrase boundary, check membership in three ways:
   - Word in `keywords` AND in `vocabulary` → wrap in `<mark class="kw-both">`.
   - Word in `keywords` only → `<mark class="kw-model">`.
   - Word in `vocabulary` only (single-word hit) → `<mark class="kw-vocab">`.
4. For multi-word vocabulary entries ("social security", "bank account"), match against consecutive segment runs using a tokenised sliding window.
5. Inject CSS for the three classes in a `st.markdown(..., unsafe_allow_html=True)` once per render.

The highlighting function is pure — takes data, returns HTML string. Testable without Streamlit.

## Error Handling

All at the Streamlit layer, rendered with `st.error()` + actionable next-step text:

| Condition | User-visible message |
|---|---|
| Non-WAV file uploaded | "Please upload a WAV file. Convert other formats with `ffmpeg -i in.mp3 out.wav`." |
| WAV too short (<1s) | "Audio too short — need at least 1 second of speech." |
| Layer 3 models missing (`FileNotFoundError` from `load_trained_detector`) | "Models not trained yet. Run training per `docs/HANDOFF.md`." |
| Vosk model missing | "Vosk ASR model not found at `models/vosk-model-small-en-us-0.15/`. See README Step 2." |
| Any other exception during pipeline | `st.error("Pipeline failed: {type}")` with full traceback in an `st.expander("Details")`. |

The `pipeline_runner.run()` catches `FileNotFoundError` and returns a typed error; everything else propagates and is caught at the app level.

## Testing Strategy

**Unit tests (pytest) — must all pass:**

- `test_highlighting.py` — table-driven tests covering:
  - Empty transcript → empty string
  - Keyword-only match → yellow class
  - Vocab-only match → blue class
  - Both → red class
  - Multi-word vocab phrase ("social security") → single span across two tokens
  - Case-insensitive matching
  - Punctuation preservation
- `test_pipeline_runner.py` — monkeypatch L1/L2/L3 calls:
  - Happy path returns `PipelineResult` with all three layers populated
  - `FileNotFoundError` from L3 loader → `PipelineResult(error="models_not_trained")`
  - Unexpected exception → re-raised (app handles)
- `test_phishing_vocabulary.py` — asserts canonical phrases are present, all lowercase, no leading/trailing whitespace.

**Not unit-tested:**

- Streamlit component rendering (UI tests are expensive, low-ROI for a single-developer dashboard). Manual smoke test: run `streamlit run streamlit_app.py`, upload `data/input/test2_16k.wav`, visually confirm all six panels render.

**Integration test:** CLI smoke test `python -c "from src.dashboard.pipeline_runner import run; r = run('data/input/test2_16k.wav'); print(r.layer3.label)"` — runnable after models are trained.

## Implementation Order (high level — exact steps in the plan)

1. Curated vocabulary constant (pure data).
2. Highlighting function (pure, TDD).
3. Pipeline runner wrapper (TDD with mocks).
4. Streamlit component helpers (verdict, chart, chips, audio).
5. `streamlit_app.py` orchestration.
6. Manual smoke test, README update, commit.

## Open Questions

None — all prior gating decisions (stack, scope, highlighting) are locked.
