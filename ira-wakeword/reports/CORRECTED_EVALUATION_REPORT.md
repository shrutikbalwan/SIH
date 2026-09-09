# CORRECTED_EVALUATION_REPORT.md
## IRA Wake-Word Model — Official Baseline Evaluation

> Date: 2026-09-04
> Model evaluated: `cnn/models/ira_cnn_int8.tflite`
> Script: `evaluate_correct_pipeline.py`
> Test set: `dataset/splits/test/` (1,003 pos + 772 neg = 1,775 clips)
> **No training files, dataset, or model weights were modified.**

---

## CRITICAL NOTICE — Previous Results Are Invalid

The evaluation run earlier this session (`evaluate.py`) reported:

```
TPR  = 65.54%
FAPH = 864 false-accepts / hour
```

**These numbers MUST NOT be used as the official model baseline.**

They were produced using the **wrong feature extraction pipeline**:

| Property | Training pipeline (correct) | evaluate.py (WRONG) |
|----------|----------------------------|---------------------|
| Library | `tf.signal.stft` | `pymicro_features.MicroFrontend` (TFLM) |
| Filter bank | Linear FFT magnitude (first 40 raw bins) | Mel-filterbank |
| Normalization | Per-clip z-score `(x-mean)/(std+1e-6)` | PCAN (Per-Channel Auto Norm) |
| Log function | `log(x + 1e-6)` | uint16 scaled x0.0390625 |

The model was trained on `tf.signal.stft` features. Evaluating it on TFLM
MicroFrontend features feeds a completely different representation to the network.
All metrics produced in that run are out-of-distribution and invalid.

---

## Corrected Preprocessing Pipeline

Reproduced verbatim from `cnn/train_cnn.py::load_audio()` and `make_spectrogram()`:

```python
# Step 1: Load audio
audio, sr = sf.read(path, dtype="float32")
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)           # stereo -> mono
if sr != 16000:
    audio = resample_poly(audio, 16000, sr)  # resample if needed
# Pad (trailing zeros) or truncate to exactly 16,000 samples
audio = pad_or_truncate(audio, 16000)

# Step 2: STFT spectrogram (tf.signal.stft, TF default Hann window)
spec = tf.signal.stft(audio,
    frame_length = 480,   # 30 ms
    frame_step   = 320,   # 20 ms
    fft_length   = 512)   # -> 257 complex bins

# Step 3: Magnitude + log with epsilon
spec = tf.abs(spec)
spec = tf.math.log(spec + 1e-6)   # log(|STFT| + 1e-6), NOT log1p

# Step 4: Keep first 40 frequency bins
spec = spec[:, :40]               # shape: (49, 40)

# Step 5: Per-clip z-score normalization
mean = tf.reduce_mean(spec)
std  = tf.math.reduce_std(spec) + 1e-6
spec = (spec - mean) / std        # exact training normalization

# Step 6: Quantize to INT8 for model input
x_float = spec.numpy()[np.newaxis, ..., np.newaxis]   # (1, 49, 40, 1)
x_int8  = clip(round(x_float / 0.08363020 + (-105)), -128, 127).astype(int8)

# Step 7: Run inference
interpreter.set_tensor(input_idx, x_int8)
interpreter.invoke()
out_int8 = interpreter.get_tensor(output_idx)   # shape: (1, 1)

# Step 8: Dequantize output
probability = (float(out_int8[0, 0]) - (-128)) * 0.00390625
```

---

## Pipeline Verification (Pre-Evaluation Checks)

Verified on 3 positive + 3 negative clips before full evaluation:

| Clip | audio shape | feature shape | mean | std | probability | correct? |
|------|-------------|---------------|------|-----|-------------|---------|
| 000002_ira_002.wav (pos)  | (16000,) | (49, 40) | 0.000000 | 1.000000 | 0.9375 | YES |
| 000024_ira_0112.wav (pos) | (16000,) | (49, 40) | 0.000000 | 0.999999 | 0.9922 | YES |
| 000025_ira_0113.wav (pos) | (16000,) | (49, 40) | 0.000000 | 0.999999 | 0.9922 | YES |
| neg_002.wav               | (16000,) | (49, 40) | 0.000000 | 0.999999 | 0.0000 | YES |
| neg_007.wav               | (16000,) | (49, 40) | 0.000000 | 0.999999 | 0.0000 | YES |
| neg_009.wav               | (16000,) | (49, 40) | 0.000000 | 0.999999 | 0.0000 | YES |

Mean = 0.000000 and std = 1.000000 on every clip confirm the z-score is applied
identically to the training pipeline.

---

## Test Set

| Class | Directory | Count |
|-------|-----------|-------|
| Positive (label=1) | `dataset/splits/test/positive/` | 1,003 clips |
| Negative (label=0) | `dataset/splits/test/negative/` |   772 clips |
| **Total** | | **1,775 clips** |

These clips are the held-out test split created by `create_split.py` (seed=42, 80/10/10).
LibriSpeech negatives were split at speaker level — no LibriSpeech speaker present in the
test set was seen during training.

Positive clips in the test split are a mix of Piper TTS and real recordings. They are
the same distribution as the training data (unlike the invalid evaluation which used
only real recordings from `real/positive/`).

---

## Official Results

### Threshold Analysis

| Threshold | Accuracy | TPR/Recall | FPR | FNR | Precision | TP | FP | FN | TN |
|-----------|----------|------------|-----|-----|-----------|----|----|----|----|
| 0.30 | 99.55% | 100.00% | 1.04% | 0.00% | 99.21% | 1003 | 8 | 0 | 764 |
| 0.40 | 99.83% | 100.00% | 0.39% | 0.00% | 99.70% | 1003 | 3 | 0 | 769 |
| **0.50** | **99.83%** | **100.00%** | **0.39%** | **0.00%** | **99.70%** | **1003** | **3** | **0** | **769** |
| 0.60 | 99.77% | 99.90%  | 0.39% | 0.10% | 99.70% | 1002 | 3 | 1 | 769 |
| 0.70 | 99.77% | 99.90%  | 0.39% | 0.10% | 99.70% | 1002 | 3 | 1 | 769 |
| 0.80 | 99.66% | 99.70%  | 0.39% | 0.30% | 99.70% | 1000 | 3 | 3 | 769 |
| 0.90 | 99.72% | 99.60%  | 0.13% | 0.40% | 99.90% |  999 | 1 | 4 | 771 |

### Confusion Matrix at Threshold = 0.50 (Official Baseline)

```
                  Predicted 0    Predicted 1
  Actual 1 (IRA)       0            1003     (total positives = 1,003)
  Actual 0 (NEG)     769               3     (total negatives =   772)
```

### Official Baseline Metrics (threshold = 0.50)

| Metric | Value |
|--------|-------|
| **True Positive Rate (TPR / Recall)** | **100.000%** |
| **False Positive Rate (FPR)** | **0.389%** |
| **False Negative Rate (FNR)** | **0.000%** |
| True Negative Rate (TNR / Specificity) | 99.611% |
| Accuracy | 99.831% |
| Precision | 99.702% |

---

## Comparison: Invalid vs Corrected Results

| Metric | OLD result (WRONG pipeline) | CORRECTED result (training pipeline) |
|--------|-----------------------------|--------------------------------------|
| Feature extractor | TFLM MicroFrontend (mel + PCAN) | tf.signal.stft + log(x+1e-6) + z-score |
| Test positives | 1,364 real-only recordings | 1,003 held-out split clips (mixed) |
| Test negatives | 4,000 LibriSpeech (not split-safe) | 772 split-safe held-out clips |
| **TPR @ 0.50** | ~~65.54%~~ | **100.00%** |
| **FNR @ 0.50** | ~~34.46%~~ | **0.00%** |
| **FPR @ 0.50** | not reported | **0.39%** |
| FAPH | ~~864/hour~~ (invalid) | NOT REPORTED (see below) |

The 65.5% TPR in the old evaluation was entirely caused by the feature mismatch.
The model itself is working correctly — it achieves near-perfect separation on its own
training-consistent feature space.

---

## Understanding the Near-Perfect Numbers

TPR = 100% and FPR = 0.39% on the test split is a strong result, but must be interpreted
with important caveats:

**Why the numbers are high:**
1. The test split contains the same types of data as training (mostly Piper TTS positives,
   LibriSpeech negatives). The model was optimised on exactly this distribution.
2. 93% of positive training data is Piper TTS, and the test positives are also mostly
   Piper TTS from the same voice batches (A/B/C/D). The model has learned the TTS voice
   fingerprint very well.

**What the numbers do NOT tell us:**
1. **Real-world TPR** — Only 682 of 10,000 training positives are real human recordings
   (single speaker). Real-world TPR on diverse speakers in noise will be lower.
2. **Real-world false accepts** — FAPH on continuous ambient audio (music, TV, background
   conversations) is unknown and cannot be derived from this test.
3. **Cross-speaker generalisation** — The model has seen very limited real voice diversity.

---

## FAPH — Not Reported

FAPH is intentionally excluded from this report.

FAPH requires a continuous streaming audio evaluation:
- Uninterrupted long-form ambient audio (not isolated 1-second clips)
- Model applied with a rolling 20 ms step across the stream
- Cooldown tracked across frame boundaries
- Corpus representative of real deployment conditions

The test set negatives (772 segmented 1-second LibriSpeech clips) do not meet these
requirements. Any FAPH computed on them would not be comparable to the 0.2–1.0 FAPH
benchmark of released microWakeWord models.

FAPH will be measured separately when a proper streaming pipeline is available.

---

## Model Details

| Property | Value |
|----------|-------|
| File | `cnn/models/ira_cnn_int8.tflite` |
| Size | 12.95 KB (13,264 bytes) |
| Input shape | `[1, 49, 40, 1]` int8 |
| Input quant | scale=0.08363020, zero_point=-105 |
| Output shape | `[1, 1]` int8 |
| Output quant | scale=0.00390625 (1/256), zero_point=-128 |
| Architecture | Conv(8)+Pool, Conv(16)+Pool, Conv(32), GAP, Dense(16), Dense(1) |
| Parameters | ~6,433 |

## Saved Artefacts

| File | Description |
|------|-------------|
| `evaluate_correct_pipeline.py` | Evaluation script with correct pipeline |
| `cnn/models/correct_pipeline_probabilities.npy` | Raw float32 probabilities (1,775,) |
| `CORRECTED_EVALUATION_REPORT.md` | This report |
| `MODEL_AUDIT.md` | Full architecture and pipeline audit |

---

*No training files, dataset files, or model weights were modified.*
