# MODEL_AUDIT.md — IRA Wake-Word Project

> Generated: 2026-09-04
> Audited by: static source analysis + live TFLite tensor inspection
> **No files were modified during this audit.**

---

## Forensic Summary: Where the reported numbers came from

| Number | Origin | Tool |
|--------|--------|------|
| TPR ≈ 65.5% | Real recordings in `real/positive/` (1,364 clips), TFLM MicroFrontend, threshold 0.50 | `evaluate.py` (written this session) |
| FAPH ≈ 864/hour | `ira_cnn_int8.tflite` on 4,000 LibriSpeech clips (~66.7 min), same frontend | `evaluate.py` (written this session) |
| FAPH ≈ 511/hour | `ira_cnn_quantized_v3.tflite` on same negative set | `evaluate.py` (written this session) |

**Neither number existed in the project before this session.** No prior training log,
evaluation output, or saved results file contained these figures.

---

## 1. CNN Architecture

**Source:** `cnn/train_cnn.py`
**Type:** Sequential 2D CNN (fixed-window, non-streaming)

```
Input: (49, 40, 1)

Conv2D(8,  (3,3), same, ReLU)  →  (49, 40,  8)
MaxPooling2D(2,2)              →  (24, 20,  8)
Conv2D(16, (3,3), same, ReLU)  →  (24, 20, 16)
MaxPooling2D(2,2)              →  (12, 10, 16)
Conv2D(32, (3,3), same, ReLU)  →  (12, 10, 32)
GlobalAveragePooling2D         →  (32,)
Dense(16, ReLU)                →  (16,)
Dropout(0.20)
Dense(1, Sigmoid)              →  (1,)
```

Key properties:
- No BatchNorm, no residual connections
- Processes an entire 1-second window at once (not streaming/stateful)
- 49 time frames = floor((16000 - 480) / 320) + 1

---

## 2. Number of Parameters

| Layer          | Weights | Biases | Total  |
|----------------|---------|--------|--------|
| Conv2D(8,3x3)  | 72      | 8      | 80     |
| Conv2D(16,3x3) | 1,152   | 16     | 1,168  |
| Conv2D(32,3x3) | 4,608   | 32     | 4,640  |
| Dense(16)      | 512     | 16     | 528    |
| Dense(1)       | 16      | 1      | 17     |
| **Total**      |         |        | **6,433** |

Consistent with 12.95 KB INT8 file size (1 byte/weight).

---

## 3. Input Tensor Shape

Verified live on all three TFLite files:

| Model                         | Shape         | dtype   |
|-------------------------------|---------------|---------|
| ira_cnn_int8.tflite           | [1, 49, 40, 1] | int8   |
| ira_cnn_int8_v2.tflite        | [1, 49, 40, 1] | int8   |
| ira_cnn_quantized_v3.tflite   | [1, 49, 40, 1] | float32|

Dimensions: batch=1, time=49 frames, freq=40 bins, channel=1

---

## 4. Output Tensor Shape

| Model                         | Shape  | dtype   |
|-------------------------------|--------|---------|
| ira_cnn_int8.tflite           | [1, 1] | int8    |
| ira_cnn_int8_v2.tflite        | [1, 1] | int8    |
| ira_cnn_quantized_v3.tflite   | [1, 1] | float32 |

Single sigmoid score per inference. INT8 output must be dequantized before thresholding.

---

## 5. INT8 Quantization Details

### ira_cnn_int8.tflite and ira_cnn_int8_v2.tflite (identical files)

Method: Full integer quantization (TFLite BUILTINS INT8, int8 input, int8 output)
Source: ira_cnn_best.keras (best val_recall checkpoint)
Calibration: up to 500 samples from dataset/splits/test/ (mixed pos+neg)

Measured parameters (live):
  Input  scale=0.083630204  zero_point=-105
  Output scale=0.003906250  zero_point=-128  (= 1/256, output_zero=-128)

Inference dequantization:
  x_int8    = clip(round(x_float / 0.083630 + (-105)), -128, 127)
  prob_float = (output_int8.astype(float32) - (-128)) * 0.00390625

NOTE: ira_cnn_int8.tflite and ira_cnn_int8_v2.tflite are bit-for-bit identical
(same size: 13,264 bytes, same quantization params). The v2 script only added
explicit input reshape during calibration — no change to the weights or the output.

### ira_cnn_quantized_v3.tflite

Method: INT8 weights/ops, FLOAT32 input+output (diagnostic/hybrid)
Size: 13,616 bytes (352 bytes larger — extra dequant ops in flatbuffer)
No manual quantize/dequantize needed at runtime.

---

## 6. Audio Sample Rate

16,000 Hz (16 kHz) — defined in every script.
All dataset WAVs stored as 16 kHz, mono, 16-bit PCM.

---

## 7. Audio Duration / Window Size

Fixed clip length: 1.0 second = 16,000 samples
All audio is zero-padded or truncated to exactly 16,000 samples before feature extraction.

Live inference (threshold_test.py): rolling 1-second buffer, model runs every 0.5 s
(block_size=8,000 samples).

---

## 8. Feature Extraction Method

=== CRITICAL: FEATURE EXTRACTOR MISMATCH ===

A. TRAINING (train_cnn.py) — tf.signal.stft
   frame_length=480 (30 ms), frame_step=320 (20 ms), fft_length=512
   magnitude = abs(STFT)
   log_spec   = log(magnitude + 1e-6)       ← log with epsilon
   keep first 40 bins
   normalize: (spec - mean) / (std + 1e-6)  ← per-clip z-score
   output shape: (49, 40)

B. EVALUATE.PY (this session) — pymicro_features MicroFrontend (TFLM)
   Same 30 ms window / 20 ms hop timing
   Uses PCAN normalization (not z-score)
   Uses mel-filterbank (not raw FFT magnitude)
   Output is uint16, scaled by x0.0390625 to float32

   The model was trained on A but evaluated using B.
   ALL metrics produced by evaluate.py reflect this mismatch.

C. cnn/test_int8.py — scipy.signal.stft
   Same STFT config as training, BUT:
   log1p(x) instead of log(x + 1e-6)  ← second inconsistency

Correct feature pipeline for eval: use tf.signal.stft + log(x+1e-6) as in train_cnn.py.

---

## 9. Positive Dataset

Total positive samples: 10,000

| Group            | Source              | Count |
|------------------|---------------------|-------|
| real_quiet_20cm  | Real recordings, 20 cm | 230 |
| real_quiet_50cm  | Real recordings, 50 cm | 225 |
| real_quiet_1m    | Real recordings, 1 m   | 227 |
| piper_original   | Piper TTS              | 1,710|
| piper_batch_A    | Piper TTS              | 2,000|
| piper_batch_B    | Piper TTS              | 2,000|
| piper_batch_C    | Piper TTS              | 2,000|
| piper_batch_D    | Piper TTS              | 1,608|

Real recordings: 682 clips, single speaker, 3 quiet acoustic conditions
Synthetic TTS: 9,318 clips (~93% of all positives)

evaluate.py tested against 1,364 clips from real/positive/ (includes additional
conditions: fan_noise, different_environment, conversation_noise).

---

## 10. Negative Dataset

Total negative samples: 6,000

| Group           | Source                     | Count |
|-----------------|----------------------------|-------|
| librispeech     | 28 speakers, 1s windows    | 4,000 |
| synthetic_noise | Ambient/background noise   | 2,000 |

FAPH evaluation used only the 4,000 LibriSpeech speech clips.
The 2,000 background noise clips were not included.

---

## 11. Train / Validation / Test Split

Source: create_split.py
Ratio: 80% / 10% / 10%

| Split      | Total  | Positive | Negative |
|------------|--------|----------|----------|
| train      | 12,741 | 7,999    | 4,742    |
| validation |  1,484 |   998    |   486    |
| test       |  1,775 | 1,003    |   772    |
| Total      | 16,000 | 10,000   |  6,000   |

LibriSpeech negatives: split at SPEAKER level (speaker-safe):
  28 speakers → 22 train / 3 validation / 3 test
  (no speaker heard in training appears in val/test)

All other groups: split at clip level (random.shuffle, seed=42).

---

## 12. Data Augmentation

NONE was applied during training.

train_cnn.py performs no tf.data augmentation, no time-shifting,
no noise mixing, no SpecAugment, no pitch variation.

Augmentation scripts exist in the project (scripts/test_augmentation.py,
dataset/augmentation_test/) but are NOT wired into the training pipeline.

Piper TTS batches provide speaker variety for positives, but this is source
diversity, not runtime augmentation.

---

## 13. Loss Function

Binary Cross-Entropy: L = -[y*log(p) + (1-y)*log(1-p)]

Class weighting was applied (train split: 7,999 pos / 4,742 neg):
  class_weight = { 0: total / (2 * neg_count),   # ~1.342
                   1: total / (2 * pos_count) }   # ~0.797

---

## 14. Optimizer

Adam (TF defaults: beta_1=0.9, beta_2=0.999, epsilon=1e-7)
No scheduler, no decay, no warmup.

---

## 15. Learning Rate

0.001 (constant throughout training)

---

## 16. Batch Size

64 (used for both fit() and evaluate())

---

## 17. Epochs / Training Steps

Maximum: 30 epochs
Early stopping: patience=5 on val_loss (restore_best_weights=True)
Best checkpoint: saved by max val_recall

Actual epochs run: UNKNOWN — no training log was saved.
Estimated steps/epoch: 12,741 / 64 ≈ 199 steps/epoch

---

## 18. Detection Threshold

No canonical threshold is defined in the project.

| Script              | Threshold         | Context              |
|---------------------|-------------------|----------------------|
| evaluate_cnn.py     | auto (recall-FAR) | Float Keras eval     |
| test_int8.py        | 0.90 (final)      | INT8 TFLite eval     |
| compare_float_int8  | 0.50              | Float/INT8 compare   |
| threshold_test.py   | None (raw output) | Live mic test        |
| evaluate.py         | 0.50 (default)    | FAPH eval (session)  |

---

## 19. How TPR / Accuracy Was Calculated

Source of 65.5% figure: evaluate.py (written this session)

Test set: 1,364 real recordings from real/positive/ (all conditions)
Feature extractor: pymicro_features MicroFrontend (NOT the training frontend)
Threshold: 0.50

Algorithm:
  1. WAV -> MicroFrontend -> predict_spectrogram(stride=1) -> ~49 predictions
  2. 5-frame moving average over predictions
  3. max(smoothed) as clip score
  4. TP if max_score >= 0.50

Results:
  TPR = 894 / 1364 = 65.54%   (quiet_20cm best: 70.9%; quiet_1m/50cm: ~63%)
  FNR = 470 / 1364 = 34.46%

The project's own evaluate_cnn.py (correct feature pipeline, test split) was
never run during this session and produced no saved output. Those accuracy numbers
do not exist in any file in the project.

---

## 20. How False Accepts Per Hour Was Calculated

Source of 511-864/hour figures: evaluate.py (written this session)
Uses: microwakeword/test.py -> compute_false_accepts_per_hour()

Test audio: 4,000 LibriSpeech speech clips (~66.7 min / 1.111 h total)
Feature extractor: pymicro_features MicroFrontend

Algorithm per clip:
  1. WAV -> MicroFrontend -> predict_spectrogram(stride=1) -> ~49 predictions
  2. 5-frame sliding window -> ~45 smoothed probabilities
  3. Iterate: if prob > threshold AND cooldown==0 -> count FA, reset cooldown=25
  4. FAPH = total_FAs / total_hours

Results at threshold 0.50:
  ira_cnn_int8.tflite        : 864 FAPH
  ira_cnn_int8_v2.tflite     : 864 FAPH  (identical model)
  ira_cnn_quantized_v3.tflite: 511 FAPH

Benchmark: ~0.2 - 1.0 FAPH (released microWakeWord models)
Observed: 500-800x higher than benchmark.

=== CRITICAL ISSUES WITH THESE FAPH NUMBERS ===

1. Feature mismatch: trained on tf.signal.stft, evaluated on TFLM MicroFrontend.
   All outputs are out-of-distribution for this model.

2. Short isolated clips: FAPH assumes continuous audio. Using 4,000 independent
   1-second clips resets model/cooldown state each time.

3. Partial negative set: only LibriSpeech speech used; 2,000 background noise clips excluded.

4. Possible training-set contamination: only ~772 clips were truly held-out test-split
   negatives. The remaining ~3,228 may have been seen during training.

---

## Key Problems Identified (Findings Only — No Files Changed)

1. FEATURE MISMATCH (critical): Training uses tf.signal.stft + z-score.
   evaluate.py uses TFLM MicroFrontend (mel filterbank + PCAN). Every metric is suspect.

2. LOG FUNCTION INCONSISTENCY: train_cnn.py uses log(x+1e-6).
   test_int8.py and make_calibration.py use log1p(x). Different at small values.

3. 93% SYNTHETIC POSITIVES: Only 682 of 10,000 positives are real recordings.
   Model has minimal exposure to real human voices.

4. SINGLE SPEAKER: All 682 real recordings are one person in one room. No speaker diversity.

5. NO AUGMENTATION: No noise mixing, time-shifting, reverb, or SpecAugment during training.

6. NO STREAMING ARCHITECTURE: Fixed 1-second window model. Real deployment requires
   sliding window; stride choice at inference changes effective behaviour significantly.

7. NO SAVED TRAINING METRICS: No log file, no history JSON, no test_probabilities.npy.
   Training-time accuracy and loss curves cannot be recovered.

---

*End of audit. No project files were modified.*
