# STREAMING_EVALUATION_REPORT.md
## IRA Wake-Word Model — Continuous Streaming Evaluation

> Date: 2026-09-04
> Model evaluated: `cnn/models/ira_cnn_int8.tflite`
> Script: `evaluate_streaming.py`
> **No training files, dataset, or model weights were modified.**

---

## Executive Summary

Following the clip-level baseline verification, a proper **continuous streaming evaluation** was performed. Unlike the isolated clip-level test, this evaluation mimics real-world microphone behavior by sliding an analysis window over continuous long-form audio and applying standard trigger and cooldown logic.

At a threshold of 0.50, the model generated **47.35 False Accepts Per Hour (FAPH)**. While this is significantly lower than the invalid ~864 FAPH reported earlier under a mismatched pipeline, it remains higher than the typical production target of 0.2–1.0 FAPH.

---

## A. Model Information

| Property | Value |
|----------|-------|
| Model file | `cnn/models/ira_cnn_int8.tflite` |
| File size | 12.95 KB (13,264 bytes) |
| Architecture | Sequential 2D CNN (Conv8 -> Conv16 -> Conv32 -> Dense) |
| Input shape | `[1, 49, 40, 1]` |
| Input dtype | `int8` (scale=0.08363020, zero_point=-105) |
| Output shape | `[1, 1]` |
| Output dtype | `int8` (scale=0.00390625, zero_point=-128) |

## B. Exact Preprocessing

The preprocessing pipeline exactly matches the training pipeline `cnn/train_cnn.py`:

1.  **Audio Setup:** 16 kHz, float32, mono
2.  **STFT:** `tf.signal.stft(frame_length=480, frame_step=320, fft_length=512)`
3.  **Magnitude + Log:** `tf.math.log(tf.abs(stft) + 1e-6)`
4.  **Frequency Cropping:** Keep first 40 bins
5.  **Normalization:** Per-window z-score `(x - mean) / (std + 1e-6)`
6.  **Quantization:** `int8` using the exact TFLite scaling parameters.

## C. Sliding-Window Configuration

Real-world deployments don't feed perfectly centered 1-second clips to the model. They sample a rolling buffer continuously.

*   **Window Size:** 1.0 second (16,000 samples)
*   **Inference Stride:** 100 ms (1,600 samples)
*   The model evaluates a new 1-second window 10 times per second.

## D. Trigger / Debounce Logic

A raw positive inference frame is not automatically a trigger event.

*   **Detection Threshold:** Configurable (see Sweep below)
*   **Consecutive Positives Required:** 1
*   **Cooldown Period (Refractory):** 1,500 ms (1.5 seconds)

If the model predicts above the threshold, a "False Trigger Event" is counted. The system then enters a 1.5-second cooldown period where subsequent positive frames are ignored. This ensures a single acoustic event (like a loud clap) is counted as one false accept, rather than 15 separate false accepts.

## E. Negative Datasets Used

Only continuous audio not containing the word "IRA" was evaluated.

| Dataset | Type | Source | Count | Duration |
| :--- | :--- | :--- | :--- | :--- |
| `dataset/negative/extracted/LibriSpeech` | Continuous Speech | 4 hold-out speakers | 177 files | ~38.9 min |
| `dataset/negative/background` | Continuous Noise | Pink/Brown/White/Room | 2,000 files | ~33.3 min |

**Note:** The LibriSpeech speakers used (`1088`, `1737`, `5789`, `6848`) are exclusively from the test split and were **never seen during training**.

## F. Total Evaluation Duration

*   **Total Files Evaluated:** 2,177 files
*   **Total Duration:** 4333.5 seconds
*   **Total Minutes:** 72.23 min
*   **Total Hours:** **1.2038 hours**

---

## G & H. Results & FAPH Calculation

**False Accepts Per Hour (FAPH) = (False Trigger Events) / (Negative Hours)**

| Threshold | Raw Positive Windows | False Trigger Events | Negative Hours | FAPH |
| :--- | :--- | :--- | :--- | :--- |
| 0.30 | 214 | 75 | 1.2038 | 62.3046 |
| 0.40 | 163 | 64 | 1.2038 | 53.1666 |
| **0.50** | **143** | **57** | **1.2038** | **47.3515** |
| 0.60 | 117 | 51 | 1.2038 | 42.3671 |
| 0.70 | 93 | 41 | 1.2038 | 34.0598 |
| 0.80 | 78 | 32 | 1.2038 | 26.5833 |
| 0.90 | 50 | 18 | 1.2038 | 14.9531 |

*(Reminder: The previously reported 864 FAPH was invalid due to a mismatched feature pipeline and improper non-streaming event counting).*

---

## I. False-Trigger Examples

57 false trigger events were detected at a threshold of 0.50. Audio clips representing the trigger (-1 sec to +2 sec) have been saved to the directory:
`streaming_false_triggers/threshold_50/`

A corresponding CSV index is located at:
`streaming_false_triggers_thr50.csv`

These clips represent the "hard negatives" that the model struggled with in its proper feature space. They can be mined for future dataset improvements.

---

## J. Limitations

1.  **Limited Duration:** 1.2 hours of negative audio is sufficient to establish a baseline, but 24–100 hours is typically required for statistically robust FAPH reporting (especially for very low-FAPH models).
2.  **Lack of Real-World Diversity:** The background noise consists of purely synthetic/generated noise (pink, brown, sine waves). It lacks real-world ambient sounds like TV broadcasts, traffic, or kitchen noises.
3.  **Single Threshold Check:** This test evaluates FAPH in a vacuum. A model could achieve 0 FAPH simply by never triggering. We must contextualize this with real-world TPR.

## K. Recommended Next Experiment

**Do not claim real-world TPR or competition-ready performance yet.**

The clip-level evaluation proved the model can easily recognize its training distribution (100% TPR on held-out Piper TTS). However, we do not know if the model will actually trigger in a continuous streaming scenario when a *real* human speaks the wake-word organically.

**Next Step:** Inject known "IRA" utterances into continuous ambient audio at known timestamps. Run the streaming evaluator over this mixed audio to simultaneously measure:
1.  **True Positive Rate (TPR) / Wake-word recall**
2.  **Detection Latency** (how many milliseconds after the word finishes does the model trigger)

---

## Appendix: Corrected Clip-Level Baseline

For historical tracking, the isolated clip-level metrics on the exact same feature pipeline are preserved below. *These are not streaming metrics.*

*   **TPR = 100.00%** (1003 / 1003)
*   **FPR = 0.39%** (3 / 772)
*   **Accuracy = 99.83%**

*(Evaluated on `dataset/splits/test/`, using centered 1-second clips.)*
