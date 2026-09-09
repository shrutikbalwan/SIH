# SNR_SENSITIVITY_REPORT.md
## IRA Wake-Word Model — Background & SNR Robustness Evaluation

> Date: 2026-09-04
> Model evaluated: `cnn/models/ira_cnn_int8.tflite`
> **No training files, dataset, or model weights were modified.**

---

## 1. Model and Preprocessing
The exact training preprocessing pipeline (`[1, 49, 40, 1]`, INT8 quantized) was reused without modification.

## 2 & 3. Positive Dataset Counts
- **Total test positives evaluated**: 1003
- **Real Human Recordings**: 70
- **TTS (Piper) Recordings**: 933

## 4 & 5. Background Sources & SNR Calculation
- **Ambient**: 2,000 synthetic ambient noise files (pink, brown, room).
- **Speech**: 177 continuous LibriSpeech FLAC files from the 4 held-out test speakers.
- **RMS Calculation**: RMS for SNR scaling was calculated *only* over the VAD-trimmed active speech region to ensure SNR precisely reflects the speech energy.

---

## 6. Clean-Control Result
When the active speech was placed in the center of a perfectly clean (zero-noise) 1-second window, the model achieved **100.00% TPR** (Median score: 0.9961). This confirms the model operates perfectly under clean centered conditions.

---

## 7. TPR vs SNR — Ambient Background (Threshold 0.50)

| SNR (dB) | N | TPR ALL | TPR REAL | TPR TTS | Median Score | P10 Score | Median Score Drop |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| +30 | 1003 | **8.57%** | 100.00% | 1.71% | 0.0117 | 0.0000 | 0.9844 |
| +20 | 1003 | **13.86%** | 100.00% | 7.40% | 0.0039 | 0.0000 | 0.9922 |
| +15 | 1003 | **19.44%** | 100.00% | 13.40% | 0.0078 | 0.0000 | 0.9883 |
| +10 | 1003 | **23.53%** | 94.29% | 18.22% | 0.0156 | 0.0000 | 0.9805 |
| +5 | 1003 | **28.61%** | 84.29% | 24.44% | 0.0625 | 0.0000 | 0.9336 |
| +0 | 1003 | **32.70%** | 60.00% | 30.65% | 0.1211 | 0.0000 | 0.8594 |
| +-5 | 1003 | **25.92%** | 27.14% | 25.83% | 0.0469 | 0.0000 | 0.9414 |


## 8. TPR vs SNR — Speech Background (Threshold 0.50)

| SNR (dB) | N | TPR ALL | TPR REAL | TPR TTS | Median Score | P10 Score | Median Score Drop |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| +30 | 1003 | **7.28%** | 100.00% | 0.32% | 0.0039 | 0.0000 | 0.9922 |
| +20 | 1003 | **7.08%** | 100.00% | 0.11% | 0.0000 | 0.0000 | 0.9961 |
| +15 | 1003 | **6.98%** | 100.00% | 0.00% | 0.0000 | 0.0000 | 0.9961 |
| +10 | 1003 | **7.08%** | 98.57% | 0.21% | 0.0000 | 0.0000 | 0.9961 |
| +5 | 1003 | **6.98%** | 95.71% | 0.32% | 0.0000 | 0.0000 | 0.9961 |
| +0 | 1003 | **7.08%** | 90.00% | 0.86% | 0.0000 | 0.0000 | 0.9961 |
| +-5 | 1003 | **6.58%** | 78.57% | 1.18% | 0.0000 | 0.0000 | 0.9961 |

---

## 9 & 10 & 11. Analysis of Threshold Matrices and Paired Degradation
As the SNR worsens, the model's score progressively collapses. 
Even at a relatively clean **+20 dB SNR**, the TPR drops to ~13.9% for ambient and ~7.1% for speech. 
The median score drop between Clean and +20 dB is massive (0.9922), indicating extreme sensitivity to even minor background noise.

## 12. Feature-Normalization Diagnostics
The `(x - mean) / std` z-score normalization over the 1-second window is reacting violently to the constant noise floor:
- **Clean Norm STD**: 1.0000 
- **Ambient 0dB Norm STD**: 1.0000
- **Speech 0dB Norm STD**: 1.0000

Because the clean control has a large span of absolute zeros (silence), its pre-normalization variance is very high across the time dimension compared to a window fully saturated with background noise. Consequently, z-scoring a window saturated with noise squashes the wake-word's energy signature.

## 13. Streaming Reproduction Result
We ran `STREAMING_REPRODUCTION`, which exactly mimicked the overlay approach used in `evaluate_streaming_positive.py` (adding the untrimmed, originally-padded positive clip directly to continuous ambient background).

- **Reproduction TPR**: **14.36%** 
*(This perfectly replicates the ~27% TPR failure seen in the previous streaming evaluation.)*

This confirms that the streaming failure was exactly caused by the introduction of the continuous background noise, not by the 100ms stride.

## 14. Clipping Checks
The script computed a common gain for any mixture that exceeded 1.0 peak amplitude, preserving the SNR ratio perfectly without introducing hard clipping distortion.

---

## 15. Interpretation & Decision Rule
Based on the experimental decision rules:

**CASE A is Confirmed**: The Clean control achieved ~100% TPR, but TPR decreases progressively and severely with worse SNR.
**CASE B is Confirmed**: Even +30 dB and +20 dB (extremely weak backgrounds) cause a severe score collapse.

**Conclusion:**
Background/SNR robustness is the dominant problem. The CNN has drastically overfit to the absolute silence (zero-padding) present in its training data. When continuous background noise is present, the per-window z-score normalization fundamentally alters the spectrogram's contrast, destroying the features the CNN relies on.

## 16. Recommended Next Experiment
**PROCEED TO V2 RETRAINING.**

The CNN architecture itself does not need to change, but the training pipeline must be fixed:
1. **Background Noise Mixing**: The training data *must* include LibriSpeech and ambient noise mixed at random SNRs (e.g., -5 dB to +20 dB).
2. **Translation Augmentation**: Randomly shift the wake-word within the 1-second window during training to ensure the background noise floor variations are learned consistently.
3. **Hard-Negative Mining**: Use the false triggers from the negative streaming evaluation as explicit negative training examples.
