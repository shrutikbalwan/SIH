# IRA CNN V2.1 Training Report

## 1. Experiment Goal
Determine whether introducing harder, mined speech negatives into the training distribution can fix the massive V2 speech false-positive regression while preserving positive recall.

## 2. Hard Speech Mining
- **Source**: Train-split speech negatives (LibriSpeech speakers). V1 streaming hard-negatives were **explicitly excluded** because they leaked test-set speakers.
- **Criteria**: Scored with `ira_cnn_v2_best_loss.keras`. Clips scoring `>= 0.30` were added to the hard pool.
- **Yield**: 644 clips (~20.5% of train speech negatives) across 22 unique speakers.

## 3. Training History & Augmentation
- **Architecture/Preprocessing**: Exact V2 replica (randomly initialized).
- **Positive Augmentation**: Preserved V2 setup (15% clean, 85% mixed). 
- **Negative Batching**: Batches were forced to contain 15% mined hard speech, 59% ordinary speech, and 19% ambient noise out of their negative slots.
- **Selection Rule**: Instead of `val_loss`, the final checkpoint `ira_cnn_v2_1_best_speech_fpr.keras` was chosen by tracking per-epoch validation FPRs and selecting the epoch with `val_tpr >= 95%` and the lowest `val_speech_fpr` (Epoch 21).

## 4. Validation Comparison (V2 vs V2.1)
The primary go/no-go metric on the exact same validation set at threshold 0.50:

| Metric | V2 (`best_loss`) | V2.1 (`best_speech_fpr`) |
| :--- | :--- | :--- |
| **TPR** | 98.10% | 95.79% |
| **Speech FPR** | 11.89% | 0.35% |
| **Ambient FPR**| 0.00% | 0.00% |
| **Overall FPR**| 7.00% | 0.21% |

**Conclusion**: Massive improvement in speech-FPR on validation data (11.89% → 0.35%), triggering the go-ahead for diagnostic historical evaluation.

## 5. Historical Diagnostic Test Evaluation
Since validation succeeded, V2.1 was evaluated on the historical 772-negative / 1003-positive set (now strictly labelled as a development/diagnostic set).

**V2.1 @ thresholds 0.50-0.90:**
| Thresh | All TPR | Real TPR | TTS TPR | All FPR | Speech FPR | Ambient FPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.50** | 95.51% | 84.29% | 96.36% | **0.00%** | **0.00%** | **0.00%** |
| **0.60** | 95.31% | 81.43% | 96.36% | 0.00% | 0.00% | 0.00% |
| **0.70** | 94.92% | 78.57% | 96.14% | 0.00% | 0.00% | 0.00% |
| **0.80** | 94.22% | 75.71% | 95.61% | 0.00% | 0.00% | 0.00% |
| **0.90** | 91.33% | 70.00% | 92.93% | 0.00% | 0.00% | 0.00% |

## 6. Key Findings
1. **Speech False Positives are Solved**: V2.1 yields a 0.00% FPR across all 572 historical LibriSpeech test negatives (down from 19.58% in V2!).
2. **Real Positive Regression**: While TTS recall remains strong (96.36%), the *Real* human positive TPR has dropped to **84.29%** (down from 100% in V1/V2).

By heavily penalizing hard speech negatives, the model learned to reject confusable human speech but sacrificed its sensitivity to real human recordings of the wake word.
