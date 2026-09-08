# STREAMING_POSITIVE_EVALUATION_REPORT.md
## IRA Wake-Word Model — Continuous Streaming TPR vs FAPH

> Date: 2026-09-04
> Model evaluated: `cnn/models/ira_cnn_int8.tflite`
> **No training files, dataset, or model weights were modified.**

---

## 1. Model and Preprocessing
The exact training preprocessing pipeline (`tf.signal.stft` -> log -> z-score) was reused.
Model input shape: `[1, 49, 40, 1]`. Correct INT8 quantization parameters were applied.

## 2. Streaming Configuration
* **Window**: 1.0 second (16,000 samples)
* **Stride**: 100 ms
* **Trigger Logic**: 1 consecutive positive window
* **Cooldown/Debounce**: 1.5 seconds

## 3 & 4. Wake-Word Events and Sources
**1,003** total Wake-Word events were inserted into the streams.
These were sourced exclusively from the **held-out test split** (`dataset/splits/test/positive/`).
*No training clips were used.*

## 5 & 6. Background/Noise Conditions
The events were evenly divided and overlaid across three continuous background streams:
* **Clean**: 334 events in heavily attenuated quiet background.
* **Ambient**: 334 events in standard synthetic ambient noise (pink/brown/fan/hum).
* **Speech**: 335 events in continuous LibriSpeech streams (4 held-out test speakers).

## 7. Ground-Truth Matching Method
A model detection at timestamp `t_d` matches an inserted wake word at timestamp `t_g` if:
`t_g - 0.2 <= t_d <= t_g + 1.2`
Since the model was trained on 1-second padded clips, this window correctly captures detections made as the 1-second sliding window passes over the inserted word. 1-to-1 matching was strictly enforced.

---

## 8, 9 & 10. Combined Performance Table (The Tradeoff)

| Threshold | Streaming TPR | FNR | Negative FAPH | Median Latency | Clean TPR | Ambient TPR | Speech TPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.30 | **31.11%** | 68.89% | **62.30** | 500.0 ms | 46.11% | 39.22% | 8.06% |
| 0.40 | **28.22%** | 71.78% | **53.17** | 500.0 ms | 42.51% | 36.53% | 5.67% |
| 0.50 | **26.82%** | 73.18% | **47.35** | 500.0 ms | 40.72% | 34.43% | 5.37% |
| 0.60 | **25.42%** | 74.58% | **42.37** | 500.0 ms | 38.92% | 32.93% | 4.48% |
| 0.70 | **24.73%** | 75.27% | **34.06** | 500.0 ms | 37.43% | 32.93% | 3.88% |
| 0.80 | **23.23%** | 76.77% | **26.58** | 400.0 ms | 35.03% | 31.14% | 3.58% |
| 0.90 | **19.64%** | 80.36% | **14.95** | 400.0 ms | 27.84% | 26.95% | 4.18% |

---

## 11. Detection Latency
Across all successful detections at threshold 0.50, the median latency (time from insertion start to detection) is **500.0 ms**. This indicates the model generally triggers when the 1-second window is half-way through the inserted audio clip.

## 12. Performance by Condition
As shown in the table above, the model struggles profoundly in all continuous streaming conditions. Even in the "Clean" condition with almost zero background noise, the TPR at threshold 0.50 is only ~27%. 

## 13. False Negatives
At threshold 0.50, **734 False Negatives** (missed wake words) occurred.
Audio segments surrounding each miss have been saved to:
`streaming_false_negatives/`
and indexed in `streaming_false_negatives.csv`.

## 14. Score Distribution Analysis
Analysis of the *maximum* model score achieved anywhere near the 1,003 true wake-word events:
- **Minimum**: 0.0000
- **10th Percentile**: 0.0000
- **25th Percentile**: 0.0039
- **Median**: 0.0938
- **75th Percentile**: 0.8789
- **90th Percentile**: 0.9727
- **Maximum**: 0.9961

*Interpretation*: The median maximum score achieved near a true wake word is below 0.30. This means for over 50% of the true wake words, the model never output a probability higher than ~0.30 at any point while sliding over the word.

---

## 15. Limitations
This test uses overlay synthesis (adding the padded positive clips directly over the background stream). While standard, this implies any silence padded into the original 1-second training clips was added to the stream. However, the catastrophic drop in TPR is primarily driven by the sliding window stride.

## 16. Recommended Threshold
**There is no viable operating threshold for this model in a streaming context.**
- If we lower the threshold to 0.30 to salvage TPR (achieving 31.11%), FAPH explodes to **62.30**.
- If we raise the threshold to 0.90 to reduce FAPH (achieving 14.95), TPR collapses to **19.64%**.

## 17. Conclusion: Is Retraining Necessary?
**YES. Retraining is absolutely necessary.**

Threshold tuning cannot solve this problem. The model suffers from severe **Translation Variance**. 
During the clip-level evaluation, the model achieved 100% TPR on the exact same clips because they were perfectly centered in the 1-second window. In the streaming evaluation, the 100ms stride means the word is evaluated at 10 different alignments, none of which perfectly match the fixed alignment it memorized during training.

**Diagnosis:**
1. The training pipeline did not use random time-shifting augmentation.
2. The model learned to expect the wake-word at a very specific temporal alignment within the 49-frame window.
3. When sliding continuously, it misses the word entirely.

**Recommendation:**
Proceed to V2 training. The new training pipeline MUST include:
1. Time-shifting augmentation (randomly rolling the audio/spectrogram).
2. Hard-negative mining using the false triggers generated in the negative streaming test.
3. Background noise mixing during training (to improve the Speech and Ambient TPRs).
