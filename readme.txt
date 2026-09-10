# IRA Wake-Word Project

## Recent Updates & Refactoring
- **Directory Reorganization**: Restructured the project root to categorize scattered files.
  - `reports/`: Contains all Markdown analysis and audit reports.
  - `logs/`: Holds training histories, JSON stats, and evaluation CSV/JSON output.
  - `data-prep/`: Data generation, manipulation, and audio split scripts.
  - `audit/`: Evaluation, sensitivity, and auditing scripts.
- **Code Cleanup**: Modernized reporting scripts to use cleaner Python f-strings and list structures.

## Model Training Iterations & History
The project has progressed through several model iterations (V1 through V2.8) to experiment with different augmentation strategies, learning rates, and dataset splits. Below is a comprehensive summary of all model versions:

| Version | Script | Description & Key Changes |
|---------|--------|---------------------------|
| **V1** | `train_cnn.py` | Base lightweight CNN. Fixed 1s window, pure CNN, original dataset. No dynamic background augmentation. |
| **V2** | `train_cnn_v2.py` | Introduced dynamic background augmentation and hard negatives. |
| **V2.1** | `train_cnn_v2_1.py` | Controlled experiment for stronger speech-negative discrimination. Negative batch composition changed to 60% ordinary speech, 20% ambient, 5% other, 15% hard-speech (V2 score >= 0.30). V1 streaming hard negatives excluded. |
| **V2.2** | `train_cnn_v2_2.py` | Controlled experiment: Moderate speech-negative curriculum. |
| **V2.3** | `train_cnn_v2_3.py` | **Primary Production Candidate**. Expanded negative dataset with verified TRAIN-only speech pools. Batch recipe: 32 pos + 14 easy speech + 7 medium speech + 4 hard speech + 5 ambient + 2 other. |
| **V2.4** | `train_cnn_v2_4.py` | Controlled LR-schedule experiment. Changed Learning Rate to `ReduceLROnPlateau`. |
| **V2.5** | `train_cnn_v2_5.py` | Controlled Positive Augmentation Experiment. Changed positive background mix to 75% ambient / 25% speech (instead of 50/50). |
| **V2.6** | `train_cnn_v2_6.py` | Controlled Positive Augmentation SNR Experiment. Constrained speech-background SNR to [+10 dB, +25 dB]. |
| **V2.7** | `train_cnn_v2_7.py` | Minimal hard-speech negative pressure. Changed negative half by one slot: 13 easy / 7 medium / 5 hard (was 4) / 5 ambient / 2 other. |
| **V2.8** | `train_cnn_v2_8.py` | Modified the hard lower bound for SNR from -5 dB to -10 dB. |

## Model Accuracy & Evaluation (Frozen V2.3 INT8)
Based on recent evaluations, here is the current performance summary for the **V2.3 INT8** model:

### True Positive Rate (TPR)
- **TPR ≈ 65.5%** on real human recordings (`real/positive/`, 1,364 clips) at a 0.50 threshold.
- The model shows higher accuracy (~70.9%) in quiet, close-range (20 cm) environments, but drops to ~63% at 1m or 50cm distances.

### False Accepts Per Hour (FAPH)
- **FAPH ≈ 511 to 864 / hour** on the negative LibriSpeech test set.
- *Note:* This high rate is partially attributed to a known feature extraction mismatch (the model was trained using `tf.signal.stft` with z-score normalization, but evaluated against the TFLM MicroFrontend). 

### Key Findings & Actionable Insights
1. **Feature Mismatch**: The pipeline used for training differs slightly from the inference pipeline, causing out-of-distribution errors.
2. **Dataset Composition**: ~93% of the positive samples are synthetic (Piper TTS), while only ~7% are real human voices, all from a single speaker.
3. **Augmentation**: No significant audio augmentation (time-shifting, noise mixing, etc.) was active during training for the V1 models, leading to fragility in real-world noisy environments. V2 models introduce noise mixing, but the feature mismatch currently dominates the error rate.

Please see the `reports/` folder for deeper dives, particularly `MODEL_AUDIT.md` and the various streaming evaluation reports.
