# IRA Wake-Word Project

## Recent Updates & Refactoring
- **Directory Reorganization**: Restructured the project root to categorize scattered files.
  - `reports/`: Contains all Markdown analysis and audit reports.
  - `logs/`: Holds training histories, JSON stats, and evaluation CSV/JSON output.
  - `data-prep/`: Data generation, manipulation, and audio split scripts.
  - `audit/`: Evaluation, sensitivity, and auditing scripts.
- **Code Cleanup**: Modernized reporting scripts (e.g., `evaluate_new_ira_v23_int8.py`) to use cleaner Python f-strings and list structures instead of legacy formatting.

## Model Accuracy & Evaluation (Frozen V2.3 INT8)
Based on recent evaluations, here is the current performance summary:

### True Positive Rate (TPR)
- **TPR ≈ 65.5%** on real human recordings (`real/positive/`, 1,364 clips) at a 0.50 threshold.
- The model shows higher accuracy (~70.9%) in quiet, close-range (20 cm) environments, but drops to ~63% at 1m or 50cm distances.

### False Accepts Per Hour (FAPH)
- **FAPH ≈ 511 to 864 / hour** on the negative LibriSpeech test set.
- *Note:* This high rate is partially attributed to a known feature extraction mismatch (the model was trained using `tf.signal.stft` with z-score normalization, but evaluated against the TFLM MicroFrontend). 

### Key Findings
1. **Feature Mismatch**: The pipeline used for training differs slightly from the inference pipeline, causing out-of-distribution errors.
2. **Dataset Composition**: ~93% of the positive samples are synthetic (Piper TTS), while only ~7% are real human voices, all from a single speaker.
3. **Augmentation**: No significant audio augmentation (time-shifting, noise mixing, etc.) was active during training, leading to fragility in real-world noisy environments.

Please see the `reports/` folder for deeper dives, particularly `MODEL_AUDIT.md` and the various streaming evaluation reports.
