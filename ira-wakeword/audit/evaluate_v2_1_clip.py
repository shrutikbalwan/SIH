import os
import csv
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).parent
MODEL_PATH = REPO_ROOT / "cnn" / "models" / "ira_cnn_v2_1_best_speech_fpr.keras"
MANIFEST = REPO_ROOT / "dataset" / "split_manifest.csv"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000

def load_audio(path: str) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    if len(a) < WINDOW_SAMPLES:
        a = np.pad(a, (0, WINDOW_SAMPLES - len(a)))
    return a[:WINDOW_SAMPLES]

def make_spectrogram(audio: np.ndarray) -> np.ndarray:
    audio_t = tf.convert_to_tensor(audio, dtype=tf.float32)
    spec = tf.signal.stft(audio_t, frame_length=480, frame_step=320, fft_length=512)
    spec = tf.abs(spec)
    spec = tf.math.log(spec + 1e-6)
    spec = spec[:, :40]
    mean = tf.reduce_mean(spec)
    std = tf.math.reduce_std(spec) + 1e-6
    spec = (spec - mean) / std
    return spec.numpy().astype(np.float32)

def main():
    print(f"Loading V2.1 Float Model: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    
    test_pos_real, test_pos_tts = [], []
    test_neg_speech, test_neg_amb = [], []
    
    with open(MANIFEST, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] != 'test': continue
            path = REPO_ROOT / r['path']
            if not path.exists(): continue
            
            label = int(r.get('label', -1))
            group = r.get('group', '').lower()
            
            if label == 1:
                is_real = ('real' in group) or ('real' in str(path).lower())
                if not is_real and not ('piper' in group or 'piper' in str(path).lower()):
                    is_real = 'tts' not in str(path).lower()
                
                if is_real: test_pos_real.append(str(path))
                else: test_pos_tts.append(str(path))
            elif label == 0:
                if 'libri' in group or 'speech' in group:
                    test_neg_speech.append(str(path))
                elif 'ambient' in group or 'background' in group or 'noise' in group:
                    test_neg_amb.append(str(path))
                
    print(f"\nDEVELOPMENT / HISTORICAL DIAGNOSTIC SET")
    print(f"{len(test_pos_real)} Real Positives, {len(test_pos_tts)} TTS Positives")
    print(f"{len(test_neg_speech)} Speech Negatives, {len(test_neg_amb)} Ambient Negatives")
    
    def score_group(paths):
        if not paths:
            return np.array([])
        specs = []
        for p in paths:
            spec = make_spectrogram(load_audio(p))
            specs.append(np.expand_dims(spec, axis=-1))
        X = np.stack(specs, axis=0)  # (N, 49, 40, 1)
        scores = model.predict(X, batch_size=64, verbose=0).flatten()
        return scores.astype(np.float32)

    scores_real = score_group(test_pos_real)
    scores_tts = score_group(test_pos_tts)
    scores_sp = score_group(test_neg_speech)
    scores_amb = score_group(test_neg_amb)
    
    scores_pos_all = np.concatenate([scores_real, scores_tts]) if len(scores_real) and len(scores_tts) else (scores_real if len(scores_real) else scores_tts)
    scores_neg_all = np.concatenate([scores_sp, scores_amb]) if len(scores_sp) and len(scores_amb) else (scores_sp if len(scores_sp) else scores_amb)
    
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    print("\n=========================================================================")
    print("HISTORICAL DIAGNOSTIC EVALUATION (V2.1 @ thresholds 0.50-0.90)")
    print("=========================================================================")
    print("Thresh | All TPR | Real TPR | TTS TPR | All FPR | Speech FPR | Ambient FPR")
    
    for t in thresholds:
        tp_all = np.sum(scores_pos_all >= t)
        tp_real = np.sum(scores_real >= t)
        tp_tts = np.sum(scores_tts >= t)
        
        fp_all = np.sum(scores_neg_all >= t)
        fp_sp = np.sum(scores_sp >= t)
        fp_amb = np.sum(scores_amb >= t)
        
        tpr_all = tp_all / len(scores_pos_all) if len(scores_pos_all) else 0
        tpr_real = tp_real / len(scores_real) if len(scores_real) else 0
        tpr_tts = tp_tts / len(scores_tts) if len(scores_tts) else 0
        
        fpr_all = fp_all / len(scores_neg_all) if len(scores_neg_all) else 0
        fpr_sp = fp_sp / len(scores_sp) if len(scores_sp) else 0
        fpr_amb = fp_amb / len(scores_amb) if len(scores_amb) else 0
        
        print(f" {t:.2f}  | {tpr_all*100:6.2f}% | {tpr_real*100:7.2f}% | {tpr_tts*100:6.2f}% | {fpr_all*100:6.2f}% | {fpr_sp*100:9.2f}% | {fpr_amb*100:10.2f}%")

if __name__ == "__main__":
    main()
