import os
import csv
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

REPO_ROOT = Path(__file__).parent
MODEL_PATH = REPO_ROOT / "cnn" / "models" / "ira_cnn_v2_best_loss.keras"
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
    print(f"Loading V2 Float Model: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    
    test_pos_real, test_pos_tts = [], []
    test_neg = []
    
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
                test_neg.append(str(path))
                
    print(f"Test Set: {len(test_pos_real)} Real Positives, {len(test_pos_tts)} TTS Positives, {len(test_neg)} Negatives.")
    
    def score_group(paths):
        if not paths:
            return np.array([])
        specs = []
        for p in paths:
            spec = make_spectrogram(load_audio(p))
            specs.append(np.expand_dims(spec, axis=-1))
        X = np.stack(specs, axis=0)  # (N, 49, 40, 1)
        scores = model.predict(X, batch_size=64, verbose=1).flatten()
        return scores.astype(np.float32)

        
    print("Scoring Real Positives...")
    scores_real = score_group(test_pos_real)
    print("Scoring TTS Positives...")
    scores_tts = score_group(test_pos_tts)
    print("Scoring Negatives...")
    scores_neg = score_group(test_neg)
    
    scores_pos_all = np.concatenate([scores_real, scores_tts]) if len(scores_real) and len(scores_tts) else (scores_real if len(scores_real) else scores_tts)
    
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print("\n=======================================================")
    print("THRESHOLD SWEEP")
    print("=======================================================")
    print("Thresh |   TP |   FN |   TN |   FP |    TPR |    FNR |    FPR |    TNR |    Acc |   Prec")
    
    for t in thresholds:
        tp = np.sum(scores_pos_all >= t)
        fn = np.sum(scores_pos_all < t)
        tn = np.sum(scores_neg < t)
        fp = np.sum(scores_neg >= t)
        
        tpr = tp / (tp + fn) if (tp + fn) else 0
        fnr = fn / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        tnr = tn / (fp + tn) if (fp + tn) else 0
        acc = (tp + tn) / (tp + fn + tn + fp)
        prec = tp / (tp + fp) if (tp + fp) else 0
        
        print(f"{t:.2f}   | {tp:4d} | {fn:4d} | {tn:4d} | {fp:4d} | {tpr*100:6.2f} | {fnr*100:6.2f} | {fpr*100:6.2f} | {tnr*100:6.2f} | {acc*100:6.2f} | {prec*100:6.2f}")
        
    print("\n=======================================================")
    print("POSITIVE ANALYSIS (Threshold 0.50)")
    print("=======================================================")
    def analyze_pos(name, scores):
        n = len(scores)
        if n == 0: return
        tp = np.sum(scores >= 0.50)
        fn = np.sum(scores < 0.50)
        tpr = tp / n
        mean = np.mean(scores)
        median = np.median(scores)
        p10 = np.percentile(scores, 10)
        print(f"{name:10s} | N={n:<4d} | TP={tp:<4d} | FN={fn:<4d} | TPR={tpr*100:6.2f}% | Mean={mean:.4f} | Med={median:.4f} | P10={p10:.4f}")
        
    analyze_pos("REAL", scores_real)
    analyze_pos("TTS", scores_tts)
    analyze_pos("ALL", scores_pos_all)

if __name__ == "__main__":
    main()
