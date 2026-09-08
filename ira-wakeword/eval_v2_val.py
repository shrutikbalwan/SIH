import os, csv
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).parent
MANIFEST = REPO_ROOT / "dataset" / "split_manifest.csv"
V2_MODEL_PATH = REPO_ROOT / "cnn" / "models" / "ira_cnn_v2_best_loss.keras"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
STFT_FRAME_LEN = 480
STFT_FRAME_STEP = 320
STFT_FFT_LEN = 512
N_FREQ_BINS = 40

def load_audio(path):
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    if len(a) < WINDOW_SAMPLES: a = np.pad(a, (0, WINDOW_SAMPLES - len(a)))
    else: a = a[:WINDOW_SAMPLES]
    return a.astype(np.float32)

def make_spectrogram(audio):
    t = tf.convert_to_tensor(audio, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN,
                       frame_step=STFT_FRAME_STEP, fft_length=STFT_FFT_LEN)
    s = tf.abs(s)
    s = tf.math.log(s + 1e-6)
    s = s[:, :N_FREQ_BINS]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)

def main():
    val_pos, val_sp, val_amb = [], [], []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "validation": continue
            p = str(REPO_ROOT / r["path"])
            label = int(r.get("label", -1))
            g = r.get("group", "").lower()
            if label == 1:
                val_pos.append(p)
            elif label == 0:
                if "libri" in g or "speech" in g:
                    val_sp.append(p)
                elif "ambient" in g or "background" in g or "noise" in g:
                    val_amb.append(p)
                    
    print(f"Val Pos: {len(val_pos)}, Val Sp: {len(val_sp)}, Val Amb: {len(val_amb)}")
    model = tf.keras.models.load_model(str(V2_MODEL_PATH))
    
    def score_paths(paths):
        if not paths: return np.array([])
        specs = [np.expand_dims(make_spectrogram(load_audio(p)), -1) for p in paths]
        return model.predict(np.stack(specs), batch_size=64, verbose=0).flatten()
        
    s_pos = score_paths(val_pos)
    s_sp = score_paths(val_sp)
    s_amb = score_paths(val_amb)
    
    THR = 0.50
    tpr = float(np.sum(s_pos >= THR) / len(s_pos)) if len(s_pos) else 0.0
    sfpr = float(np.sum(s_sp >= THR) / len(s_sp)) if len(s_sp) else 0.0
    afpr = float(np.sum(s_amb >= THR) / len(s_amb)) if len(s_amb) else 0.0
    all_neg = np.concatenate([s_sp, s_amb])
    ofpr = float(np.sum(all_neg >= THR) / len(all_neg)) if len(all_neg) else 0.0
    
    print("\nV2 Validation metrics @ 0.50:")
    print(f"TPR:         {tpr*100:.2f}%")
    print(f"Speech FPR:  {sfpr*100:.2f}%")
    print(f"Ambient FPR: {afpr*100:.2f}%")
    print(f"Overall FPR: {ofpr*100:.2f}%")

if __name__ == "__main__":
    main()
