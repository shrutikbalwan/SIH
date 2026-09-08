import os
import csv
import json
import random
import numpy as np
import tensorflow as tf
import soundfile as sf
import hashlib
from scipy.signal import resample_poly
from pathlib import Path
import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Constants
REPO_ROOT = Path(__file__).parent.parent
MANIFEST = REPO_ROOT / "dataset" / "split_manifest_v3.csv"
OUT_DIR = REPO_ROOT / "dataset" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NPZ_PATH = OUT_DIR / "expanded_val_v2_3_frozen.npz"
JSON_PATH = OUT_DIR / "expanded_val_v2_3_frozen.json"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
STFT_FRAME_LEN = 480
STFT_FRAME_STEP = 320
STFT_FFT_LEN = 512
N_FREQ_BINS = 40

CLEAN_PROB = 0.15
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
SNR_WEIGHTS = {"easy": 1, "medium": 1, "hard": 1}
RMS_THRESH = 1e-4

def load_audio(path: str) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a.astype(np.float32)

def pad_to_window(audio: np.ndarray) -> np.ndarray:
    if len(audio) < WINDOW_SAMPLES:
        return np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    return audio[:WINDOW_SAMPLES]

def make_spectrogram(audio: np.ndarray) -> np.ndarray:
    t = tf.convert_to_tensor(audio, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN, frame_step=STFT_FRAME_STEP, fft_length=STFT_FFT_LEN)
    s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_FREQ_BINS]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)

def calc_rms(audio: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(audio ** 2)))
    return r if r > 1e-9 else 1e-9

def vad_trim(audio: np.ndarray, top_db=25) -> np.ndarray:
    rms = calc_rms(audio)
    threshold = rms / (10 ** (top_db / 20))
    above = np.where(np.abs(audio) > threshold)[0]
    if len(above) == 0: return audio
    return audio[above[0]:above[-1] + 1]

def mix_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    s_rms = calc_rms(speech)
    n_rms = calc_rms(noise)
    target_n_rms = s_rms / (10 ** (snr_db / 20))
    scale = target_n_rms / n_rms
    mixed = speech + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99: mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)

def get_random_bg_segment(bg_paths: list, target_len: int = WINDOW_SAMPLES):
    while True:
        if not bg_paths:
            return np.zeros(target_len, dtype=np.float32)
        path = random.choice(bg_paths)
        a = load_audio(path)
        if len(a) < target_len:
            seg = np.pad(a, (0, target_len - len(a)))
        elif len(a) > target_len:
            start_sample = random.randint(0, len(a) - target_len)
            seg = a[start_sample : start_sample + target_len]
        else:
            seg = a
        rms = calc_rms(seg)
        if rms >= RMS_THRESH:
            return seg, path

# Original V2.3 Augmentation
def augment_positive(path: str, amb_paths: list, sp_paths: list):
    orig = load_audio(path)
    active = vad_trim(orig)

    pos_rms = calc_rms(active)
    max_shift = max(0, WINDOW_SAMPLES - len(active))
    offset = random.randint(0, max_shift)
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    end_idx = min(offset + len(active), WINDOW_SAMPLES)
    active_len = end_idx - offset
    padded[offset:end_idx] = active[:active_len]

    if random.random() < CLEAN_PROB or (not amb_paths and not sp_paths):
        return padded, {"type": "clean", "bg_src": None, "snr_tier": None, "snr_db": None, "offset": offset}

    if sp_paths and (random.random() < 0.5 or not amb_paths):
        bg_seg, bg_src = get_random_bg_segment(sp_paths)
        bg_type = "speech"
    else:
        bg_seg, bg_src = get_random_bg_segment(amb_paths)
        bg_type = "ambient"

    snr_tier = random.choices(list(SNR_RANGES.keys()), weights=list(SNR_WEIGHTS.values()))[0]
    lo, hi = SNR_RANGES[snr_tier]
    snr_db = random.uniform(lo, hi)
    mixed = mix_snr(padded, bg_seg, snr_db)
    
    return mixed, {"type": bg_type, "bg_src": bg_src, "snr_tier": snr_tier, "snr_db": snr_db, "offset": offset}

def load_splits():
    splits = {
        "train":      {"pos": [], "speech": [], "ambient": [], "other": []},
        "validation": {"pos": [], "speech": [], "ambient": [], "other": []},
        "test":       {"pos": [], "speech": [], "ambient": [], "other": []},
    }
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split = r["split"].strip()
            if split not in splits: continue
            p = str(REPO_ROOT / r["path"])
            label = int(r.get("label", -1))
            g = r.get("group", "").lower()
            if label == 1:
                splits[split]["pos"].append(p)
            elif label == 0:
                if "libri" in g or "speech" in g: splits[split]["speech"].append(p)
                elif "ambient" in g or "background" in g or "noise" in g: splits[split]["ambient"].append(p)
                else: splits[split]["other"].append(p)
    return splits

def prefilter_positives(pos_paths: list) -> list:
    valid = []
    for p in pos_paths:
        try:
            a = load_audio(p)
            active = vad_trim(a)
            if len(active) <= WINDOW_SAMPLES: valid.append(p)
        except:
            pass
    return valid

def main():
    print("="*50)
    print("0. PERMANENTLY FREEZE EXPANDED_VAL FIRST")
    print("="*50)
    
    splits = load_splits()
    val_pos = prefilter_positives(splits["validation"]["pos"])
    val_sp = splits["validation"]["speech"]
    val_amb = splits["validation"]["ambient"]
    val_oth = splits["validation"]["other"]
    
    # Deterministic generation
    random.seed(999)
    np.random.seed(999)
    
    X = []
    labels = []
    negative_categories = []
    positive_metadata = []
    source_paths = []
    
    print("Generating positives...")
    for p in val_pos:
        a, meta = augment_positive(p, val_amb, val_sp)
        spec = make_spectrogram(a)
        X.append(np.expand_dims(spec, -1))
        labels.append(1.0)
        negative_categories.append("positive")
        positive_metadata.append(meta)
        source_paths.append(p)
        
    print("Generating speech negatives...")
    for p in val_sp:
        a = pad_to_window(load_audio(p))
        spec = make_spectrogram(a)
        X.append(np.expand_dims(spec, -1))
        labels.append(0.0)
        negative_categories.append("speech")
        positive_metadata.append(None)
        source_paths.append(p)
        
    print("Generating ambient negatives...")
    for p in val_amb:
        a = pad_to_window(load_audio(p))
        spec = make_spectrogram(a)
        X.append(np.expand_dims(spec, -1))
        labels.append(0.0)
        negative_categories.append("ambient")
        positive_metadata.append(None)
        source_paths.append(p)
        
    X = np.array(X, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
    negative_categories = np.array(negative_categories)
    positive_metadata = np.array(positive_metadata, dtype=object)
    source_paths = np.array(source_paths)
    
    pos_count = np.sum(labels == 1.0)
    sp_count = np.sum(negative_categories == "speech")
    amb_count = np.sum(negative_categories == "ambient")
    
    print(f"positive count: {pos_count}")
    print(f"speech-negative count: {sp_count}")
    print(f"ambient-negative count: {amb_count}")
    
    print(f"Saving to {NPZ_PATH} ...")
    np.savez_compressed(
        NPZ_PATH,
        X=X,
        labels=labels,
        negative_categories=negative_categories,
        positive_metadata=positive_metadata,
        source_paths=source_paths
    )
    
    # Hash ordered source_paths string
    paths_str = "|".join(list(source_paths))
    ordered_sha = hashlib.sha256(paths_str.encode('utf-8')).hexdigest()
    
    # Hash the resulting file
    h = hashlib.sha256()
    with open(NPZ_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    npz_sha = h.hexdigest()
    
    meta_info = {
        "dataset_counts": {
            "positives": int(pos_count),
            "speech_negatives": int(sp_count),
            "ambient_negatives": int(amb_count)
        },
        "construction_seed": 999,
        "augmentation_parameters": {
            "CLEAN_PROB": CLEAN_PROB,
            "SPEECH_PROB": 0.5,
            "AMBIENT_PROB": 0.5,
            "SNR_RANGES": SNR_RANGES
        },
        "ordered_example_sha256": ordered_sha,
        "npz_sha256": npz_sha
    }
    
    with open(JSON_PATH, "w") as f:
        json.dump(meta_info, f, indent=2)
        
    print(f"Metadata saved to {JSON_PATH}")
    print(json.dumps(meta_info, indent=2))
    
    print("\nVerifying V2.3 best_loss...")
    m = tf.keras.models.load_model(str(REPO_ROOT / "cnn" / "models" / "ira_cnn_v2_3_best_loss.keras"))
    
    # Evaluate directly from NPZ arrays
    pos_idx = (labels == 1.0).flatten()
    sp_idx = (negative_categories == "speech").flatten()
    amb_idx = (negative_categories == "ambient").flatten()
    
    X_pos = X[pos_idx]
    X_sp = X[sp_idx]
    X_amb = X[amb_idx]
    
    s_pos = m.predict(X_pos, batch_size=64, verbose=0).flatten()
    s_sp  = m.predict(X_sp, batch_size=64, verbose=0).flatten()
    s_amb = m.predict(X_amb, batch_size=64, verbose=0).flatten()
    
    for thr in [0.43, 0.50]:
        tpr = np.sum(s_pos >= thr) / len(s_pos) * 100
        sfpr = np.sum(s_sp >= thr) / len(s_sp) * 100
        afpr = np.sum(s_amb >= thr) / len(s_amb) * 100
        print(f"thr={thr:.2f}: TPR={tpr:.2f}%, Speech FPR={sfpr:.2f}%, Ambient FPR={afpr:.2f}%")
        
    print("Done freezing EXPANDED_VAL.")

if __name__ == "__main__":
    main()
