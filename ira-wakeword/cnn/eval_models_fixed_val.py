import os, csv, json, random
import numpy as np
import tensorflow as tf
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).parent.parent
MANIFEST = REPO_ROOT / "dataset" / "split_manifest.csv"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
STFT_FRAME_LEN = 480
STFT_FRAME_STEP = 320
STFT_FFT_LEN = 512
N_FREQ_BINS = 40
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
SNR_WEIGHTS = {"easy": 1, "medium": 1, "hard": 1}
CLEAN_PROB = 0.15

def load_audio(path: str) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a.astype(np.float32)

def pad_to_window(audio: np.ndarray) -> np.ndarray:
    if len(audio) < WINDOW_SAMPLES: return np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    return audio[:WINDOW_SAMPLES]

def make_spectrogram(audio: np.ndarray) -> np.ndarray:
    t = tf.convert_to_tensor(audio, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN, frame_step=STFT_FRAME_STEP, fft_length=STFT_FFT_LEN)
    s = tf.abs(s)
    s = tf.math.log(s + 1e-6)
    s = s[:, :N_FREQ_BINS]
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

def get_random_bg_segment(bg_paths: list, target_len: int = WINDOW_SAMPLES):
    while True:
        if not bg_paths: return np.zeros(target_len, dtype=np.float32)
        path = random.choice(bg_paths)
        a = load_audio(path)
        if len(a) < target_len: seg = np.pad(a, (0, target_len - len(a)))
        elif len(a) > target_len:
            start_sample = random.randint(0, len(a) - target_len)
            seg = a[start_sample : start_sample + target_len]
        else: seg = a
        rms = calc_rms(seg)
        if rms >= 1e-4: return seg

def mix_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    s_rms = calc_rms(speech)
    n_rms = calc_rms(noise)
    target_n_rms = s_rms / (10 ** (snr_db / 20))
    scale = target_n_rms / n_rms
    mixed = speech + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99: mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)

def augment_positive(path: str, amb_paths: list, sp_paths: list) -> np.ndarray:
    orig = load_audio(path)
    active = vad_trim(orig)
    if len(active) > WINDOW_SAMPLES: return None
    pos_rms = calc_rms(active)
    max_shift = WINDOW_SAMPLES - len(active)
    offset = random.randint(0, max_shift)
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    padded[offset:offset + len(active)] = active
    if random.random() < CLEAN_PROB or (not amb_paths and not sp_paths): return padded
    if sp_paths and (random.random() < 0.5 or not amb_paths): bg_seg = get_random_bg_segment(sp_paths)
    else: bg_seg = get_random_bg_segment(amb_paths)
    snr_tier = random.choices(list(SNR_RANGES.keys()), weights=list(SNR_WEIGHTS.values()))[0]
    lo, hi = SNR_RANGES[snr_tier]
    snr_db = random.uniform(lo, hi)
    return mix_snr(padded, bg_seg, snr_db)

print("Verifying Validation Checkpoint Rule from History...")
history_csv = REPO_ROOT / "v2_2_training_history.csv"
best_ep, best_sfpr, best_tpr, best_loss = -1, float("inf"), 0.0, float("inf")
with open(history_csv, newline="") as f:
    r = csv.DictReader(f)
    for i, row in enumerate(r):
        ep = i + 1
        tpr = float(row["val_tpr_50"])
        sfpr = float(row["val_speech_fpr_50"])
        loss = float(row["val_loss"])
        
        is_best = False
        if tpr >= 0.95:
            if sfpr < best_sfpr - 1e-4:
                is_best = True
            elif abs(sfpr - best_sfpr) <= 1e-4:
                if tpr > best_tpr + 1e-4:
                    is_best = True
                elif abs(tpr - best_tpr) <= 1e-4:
                    if loss < best_loss:
                        is_best = True
        if is_best:
            best_ep, best_sfpr, best_tpr, best_loss = ep, sfpr, tpr, loss

print(f"According to rule: TPR >= 95% -> lowest sfpr -> highest tpr -> lowest loss")
print(f"Selected Epoch: {best_ep}")
print(f"Validation metrics at selected epoch (from CSV): TPR={best_tpr*100:.2f}%, Speech FPR={best_sfpr*100:.2f}%, Val Loss={best_loss:.4f}\n")


splits = {"validation": {"pos": [], "speech": [], "ambient": [], "other": []}}
with open(MANIFEST, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["split"] != "validation": continue
        p = str(REPO_ROOT / r["path"])
        if not Path(p).exists(): continue
        label = int(r.get("label", -1))
        g = r.get("group", "").lower()
        if label == 1: splits["validation"]["pos"].append(p)
        elif label == 0:
            if "libri" in g or "speech" in g: splits["validation"]["speech"].append(p)
            elif "ambient" in g or "background" in g or "noise" in g: splits["validation"]["ambient"].append(p)
            else: splits["validation"]["other"].append(p)

val_pos = splits["validation"]["pos"]
val_sp  = splits["validation"]["speech"]
val_amb = splits["validation"]["ambient"]
val_oth = splits["validation"]["other"]

random.seed(999)
np.random.seed(999)

val_pos_X = []
for p in val_pos:
    audio = augment_positive(p, val_amb, val_sp)
    if audio is not None:
        val_pos_X.append(np.expand_dims(make_spectrogram(audio), -1))

val_sp_X = []
for p in val_sp:
    audio = pad_to_window(load_audio(p))
    val_sp_X.append(np.expand_dims(make_spectrogram(audio), -1))

val_amb_X = []
for p in val_amb:
    audio = pad_to_window(load_audio(p))
    val_amb_X.append(np.expand_dims(make_spectrogram(audio), -1))

val_pos_X = np.array(val_pos_X, dtype=np.float32)
val_sp_X = np.array(val_sp_X, dtype=np.float32)
val_amb_X = np.array(val_amb_X, dtype=np.float32)

print("Validation set built. Evaluating models...")
models_to_test = {
    "V2": REPO_ROOT / "cnn" / "models" / "ira_cnn_v2_best_loss.keras",
    "V2.1": REPO_ROOT / "cnn" / "models" / "ira_cnn_v2_1_best_speech_fpr.keras",
    "V2.2": REPO_ROOT / "cnn" / "models" / "ira_cnn_v2_2_best_speech_fpr.keras",
}

thresholds = np.arange(0.30, 0.951, 0.01)

results_data = {}

for name, path in models_to_test.items():
    print(f"\n================ {name} ================")
    if not path.exists():
        print("Model not found.")
        continue
    model = tf.keras.models.load_model(str(path))
    s_pos = model.predict(val_pos_X, batch_size=64, verbose=0).flatten()
    s_sp  = model.predict(val_sp_X, batch_size=64, verbose=0).flatten()
    s_amb = model.predict(val_amb_X, batch_size=64, verbose=0).flatten()
    s_neg = np.concatenate([s_sp, s_amb])
    
    results_data[name] = {"pos": s_pos, "sp": s_sp, "amb": s_amb, "neg": s_neg}
    
    print("THR\tTPR\tSpFPR\tAmFPR\tOvrFPR")
    for thr in thresholds:
        tpr = float(np.sum(s_pos >= thr) / len(s_pos))
        sfpr = float(np.sum(s_sp >= thr) / len(s_sp))
        afpr = float(np.sum(s_amb >= thr) / len(s_amb))
        ofpr = float(np.sum(s_neg >= thr) / len(s_neg))
        print(f"{thr:.2f}\t{tpr*100:.2f}%\t{sfpr*100:.2f}%\t{afpr*100:.2f}%\t{ofpr*100:.2f}%")
        
    if name == "V2.2":
        print("\n--- V2.2 Score Distributions ---")
        print("Positive Scores:")
        print(f"Mean:   {np.mean(s_pos):.4f}")
        print(f"Median: {np.median(s_pos):.4f}")
        print(f"P10:    {np.percentile(s_pos, 10):.4f}")
        print(f"P25:    {np.percentile(s_pos, 25):.4f}")
        print(f"P50:    {np.percentile(s_pos, 50):.4f}")
        print(f"P75:    {np.percentile(s_pos, 75):.4f}")
        print(f"P90:    {np.percentile(s_pos, 90):.4f}")
        
        print("\nSpeech Negative Scores:")
        print(f"Mean:   {np.mean(s_sp):.4f}")
        print(f"Median: {np.median(s_sp):.4f}")
        print(f"P90:    {np.percentile(s_sp, 90):.4f}")
        print(f"P95:    {np.percentile(s_sp, 95):.4f}")
        print(f"P99:    {np.percentile(s_sp, 99):.4f}")
        print(f"Max:    {np.max(s_sp):.4f}")
        
        print("\n--- Feasibility Checks (V2.2) ---")
        check1 = False
        check2 = False
        check3 = False
        
        for thr in thresholds:
            tpr = float(np.sum(s_pos >= thr) / len(s_pos))
            sfpr = float(np.sum(s_sp >= thr) / len(s_sp))
            
            if tpr >= 0.95 and sfpr <= 0.02 and not check1:
                print(f"Feasibility (TPR>=95, SpFPR<=2%): PASSED at thr={thr:.2f} (TPR={tpr*100:.2f}%, SpFPR={sfpr*100:.2f}%)")
                check1 = True
            
            if tpr >= 0.94 and sfpr <= 0.02 and not check2:
                print(f"Feasibility (TPR>=94, SpFPR<=2%): PASSED at thr={thr:.2f} (TPR={tpr*100:.2f}%, SpFPR={sfpr*100:.2f}%)")
                check2 = True
                
            if tpr >= 0.95 and sfpr <= 0.03 and not check3:
                print(f"Feasibility (TPR>=95, SpFPR<=3%): PASSED at thr={thr:.2f} (TPR={tpr*100:.2f}%, SpFPR={sfpr*100:.2f}%)")
                check3 = True
                
        if not check1: print("Feasibility (TPR>=95, SpFPR<=2%): FAILED")
        if not check2: print("Feasibility (TPR>=94, SpFPR<=2%): FAILED")
        if not check3: print("Feasibility (TPR>=95, SpFPR<=3%): FAILED")

