# -*- coding: utf-8 -*-
"""
train_cnn_v2_2.py
=================
IRA CNN V2.2 -- Controlled experiment: MODERATE speech-negative curriculum.
"""

import os, csv, json, random, time, collections
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).parent.parent
MANIFEST       = REPO_ROOT / "dataset" / "split_manifest.csv"
SCORES_CSV     = REPO_ROOT / "v2_train_speech_negative_scores.csv"

MODEL_OUT_DIR  = REPO_ROOT / "cnn" / "models"
BEST_LOSS_PATH = MODEL_OUT_DIR / "ira_cnn_v2_2_best_loss.keras"
BEST_SFPR_PATH = MODEL_OUT_DIR / "ira_cnn_v2_2_best_speech_fpr.keras"
FINAL_PATH     = MODEL_OUT_DIR / "ira_cnn_v2_2_final.keras"

HISTORY_CSV    = REPO_ROOT / "v2_2_training_history.csv"
STATS_JSON     = REPO_ROOT / "v2_2_augmentation_stats.json"
HARD_SAMPLING_JSON = REPO_ROOT / "v2_2_hard_sampling_stats.json"

# ---------------------------------------------------------------------------
# Audio / feature constants
# ---------------------------------------------------------------------------
SAMPLE_RATE    = 16000
WINDOW_SAMPLES = 16000

STFT_FRAME_LEN  = 480
STFT_FRAME_STEP = 320
STFT_FFT_LEN    = 512
N_FREQ_BINS     = 40

# ---------------------------------------------------------------------------
# Augmentation constants
# ---------------------------------------------------------------------------
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
SNR_WEIGHTS = {"easy": 1, "medium": 1, "hard": 1}
CLEAN_PROB = 0.15

N_EASY_SP   = 16
N_MED_SP    = 6
N_HARD_SP   = 3
N_AMBIENT   = 5
N_OTHER     = 2

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE     = 64
EPOCHS         = 30
LEARNING_RATE  = 0.001
POS_PER_BATCH  = BATCH_SIZE // 2
NEG_PER_BATCH  = BATCH_SIZE // 2
EARLY_STOP_PATIENCE = 5


# ===========================================================================
# Audio utilities
# ===========================================================================
def load_audio(path: str) -> np.ndarray:
    try:
        a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {path}: {e}")
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a.astype(np.float32)

def pad_to_window(audio: np.ndarray) -> np.ndarray:
    if len(audio) < WINDOW_SAMPLES:
        return np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    return audio[:WINDOW_SAMPLES]

def make_spectrogram(audio: np.ndarray) -> np.ndarray:
    t = tf.convert_to_tensor(audio, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN,
                       frame_step=STFT_FRAME_STEP, fft_length=STFT_FFT_LEN)
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

def get_random_bg_segment(bg_paths: list, stats: dict, target_len: int = WINDOW_SAMPLES):
    while True:
        if not bg_paths:
            return np.zeros(target_len, dtype=np.float32), "None", 0
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
        if rms >= 1e-4:
            return seg, path, 0
        stats["silent_bg_rejected"] += 1

def mix_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    s_rms = calc_rms(speech)
    n_rms = calc_rms(noise)
    target_n_rms = s_rms / (10 ** (snr_db / 20))
    scale = target_n_rms / n_rms
    mixed = speech + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99: mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)

def augment_positive(path: str, amb_paths: list, sp_paths: list, stats: dict) -> np.ndarray:
    orig = load_audio(path)
    active = vad_trim(orig)

    if len(active) > WINDOW_SAMPLES:
        raise ValueError(f"Positive {path} active speech > {WINDOW_SAMPLES} samples.")

    pos_rms = calc_rms(active)
    max_shift = WINDOW_SAMPLES - len(active)
    offset = random.randint(0, max_shift)
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    padded[offset:offset + len(active)] = active

    if random.random() < CLEAN_PROB or (not amb_paths and not sp_paths):
        stats["clean"] += 1
        return padded

    if sp_paths and (random.random() < 0.5 or not amb_paths):
        bg_seg, bg_src, _ = get_random_bg_segment(sp_paths, stats)
        stats["speech"] += 1
    else:
        bg_seg, bg_src, _ = get_random_bg_segment(amb_paths, stats)
        stats["ambient"] += 1

    snr_tier = random.choices(list(SNR_RANGES.keys()), weights=list(SNR_WEIGHTS.values()))[0]
    lo, hi = SNR_RANGES[snr_tier]
    snr_db = random.uniform(lo, hi)
    mixed = mix_snr(padded, bg_seg, snr_db)

    if snr_tier == "easy":   stats["easy_snr"] += 1
    elif snr_tier == "medium": stats["med_snr"] += 1
    else:                      stats["hard_snr"] += 1

    return mixed


# ===========================================================================
# Split loading
# ===========================================================================
def load_splits():
    splits = {
        "train":      {"pos": [], "ambient": [], "other": []},
        "validation": {"pos": [], "speech": [], "ambient": [], "other": []},
        "test":       {"pos": [], "speech": [], "ambient": [], "other": []}
    }
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split = r["split"]
            if split not in splits: continue
            p = str(REPO_ROOT / r["path"])
            if not Path(p).exists(): continue
            label = int(r.get("label", -1))
            g = r.get("group", "").lower()
            
            if split == "train":
                if label == 1:
                    splits["train"]["pos"].append(p)
                elif label == 0:
                    if "ambient" in g or "background" in g or "noise" in g:
                        splits["train"]["ambient"].append(p)
                    elif not ("libri" in g or "speech" in g):
                        splits["train"]["other"].append(p)
            elif split == "validation":
                if label == 1:
                    splits["validation"]["pos"].append(p)
                elif label == 0:
                    if "libri" in g or "speech" in g:
                        splits["validation"]["speech"].append(p)
                    elif "ambient" in g or "background" in g or "noise" in g:
                        splits["validation"]["ambient"].append(p)
                    else:
                        splits["validation"]["other"].append(p)
            elif split == "test":
                if label == 1:
                    splits["test"]["pos"].append(p)
                elif label == 0:
                    if "libri" in g or "speech" in g:
                        splits["test"]["speech"].append(p)
                    elif "ambient" in g or "background" in g or "noise" in g:
                        splits["test"]["ambient"].append(p)
                    else:
                        splits["test"]["other"].append(p)
    return splits

def load_speech_pools():
    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"Scores CSV not found: {SCORES_CSV}")
    
    easy, med, hard = set(), set(), set()
    spk_easy, spk_med, spk_hard = set(), set(), set()
    
    # Verify via manifest
    manifest_info = {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = str(REPO_ROOT / r["path"])
            manifest_info[p] = r
            
    with open(SCORES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            score = float(r["V2_score"])
            p = str(Path(r["path"]))
            spk = r.get("speaker_id", "unknown")
            if not Path(p).exists(): continue
            
            # Cross-check with manifest
            if p not in manifest_info: continue
            minfo = manifest_info[p]
            assert minfo["split"] == "train", f"{p} is not train split!"
            assert int(minfo["label"]) == 0, f"{p} is not negative!"
            assert "libri" in minfo["group"].lower() or "speech" in minfo["group"].lower(), f"{p} is not speech!"
            
            if score < 0.10:
                easy.add(p)
                spk_easy.add(spk)
            elif score < 0.50:
                med.add(p)
                spk_med.add(spk)
            else:
                hard.add(p)
                spk_hard.add(spk)
                
    return list(easy), list(med), list(hard), len(spk_easy), len(spk_med), len(spk_hard)

def prefilter_positives(pos_paths: list) -> list:
    valid, skipped = [], []
    for p in pos_paths:
        try:
            a = load_audio(p)
            active = vad_trim(a)
            if len(active) > WINDOW_SAMPLES: skipped.append(p)
            else: valid.append(p)
        except Exception:
            skipped.append(p)
    return valid

# ===========================================================================
# Generator
# ===========================================================================
class V22DataGenerator(tf.keras.utils.PyDataset):
    def __init__(self, positives, easy_sp, med_sp, hard_sp, ambient_negs, other_negs,
                 amb_paths, sp_paths, stats, **kwargs):
        super().__init__(**kwargs)
        self.positives    = positives
        self.easy_sp      = easy_sp
        self.med_sp       = med_sp
        self.hard_sp      = hard_sp
        self.ambient_negs = ambient_negs
        self.other_negs   = other_negs
        self.amb_paths    = amb_paths
        self.sp_paths     = sp_paths
        self.stats        = stats
        self.epoch_hard_counts = collections.defaultdict(int)
        self.hard_stats_history = []

        self.batch_count = len(self.positives) // POS_PER_BATCH
        self.on_epoch_end()

    def __len__(self):
        return self.batch_count

    def on_epoch_end(self):
        # Save hard counts for the completed epoch
        if self.epoch_hard_counts:
            counts = list(self.epoch_hard_counts.values())
            self.hard_stats_history.append({
                "unique_files_sampled": len(counts),
                "total_draws": sum(counts),
                "min_repetitions": min(counts),
                "mean_repetitions": sum(counts) / len(counts),
                "max_repetitions": max(counts)
            })
            with open(HARD_SAMPLING_JSON, "w") as f:
                json.dump(self.hard_stats_history, f, indent=2)
            self.epoch_hard_counts.clear()

        self._pos_perm  = np.random.permutation(len(self.positives))
        self._easy_perm = np.random.permutation(len(self.easy_sp))
        self._med_perm  = np.random.permutation(len(self.med_sp))
        self._hard_perm = np.random.permutation(len(self.hard_sp))
        self._amb_perm  = np.random.permutation(len(self.ambient_negs))
        self._oth_perm  = np.random.permutation(len(self.other_negs)) if self.other_negs else None
        
        self._pos_idx = self._easy_idx = self._med_idx = self._hard_idx = self._amb_idx = self._oth_idx = 0

    def _next_idx(self, perm, idx_attr, pool_len, n):
        cur = getattr(self, idx_attr)
        idxs = []
        for _ in range(n):
            if cur >= pool_len:
                perm[:] = np.random.permutation(pool_len)
                cur = 0
            idxs.append(int(perm[cur])); cur += 1
        setattr(self, idx_attr, cur)
        return idxs

    def __getitem__(self, idx):
        X = np.zeros((BATCH_SIZE, 49, N_FREQ_BINS, 1), dtype=np.float32)
        y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

        # Positives
        pos_idxs = self._next_idx(self._pos_perm, "_pos_idx", len(self.positives), POS_PER_BATCH)
        for i, pi in enumerate(pos_idxs):
            audio = augment_positive(self.positives[pi], self.amb_paths, self.sp_paths, self.stats)
            X[i, :, :, 0] = make_spectrogram(audio)
            y[i, 0] = 1.0

        # Fill Negatives
        curr_n = POS_PER_BATCH
        
        # Easy Speech
        n_e = min(N_EASY_SP, len(self.easy_sp))
        if n_e > 0:
            for i in self._next_idx(self._easy_perm, "_easy_idx", len(self.easy_sp), n_e):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.easy_sp[i])))
                curr_n += 1
                self.stats["easy_speech_sampled"] += 1

        # Medium Speech
        n_m = min(N_MED_SP, len(self.med_sp))
        if n_m > 0:
            for i in self._next_idx(self._med_perm, "_med_idx", len(self.med_sp), n_m):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.med_sp[i])))
                curr_n += 1
                self.stats["medium_speech_sampled"] += 1
                
        # Hard Speech
        n_h = min(N_HARD_SP, len(self.hard_sp))
        if n_h > 0:
            for i in self._next_idx(self._hard_perm, "_hard_idx", len(self.hard_sp), n_h):
                path = self.hard_sp[i]
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(path)))
                curr_n += 1
                self.stats["hard_speech_sampled"] += 1
                self.epoch_hard_counts[path] += 1
                
        # Ambient
        n_a = min(N_AMBIENT, len(self.ambient_negs))
        if n_a > 0:
            for i in self._next_idx(self._amb_perm, "_amb_idx", len(self.ambient_negs), n_a):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.ambient_negs[i])))
                curr_n += 1
                self.stats["ambient_sampled"] += 1
                
        # Other (fallback to easy speech if needed)
        rem = BATCH_SIZE - curr_n
        if self.other_negs and rem > 0:
            n_o = min(rem, N_OTHER)
            for i in self._next_idx(self._oth_perm, "_oth_idx", len(self.other_negs), n_o):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.other_negs[i])))
                curr_n += 1
                self.stats["other_sampled"] += 1
                
        # Fallback for any remaining slots
        rem = BATCH_SIZE - curr_n
        if rem > 0:
            for i in self._next_idx(self._easy_perm, "_easy_idx", len(self.easy_sp), rem):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.easy_sp[i])))
                curr_n += 1
                self.stats["easy_speech_sampled"] += 1

        return X, y


# ===========================================================================
# Callbacks
# ===========================================================================
class ValidationFPRCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_pos_X, val_sp_X, val_amb_X, model_path):
        super().__init__()
        self.val_pos_X = val_pos_X
        self.val_sp_X = val_sp_X
        self.val_amb_X = val_amb_X
        self.val_all_neg_X = np.concatenate([val_sp_X, val_amb_X]) if len(val_sp_X) and len(val_amb_X) else (val_sp_X if len(val_sp_X) else val_amb_X)
        self.best_sfpr_path = model_path
        self.records = []
        
        self._best_sfpr = float("inf")
        self._best_tpr = 0.0
        self._best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        THR = 0.50
        s_pos = self.model.predict(self.val_pos_X, batch_size=64, verbose=0).flatten() if len(self.val_pos_X) else np.array([])
        s_sp  = self.model.predict(self.val_sp_X, batch_size=64, verbose=0).flatten() if len(self.val_sp_X) else np.array([])
        s_amb = self.model.predict(self.val_amb_X, batch_size=64, verbose=0).flatten() if len(self.val_amb_X) else np.array([])
        all_neg = np.concatenate([s_sp, s_amb]) if len(s_sp) and len(s_amb) else (s_sp if len(s_sp) else s_amb)
        
        tpr  = float(np.sum(s_pos >= THR) / len(s_pos)) if len(s_pos) else 0.0
        sfpr = float(np.sum(s_sp  >= THR) / len(s_sp))  if len(s_sp)  else 0.0
        afpr = float(np.sum(s_amb >= THR) / len(s_amb)) if len(s_amb) else 0.0
        ofpr = float(np.sum(all_neg >= THR) / len(all_neg)) if len(all_neg) else 0.0
        vloss = logs.get("val_loss", float("inf")) if logs else float("inf")

        print(f"  [ValFPR] epoch={epoch+1}  TPR={tpr*100:.2f}%  "
              f"speech_FPR={sfpr*100:.2f}%  ambient_FPR={afpr*100:.2f}%  overall_FPR={ofpr*100:.2f}%")

        self.records.append({
            "epoch": epoch + 1,
            "val_tpr": tpr, "val_speech_fpr": sfpr,
            "val_ambient_fpr": afpr, "val_overall_fpr": ofpr,
        })
        if logs is not None:
            logs["val_tpr_50"] = tpr
            logs["val_speech_fpr_50"] = sfpr
            logs["val_ambient_fpr_50"] = afpr
            logs["val_overall_fpr_50"] = ofpr

        # Selection rule: TPR >= 95%, lowest sfpr. 
        # If tied (within 0.0001), higher TPR, then lower val_loss
        is_best = False
        if tpr >= 0.95:
            if sfpr < self._best_sfpr - 1e-4:
                is_best = True
            elif abs(sfpr - self._best_sfpr) <= 1e-4:
                if tpr > self._best_tpr + 1e-4:
                    is_best = True
                elif abs(tpr - self._best_tpr) <= 1e-4:
                    if vloss < self._best_loss:
                        is_best = True
                        
        if is_best:
            self._best_sfpr = sfpr
            self._best_tpr = tpr
            self._best_loss = vloss
            self.model.save(str(self.best_sfpr_path))
            print(f"  [ValFPR] ** Saved best checkpoint: epoch={epoch+1}, "
                  f"speech_FPR={sfpr*100:.2f}%, TPR={tpr*100:.2f}%")

class LrHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(lr)
        if logs is not None: logs["learning_rate"] = lr


# ===========================================================================
# Model architecture
# ===========================================================================
def build_model():
    inp = tf.keras.Input(shape=(49, N_FREQ_BINS, 1), name="input")
    x = tf.keras.layers.Conv2D(8, (3,3), activation="relu", padding="same")(inp)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inp, out, name="IRA_CNN_V2_2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ===========================================================================
# Main
# ===========================================================================
def main():
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading split manifest...")
    splits = load_splits()
    train_pos = prefilter_positives(splits["train"]["pos"])
    train_amb = splits["train"]["ambient"]
    train_oth = splits["train"]["other"]
    
    val_pos = splits["validation"]["pos"]
    val_sp  = splits["validation"]["speech"]
    val_amb = splits["validation"]["ambient"]
    val_oth = splits["validation"]["other"]
    
    test_amb = splits["test"]["ambient"]

    print("Loading V2-mined speech pools...")
    easy_sp, med_sp, hard_sp, u_easy, u_med, u_hard = load_speech_pools()
    all_train_sp = easy_sp + med_sp + hard_sp
    
    print("\n" + "="*65)
    print("PRE-TRAINING SANITY CHECK")
    print("="*65)
    
    # Prove non-overlap
    train_amb_set = set(train_amb)
    val_amb_set = set(val_amb)
    test_amb_set = set(test_amb)
    assert train_amb_set.isdisjoint(val_amb_set), "Overlap between train and validation ambient pools!"
    assert train_amb_set.isdisjoint(test_amb_set), "Overlap between train and test ambient pools!"
    
    easy_set = set(easy_sp)
    med_set = set(med_sp)
    hard_set = set(hard_sp)
    assert easy_set.isdisjoint(med_set), "Overlap between easy and medium speech pools!"
    assert easy_set.isdisjoint(hard_set), "Overlap between easy and hard speech pools!"
    assert med_set.isdisjoint(hard_set), "Overlap between medium and hard speech pools!"
    
    all_train_sp_set = set(all_train_sp)
    val_sp_set = set(val_sp)
    test_sp_set = set(splits["test"]["speech"])
    assert all_train_sp_set.isdisjoint(val_sp_set), "Overlap between train speech and validation speech pools!"
    assert all_train_sp_set.isdisjoint(test_sp_set), "Overlap between train speech and test speech pools!"

    print("No train/validation/test overlap detected in ambient or speech pools.")
    print("Train ambient pool counts:", len(train_amb))
    print("Validation ambient pool counts:", len(val_amb))
    print("Test ambient pool counts:", len(test_amb))
    
    print("\nBuilding fixed deterministic validation set...")
    r_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(999)
    np.random.seed(999)
    
    val_pos_stats = {"clean":0, "ambient":0, "speech":0, "easy_snr":0, "med_snr":0, "hard_snr":0, "silent_bg_rejected": 0}
    val_pos_X = []
    for p in val_pos:
        audio = augment_positive(p, val_amb, val_sp, val_pos_stats)
        val_pos_X.append(np.expand_dims(make_spectrogram(audio), -1))
    
    val_sp_X = []
    for p in val_sp:
        audio = pad_to_window(load_audio(p))
        val_sp_X.append(np.expand_dims(make_spectrogram(audio), -1))
        
    val_amb_X = []
    for p in val_amb:
        audio = pad_to_window(load_audio(p))
        val_amb_X.append(np.expand_dims(make_spectrogram(audio), -1))
        
    val_oth_X = []
    for p in val_oth:
        audio = pad_to_window(load_audio(p))
        val_oth_X.append(np.expand_dims(make_spectrogram(audio), -1))
        
    val_pos_X = np.array(val_pos_X, dtype=np.float32)
    val_sp_X = np.array(val_sp_X, dtype=np.float32)
    val_amb_X = np.array(val_amb_X, dtype=np.float32)
    val_oth_X = np.array(val_oth_X, dtype=np.float32)
    
    val_X_list = [val_pos_X, val_sp_X, val_amb_X]
    if len(val_oth_X) > 0: val_X_list.append(val_oth_X)
    val_X = np.concatenate(val_X_list)
    val_y = np.concatenate([np.ones(len(val_pos_X)), np.zeros(len(val_sp_X) + len(val_amb_X) + len(val_oth_X))]).reshape(-1, 1)

    random.setstate(r_state)
    np.random.set_state(np_state)
    
    print("Fixed deterministic validation set created successfully.")
    
    train_stats = {k: 0 for k in ["clean","ambient","speech","easy_snr","med_snr","hard_snr", "silent_bg_rejected",
                                   "easy_speech_sampled", "medium_speech_sampled", "hard_speech_sampled",
                                   "ambient_sampled", "other_sampled"]}

    train_gen = V22DataGenerator(
        train_pos, easy_sp, med_sp, hard_sp, train_amb, train_oth, train_amb, all_train_sp, train_stats
    )
    
    # Generate 3 batches for sanity check
    print("Running 3-batch generator test...")
    for i in range(3):
        X, y = train_gen[i]
        
    assert train_stats["easy_speech_sampled"] > 0
    assert train_stats["medium_speech_sampled"] > 0
    assert train_stats["hard_speech_sampled"] > 0
    assert train_stats["ambient_sampled"] > 0
    assert train_stats["clean"] > 0
    assert (train_stats["ambient"] > 0 or train_stats["speech"] > 0)
    
    # Reset stats
    for k in train_stats: train_stats[k] = 0
    train_gen.epoch_hard_counts.clear()
    
    print("V2.2 TRAINING PIPELINE VALIDATION: PASS")

    print("\n" + "="*65)
    print("PRE-TRAINING DATA REPORT (V2.2)")
    print("="*65)
    print(f"  Train positives         : {len(train_pos)}")
    print(f"  Train easy speech (<0.1): {len(easy_sp)} ({u_easy} spk)")
    print(f"  Train med speech (<0.5) : {len(med_sp)} ({u_med} spk)")
    print(f"  Train hard speech (>=.5): {len(hard_sp)} ({u_hard} spk)")
    print(f"  Train ambient negs      : {len(train_amb)}")
    print(f"  Train other negs        : {len(train_oth)}")
    print(f"  Val positives           : {len(val_pos_X)}")
    print(f"  Val speech negs         : {len(val_sp_X)}")
    print(f"  Val ambient negs        : {len(val_amb_X)}")
    print(f"  Val other negs          : {len(val_oth_X)}")
    
    model = build_model()
    fpr_cb = ValidationFPRCallback(val_pos_X, val_sp_X, val_amb_X, BEST_SFPR_PATH)
    lr_cb  = LrHistoryCallback()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_LOSS_PATH), monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        fpr_cb,
        lr_cb,
    ]

    print("\n" + "="*65)
    print("TRAINING IRA CNN V2.2")
    print("="*65)
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=(val_X, val_y),
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(FINAL_PATH))
    print(f"\nSaved final model: {FINAL_PATH}")

    with open(STATS_JSON, "w") as f:
        json.dump({"train_stats": train_stats}, f, indent=2)

    h = history.history
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(h.keys()) + ["val_tpr_50","val_speech_fpr_50","val_ambient_fpr_50","val_overall_fpr_50", "learning_rate"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ep in range(len(h["loss"])):
            row = {k: h[k][ep] for k in h}
            if ep < len(fpr_cb.records):
                rec = fpr_cb.records[ep]
                row["val_tpr_50"] = rec["val_tpr"]
                row["val_speech_fpr_50"] = rec["val_speech_fpr"]
                row["val_ambient_fpr_50"] = rec["val_ambient_fpr"]
                row["val_overall_fpr_50"] = rec["val_overall_fpr"]
            row["learning_rate"] = lr_cb.lrs[ep] if ep < len(lr_cb.lrs) else ""
            w.writerow(row)

    print("\nDone. Ready for V2.2 evaluation.")

if __name__ == "__main__":
    main()
