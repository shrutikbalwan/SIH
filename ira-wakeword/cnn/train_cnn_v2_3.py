# -*- coding: utf-8 -*-
"""
train_cnn_v2_3.py
=================
IRA CNN V2.3 -- Expanded negative dataset with verified TRAIN-only speech pools.

Batch recipe  (32 neg per batch):
  14 easy speech  (V2 score < 0.10)
   7 medium speech (0.10 <= score < 0.50)
   4 hard speech   (score >= 0.50)
   5 ambient
   2 other
  ---
  32 total negatives

  32 positives
  ---
  64 total per batch

Key improvements over V2.2:
  - Uses split_manifest_v3.csv (148 speakers across 4 splits)
  - Difficulty pools built from audit scores on 12142 TRAIN clips
  - Per-epoch sampling statistics reported for ALL three pools
  - No unseen_test or historical test involvement whatsoever
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
MANIFEST       = REPO_ROOT / "dataset" / "split_manifest_v3.csv"

MODEL_OUT_DIR  = REPO_ROOT / "cnn" / "models"
BEST_LOSS_PATH = MODEL_OUT_DIR / "ira_cnn_v2_3_best_loss.keras"
BEST_SFPR_PATH = MODEL_OUT_DIR / "ira_cnn_v2_3_best_speech_fpr.keras"
FINAL_PATH     = MODEL_OUT_DIR / "ira_cnn_v2_3_final.keras"

HISTORY_CSV        = REPO_ROOT / "v2_3_training_history.csv"
STATS_JSON         = REPO_ROOT / "v2_3_augmentation_stats.json"
SAMPLING_STATS_JSON= REPO_ROOT / "v2_3_sampling_stats.json"

# V2 model used for scoring difficulty pools (audit-verified)
V2_MODEL_PATH  = MODEL_OUT_DIR / "ira_cnn_v2_best_loss.keras"

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
# Augmentation constants (unchanged from V2.2)
# ---------------------------------------------------------------------------
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
SNR_WEIGHTS = {"easy": 1, "medium": 1, "hard": 1}
CLEAN_PROB = 0.15

# V2.3 batch recipe
N_EASY_SP   = 14
N_MED_SP    =  7
N_HARD_SP   =  4
N_AMBIENT   =  5
N_OTHER     =  2

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE     = 64
EPOCHS         = 30
LEARNING_RATE  = 0.001
POS_PER_BATCH  = BATCH_SIZE // 2   # 32
NEG_PER_BATCH  = BATCH_SIZE // 2   # 32
EARLY_STOP_PATIENCE = 5

# ---------------------------------------------------------------------------
# V2 difficulty thresholds (audit-verified)
# ---------------------------------------------------------------------------
EASY_THRESH = 0.10
HARD_THRESH = 0.50
RMS_THRESH  = 1e-4


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
        if rms >= RMS_THRESH:
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

    if snr_tier == "easy":     stats["easy_snr"] += 1
    elif snr_tier == "medium": stats["med_snr"] += 1
    else:                      stats["hard_snr"] += 1

    return mixed


# ===========================================================================
# Split loading from manifest_v3
# ===========================================================================
def load_splits():
    """
    Loads train, validation from split_manifest_v3.csv.
    unseen_test is NEVER loaded — we skip it entirely.
    historical test ('test') is loaded only for ambient disjoint assertion.
    """
    splits = {
        "train":      {"pos": [], "speech": [], "ambient": [], "other": []},
        "validation": {"pos": [], "speech": [], "ambient": [], "other": []},
        "test":       {"pos": [], "speech": [], "ambient": [], "other": []},
    }
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split = r["split"].strip()
            if split == "unseen_test":
                continue  # NEVER loaded during training
            if split not in splits:
                continue
            p = str(REPO_ROOT / r["path"])
            if not Path(p).exists():
                continue
            label = int(r.get("label", -1))
            g = r.get("group", "").lower()

            if label == 1:
                splits[split]["pos"].append(p)
            elif label == 0:
                if "libri" in g or "speech" in g:
                    splits[split]["speech"].append(p)
                elif "ambient" in g or "background" in g or "noise" in g:
                    splits[split]["ambient"].append(p)
                else:
                    splits[split]["other"].append(p)
    return splits


def build_difficulty_pools_from_manifest(train_speech_paths: list):
    """
    Score all train speech negatives using the V2 baseline model.
    This replicates the verified audit scoring from audit_integrity.py.
    Pools are TRAIN-only by construction (train_speech_paths is from manifest train split).
    """
    print("  Loading V2 baseline for difficulty scoring...")
    v2_model = tf.keras.models.load_model(str(V2_MODEL_PATH))

    print(f"  Scoring {len(train_speech_paths)} train speech clips...")
    X = []
    for p in train_speech_paths:
        a = pad_to_window(load_audio(p))
        t = tf.convert_to_tensor(a, dtype=tf.float32)
        s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN, frame_step=STFT_FRAME_STEP, fft_length=STFT_FFT_LEN)
        s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_FREQ_BINS]
        m = tf.reduce_mean(s); d = tf.math.reduce_std(s) + 1e-6
        X.append(np.expand_dims(((s - m)/d).numpy(), -1))

    X = np.array(X, dtype=np.float32)
    scores = v2_model.predict(X, batch_size=128, verbose=0).flatten()
    del v2_model, X  # free memory

    easy, med, hard = [], [], []
    for i, score in enumerate(scores):
        if score < EASY_THRESH:
            easy.append(train_speech_paths[i])
        elif score < HARD_THRESH:
            med.append(train_speech_paths[i])
        else:
            hard.append(train_speech_paths[i])

    def pool_speakers(paths):
        spks = set()
        for p in paths:
            spk = os.path.basename(p).split("_")[0]
            if spk.isdigit(): spks.add(spk)
        return spks

    spk_easy  = pool_speakers(easy)
    spk_med   = pool_speakers(med)
    spk_hard  = pool_speakers(hard)

    print(f"  Easy  (< {EASY_THRESH}): {len(easy):5d} clips, {len(spk_easy):3d} speakers")
    print(f"  Med   ({EASY_THRESH} - {HARD_THRESH}): {len(med):5d} clips, {len(spk_med):3d} speakers")
    print(f"  Hard  (>= {HARD_THRESH}): {len(hard):5d} clips, {len(spk_hard):3d} speakers")
    print(f"  Score dist  mean={np.mean(scores):.4f}  median={np.median(scores):.4f}"
          f"  P90={np.percentile(scores,90):.4f}  P95={np.percentile(scores,95):.4f}"
          f"  P99={np.percentile(scores,99):.4f}")

    return easy, med, hard, spk_easy, spk_med, spk_hard


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
    if skipped:
        print(f"  Skipped {len(skipped)} positives (too long or unreadable).")
    return valid


# ===========================================================================
# Generator
# ===========================================================================
class V23DataGenerator(tf.keras.utils.PyDataset):
    """
    Batch recipe (verified at sanity check):
      32 positives (with augmentation)
      14 easy speech negatives
       7 medium speech negatives
       4 hard speech negatives
       5 ambient negatives
       2 other negatives
      ---
      64 total
    """
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

        # Per-epoch sampling tracking for all three pools
        self._epoch_easy_counts = collections.defaultdict(int)
        self._epoch_med_counts  = collections.defaultdict(int)
        self._epoch_hard_counts = collections.defaultdict(int)
        self.sampling_history   = []

        self.batch_count = len(self.positives) // POS_PER_BATCH
        self.on_epoch_end()

    def __len__(self):
        return self.batch_count

    def on_epoch_end(self):
        # Record sampling stats for completed epoch
        for pool_name, counts_dict in [
            ("easy",   self._epoch_easy_counts),
            ("medium", self._epoch_med_counts),
            ("hard",   self._epoch_hard_counts),
        ]:
            if counts_dict:
                vals = list(counts_dict.values())
                self.sampling_history.append({
                    "pool": pool_name,
                    "unique_clips_sampled": len(vals),
                    "total_draws": sum(vals),
                    "unique_speakers": len({os.path.basename(p).split("_")[0]
                                            for p in counts_dict.keys()
                                            if os.path.basename(p).split("_")[0].isdigit()}),
                    "min_reps": min(vals),
                    "mean_reps": round(sum(vals)/len(vals), 2),
                    "max_reps": max(vals),
                })

        if self.sampling_history:
            with open(SAMPLING_STATS_JSON, "w") as f:
                json.dump(self.sampling_history, f, indent=2)

        self._epoch_easy_counts.clear()
        self._epoch_med_counts.clear()
        self._epoch_hard_counts.clear()

        # Shuffle permutations
        self._pos_perm  = np.random.permutation(len(self.positives))
        self._easy_perm = np.random.permutation(len(self.easy_sp))
        self._med_perm  = np.random.permutation(len(self.med_sp))
        self._hard_perm = np.random.permutation(len(self.hard_sp))
        self._amb_perm  = np.random.permutation(len(self.ambient_negs))
        self._oth_perm  = np.random.permutation(len(self.other_negs)) if self.other_negs else None

        self._pos_idx = self._easy_idx = self._med_idx = 0
        self._hard_idx = self._amb_idx = self._oth_idx = 0

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

        # --- Positives (first 32 slots) ---
        pos_idxs = self._next_idx(self._pos_perm, "_pos_idx", len(self.positives), POS_PER_BATCH)
        for i, pi in enumerate(pos_idxs):
            audio = augment_positive(self.positives[pi], self.amb_paths, self.sp_paths, self.stats)
            X[i, :, :, 0] = make_spectrogram(audio)
            y[i, 0] = 1.0

        curr_n = POS_PER_BATCH

        # --- Easy Speech (14) ---
        n_e = min(N_EASY_SP, len(self.easy_sp))
        if n_e > 0:
            for i in self._next_idx(self._easy_perm, "_easy_idx", len(self.easy_sp), n_e):
                path = self.easy_sp[i]
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(path)))
                self._epoch_easy_counts[path] += 1
                self.stats["easy_speech_sampled"] += 1
                curr_n += 1

        # --- Medium Speech (7) ---
        n_m = min(N_MED_SP, len(self.med_sp))
        if n_m > 0:
            for i in self._next_idx(self._med_perm, "_med_idx", len(self.med_sp), n_m):
                path = self.med_sp[i]
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(path)))
                self._epoch_med_counts[path] += 1
                self.stats["medium_speech_sampled"] += 1
                curr_n += 1

        # --- Hard Speech (4) ---
        n_h = min(N_HARD_SP, len(self.hard_sp))
        if n_h > 0:
            for i in self._next_idx(self._hard_perm, "_hard_idx", len(self.hard_sp), n_h):
                path = self.hard_sp[i]
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(path)))
                self._epoch_hard_counts[path] += 1
                self.stats["hard_speech_sampled"] += 1
                curr_n += 1

        # --- Ambient (5) ---
        n_a = min(N_AMBIENT, len(self.ambient_negs))
        if n_a > 0:
            for i in self._next_idx(self._amb_perm, "_amb_idx", len(self.ambient_negs), n_a):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.ambient_negs[i])))
                self.stats["ambient_sampled"] += 1
                curr_n += 1

        # --- Other (2) ---
        rem = BATCH_SIZE - curr_n
        if self.other_negs and rem > 0:
            n_o = min(rem, N_OTHER)
            for i in self._next_idx(self._oth_perm, "_oth_idx", len(self.other_negs), n_o):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.other_negs[i])))
                self.stats["other_sampled"] += 1
                curr_n += 1

        # --- Fallback (fill any remainder with easy) ---
        rem = BATCH_SIZE - curr_n
        if rem > 0:
            for i in self._next_idx(self._easy_perm, "_easy_idx", len(self.easy_sp), rem):
                path = self.easy_sp[i]
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(path)))
                self._epoch_easy_counts[path] += 1
                self.stats["easy_speech_sampled"] += 1
                curr_n += 1

        return X, y


# ===========================================================================
# Callbacks
# ===========================================================================
class ValidationFPRCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_pos_X, val_sp_X, val_amb_X, model_path):
        super().__init__()
        self.val_pos_X = val_pos_X
        self.val_sp_X  = val_sp_X
        self.val_amb_X = val_amb_X
        self.best_sfpr_path = model_path
        self.records = []

        self._best_sfpr = float("inf")
        self._best_tpr  = 0.0
        self._best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        THR = 0.50
        s_pos = self.model.predict(self.val_pos_X, batch_size=64, verbose=0).flatten()
        s_sp  = self.model.predict(self.val_sp_X,  batch_size=64, verbose=0).flatten()
        s_amb = self.model.predict(self.val_amb_X,  batch_size=64, verbose=0).flatten()
        all_neg = np.concatenate([s_sp, s_amb])

        tpr  = float(np.sum(s_pos >= THR) / len(s_pos)) if len(s_pos) else 0.0
        sfpr = float(np.sum(s_sp  >= THR) / len(s_sp))  if len(s_sp)  else 0.0
        afpr = float(np.sum(s_amb >= THR) / len(s_amb)) if len(s_amb) else 0.0
        ofpr = float(np.sum(all_neg >= THR) / len(all_neg)) if len(all_neg) else 0.0
        vloss = logs.get("val_loss", float("inf")) if logs else float("inf")

        print(f"  [ValFPR] ep={epoch+1}  TPR={tpr*100:.2f}%  "
              f"spFPR={sfpr*100:.2f}%  ambFPR={afpr*100:.2f}%  ovrFPR={ofpr*100:.2f}%")

        rec = {"epoch": epoch+1, "val_tpr": tpr, "val_speech_fpr": sfpr,
               "val_ambient_fpr": afpr, "val_overall_fpr": ofpr}
        self.records.append(rec)

        if logs is not None:
            logs["val_tpr_50"]          = tpr
            logs["val_speech_fpr_50"]   = sfpr
            logs["val_ambient_fpr_50"]  = afpr
            logs["val_overall_fpr_50"]  = ofpr

        # Checkpoint rule: TPR >= 95% -> lowest sfpr -> highest tpr -> lowest val_loss
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
            self._best_tpr  = tpr
            self._best_loss = vloss
            self.model.save(str(self.best_sfpr_path))
            print(f"  [ValFPR] ** Saved best checkpoint: ep={epoch+1}  "
                  f"spFPR={sfpr*100:.2f}%  TPR={tpr*100:.2f}%")


class LrHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(lr)
        if logs is not None: logs["learning_rate"] = lr


# ===========================================================================
# Model architecture (identical to V2/V2.1/V2.2)
# ===========================================================================
def build_model():
    inp = tf.keras.Input(shape=(49, N_FREQ_BINS, 1), name="input")
    x = tf.keras.layers.Conv2D(8,  (3,3), activation="relu", padding="same")(inp)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inp, out, name="IRA_CNN_V2_3")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ===========================================================================
# Post-training: threshold frontier + comparison
# ===========================================================================
def run_post_training_eval(val_pos_X, val_sp_X, val_amb_X):
    print("\n" + "="*65)
    print("POST-TRAINING EVALUATION")
    print("="*65)

    model_paths = {
        "V2":  MODEL_OUT_DIR / "ira_cnn_v2_best_loss.keras",
        "V2.1": MODEL_OUT_DIR / "ira_cnn_v2_1_best_speech_fpr.keras",
        "V2.2": MODEL_OUT_DIR / "ira_cnn_v2_2_best_speech_fpr.keras",
        "V2.3": BEST_SFPR_PATH,
    }

    KNOWN_REF = {
        "V2":  {"tpr": 98.20, "sfpr": 11.89, "afpr": 0.00, "ofpr": 7.00},
        "V2.1": {"tpr": 85.27, "sfpr":  0.35, "afpr": 0.00, "ofpr": 0.21},
        "V2.2": {"tpr": 95.29, "sfpr":  5.59, "afpr": 0.50, "ofpr": 3.50},
    }

    print("\n--- @ threshold 0.50 comparison ---")
    print(f"{'Model':<8}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")
    print("-"*50)

    v23_scores = None
    for name, mp in model_paths.items():
        if not mp.exists():
            print(f"{name:<8}  model file not found: {mp}")
            continue
        m = tf.keras.models.load_model(str(mp))
        s_pos = m.predict(val_pos_X, batch_size=64, verbose=0).flatten()
        s_sp  = m.predict(val_sp_X,  batch_size=64, verbose=0).flatten()
        s_amb = m.predict(val_amb_X, batch_size=64, verbose=0).flatten()
        if name == "V2.3":
            v23_scores = (s_pos, s_sp, s_amb)
        thr = 0.50
        tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
        sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
        afpr = np.sum(s_amb >= thr) / len(s_amb) * 100
        neg  = np.concatenate([s_sp, s_amb])
        ofpr = np.sum(neg   >= thr) / len(neg)   * 100
        print(f"{name:<8}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")

        if name in KNOWN_REF:
            ref = KNOWN_REF[name]
            tpr_ok  = abs(tpr  - ref["tpr"])  < 0.5
            sfpr_ok = abs(sfpr - ref["sfpr"]) < 0.5
            status = "OK" if (tpr_ok and sfpr_ok) else "WARN"
            print(f"         [ref TPR={ref['tpr']}%  spFPR={ref['sfpr']}%  {status}]")

    # Threshold frontier for V2.3
    if v23_scores is not None:
        s_pos, s_sp, s_amb = v23_scores
        print("\n--- V2.3 Threshold Frontier (0.30 - 0.95) ---")
        print(f"{'THR':>4}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")
        thresholds = np.arange(0.30, 0.951, 0.01)
        neg = np.concatenate([s_sp, s_amb])

        check_A = False  # TPR>=95 AND SpFPR<=2
        check_B = False  # TPR>=95 AND SpFPR<=3
        best_near = None

        for thr in thresholds:
            tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
            sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
            afpr = np.sum(s_amb >= thr) / len(s_amb) * 100
            ofpr = np.sum(neg   >= thr) / len(neg)   * 100
            print(f"{thr:.2f}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")

            if tpr >= 95.0 and sfpr <= 2.0 and not check_A:
                print(f"  -> FEASIBILITY A (TPR>=95% & SpFPR<=2%): PASS at thr={thr:.2f}")
                check_A = True
            if tpr >= 95.0 and sfpr <= 3.0 and not check_B:
                print(f"  -> FEASIBILITY B (TPR>=95% & SpFPR<=3%): PASS at thr={thr:.2f}")
                check_B = True

            # Track best near-miss for report
            if tpr >= 95.0:
                if best_near is None or sfpr < best_near[1]:
                    best_near = (thr, sfpr, tpr)

        print("\n--- FEASIBILITY SUMMARY (V2.3) ---")
        if not check_A: print("  FEASIBILITY A (TPR>=95% & SpFPR<=2%): FAILED")
        if not check_B: print("  FEASIBILITY B (TPR>=95% & SpFPR<=3%): FAILED")
        if best_near:
            print(f"  Best when TPR>=95%: thr={best_near[0]:.2f}  SpFPR={best_near[1]:.2f}%  TPR={best_near[2]:.2f}%")

        if check_A:
            print("\n  OVERALL: PASS - V2.3 meets TPR>=95% AND SpFPR<=2%")
        elif check_B:
            print("\n  OVERALL: PARTIAL PASS - V2.3 meets TPR>=95% AND SpFPR<=3%")
        else:
            print("\n  OVERALL: FAIL - V2.3 does not meet either feasibility target at any threshold")


# ===========================================================================
# Main
# ===========================================================================
def main():
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading split manifest v3...")
    splits = load_splits()
    train_pos  = prefilter_positives(splits["train"]["pos"])
    train_sp   = splits["train"]["speech"]
    train_amb  = splits["train"]["ambient"]
    train_oth  = splits["train"]["other"]

    val_pos = splits["validation"]["pos"]
    val_sp  = splits["validation"]["speech"]
    val_amb = splits["validation"]["ambient"]
    val_oth = splits["validation"]["other"]

    test_amb = splits["test"]["ambient"]

    # Build difficulty pools from TRAIN speech negatives only
    print("\nBuilding V2.3 difficulty pools (TRAIN speech only)...")
    easy_sp, med_sp, hard_sp, spk_easy, spk_med, spk_hard = \
        build_difficulty_pools_from_manifest(train_sp)

    all_train_sp = easy_sp + med_sp + hard_sp

    # -------------------------------------------------------------------------
    # SANITY CHECK
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("PRE-TRAINING SANITY CHECK (V2.3)")
    print("="*65)

    # 1. No overlap in speaker IDs between train / val / test / unseen_test
    def get_speakers_from_manifest(target_splits):
        spks = collections.defaultdict(set)
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sp = r["split"].strip()
                if sp not in target_splits: continue
                if r["label"] != "0": continue
                grp = r.get("group","").lower()
                if "libri" not in grp and "speech" not in grp: continue
                spk = r.get("speaker_id","").strip()
                if spk: spks[sp].add(spk)
        return spks

    all_sp = get_speakers_from_manifest(["train","validation","test","unseen_test"])
    for s1 in ["train","validation","test","unseen_test"]:
        for s2 in ["train","validation","test","unseen_test"]:
            if s1 >= s2: continue
            inter = all_sp[s1].intersection(all_sp[s2])
            assert len(inter) == 0, f"LEAKAGE: {s1} and {s2} share speakers {inter}"
    print("  Speaker overlap check: PASS (0 overlap across train/val/test/unseen_test)")

    # 2. ambient pool disjoint
    train_amb_set = set(train_amb)
    val_amb_set   = set(val_amb)
    test_amb_set  = set(test_amb)
    assert train_amb_set.isdisjoint(val_amb_set),  "Overlap: train-amb vs val-amb!"
    assert train_amb_set.isdisjoint(test_amb_set), "Overlap: train-amb vs test-amb!"
    assert val_amb_set.isdisjoint(test_amb_set),   "Overlap: val-amb vs test-amb!"
    print("  Ambient pool disjoint check: PASS")

    # 3. Speech pool disjoint
    easy_set = set(easy_sp); med_set = set(med_sp); hard_set = set(hard_sp)
    assert easy_set.isdisjoint(med_set),  "easy ∩ med != empty!"
    assert easy_set.isdisjoint(hard_set), "easy ∩ hard != empty!"
    assert med_set.isdisjoint(hard_set),  "med  ∩ hard != empty!"
    print("  Speech pool (easy/med/hard) disjoint check: PASS")

    # 4. Train speech does NOT overlap validation or test speech
    val_sp_set  = set(val_sp)
    test_sp_set = set(splits["test"]["speech"])
    all_train_sp_set = set(all_train_sp)
    assert all_train_sp_set.isdisjoint(val_sp_set),  "Overlap: train sp vs val sp!"
    assert all_train_sp_set.isdisjoint(test_sp_set), "Overlap: train sp vs test sp!"
    print("  Train speech vs val/test speech disjoint check: PASS")

    print(f"\n  Train positives           : {len(train_pos)}")
    print(f"  Train easy speech (<0.10) : {len(easy_sp):5d} clips, {len(spk_easy):3d} speakers")
    print(f"  Train med  speech (<0.50) : {len(med_sp):5d} clips, {len(spk_med):3d} speakers")
    print(f"  Train hard speech (>=0.50): {len(hard_sp):5d} clips, {len(spk_hard):3d} speakers")
    print(f"  Train ambient             : {len(train_amb)}")
    print(f"  Train other               : {len(train_oth)}")

    # Build fixed deterministic validation arrays (seed=999, same as all previous experiments)
    print("\n  Building fixed deterministic validation set (seed=999)...")
    r_state  = random.getstate()
    np_state = np.random.get_state()
    random.seed(999); np.random.seed(999)

    _vstats = {k: 0 for k in ["clean","ambient","speech","easy_snr","med_snr","hard_snr","silent_bg_rejected"]}
    val_pos_X = np.array([
        np.expand_dims(make_spectrogram(augment_positive(p, val_amb, val_sp, _vstats)), -1)
        for p in val_pos
    ], dtype=np.float32)
    val_sp_X  = np.array([
        np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1)
        for p in val_sp
    ], dtype=np.float32)
    val_amb_X = np.array([
        np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1)
        for p in val_amb
    ], dtype=np.float32)
    val_oth_X = np.array([
        np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1)
        for p in val_oth
    ], dtype=np.float32) if val_oth else np.zeros((0, 49, N_FREQ_BINS, 1), dtype=np.float32)

    random.setstate(r_state); np.random.set_state(np_state)

    val_X = np.concatenate([val_pos_X, val_sp_X, val_amb_X] + ([val_oth_X] if len(val_oth_X) else []))
    val_y = np.concatenate([np.ones(len(val_pos_X)),
                            np.zeros(len(val_sp_X)+len(val_amb_X)+len(val_oth_X))]).reshape(-1,1)
    print(f"  Val pos={len(val_pos_X)}, val_sp={len(val_sp_X)}, val_amb={len(val_amb_X)}, val_oth={len(val_oth_X)}")

    # 3-batch generator dry run
    train_stats = {k: 0 for k in [
        "clean","ambient","speech","easy_snr","med_snr","hard_snr","silent_bg_rejected",
        "easy_speech_sampled","medium_speech_sampled","hard_speech_sampled",
        "ambient_sampled","other_sampled"
    ]}
    train_gen = V23DataGenerator(
        train_pos, easy_sp, med_sp, hard_sp, train_amb, train_oth,
        train_amb, all_train_sp, train_stats
    )

    print("  Running 3-batch sanity pass...")
    for i in range(3):
        Xb, yb = train_gen[i]
    assert train_stats["easy_speech_sampled"]   > 0, "No easy speech in 3 batches!"
    assert train_stats["medium_speech_sampled"]  > 0, "No medium speech in 3 batches!"
    assert train_stats["hard_speech_sampled"]    > 0, "No hard speech in 3 batches!"
    assert train_stats["ambient_sampled"]        > 0, "No ambient in 3 batches!"
    assert train_stats["clean"]                  > 0, "No clean positives in 3 batches!"
    assert (train_stats["ambient"] + train_stats["speech"]) > 0, "No mixed positives!"
    print(f"  Sanity batch counts: easy={train_stats['easy_speech_sampled']}  "
          f"med={train_stats['medium_speech_sampled']}  hard={train_stats['hard_speech_sampled']}  "
          f"amb={train_stats['ambient_sampled']}")
    print(f"  Positive augmentation: clean={train_stats['clean']}  "
          f"amb_mix={train_stats['ambient']}  sp_mix={train_stats['speech']}  "
          f"easy_snr={train_stats['easy_snr']}  med_snr={train_stats['med_snr']}  hard_snr={train_stats['hard_snr']}")

    # Reset before real training
    for k in train_stats: train_stats[k] = 0
    train_gen._epoch_easy_counts.clear()
    train_gen._epoch_med_counts.clear()
    train_gen._epoch_hard_counts.clear()
    train_gen.sampling_history.clear()
    train_gen.on_epoch_end()

    print("\nV2.3 TRAINING PIPELINE VALIDATION: PASS")

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("TRAINING IRA CNN V2.3")
    print("="*65)
    print(f"Batch: {POS_PER_BATCH} pos + 14 easy + 7 med + 4 hard + 5 amb + 2 other = {BATCH_SIZE}")

    model  = build_model()
    fpr_cb = ValidationFPRCallback(val_pos_X, val_sp_X, val_amb_X, BEST_SFPR_PATH)
    lr_cb  = LrHistoryCallback()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_LOSS_PATH), monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1),
        fpr_cb,
        lr_cb,
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=(val_X, val_y),
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(FINAL_PATH))
    print(f"\nSaved final model: {FINAL_PATH}")

    # Save augmentation stats
    with open(STATS_JSON, "w") as f:
        json.dump({"train_stats": train_stats}, f, indent=2)

    # Save training history CSV
    h = history.history
    fieldnames = (list(h.keys()) +
                  ["val_tpr_50","val_speech_fpr_50","val_ambient_fpr_50","val_overall_fpr_50","learning_rate"])
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ep in range(len(h["loss"])):
            row = {k: h[k][ep] for k in h}
            if ep < len(fpr_cb.records):
                rec = fpr_cb.records[ep]
                row["val_tpr_50"]         = rec["val_tpr"]
                row["val_speech_fpr_50"]  = rec["val_speech_fpr"]
                row["val_ambient_fpr_50"] = rec["val_ambient_fpr"]
                row["val_overall_fpr_50"] = rec["val_overall_fpr"]
            row["learning_rate"] = lr_cb.lrs[ep] if ep < len(lr_cb.lrs) else ""
            w.writerow(row)

    # Verify selected epoch
    print("\n--- Verifying checkpoint selection rule ---")
    best_ep, best_sfpr, best_tpr, best_loss = -1, float("inf"), 0.0, float("inf")
    with open(HISTORY_CSV, newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            ep   = i + 1
            tpr  = float(row["val_tpr_50"])
            sfpr = float(row["val_speech_fpr_50"])
            loss = float(row["val_loss"])
            if tpr >= 0.95:
                is_best = False
                if sfpr < best_sfpr - 1e-4: is_best = True
                elif abs(sfpr - best_sfpr) <= 1e-4:
                    if tpr > best_tpr + 1e-4: is_best = True
                    elif abs(tpr - best_tpr) <= 1e-4 and loss < best_loss: is_best = True
                if is_best:
                    best_ep, best_sfpr, best_tpr, best_loss = ep, sfpr, tpr, loss
    print(f"  Rule: TPR>=95% -> lowest spFPR -> highest TPR -> lowest val_loss")
    print(f"  Selected Epoch : {best_ep}")
    print(f"  Val TPR        : {best_tpr*100:.2f}%")
    print(f"  Val Speech FPR : {best_sfpr*100:.2f}%")
    print(f"  Val Loss       : {best_loss:.4f}")

    # -------------------------------------------------------------------------
    # POST-TRAINING EVAL
    # -------------------------------------------------------------------------
    run_post_training_eval(val_pos_X, val_sp_X, val_amb_X)
    print("\nDone. V2.3 complete. Do NOT evaluate unseen_test yet.")


if __name__ == "__main__":
    main()
