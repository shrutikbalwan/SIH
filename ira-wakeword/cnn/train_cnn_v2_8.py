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
FROZEN_VAL_NPZ = REPO_ROOT / "dataset" / "evaluation" / "expanded_val_v2_3_frozen.npz"

MODEL_OUT_DIR  = REPO_ROOT / "cnn" / "models"
BEST_LOSS_PATH = MODEL_OUT_DIR / "ira_cnn_v2_8_best_loss.keras"
BEST_SFPR_PATH = MODEL_OUT_DIR / "ira_cnn_v2_8_best_speech_fpr.keras"
FINAL_PATH     = MODEL_OUT_DIR / "ira_cnn_v2_8_final.keras"

HISTORY_CSV        = REPO_ROOT / "v2_8_training_history.csv"
STATS_JSON         = REPO_ROOT / "v2_8_augmentation_stats.json"
SAMPLING_STATS_JSON= REPO_ROOT / "v2_8_sampling_stats.json"

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
# V2.8 SOLE INTERVENTION: hard lower bound -5 dB -> -10 dB. Nothing else.
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-10, 5)}
V2_3_SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}  # control, audit only
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

    # audit-only bookkeeping; does not affect sampling
    stats.setdefault("snr_vals", []).append(snr_db)
    stats.setdefault("snr_by_tier", {}).setdefault(snr_tier, []).append(snr_db)
    stats.setdefault("bg_kind", []).append("speech" if bg_src in sp_paths else "ambient")

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
class V28DataGenerator(tf.keras.utils.PyDataset):
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
    model = tf.keras.Model(inp, out, name="IRA_CNN_V2_8")
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
def _scores(mp, pos_X, sp_X, amb_X):
    m = tf.keras.models.load_model(str(mp))
    o = (m.predict(pos_X, batch_size=64, verbose=0).flatten(),
         m.predict(sp_X, batch_size=64, verbose=0).flatten(),
         m.predict(amb_X, batch_size=64, verbose=0).flatten())
    del m
    return o


def _at(sc, thr):
    s_pos, s_sp, s_amb = sc
    neg = np.concatenate([s_sp, s_amb])
    return {"thr": float(thr),
            "TPR": float((s_pos >= thr).sum() / len(s_pos) * 100),
            "SpFPR": float((s_sp >= thr).sum() / len(s_sp) * 100),
            "AmbFPR": float((s_amb >= thr).sum() / len(s_amb) * 100),
            "OvrFPR": float((neg >= thr).sum() / len(neg) * 100)}


def _bce(sc):
    s_pos, s_sp, s_amb = sc
    y = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_sp) + len(s_amb))])
    pr = np.clip(np.concatenate([s_pos, s_sp, s_amb]), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))


def run_post_training_eval(val_pos_X, val_sp_X, val_amb_X):
    print("\n" + "=" * 72)
    print("POST-TRAINING EVALUATION -- FROZEN VALIDATION ONLY")
    print("=" * 72)
    print("  No consumed real-human recordings. No consumed unseen_test.")

    models = {
        "V2.3 best_loss": MODEL_OUT_DIR / "ira_cnn_v2_3_best_loss.keras",
        "V2.3 best_sfpr": MODEL_OUT_DIR / "ira_cnn_v2_3_best_speech_fpr.keras",
        "V2.8 best_loss": BEST_LOSS_PATH,
        "V2.8 best_sfpr": BEST_SFPR_PATH,
    }
    sc, report = {}, {}
    for name, mp in models.items():
        if mp.exists():
            sc[name] = _scores(mp, val_pos_X, val_sp_X, val_amb_X)

    print("\n  --- (4) FROZEN VALIDATION METRICS ---")
    print(f"  {'model':<16} {'BCE':>8} {'thr':>5} {'TPR':>8} {'SpFPR':>8} {'AmbFPR':>8} {'OvrFPR':>8}")
    for name in models:
        if name not in sc:
            print(f"  {name:<16} NOT FOUND")
            continue
        b = _bce(sc[name])
        for thr in (0.43, 0.50):
            m = _at(sc[name], thr)
            print(f"  {name:<16} {b:>8.4f} {thr:>5.2f} {m['TPR']:>7.2f}% {m['SpFPR']:>7.2f}% "
                  f"{m['AmbFPR']:>7.2f}% {m['OvrFPR']:>7.2f}%")
        report[name] = {"bce": b, "at_0.43": _at(sc[name], 0.43), "at_0.50": _at(sc[name], 0.50)}

    THRS = np.arange(0.30, 0.951, 0.01)

    print("\n  --- (6) V2.8 THRESHOLD FRONTIER (0.30-0.95, step 0.01) ---")
    for name in ["V2.8 best_loss", "V2.8 best_sfpr"]:
        if name not in sc:
            continue
        print(f"\n  {name}")
        print(f"  {'THR':>5} {'TPR':>8} {'SpFPR':>8} {'AmbFPR':>8} {'OvrFPR':>8}")
        rows = []
        for t in THRS:
            m = _at(sc[name], t)
            rows.append(m)
            print(f"  {t:>5.2f} {m['TPR']:>7.2f}% {m['SpFPR']:>7.2f}% {m['AmbFPR']:>7.2f}% {m['OvrFPR']:>7.2f}%")
        report.setdefault(name, {})["frontier"] = rows

    # ---- (5) matched-recall comparison ----
    print("\n" + "=" * 72)
    print("(5) MATCHED-RECALL COMPARISON vs V2.3 best_loss (TPR = 95.09%)")
    print("=" * 72)
    TARGET = 95.09
    ref = _at(sc["V2.3 best_loss"], 0.43)
    print(f"  V2.3 best_loss @0.43 : TPR={ref['TPR']:.2f}%  SpFPR={ref['SpFPR']:.2f}%  "
          f"AmbFPR={ref['AmbFPR']:.2f}%  OvrFPR={ref['OvrFPR']:.2f}%")
    matched = {}
    for name in ["V2.8 best_loss", "V2.8 best_sfpr"]:
        if name not in sc:
            continue
        cand = [_at(sc[name], t) for t in THRS]
        ok = [m for m in cand if m["TPR"] >= TARGET]
        if ok:
            best = min(ok, key=lambda m: (m["SpFPR"], -m["TPR"]))
            print(f"  {name:<16} @{best['thr']:.2f}: TPR={best['TPR']:.2f}%  "
                  f"SpFPR={best['SpFPR']:.2f}%  AmbFPR={best['AmbFPR']:.2f}%  OvrFPR={best['OvrFPR']:.2f}%")
            print(f"  {'':<16}   vs V2.3: SpFPR {best['SpFPR']-ref['SpFPR']:+.2f} pp, "
                  f"AmbFPR {best['AmbFPR']-ref['AmbFPR']:+.2f} pp")
            matched[name] = best
        else:
            mx = max(cand, key=lambda m: m["TPR"])
            print(f"  {name:<16} CANNOT REACH TPR {TARGET}% anywhere in 0.30-0.95 "
                  f"(max TPR {mx['TPR']:.2f}% @{mx['thr']:.2f})")
            matched[name] = None
    report["matched_recall"] = {"target_TPR": TARGET, "v2_3_reference": ref,
                                "v2_8": {k: v for k, v in matched.items()}}

    # ---- (7) GO / NO-GO ----
    print("\n" + "=" * 72)
    print("(7) GO / NO-GO")
    print("=" * 72)
    print("  Rule: GO only if the frozen-validation frontier is at least competitive")
    print("        with V2.3 WITHOUT materially sacrificing positive recall.")
    go = False
    for name, best in matched.items():
        if best is None:
            print(f"  {name:<16} NO-GO (cannot hold TPR >= {TARGET}%)")
            continue
        better = best["SpFPR"] <= ref["SpFPR"] + 1e-9 and best["AmbFPR"] <= 1.0
        print(f"  {name:<16} matched-recall SpFPR {best['SpFPR']:.2f}% vs V2.3 {ref['SpFPR']:.2f}%"
              f"  -> {'competitive/better' if better else 'WORSE'}")
        go = go or better
    print()
    print("  OVERALL V2.8: " + ("GO FOR NEW UNTOUCHED EVALUATION" if go else "NO-GO -- RETAIN V2.3"))
    if not go:
        print("  V2.3 remains the standing candidate.")
    print()
    print("  (8) Consumed data confirmation: the 831 real-human recordings, their 90 FNs,")
    print("      their 46 hard misses, the one-shot unseen_test speech negatives, and")
    print("      historical-test FPs were NOT used for training, checkpoint selection,")
    print("      threshold selection, or any comparison above.")
    print("  Not quantized. V2.3 artifacts not overwritten.")

    json.dump(report, open(REPO_ROOT / "v2_8_frozen_validation_report.json", "w"), indent=2, default=float)
    print(f"\n  wrote {REPO_ROOT / 'v2_8_frozen_validation_report.json'}")



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
    # -------------------------------------------------------------------
    # VALIDATION: permanently frozen artifact, READ-ONLY. Never reconstructed.
    # -------------------------------------------------------------------
    print("\n  Loading FROZEN validation artifact (read-only)...")
    if not FROZEN_VAL_NPZ.exists():
        raise SystemExit("STOP: frozen validation NPZ not found: %s" % FROZEN_VAL_NPZ)
    _vd = np.load(str(FROZEN_VAL_NPZ), allow_pickle=True)
    _vX, _vlab, _vcat = _vd["X"], _vd["labels"], _vd["negative_categories"]
    val_pos_X = _vX[(_vlab == 1.0).flatten()]
    val_sp_X  = _vX[(_vcat == "speech")]
    val_amb_X = _vX[(_vcat == "ambient")]
    val_X, val_y = _vX, _vlab
    print(f"  frozen val: pos={len(val_pos_X)} speech={len(val_sp_X)} ambient={len(val_amb_X)}")
    if (len(val_pos_X), len(val_sp_X), len(val_amb_X)) != (998, 1786, 200):
        raise SystemExit("STOP: frozen validation is not 998/1786/200.")
    print("  augment_positive() is NOT called for validation.")

    # 3-batch generator dry run
    train_stats = {k: 0 for k in [
        "clean","ambient","speech","easy_snr","med_snr","hard_snr","silent_bg_rejected",
        "easy_speech_sampled","medium_speech_sampled","hard_speech_sampled",
        "ambient_sampled","other_sampled"
    ]}
    train_stats["snr_vals"] = []
    train_stats["snr_by_tier"] = {}
    train_stats["bg_kind"] = []
    train_gen = V28DataGenerator(
        train_pos, easy_sp, med_sp, hard_sp, train_amb, train_oth,
        train_amb, all_train_sp, train_stats
    )

    DRY = 40
    print(f"  Running {DRY}-batch pre-training audit...")
    for i in range(DRY):
        train_gen[i]

    audit_fail = []

    # ---- (10) config diff vs V2.3 ----
    print("\n  --- (10) CONFIG DIFF vs V2.3 (training distribution) ---")
    diffs = []
    for k in ["easy", "medium", "hard"]:
        if SNR_RANGES[k] != V2_3_SNR_RANGES[k]:
            diffs.append((k, V2_3_SNR_RANGES[k], SNR_RANGES[k]))
        mark = "CHANGED" if SNR_RANGES[k] != V2_3_SNR_RANGES[k] else "unchanged"
        print(f"    SNR {k:<7} V2.3={str(V2_3_SNR_RANGES[k]):<10} V2.8={str(SNR_RANGES[k]):<10} {mark}")
    print(f"    SNR_WEIGHTS      {SNR_WEIGHTS}  (equal thirds, unchanged)")
    print(f"    CLEAN_PROB       {CLEAN_PROB}   (unchanged)")
    print(f"    background split 50/50 speech/ambient (unchanged)")
    print(f"    recipe constants {N_EASY_SP}/{N_MED_SP}/{N_HARD_SP}/{N_AMBIENT}/{N_OTHER} (unchanged)")
    print(f"    SEED={SEED} BATCH={BATCH_SIZE} LR={LEARNING_RATE} EPOCHS={EPOCHS} "
          f"PATIENCE={EARLY_STOP_PATIENCE} (unchanged)")
    if len(diffs) != 1 or diffs[0][0] != "hard" or diffs[0][2] != (-10, 5):
        audit_fail.append(f"config diff is not exactly the hard SNR bound: {diffs}")
    else:
        print(f"    => EXACTLY ONE CHANGE: hard SNR lower bound {diffs[0][1][0]} -> {diffs[0][2][0]} dB")

    # ---- (1,2) provenance ----
    print("\n  --- (1,2) TRAINING DATA PROVENANCE ---")
    all_train = set(train_pos) | set(all_train_sp) | set(train_amb) | set(train_oth)
    man_split = {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            man_split[str(REPO_ROOT / r["path"])] = r["split"].strip()
    by_split = collections.Counter(man_split.get(q, "NOT_IN_MANIFEST") for q in all_train)
    print(f"    training files by manifest split: {dict(by_split)}")
    if set(by_split) != {"train"}:
        audit_fail.append(f"training pool contains non-train splits: {dict(by_split)}")
    consumed = [q for q in all_train if "ira words" in q.lower() or "today ira wav" in q.lower()]
    print(f"    consumed 831 real-human recordings in training pool: {len(consumed)}")
    if consumed:
        audit_fail.append("consumed real-human diagnostic recordings present in training")
    print("    unseen_test / historical test / mined real-human FPs: NONE (never loaded)")

    # ---- (3,4,5) SNR tiers and ranges ----
    print("\n  --- (3,4,5) POSITIVE SNR TIERS AND OBSERVED RANGES ---")
    tiers = train_stats["snr_by_tier"]
    tot_mixed = sum(len(v) for v in tiers.values())
    print(f"    {'tier':<8} {'count':>6} {'share':>8} {'obs min':>10} {'obs max':>10} {'declared':>12}")
    for k in ["easy", "medium", "hard"]:
        v = tiers.get(k, [])
        lo, hi = SNR_RANGES[k]
        mn = min(v) if v else float("nan")
        mx = max(v) if v else float("nan")
        print(f"    {k:<8} {len(v):>6} {len(v)/tot_mixed*100:>7.1f}% {mn:>10.3f} {mx:>10.3f} {str(SNR_RANGES[k]):>12}")
        if v and (mn < lo - 1e-9 or mx > hi + 1e-9):
            audit_fail.append(f"{k} SNR outside declared range")
        if not (0.28 <= len(v) / tot_mixed <= 0.39):
            audit_fail.append(f"{k} tier share {len(v)/tot_mixed:.3f} not ~1/3")
    allv = train_stats["snr_vals"]
    print(f"    global observed SNR: {min(allv):.3f} .. {max(allv):.3f} dB")
    print("    convention: random.uniform(lo, hi) -> lo inclusive, hi effectively exclusive")
    if min(allv) < -10 - 1e-9 or max(allv) > 30 + 1e-9:
        audit_fail.append("SNR outside [-10, 30]")
    below = sum(1 for x in allv if x < -5)
    print(f"    NEW territory (SNR < -5 dB, impossible under V2.3): {below} "
          f"({below/len(allv)*100:.1f}% of mixed positives)")

    # ---- (6,7) clean/mixed and background type ----
    print("\n  --- (6,7) CLEAN/MIXED AND BACKGROUND TYPE ---")
    pos_tot = train_stats["clean"] + train_stats["ambient"] + train_stats["speech"]
    mixed = train_stats["ambient"] + train_stats["speech"]
    print(f"    positives drawn : {pos_tot} (expected {POS_PER_BATCH*DRY})")
    print(f"    clean           : {train_stats['clean']} ({train_stats['clean']/pos_tot*100:.1f}%, target 15%)")
    print(f"    mixed           : {mixed} ({mixed/pos_tot*100:.1f}%, target 85%)")
    print(f"    background      : {train_stats['ambient']/mixed*100:.1f}% ambient / "
          f"{train_stats['speech']/mixed*100:.1f}% speech (target 50/50)")
    if pos_tot != POS_PER_BATCH * DRY:
        audit_fail.append("positives per batch != 32")
    if not (0.10 <= train_stats["clean"] / pos_tot <= 0.20):
        audit_fail.append("clean fraction not ~15%")
    if not (0.40 <= train_stats["speech"] / mixed <= 0.60):
        audit_fail.append("background split not ~50/50")

    # ---- (8) effective negative recipe ----
    print("\n  --- (8) EFFECTIVE NEGATIVE RECIPE (per 32-negative half) ---")
    other_avail = len(train_oth) > 0
    eff = {"easy": N_EASY_SP + (0 if other_avail else N_OTHER), "medium": N_MED_SP,
           "hard": N_HARD_SP, "ambient": N_AMBIENT, "other": N_OTHER if other_avail else 0}
    obs = {"easy": train_stats["easy_speech_sampled"], "medium": train_stats["medium_speech_sampled"],
           "hard": train_stats["hard_speech_sampled"], "ambient": train_stats["ambient_sampled"],
           "other": train_stats["other_sampled"]}
    print(f"    OTHER pool empty: {not other_avail} -> its {N_OTHER} slots fall back to easy speech")
    print(f"    {'pool':<9} {'expected':>9} {'observed/batch':>15}")
    for k in ["easy", "medium", "hard", "ambient", "other"]:
        print(f"    {k:<9} {eff[k]:>9} {obs[k]/DRY:>15.2f}")
        if obs[k] != eff[k] * DRY:
            audit_fail.append(f"negative recipe {k}: {obs[k]/DRY:.2f}/batch != {eff[k]}")
    if (eff["easy"], eff["medium"], eff["hard"], eff["ambient"], eff["other"]) != (16, 7, 4, 5, 0):
        audit_fail.append(f"effective recipe is not 16/7/4/5/0: {eff}")
    else:
        print("    => effective V2.3 recipe 16/7/4/5/0 CONFIRMED (not V2.7 15/7/5/5/0)")

    # ---- (9) placement ----
    print("\n  --- (9) POSITIVE TEMPORAL PLACEMENT ---")
    print("    unchanged from V2.3: vad_trim -> random offset in [0, 16000-len(active)]")

    # ---- verdict ----
    print("\n  --- AUDIT VERDICT ---")
    if audit_fail:
        for fmsg in audit_fail:
            print(f"    FAIL: {fmsg}")
        raise SystemExit("STOP: pre-training audit FAILED. V2.8 not trained.")
    print("    ALL CHECKS PASS")

    for k in list(train_stats):
        if isinstance(train_stats[k], list):
            train_stats[k] = []
        elif isinstance(train_stats[k], dict):
            train_stats[k] = {}
        else:
            train_stats[k] = 0
    train_gen._epoch_easy_counts.clear()
    train_gen._epoch_med_counts.clear()
    train_gen._epoch_hard_counts.clear()
    train_gen.sampling_history.clear()
    train_gen.on_epoch_end()
    print("    all sanity statistics reset.")

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
    print("\nDone. V2.8 complete. Not quantized. No consumed datasets touched.")


if __name__ == "__main__":
    main()
