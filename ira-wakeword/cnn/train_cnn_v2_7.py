# -*- coding: utf-8 -*-
"""
train_cnn_v2_7.py
=================
IRA CNN V2.7 -- Minimal hard-speech negative pressure experiment.

ONLY change from V2.3:
  Negative half re-balanced by ONE slot:
      V2.3:  14 easy / 7 medium / 4 hard / 5 ambient / 2 other
      V2.7:  13 easy / 7 medium / 5 hard / 5 ambient / 2 other

Everything else is IDENTICAL to V2.3:
  - split_manifest_v3.csv
  - Original V2.3 positive augmentation (15% clean, 85% mixed;
    within mixed 50% ambient / 50% unrelated speech; original SNR tiers)
  - Same difficulty pools (V2 baseline scores, TRAIN-only)
  - Same architecture, preprocessing, Adam, constant LR = 0.001, BCE,
    no class weights, batch 64, seed 42, max 30 epochs, EarlyStopping(5)

Explicitly NOT used:
  - V2.5's 75/25 positive-background ratio
  - V2.6's +10..+25 dB speech-only SNR restriction
  - ReduceLROnPlateau
  - unseen_test

Validation is READ-ONLY: it EXCLUSIVELY loads the permanently frozen
expanded_val_v2_3_frozen.npz. augment_positive() is never called for
validation.
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
FROZEN_VAL_JSON= REPO_ROOT / "dataset" / "evaluation" / "expanded_val_v2_3_frozen.json"

MODEL_OUT_DIR  = REPO_ROOT / "cnn" / "models"
BEST_LOSS_PATH = MODEL_OUT_DIR / "ira_cnn_v2_7_best_loss.keras"
BEST_SFPR_PATH = MODEL_OUT_DIR / "ira_cnn_v2_7_best_speech_fpr.keras"
FINAL_PATH     = MODEL_OUT_DIR / "ira_cnn_v2_7_final.keras"

HISTORY_CSV        = REPO_ROOT / "v2_7_training_history.csv"
STATS_JSON         = REPO_ROOT / "v2_7_augmentation_stats.json"
SAMPLING_STATS_JSON= REPO_ROOT / "v2_7_sampling_stats.json"

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
# Augmentation constants
# ---------------------------------------------------------------------------
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
SNR_WEIGHTS = {"easy": 1, "medium": 1, "hard": 1}
CLEAN_PROB = 0.15

# V2.7 batch recipe -- THE ONLY INTERVENTION vs V2.3
# V2.3 was: 14 easy / 7 med / 4 hard / 5 ambient / 2 other
N_EASY_SP   = 13
N_MED_SP    =  7
N_HARD_SP   =  5
N_AMBIENT   =  5
N_OTHER     =  2
assert N_EASY_SP + N_MED_SP + N_HARD_SP + N_AMBIENT + N_OTHER == 32,     "V2.7 negative recipe must total 32"

# Nominal recipes (as specified). NOTE: split_manifest_v3.csv contains NO
# train negatives outside speech/ambient, so the "other" pool is empty and the
# generator's tail fallback reassigns those slots to easy speech. This has been
# true since V2.3, so the effective recipes below are what actually train.
# The V2.7 intervention (one slot easy -> hard) is preserved either way.
V2_3_RECIPE = {"easy": 14, "medium": 7, "hard": 4, "ambient": 5, "other": 2}
V2_7_RECIPE = {"easy": N_EASY_SP, "medium": N_MED_SP, "hard": N_HARD_SP,
               "ambient": N_AMBIENT, "other": N_OTHER}

def effective_recipe(nominal: dict, other_pool_available: bool) -> dict:
    """Fold the empty-'other' fallback (-> easy speech) into the recipe."""
    eff = dict(nominal)
    if not other_pool_available:
        eff["easy"] += eff["other"]
        eff["other"] = 0
    return eff

# Frozen-validation reference for V2.3 best_loss (must hold before training)
V2_3_REF_THR  = 0.43
V2_3_REF_TPR  = 95.09
V2_3_REF_SFPR = 3.25
V2_3_REF_AFPR = 0.00
REF_TOL       = 0.30   # percentage points

# Expected difficulty pool sizes (TRAIN-only V2 baseline scores)
EXPECTED_POOLS = {"easy": 6616, "medium": 3294, "hard": 2232}

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
    max_shift = max(0, WINDOW_SAMPLES - len(active))
    offset = random.randint(0, max_shift)
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    end_idx = min(offset + len(active), WINDOW_SAMPLES)
    active_len = end_idx - offset
    padded[offset:end_idx] = active[:active_len]

    if random.random() < CLEAN_PROB or (not amb_paths and not sp_paths):
        stats["clean"] += 1
        return padded

    # V2.7: EXACT V2.3 background composition -- 50% speech / 50% ambient
    if sp_paths and (random.random() < 0.5 or not amb_paths):
        bg_seg, bg_src, _ = get_random_bg_segment(sp_paths, stats)
        stats["speech"] += 1
        is_speech_bg = True
    else:
        bg_seg, bg_src, _ = get_random_bg_segment(amb_paths, stats)
        stats["ambient"] += 1
        is_speech_bg = False

    # V2.7: EXACT V2.3 SNR distribution -- same tiers for BOTH background types
    snr_tier = random.choices(list(SNR_RANGES.keys()), weights=list(SNR_WEIGHTS.values()))[0]
    lo, hi = SNR_RANGES[snr_tier]
    snr_db = random.uniform(lo, hi)

    if snr_tier == "easy":     stats["easy_snr"] += 1
    elif snr_tier == "medium": stats["med_snr"] += 1
    else:                      stats["hard_snr"] += 1

    # Audit-only bookkeeping (does not affect sampling)
    if is_speech_bg: stats["speech_snr_vals"].append(snr_db)
    else:            stats["ambient_snr_vals"].append(snr_db)

    mixed = mix_snr(padded, bg_seg, snr_db)
    return mixed


# ===========================================================================
# Split loading from manifest_v3
# ===========================================================================
def load_splits():
    splits = {
        "train":      {"pos": [], "speech": [], "ambient": [], "other": []}
    }
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split = r["split"].strip()
            if split != "train":
                continue # We only load train here, validation is loaded from frozen NPZ!
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
    del v2_model, X

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
class V27DataGenerator(tf.keras.utils.PyDataset):
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

        self._epoch_easy_counts = collections.defaultdict(int)
        self._epoch_med_counts  = collections.defaultdict(int)
        self._epoch_hard_counts = collections.defaultdict(int)
        self.sampling_history   = []

        self.batch_count = len(self.positives) // POS_PER_BATCH
        self.on_epoch_end()

    def __len__(self):
        return self.batch_count

    def on_epoch_end(self):
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

        # Positives
        pos_idxs = self._next_idx(self._pos_perm, "_pos_idx", len(self.positives), POS_PER_BATCH)
        for i, pi in enumerate(pos_idxs):
            audio = augment_positive(self.positives[pi], self.amb_paths, self.sp_paths, self.stats)
            X[i, :, :, 0] = make_spectrogram(audio)
            y[i, 0] = 1.0

        curr_n = POS_PER_BATCH

        # Easy Speech
        n_e = min(N_EASY_SP, len(self.easy_sp))
        if n_e > 0:
            for i in self._next_idx(self._easy_perm, "_easy_idx", len(self.easy_sp), n_e):
                path = self.easy_sp[i]
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(path)))
                self._epoch_easy_counts[path] += 1
                self.stats["easy_speech_sampled"] += 1
                curr_n += 1

        # Medium Speech
        n_m = min(N_MED_SP, len(self.med_sp))
        if n_m > 0:
            for i in self._next_idx(self._med_perm, "_med_idx", len(self.med_sp), n_m):
                path = self.med_sp[i]
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(path)))
                self._epoch_med_counts[path] += 1
                self.stats["medium_speech_sampled"] += 1
                curr_n += 1

        # Hard Speech
        n_h = min(N_HARD_SP, len(self.hard_sp))
        if n_h > 0:
            for i in self._next_idx(self._hard_perm, "_hard_idx", len(self.hard_sp), n_h):
                path = self.hard_sp[i]
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(path)))
                self._epoch_hard_counts[path] += 1
                self.stats["hard_speech_sampled"] += 1
                curr_n += 1

        # Ambient
        n_a = min(N_AMBIENT, len(self.ambient_negs))
        if n_a > 0:
            for i in self._next_idx(self._amb_perm, "_amb_idx", len(self.ambient_negs), n_a):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.ambient_negs[i])))
                self.stats["ambient_sampled"] += 1
                curr_n += 1

        # Other
        rem = BATCH_SIZE - curr_n
        if self.other_negs and rem > 0:
            n_o = min(rem, N_OTHER)
            for i in self._next_idx(self._oth_perm, "_oth_idx", len(self.other_negs), n_o):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.other_negs[i])))
                self.stats["other_sampled"] += 1
                curr_n += 1

        # Fallback (fill any remainder with easy)
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

        rec = {"epoch": epoch+1, "val_loss": vloss, "val_tpr": tpr, "val_speech_fpr": sfpr,
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
# Model architecture
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
    model = tf.keras.Model(inp, out, name="IRA_CNN_V2_6")
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
def threshold_sweep(model, pos_X, sp_X, amb_X, label):
    s_pos = model.predict(pos_X, batch_size=64, verbose=0).flatten()
    s_sp  = model.predict(sp_X,  batch_size=64, verbose=0).flatten()
    s_amb = model.predict(amb_X, batch_size=64, verbose=0).flatten()
    s_neg = np.concatenate([s_sp, s_amb])
    thresholds = np.arange(0.30, 0.951, 0.01)

    print(f"\n--- Threshold Frontier: {label} (frozen EXPANDED_VAL) ---")
    print(f"{'THR':>4}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}   flags")

    feasible_primary, feasible_strict = [], []
    best_at_tpr95 = None

    for thr in thresholds:
        tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
        sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
        afpr = np.sum(s_amb >= thr) / len(s_amb) * 100
        ofpr = np.sum(s_neg >= thr) / len(s_neg) * 100

        row = (float(thr), tpr, sfpr, afpr, ofpr)
        flags = ""
        if tpr >= 95.0 and afpr <= 1.0 and sfpr <= 3.0:
            feasible_primary.append(row); flags += " PRIMARY"
        if tpr >= 95.0 and afpr <= 1.0 and sfpr <= 2.0:
            feasible_strict.append(row); flags += " STRICT"
        if tpr >= 95.0 and (best_at_tpr95 is None or sfpr < best_at_tpr95[2]):
            best_at_tpr95 = row

        print(f"{thr:.2f}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%  {flags}")

    print(f"\n  [{label}] FEASIBILITY")
    if feasible_primary:
        print(f"  PRIMARY (TPR>=95 & SpFPR<=3 & AmbFPR<=1): PASS -- {len(feasible_primary)} feasible threshold(s)")
        for t, tpr, sfpr, afpr, ofpr in feasible_primary:
            print(f"      thr={t:.2f}  TPR={tpr:.2f}%  SpFPR={sfpr:.2f}%  AmbFPR={afpr:.2f}%  OvrFPR={ofpr:.2f}%")
    else:
        print("  PRIMARY (TPR>=95 & SpFPR<=3 & AmbFPR<=1): FAILED at every threshold")

    if feasible_strict:
        print(f"  STRICT  (TPR>=95 & SpFPR<=2 & AmbFPR<=1): PASS -- {len(feasible_strict)} feasible threshold(s)")
        for t, tpr, sfpr, afpr, ofpr in feasible_strict:
            print(f"      thr={t:.2f}  TPR={tpr:.2f}%  SpFPR={sfpr:.2f}%  AmbFPR={afpr:.2f}%  OvrFPR={ofpr:.2f}%")
    else:
        print("  STRICT  (TPR>=95 & SpFPR<=2 & AmbFPR<=1): FAILED at every threshold")

    if best_at_tpr95:
        t, tpr, sfpr, afpr, ofpr = best_at_tpr95
        print(f"  LOWEST Speech FPR achievable while TPR>=95%: {sfpr:.2f}%  "
              f"(thr={t:.2f}  TPR={tpr:.2f}%  AmbFPR={afpr:.2f}%  OvrFPR={ofpr:.2f}%)")
    else:
        print("  Model never reaches TPR>=95% in the swept range.")

    return (feasible_primary[0] if feasible_primary else None)



# ===========================================================================
# Main
# ===========================================================================
def load_frozen_val():
    """Validation is READ-ONLY. augment_positive() is NEVER called here."""
    print("\n" + "="*65)
    print("STEP 1 -- LOADING FROZEN EXPANDED_VAL (READ-ONLY)")
    print("="*65)
    if not FROZEN_VAL_NPZ.exists():
        raise SystemExit(f"STOP: frozen validation artifact not found: {FROZEN_VAL_NPZ}")

    d = np.load(str(FROZEN_VAL_NPZ), allow_pickle=True)
    X   = d["X"]
    lab = d["labels"]
    cat = d["negative_categories"]

    pos_X = X[(lab == 1.0).flatten()]
    sp_X  = X[(cat == "speech").flatten()]
    amb_X = X[(cat == "ambient").flatten()]

    print(f"  Artifact : {FROZEN_VAL_NPZ}")
    print(f"  positives={len(pos_X)}  speech_neg={len(sp_X)}  ambient_neg={len(amb_X)}")
    if FROZEN_VAL_JSON.exists():
        meta = json.load(open(FROZEN_VAL_JSON))
        print(f"  npz_sha256 (recorded) = {meta.get('npz_sha256')}")
    if (len(pos_X), len(sp_X), len(amb_X)) != (998, 1786, 200):
        raise SystemExit("STOP: frozen EXPANDED_VAL is not 998/1786/200.")
    return X, lab, pos_X, sp_X, amb_X


def assert_v2_3_reference(pos_X, sp_X, amb_X):
    """Gate: V2.3 best_loss must reproduce its frozen-validation reference."""
    print("\n" + "="*65)
    print("STEP 2 -- V2.3 BASELINE REFERENCE ASSERTION (frozen EXPANDED_VAL)")
    print("="*65)
    mp = MODEL_OUT_DIR / "ira_cnn_v2_3_best_loss.keras"
    if not mp.exists():
        raise SystemExit(f"STOP: V2.3 best_loss checkpoint not found: {mp}")

    m = tf.keras.models.load_model(str(mp))
    s_pos = m.predict(pos_X, batch_size=64, verbose=0).flatten()
    s_sp  = m.predict(sp_X,  batch_size=64, verbose=0).flatten()
    s_amb = m.predict(amb_X, batch_size=64, verbose=0).flatten()
    del m

    thr  = V2_3_REF_THR
    tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
    sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
    afpr = np.sum(s_amb >= thr) / len(s_amb) * 100

    print(f"  V2.3 best_loss @ thr={thr:.2f}")
    print(f"    TPR         = {tpr:.2f}%   (expected ~{V2_3_REF_TPR:.2f}%)")
    print(f"    Speech FPR  = {sfpr:.2f}%   (expected ~{V2_3_REF_SFPR:.2f}%)")
    print(f"    Ambient FPR = {afpr:.2f}%   (expected ~{V2_3_REF_AFPR:.2f}%)")

    if not (abs(tpr - V2_3_REF_TPR) <= REF_TOL and
            abs(sfpr - V2_3_REF_SFPR) <= REF_TOL and
            abs(afpr - V2_3_REF_AFPR) <= REF_TOL):
        raise SystemExit("STOP: V2.3 reference assertion FAILED on frozen EXPANDED_VAL. "
                         "V2.7 will not be trained.")
    print("  V2.3 REFERENCE ASSERTION: PASS")


def main():
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # STEP 1+2: read-only validation load, then baseline gate BEFORE any training
    val_X, val_y, val_pos_X, val_sp_X, val_amb_X = load_frozen_val()
    assert_v2_3_reference(val_pos_X, val_sp_X, val_amb_X)

    print("\nLoading split manifest v3 for TRAIN only...")
    splits = load_splits()
    train_pos  = prefilter_positives(splits["train"]["pos"])
    train_sp   = splits["train"]["speech"]
    train_amb  = splits["train"]["ambient"]
    train_oth  = splits["train"]["other"]

    print("\nBuilding V2.7 difficulty pools (TRAIN speech only)...")
    easy_sp, med_sp, hard_sp, spk_easy, spk_med, spk_hard = \
        build_difficulty_pools_from_manifest(train_sp)

    all_train_sp = easy_sp + med_sp + hard_sp

    print("  Pool sizes vs expected (V2 baseline scoring, TRAIN-only):")
    for k, got in [("easy", len(easy_sp)), ("medium", len(med_sp)), ("hard", len(hard_sp))]:
        exp = EXPECTED_POOLS[k]
        print(f"    {k:<7} got={got:<6} expected~{exp:<6} delta={got-exp:+d}")

    # -------------------------------------------------------------------------
    # SANITY CHECK
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("PRE-TRAINING SANITY CHECK (V2.7)")
    print("="*65)

    train_stats = {k: 0 for k in [
        "clean","ambient","speech","easy_snr","med_snr","hard_snr","silent_bg_rejected",
        "easy_speech_sampled","medium_speech_sampled","hard_speech_sampled",
        "ambient_sampled","other_sampled"
    ]}
    train_stats["ambient_snr_vals"] = []
    train_stats["speech_snr_vals"] = []
    
    # Run a quick 20-batch dry run
    gen = V27DataGenerator(train_pos, easy_sp, med_sp, hard_sp, train_amb, train_oth,
                           train_amb, all_train_sp, train_stats)

    DRY_BATCHES = 20
    print(f"  Running {DRY_BATCHES}-batch dry run...")
    for i in range(DRY_BATCHES): gen[i]

    # ---- 6a. Negative-half composition (THE V2.7 INTERVENTION) ----
    observed = {
        "easy":    train_stats["easy_speech_sampled"],
        "medium":  train_stats["medium_speech_sampled"],
        "hard":    train_stats["hard_speech_sampled"],
        "ambient": train_stats["ambient_sampled"],
        "other":   train_stats["other_sampled"],
    }
    other_avail = len(train_oth) > 0
    eff_v2_3 = effective_recipe(V2_3_RECIPE, other_avail)
    eff_v2_7 = effective_recipe(V2_7_RECIPE, other_avail)

    print("\n  --- Negative half composition (per batch) ---")
    if not other_avail:
        print("  NOTE: train 'other' pool is EMPTY in split_manifest_v3.csv; those slots")
        print("        fall back to easy speech. This was equally true for V2.3, so the")
        print("        one-slot easy->hard intervention is unaffected.")
    print(f"  {'pool':<9} {'V2.3 nom':>9} {'V2.7 nom':>9} {'V2.3 eff':>9} {'V2.7 eff':>9} {'observed':>9} {'total':>7}")
    for k in ["easy","medium","hard","ambient","other"]:
        per_batch = observed[k] / DRY_BATCHES
        print(f"  {k:<9} {V2_3_RECIPE[k]:>9} {V2_7_RECIPE[k]:>9} {eff_v2_3[k]:>9} "
              f"{eff_v2_7[k]:>9} {per_batch:>9.2f} {observed[k]:>7}")
        assert observed[k] == eff_v2_7[k] * DRY_BATCHES, (
            f"Negative recipe violation: {k} = {per_batch:.2f}/batch, expected {eff_v2_7[k]}")
    assert sum(observed.values()) == 32 * DRY_BATCHES, "Negative half is not 32 per batch"

    # The intervention itself: exactly one slot moved easy -> hard vs V2.3
    assert eff_v2_7["hard"]   == eff_v2_3["hard"] + 1,   "hard slot did not increase by exactly 1"
    assert eff_v2_7["easy"]   == eff_v2_3["easy"] - 1,   "easy slot did not decrease by exactly 1"
    for k in ["medium","ambient","other"]:
        assert eff_v2_7[k] == eff_v2_3[k], f"{k} slot changed -- not a one-slot intervention"
    print(f"  Effective V2.7 negative half: "
          f"{eff_v2_7['easy']} easy / {eff_v2_7['medium']} medium / {eff_v2_7['hard']} hard / "
          f"{eff_v2_7['ambient']} ambient / {eff_v2_7['other']} other = 32 : VERIFIED EXACTLY")
    print(f"  vs V2.3 effective ({eff_v2_3['easy']}/{eff_v2_3['medium']}/{eff_v2_3['hard']}/"
          f"{eff_v2_3['ambient']}/{eff_v2_3['other']}): exactly ONE slot moved easy -> hard")

    # ---- 6b. Unique clips / speakers actually drawn ----
    def uniq_spk(d):
        return len({os.path.basename(x).split("_")[0] for x in d
                    if os.path.basename(x).split("_")[0].isdigit()})
    for name, d in [("easy", gen._epoch_easy_counts),
                    ("medium", gen._epoch_med_counts),
                    ("hard", gen._epoch_hard_counts)]:
        print(f"  {name:<7} draws={sum(d.values()):>5}  unique_clips={len(d):>5}  unique_speakers={uniq_spk(d):>3}")

    # ---- 6c. V2.3 positive augmentation restored ----
    print("\n  --- Positive augmentation (must match V2.3) ---")
    pos_total = train_stats["clean"] + train_stats["ambient"] + train_stats["speech"]
    mixed_total = train_stats["ambient"] + train_stats["speech"]
    clean_pct = train_stats["clean"] / pos_total * 100
    amb_pct = train_stats["ambient"] / mixed_total * 100
    sp_pct  = train_stats["speech"]  / mixed_total * 100
    print(f"  positives drawn     : {pos_total} (expected {POS_PER_BATCH*DRY_BATCHES})")
    print(f"  clean               : {train_stats['clean']} ({clean_pct:.1f}%, target 15%)")
    print(f"  mixed               : {mixed_total} ({100-clean_pct:.1f}%, target 85%)")
    print(f"  mixed composition   : {amb_pct:.1f}% ambient / {sp_pct:.1f}% speech (target 50/50)")
    print(f"  SNR tiers           : easy={train_stats['easy_snr']} med={train_stats['med_snr']} hard={train_stats['hard_snr']}")
    all_snr = train_stats["ambient_snr_vals"] + train_stats["speech_snr_vals"]
    sp_snr  = train_stats["speech_snr_vals"]
    print(f"  SNR range (all)     : min {min(all_snr):.2f} dB / max {max(all_snr):.2f} dB")
    print(f"  SNR range (speech bg): min {min(sp_snr):.2f} dB / max {max(sp_snr):.2f} dB")

    assert pos_total == POS_PER_BATCH * DRY_BATCHES, "Positive count per batch is not 32"
    assert train_stats["clean"] > 0, "No clean positives"
    assert train_stats["ambient"] > 0, "No ambient-mixed positives"
    assert train_stats["speech"] > 0, "No speech-mixed positives"
    assert 40 <= sp_pct <= 60, "Mixed background composition is not approximately 50/50 (V2.3)"
    # V2.3 SNR distribution spans the full -5..30 dB tier set for BOTH bg types.
    # V2.6's +10..+25 speech-only restriction must NOT be in effect.
    assert min(sp_snr) < 10.0, "Speech-bg SNR floor looks restricted -- V2.6 behaviour leaked in"
    assert max(all_snr) > 25.0, "SNR ceiling looks restricted -- V2.6 behaviour leaked in"
    print("  V2.3 positive augmentation (15/85, 50/50, original SNR tiers): RESTORED")

    # ---- Reset all counters before real training ----
    for k in train_stats:
        train_stats[k] = [] if isinstance(train_stats[k], list) else 0
    gen._epoch_easy_counts.clear()
    gen._epoch_med_counts.clear()
    gen._epoch_hard_counts.clear()
    gen.sampling_history.clear()
    gen.on_epoch_end()
    assert all((v == [] if isinstance(v, list) else v == 0) for v in train_stats.values())
    print("  All counters reset.")

    print("\nV2.7 TRAINING PIPELINE VALIDATION: PASS")

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("TRAINING IRA CNN V2.7")
    print("="*65)

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
        gen,
        epochs=EPOCHS,
        validation_data=(val_X, val_y),
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(FINAL_PATH))
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

    # -------------------------------------------------------------------------
    # POST-TRAINING EVAL & SWEEPS
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("POST-TRAINING THRESHOLD SWEEPS (FROZEN EXPANDED_VAL)")
    print("="*65)
    
    go_target_sfpr = None
    if BEST_SFPR_PATH.exists():
        m_sfpr = tf.keras.models.load_model(str(BEST_SFPR_PATH))
        go_target_sfpr = threshold_sweep(m_sfpr, val_pos_X, val_sp_X, val_amb_X, "V2.7 best_speech_fpr")
    else:
        print("  best_speech_fpr model not found (no eligible epoch).")

    go_target_bl = None
    if BEST_LOSS_PATH.exists():
        m_bl = tf.keras.models.load_model(str(BEST_LOSS_PATH))
        go_target_bl = threshold_sweep(m_bl, val_pos_X, val_sp_X, val_amb_X, "V2.7 best_loss")
    else:
        print("  best_loss model not found.")

    # -------------------------------------------------------------------------
    # COMPARE V2.3 vs V2.7
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("V2.3 vs V2.7 COMPARISON (FROZEN EXPANDED_VAL)")
    print("="*65)
    
    compare_models = {
        "V2.3 best_loss": MODEL_OUT_DIR / "ira_cnn_v2_3_best_loss.keras",
        "V2.3 best_sfpr": MODEL_OUT_DIR / "ira_cnn_v2_3_best_speech_fpr.keras",
        "V2.7 best_loss": BEST_LOSS_PATH,
        "V2.7 best_sfpr": BEST_SFPR_PATH,
    }
    
    for name, mp in compare_models.items():
        print("\n" + "-"*65)
        if not mp.exists():
            print(f"{name}: NOT FOUND ({mp})")
            continue
        m = tf.keras.models.load_model(str(mp))
        s_pos = m.predict(val_pos_X, batch_size=64, verbose=0).flatten()
        s_sp  = m.predict(val_sp_X,  batch_size=64, verbose=0).flatten()
        s_amb = m.predict(val_amb_X, batch_size=64, verbose=0).flatten()
        del m
        s_neg = np.concatenate([s_sp, s_amb])

        # optimal threshold = lowest Speech FPR subject to TPR >= 95%
        best_thr, best_sfpr = None, float("inf")
        for thr in np.arange(0.30, 0.951, 0.01):
            tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
            sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
            if tpr >= 95.0 and sfpr < best_sfpr:
                best_sfpr, best_thr = sfpr, float(thr)

        if best_thr is None:
            print(f"{name}: never attains TPR >= 95% in 0.30-0.95")
            continue

        thr = best_thr
        tp = int(np.sum(s_pos >= thr)); fn = len(s_pos) - tp
        sp_fp = int(np.sum(s_sp >= thr)); sp_tn = len(s_sp) - sp_fp
        am_fp = int(np.sum(s_amb >= thr)); am_tn = len(s_amb) - am_fp
        n_fp  = int(np.sum(s_neg >= thr))

        vloss = float(tf.keras.losses.binary_crossentropy(
            tf.constant(np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))]), dtype=tf.float32),
            tf.constant(np.concatenate([s_pos, s_neg]), dtype=tf.float32)).numpy().mean())

        print(f"{name}")
        print(f"  threshold      : {thr:.2f}")
        print(f"  TP             : {tp}")
        print(f"  FN             : {fn}")
        print(f"  TPR            : {tp/len(s_pos)*100:.2f}%")
        print(f"  speech FP      : {sp_fp}")
        print(f"  speech TN      : {sp_tn}")
        print(f"  Speech FPR     : {sp_fp/len(s_sp)*100:.2f}%")
        print(f"  ambient FP     : {am_fp}")
        print(f"  ambient TN     : {am_tn}")
        print(f"  Ambient FPR    : {am_fp/len(s_amb)*100:.2f}%")
        print(f"  Overall FPR    : {n_fp/len(s_neg)*100:.2f}%")
        print(f"  BCE on frozen val (all thresholds): {vloss:.4f}")


    print("\n" + "="*65)
    print("GO / NO-GO")
    print("="*65)
    print("  Criterion: TPR>=95% & Speech FPR<=3% & Ambient FPR<=1% on frozen EXPANDED_VAL")
    hits = [("V2.7 best_loss", go_target_bl), ("V2.7 best_speech_fpr", go_target_sfpr)]
    any_go = False
    for nm, r in hits:
        if r is None:
            print(f"  {nm:<22}: no feasible threshold")
        else:
            t, tpr, sfpr, afpr, ofpr = r
            any_go = True
            print(f"  {nm:<22}: FEASIBLE  thr={t:.2f}  TPR={tpr:.2f}%  "
                  f"SpFPR={sfpr:.2f}%  AmbFPR={afpr:.2f}%  OvrFPR={ofpr:.2f}%")
    print()
    print("  OVERALL V2.7: " + ("GO" if any_go else "NO-GO"))
    print("  STOP. unseen_test NOT evaluated. No quantization, streaming or deployment.")

    print("\nDone. V2.7 complete.")

if __name__ == "__main__":
    main()
