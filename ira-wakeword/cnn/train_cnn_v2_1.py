# -*- coding: utf-8 -*-
"""
train_cnn_v2_1.py
=================
IRA CNN V2.1 -- Controlled experiment: stronger speech-negative discrimination.

KEY CHANGES vs V2:
  - Negative batch composition: 60% ordinary speech, 20% ambient, 5% other, 15% hard-speech
  - Hard-speech pool: train-split clips with V2 score >= 0.30 (mined by mine_v2_train_speech_hard_negatives.py)
  - Per-epoch validation speech FPR / ambient FPR / overall FPR reported
  - Extra diagnostic checkpoint: among epochs with val_recall >= 0.95, pick lowest val_speech_FPR
  - V1 streaming hard negatives EXCLUDED (source speakers overlap with test set)
  - Train from RANDOM init (no V2 weights)
  - All else identical to V2

IMPORTANT NOTES:
  - V2 model files are NOT overwritten.
  - The 772 historical test negatives are labelled DEVELOPMENT/DIAGNOSTIC.
    They are used post-hoc for analysis only, not for checkpoint selection.
  - No quantization in this script.
"""

import os, csv, json, random, math, time
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
HARD_SPEECH_CSV = REPO_ROOT / "v2_train_speech_negative_scores.csv"
HARD_SPEECH_THRESHOLD = 0.30   # clips with V2 score >= this go into hard-speech pool

MODEL_OUT_DIR  = REPO_ROOT / "cnn" / "models"
BEST_LOSS_PATH = MODEL_OUT_DIR / "ira_cnn_v2_1_best_loss.keras"
BEST_SFPR_PATH = MODEL_OUT_DIR / "ira_cnn_v2_1_best_speech_fpr.keras"
FINAL_PATH     = MODEL_OUT_DIR / "ira_cnn_v2_1_final.keras"

HISTORY_CSV    = REPO_ROOT / "v2_1_training_history.csv"
STATS_JSON     = REPO_ROOT / "v2_1_augmentation_stats.json"

BG_DIR         = REPO_ROOT / "dataset" / "negative" / "background"

# ---------------------------------------------------------------------------
# Audio / feature constants -- identical to V1 and V2
# ---------------------------------------------------------------------------
SAMPLE_RATE    = 16000
WINDOW_SAMPLES = 16000

STFT_FRAME_LEN  = 480
STFT_FRAME_STEP = 320
STFT_FFT_LEN    = 512
N_FREQ_BINS     = 40

# ---------------------------------------------------------------------------
# Augmentation constants -- identical to V2
# ---------------------------------------------------------------------------
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
SNR_WEIGHTS = {"easy": 1, "medium": 1, "hard": 1}

# Positive augmentation: ~15% clean, ~85% mixed
CLEAN_PROB = 0.15

# Negative batch fractions (within the 50% negative half of each batch)
# Target: 60% ordinary speech, 20% ambient, 5% other, 15% hard-speech
NEG_FRAC_SPEECH  = 0.60   # ordinary speech (including easy-score speech)
NEG_FRAC_AMBIENT = 0.20
NEG_FRAC_OTHER   = 0.05
NEG_FRAC_HARD    = 0.15   # hard-speech pool (score >= 0.30)

# Bounded oversampling cap: hard-speech clips can appear at most this many
# times per epoch across all batches (prevent tiny pool dominating)
HARD_SPEECH_MAX_OVERSAMPLE = 5

# ---------------------------------------------------------------------------
# Training hyperparameters -- identical to V2
# ---------------------------------------------------------------------------
BATCH_SIZE     = 64
EPOCHS         = 30
LEARNING_RATE  = 0.001
POS_PER_BATCH  = BATCH_SIZE // 2   # 32 positives
NEG_PER_BATCH  = BATCH_SIZE // 2   # 32 negatives

EARLY_STOP_PATIENCE = 5


# ===========================================================================
# Audio utilities
# ===========================================================================
def load_audio(path: str) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    if len(a) < WINDOW_SAMPLES: a = np.pad(a, (0, WINDOW_SAMPLES - len(a)))
    else: a = a[:WINDOW_SAMPLES]
    return a.astype(np.float32)


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


def vad_trim(audio: np.ndarray, top_db=20) -> np.ndarray:
    """Simple energy VAD: keep samples above threshold."""
    rms = calc_rms(audio)
    threshold = rms / (10 ** (top_db / 20))
    above = np.where(np.abs(audio) > threshold)[0]
    if len(above) == 0: return audio
    return audio[above[0]:above[-1] + 1]


def get_random_bg_segment(bg_paths):
    """Return (waveform, source_path, start_sample) for a random background segment."""
    path = random.choice(bg_paths)
    audio = load_audio(path)
    if len(audio) <= WINDOW_SAMPLES:
        return audio, path, 0
    start = random.randint(0, len(audio) - WINDOW_SAMPLES)
    return audio[start:start + WINDOW_SAMPLES], path, start


def mix_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix speech with noise at a target SNR (dB)."""
    s_rms = calc_rms(speech)
    n_rms = calc_rms(noise)
    target_n_rms = s_rms / (10 ** (snr_db / 20))
    scale = target_n_rms / n_rms
    mixed = speech + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99: mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


# ===========================================================================
# Positive augmentation (identical to V2)
# ===========================================================================
def augment_positive(path: str, amb_paths: list, sp_paths: list, stats: dict) -> np.ndarray:
    """Load, VAD-trim, and augment a positive clip. Raises on invalid input."""
    orig = load_audio(path)
    active = vad_trim(orig)

    if len(active) > WINDOW_SAMPLES:
        raise ValueError(f"Positive {path} active speech > {WINDOW_SAMPLES} samples. Pre-filter failed.")

    pos_rms = calc_rms(active)
    max_shift = WINDOW_SAMPLES - len(active)
    offset = random.randint(0, max_shift)
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    padded[offset:offset + len(active)] = active

    # Decide augmentation mode
    if random.random() < CLEAN_PROB or (not amb_paths and not sp_paths):
        stats["clean"] += 1
        return padded

    if sp_paths and (random.random() < 0.5 or not amb_paths):
        bg_seg, bg_src, _ = get_random_bg_segment(sp_paths)
        stats["speech"] += 1
    else:
        bg_seg, bg_src, _ = get_random_bg_segment(amb_paths)
        stats["ambient"] += 1

    snr_tier = random.choices(list(SNR_RANGES.keys()),
                              weights=list(SNR_WEIGHTS.values()))[0]
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
    """Load manifest and return structured split dicts."""
    splits = {
        "train":      {"pos": [], "speech": [], "ambient": [], "other": []},
        "validation": {"pos": [], "speech": [], "ambient": [], "other": []},
        "test":       {"pos": [], "speech": [], "ambient": [], "other": []},
    }
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split = r["split"]
            if split not in splits: continue
            p = str(REPO_ROOT / r["path"])
            if not Path(p).exists(): continue
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


def load_hard_speech_pool():
    """Load the V2-mined hard speech negative pool (train-only)."""
    if not HARD_SPEECH_CSV.exists():
        raise FileNotFoundError(f"Hard speech CSV not found: {HARD_SPEECH_CSV}\n"
                                "Run mine_v2_train_speech_hard_negatives.py first.")
    hard = []
    with open(HARD_SPEECH_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["V2_score"]) >= HARD_SPEECH_THRESHOLD:
                p = r["path"]
                if Path(p).exists():
                    hard.append(p)
    return list(set(hard))  # deduplicate


def load_bg_pool():
    """Load all background audio files for positive augmentation."""
    if not BG_DIR.exists(): return []
    return [str(p) for p in BG_DIR.rglob("*.wav")]


# ===========================================================================
# Pre-filter positives (remove clips > WINDOW_SAMPLES after VAD)
# ===========================================================================
def prefilter_positives(pos_paths: list) -> list:
    valid, skipped = [], []
    for p in pos_paths:
        try:
            a = load_audio(p)
            active = vad_trim(a)
            if len(active) > WINDOW_SAMPLES:
                skipped.append(p)
            else:
                valid.append(p)
        except Exception as e:
            skipped.append(p)
    if skipped:
        print(f"  Pre-filter: skipped {len(skipped)} positives (VAD active > {WINDOW_SAMPLES})")
    return valid


# ===========================================================================
# Generator
# ===========================================================================
class V21DataGenerator(tf.keras.utils.PyDataset):
    """
    Balanced 50/50 generator with explicit speech-dominated negative batching.

    Negative composition per batch (of NEG_PER_BATCH=32):
      hard_n  = round(NEG_PER_BATCH * 0.15) = 5   <- hard speech pool
      speech_n = round(NEG_PER_BATCH * 0.60) = 19  <- ordinary speech
      ambient_n = round(NEG_PER_BATCH * 0.20) = 6  <- ambient
      other_n  = NEG_PER_BATCH - hard_n - speech_n - ambient_n = 2
    """
    def __init__(self, positives, speech_negs, ambient_negs, other_negs,
                 hard_speech, amb_paths, sp_paths, stats, **kwargs):
        super().__init__(**kwargs)
        self.positives    = positives
        self.speech_negs  = speech_negs
        self.ambient_negs = ambient_negs
        self.other_negs   = other_negs
        self.hard_speech  = hard_speech
        self.amb_paths    = amb_paths
        self.sp_paths     = sp_paths
        self.stats        = stats

        # Per-batch allocation
        self.hard_n   = max(1, round(NEG_PER_BATCH * NEG_FRAC_HARD))
        self.speech_n = max(1, round(NEG_PER_BATCH * NEG_FRAC_SPEECH))
        self.ambient_n= max(1, round(NEG_PER_BATCH * NEG_FRAC_AMBIENT))
        self.other_n  = max(1, NEG_PER_BATCH - self.hard_n - self.speech_n - self.ambient_n)

        # Cap hard-speech usage per epoch
        hard_budget = min(len(self.hard_speech) * HARD_SPEECH_MAX_OVERSAMPLE,
                          len(self.positives))  # can't be more than epoch size
        self.batch_count = min(len(self.positives) // POS_PER_BATCH,
                               hard_budget // max(1, self.hard_n))

        self.on_epoch_end()

    def __len__(self):
        return self.batch_count

    def on_epoch_end(self):
        self._pos_perm  = np.random.permutation(len(self.positives))
        self._sp_perm   = np.random.permutation(len(self.speech_negs))
        self._amb_perm  = np.random.permutation(len(self.ambient_negs))
        self._oth_perm  = np.random.permutation(len(self.other_negs)) if self.other_negs else None
        self._hard_perm = np.random.permutation(len(self.hard_speech))
        self._sp_idx = self._amb_idx = self._oth_idx = self._hard_idx = self._pos_idx = 0

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
            spec  = make_spectrogram(audio)
            X[i, :, :, 0] = spec
            y[i, 0] = 1.0

        # Negatives -- hard speech
        hard_idxs = self._next_idx(self._hard_perm, "_hard_idx", len(self.hard_speech), self.hard_n)
        for j, hi in enumerate(hard_idxs):
            spec = make_spectrogram(load_audio(self.hard_speech[hi]))
            X[POS_PER_BATCH + j, :, :, 0] = spec
            self.stats["hard_negatives_sampled"] += 1

        # Negatives -- ordinary speech
        sp_idxs = self._next_idx(self._sp_perm, "_sp_idx", len(self.speech_negs), self.speech_n)
        for j, si in enumerate(sp_idxs):
            spec = make_spectrogram(load_audio(self.speech_negs[si]))
            X[POS_PER_BATCH + self.hard_n + j, :, :, 0] = spec
            self.stats["ordinary_speech_sampled"] += 1

        # Negatives -- ambient
        amb_idxs = self._next_idx(self._amb_perm, "_amb_idx", len(self.ambient_negs), self.ambient_n)
        for j, ai in enumerate(amb_idxs):
            spec = make_spectrogram(load_audio(self.ambient_negs[ai]))
            X[POS_PER_BATCH + self.hard_n + self.speech_n + j, :, :, 0] = spec
            self.stats["ordinary_ambient_sampled"] += 1

        # Negatives -- other (fill remaining)
        base = POS_PER_BATCH + self.hard_n + self.speech_n + self.ambient_n
        if self.other_negs and self.other_n > 0:
            oth_idxs = self._next_idx(self._oth_perm, "_oth_idx", len(self.other_negs), self.other_n)
            for j, oi in enumerate(oth_idxs):
                if base + j < BATCH_SIZE:
                    spec = make_spectrogram(load_audio(self.other_negs[oi]))
                    X[base + j, :, :, 0] = spec
                    self.stats["ordinary_other_sampled"] += 1
        # If no other negs, fill remaining with extra speech
        else:
            for j in range(self.other_n):
                if base + j < BATCH_SIZE:
                    ri = random.randint(0, len(self.speech_negs) - 1)
                    spec = make_spectrogram(load_audio(self.speech_negs[ri]))
                    X[base + j, :, :, 0] = spec
                    self.stats["ordinary_speech_sampled"] += 1

        return X, y


# ===========================================================================
# Per-epoch validation FPR callback
# ===========================================================================
class ValidationFPRCallback(tf.keras.callbacks.Callback):
    """
    At end of each epoch, compute:
      - val positive TPR @ 0.50
      - val speech FPR @ 0.50
      - val ambient FPR @ 0.50
      - val overall FPR @ 0.50
    Logs these as custom metrics for history saving.
    Also tracks the best checkpoint for speech-FPR-aware selection.
    """
    def __init__(self, val_pos, val_speech, val_ambient, model_path):
        super().__init__()
        self.val_pos     = val_pos
        self.val_speech  = val_speech
        self.val_ambient = val_ambient
        self.best_sfpr_path = model_path
        # epoch records: list of {epoch, tpr, speech_fpr, ambient_fpr, overall_fpr}
        self.records = []
        # track best: among epochs with TPR >= 95%, lowest speech FPR
        self._best_sfpr = float("inf")
        self._best_tpr_for_sfpr = 0.0

    def _batch_score(self, paths):
        if not paths: return np.array([])
        specs = [np.expand_dims(make_spectrogram(load_audio(p)), -1) for p in paths]
        X = np.stack(specs)
        return self.model.predict(X, batch_size=64, verbose=0).flatten()

    def on_epoch_end(self, epoch, logs=None):
        THR = 0.50
        s_pos  = self._batch_score(self.val_pos)
        s_sp   = self._batch_score(self.val_speech)
        s_amb  = self._batch_score(self.val_ambient)

        tpr     = float(np.sum(s_pos >= THR) / len(s_pos)) if len(s_pos) else 0.0
        sfpr    = float(np.sum(s_sp  >= THR) / len(s_sp))  if len(s_sp)  else 0.0
        afpr    = float(np.sum(s_amb >= THR) / len(s_amb)) if len(s_amb) else 0.0
        all_neg = np.concatenate([s_sp, s_amb]) if len(s_sp) and len(s_amb) else (s_sp if len(s_sp) else s_amb)
        ofpr    = float(np.sum(all_neg >= THR) / len(all_neg)) if len(all_neg) else 0.0

        print(f"  [ValFPR] epoch={epoch+1}  TPR={tpr*100:.2f}%  "
              f"speech_FPR={sfpr*100:.2f}%  ambient_FPR={afpr*100:.2f}%  overall_FPR={ofpr*100:.2f}%")

        self.records.append({
            "epoch": epoch + 1,
            "val_tpr": tpr, "val_speech_fpr": sfpr,
            "val_ambient_fpr": afpr, "val_overall_fpr": ofpr,
        })
        if logs is not None:
            logs["val_tpr_50"]         = tpr
            logs["val_speech_fpr_50"]  = sfpr
            logs["val_ambient_fpr_50"] = afpr
            logs["val_overall_fpr_50"] = ofpr

        # Save best speech-FPR checkpoint (rule: TPR >= 95%, lowest speech FPR)
        if tpr >= 0.95 and sfpr < self._best_sfpr:
            self._best_sfpr = sfpr
            self._best_tpr_for_sfpr = tpr
            self.model.save(str(self.best_sfpr_path))
            print(f"  [ValFPR] ** Saved best-speech-FPR checkpoint: epoch={epoch+1}, "
                  f"speech_FPR={sfpr*100:.2f}%, TPR={tpr*100:.2f}%")


# ===========================================================================
# LR history callback
# ===========================================================================
class LrHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(lr)
        if logs is not None:
            logs["learning_rate"] = lr


# ===========================================================================
# Model architecture (identical to V1 / V2)
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
    model = tf.keras.Model(inp, out, name="IRA_CNN_V2_1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ===========================================================================
# Main
# ===========================================================================
def main():
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # --- Load splits ---
    print("\nLoading split manifest...")
    splits = load_splits()
    train_pos   = splits["train"]["pos"]
    train_sp    = splits["train"]["speech"]
    train_amb   = splits["train"]["ambient"]
    train_oth   = splits["train"]["other"]
    val_pos     = splits["validation"]["pos"]
    val_sp      = splits["validation"]["speech"]
    val_amb     = splits["validation"]["ambient"]

    # --- Load hard speech pool ---
    print("Loading V2-mined hard speech pool...")
    hard_speech = load_hard_speech_pool()
    print(f"  Hard-speech pool size: {len(hard_speech)} clips (V2 score >= {HARD_SPEECH_THRESHOLD})")

    # --- V1 streaming FTs: explicitly excluded ---
    print("  V1 streaming hard-negatives: EXCLUDED (source speakers overlap with test split)")
    v1_hard_count = 0   # 0 used

    # --- Background pool for positive augmentation ---
    bg_paths = load_bg_pool()
    print(f"  Background pool for positive augmentation: {len(bg_paths)} files")

    # --- Pre-filter positives ---
    print("\nPre-filtering positives (VAD > 1 sec check)...")
    train_pos = prefilter_positives(train_pos)

    # --- Isolation assertions ---
    train_set = set(train_pos + train_sp + train_amb + train_oth)
    val_set   = set(val_pos   + val_sp   + val_amb)
    test_set  = set(splits["test"]["pos"] + splits["test"]["speech"] +
                    splits["test"]["ambient"] + splits["test"]["other"])
    hard_set  = set(hard_speech)

    tv = train_set & val_set
    te = train_set & test_set
    ve = val_set   & test_set
    ht = hard_set  & test_set

    if tv: raise AssertionError(f"Train/Val overlap: {len(tv)} files")
    if te: raise AssertionError(f"Train/Test overlap: {len(te)} files")
    if ve: raise AssertionError(f"Val/Test overlap: {len(ve)} files")
    if ht: raise AssertionError(f"Hard-speech/Test overlap: {len(ht)} files -- LEAKAGE")
    print("  Split isolation: PASS")
    if hard_set & val_set:
        print(f"  WARNING: {len(hard_set & val_set)} hard-speech clips appear in validation.")

    # --- Pre-training report ---
    print("\n" + "="*65)
    print("PRE-TRAINING DATA REPORT")
    print("="*65)
    print(f"  Train positives         : {len(train_pos)}")
    print(f"  Train speech negs       : {len(train_sp)}")
    print(f"  Train ambient negs      : {len(train_amb)}")
    print(f"  Train other negs        : {len(train_oth)}")
    print(f"  V2-mined hard-speech    : {len(hard_speech)}")
    print(f"  V1 streaming hard-negs  : {v1_hard_count} (excluded -- test speaker overlap)")
    print(f"  Val positives           : {len(val_pos)}")
    print(f"  Val speech negs         : {len(val_sp)}")
    print(f"  Val ambient negs        : {len(val_amb)}")
    print(f"  BG pool (pos aug)       : {len(bg_paths)}")
    print()
    print(f"  Negative batch allocation (per {NEG_PER_BATCH} neg slots):")
    hard_n   = max(1, round(NEG_PER_BATCH * NEG_FRAC_HARD))
    speech_n = max(1, round(NEG_PER_BATCH * NEG_FRAC_SPEECH))
    ambient_n= max(1, round(NEG_PER_BATCH * NEG_FRAC_AMBIENT))
    other_n  = max(1, NEG_PER_BATCH - hard_n - speech_n - ambient_n)
    print(f"    Hard speech           : {hard_n}  ({hard_n/NEG_PER_BATCH*100:.0f}%)")
    print(f"    Ordinary speech       : {speech_n}  ({speech_n/NEG_PER_BATCH*100:.0f}%)")
    print(f"    Ambient               : {ambient_n}   ({ambient_n/NEG_PER_BATCH*100:.0f}%)")
    print(f"    Other                 : {other_n}   ({other_n/NEG_PER_BATCH*100:.0f}%)")
    print(f"  Hard-speech max oversample: {HARD_SPEECH_MAX_OVERSAMPLE}x")
    print()

    # --- Test split report (for reference, labelled DIAGNOSTIC) ---
    print("  DIAGNOSTIC (historical test set -- NOT used for selection):")
    print(f"    Test positives        : {len(splits['test']['pos'])}")
    print(f"    Test speech negs      : {len(splits['test']['speech'])}")
    print(f"    Test ambient negs     : {len(splits['test']['ambient'])}")

    # --- Build generator and run sanity check ---
    train_stats = {k: 0 for k in ["clean","ambient","speech","easy_snr","med_snr","hard_snr",
                                   "hard_negatives_sampled","ordinary_speech_sampled",
                                   "ordinary_ambient_sampled","ordinary_other_sampled",
                                   "skipped_too_long"]}

    train_gen = V21DataGenerator(
        train_pos, train_sp, train_amb, train_oth, hard_speech, bg_paths, train_sp, train_stats
    )
    print(f"\nBatches per epoch: {len(train_gen)}")
    print("\nRunning 5-batch sanity check...")
    all_ok = True
    for i in range(min(5, len(train_gen))):
        Xb, yb = train_gen[i]
        if Xb.shape[1:] != (49, N_FREQ_BINS, 1):
            print(f"  FAIL batch {i}: shape {Xb.shape}"); all_ok = False
        if np.any(np.isnan(Xb)) or np.any(np.isinf(Xb)):
            print(f"  FAIL batch {i}: NaN/Inf"); all_ok = False

    required = ["clean","ambient","speech","easy_snr","med_snr","hard_snr",
                "hard_negatives_sampled","ordinary_speech_sampled","ordinary_ambient_sampled"]
    for k in required:
        if train_stats[k] <= 0:
            print(f"  FAIL: cumulative stats missing {k}"); all_ok = False

    if not all_ok:
        raise RuntimeError("V2.1 TRAINING PIPELINE VALIDATION FAILED.")
    print("V2.1 TRAINING PIPELINE VALIDATION: PASS\n")

    # Reset stats so JSON contains actual training counts only
    for k in train_stats: train_stats[k] = 0

    # --- Build model (random init) ---
    model = build_model()
    print("Model summary:")
    model.summary(print_fn=lambda x: print("  " + x))
    print(f"\nClass weights: None (generator explicitly balances 50/50)")

    # --- Callbacks ---
    fpr_cb = ValidationFPRCallback(val_pos, val_sp, val_amb, BEST_SFPR_PATH)
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

    # Build validation data for Keras (positives + negatives flat)
    print("\nBuilding validation data array...")
    val_all_neg = val_sp + val_amb
    val_specs, val_labels = [], []
    for p in val_pos:
        spec = make_spectrogram(load_audio(p))
        val_specs.append(np.expand_dims(spec, -1)); val_labels.append(1.0)
    for p in val_all_neg:
        spec = make_spectrogram(load_audio(p))
        val_specs.append(np.expand_dims(spec, -1)); val_labels.append(0.0)
    val_X = np.stack(val_specs); val_y = np.array(val_labels, dtype=np.float32).reshape(-1,1)
    print(f"  Val samples: {len(val_labels)} ({sum(val_labels):.0f} pos / {len(val_labels)-int(sum(val_labels))} neg)")

    # --- Train ---
    print("\n" + "="*65)
    print("TRAINING IRA CNN V2.1")
    print("="*65)
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=(val_X, val_y),
        callbacks=callbacks,
        verbose=1,
    )

    # --- Save final model ---
    model.save(str(FINAL_PATH))
    print(f"\nSaved final model: {FINAL_PATH}")

    # --- Save training history CSV ---
    h = history.history
    h_len = len(h["loss"])
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(h.keys()) + ["val_tpr_50","val_speech_fpr_50","val_ambient_fpr_50","val_overall_fpr_50"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ep in range(h_len):
            row = {k: h[k][ep] for k in h}
            # Append FPR records
            if ep < len(fpr_cb.records):
                rec = fpr_cb.records[ep]
                row["val_tpr_50"]         = rec["val_tpr"]
                row["val_speech_fpr_50"]  = rec["val_speech_fpr"]
                row["val_ambient_fpr_50"] = rec["val_ambient_fpr"]
                row["val_overall_fpr_50"] = rec["val_overall_fpr"]
            w.writerow(row)
    print(f"History saved: {HISTORY_CSV}")

    # Add LR column
    lrs = lr_cb.lrs
    rows = []
    with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        row["learning_rate"] = lrs[i] if i < len(lrs) else ""
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # --- Save augmentation stats ---
    with open(STATS_JSON, "w") as f:
        json.dump({"train_stats": train_stats}, f, indent=2)
    print(f"Stats saved: {STATS_JSON}")

    # --- Print training summary ---
    best_epoch = int(np.argmin(h["val_loss"])) + 1
    best_val_loss = min(h["val_loss"])
    best_val_prec = h["val_precision"][np.argmin(h["val_loss"])]
    best_val_rec  = h["val_recall"][np.argmin(h["val_loss"])]

    print("\n" + "="*65)
    print("TRAINING SUMMARY")
    print("="*65)
    print(f"  Best epoch (val_loss)   : {best_epoch}")
    print(f"  Best val_loss           : {best_val_loss:.5f}")
    print(f"  Best val_precision      : {best_val_prec:.4f}")
    print(f"  Best val_recall         : {best_val_rec:.4f}")
    print(f"  Total epochs run        : {h_len}")
    print(f"  Training time           : {(time.time()-t_start)/60:.1f} min")
    print()
    print(f"  Best-speech-FPR checkpoint rule:")
    print(f"    Among epochs with val TPR >= 95%, lowest val speech FPR")
    if fpr_cb._best_sfpr < float('inf'):
        print(f"    -> speech FPR = {fpr_cb._best_sfpr*100:.2f}%  "
              f"   TPR = {fpr_cb._best_tpr_for_sfpr*100:.2f}%")
        print(f"    Saved: {BEST_SFPR_PATH}")
    else:
        print(f"    -> No epoch achieved val TPR >= 95% (best-speech-FPR checkpoint not saved)")
    print()

    # --- Print per-epoch FPR summary ---
    print("  Per-epoch validation FPR summary:")
    print(f"  {'Epoch':>5}  {'val_loss':>9}  {'TPR@50':>8}  {'sp_FPR@50':>10}  {'amb_FPR@50':>11}  {'all_FPR@50':>11}")
    for ep_idx in range(h_len):
        vl = h["val_loss"][ep_idx]
        if ep_idx < len(fpr_cb.records):
            r = fpr_cb.records[ep_idx]
            print(f"  {ep_idx+1:>5}  {vl:>9.5f}  {r['val_tpr']*100:>7.2f}%  "
                  f"{r['val_speech_fpr']*100:>9.2f}%  {r['val_ambient_fpr']*100:>10.2f}%  "
                  f"{r['val_overall_fpr']*100:>10.2f}%")
        else:
            print(f"  {ep_idx+1:>5}  {vl:>9.5f}  (FPR not recorded)")

    print(f"\nAugmentation stats (actual training): {train_stats}")
    print("\nDone. Evaluate val first, then diagnostic test if val passes.")


if __name__ == "__main__":
    main()

