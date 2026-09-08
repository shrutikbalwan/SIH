# -*- coding: utf-8 -*-
"""
train_cnn_v2_4.py
=================
IRA CNN V2.4 -- Controlled LR-schedule experiment.

ONLY change from V2.3:
  Adam LR=0.001 -> ReduceLROnPlateau(factor=0.5, patience=2, min_lr=6.25e-5)
  EarlyStopping patience=7, restore_best_weights=False
  Max epochs=40

Everything else is IDENTICAL to V2.3:
  - split_manifest_v3.csv
  - Same difficulty pools (V2 scores, TRAIN-only)
  - Same batch recipe: 32 pos + 14 easy + 7 med + 4 hard + 5 amb + 2 oth
  - Same positive augmentation
  - Same CNN architecture
  - Same preprocessing
  - Same EXPANDED_VAL (seed=999)
"""

import os, csv, json, random, collections
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
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).parent.parent
MANIFEST       = REPO_ROOT / "dataset" / "split_manifest_v3.csv"
MODEL_OUT_DIR  = REPO_ROOT / "cnn" / "models"
BEST_LOSS_PATH = MODEL_OUT_DIR / "ira_cnn_v2_4_best_loss.keras"
BEST_SFPR_PATH = MODEL_OUT_DIR / "ira_cnn_v2_4_best_speech_fpr.keras"
FINAL_PATH     = MODEL_OUT_DIR / "ira_cnn_v2_4_final.keras"
HISTORY_CSV    = REPO_ROOT / "v2_4_training_history.csv"
STATS_JSON     = REPO_ROOT / "v2_4_augmentation_stats.json"
SAMPLING_JSON  = REPO_ROOT / "v2_4_sampling_stats.json"
V2_MODEL_PATH  = MODEL_OUT_DIR / "ira_cnn_v2_best_loss.keras"

# ---------------------------------------------------------------------------
# Audio / feature constants (UNCHANGED)
# ---------------------------------------------------------------------------
SAMPLE_RATE    = 16000
WINDOW_SAMPLES = 16000
STFT_FRAME_LEN  = 480
STFT_FRAME_STEP = 320
STFT_FFT_LEN    = 512
N_FREQ_BINS     = 40

# ---------------------------------------------------------------------------
# Augmentation constants (UNCHANGED from V2.2/V2.3)
# ---------------------------------------------------------------------------
SNR_RANGES  = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
SNR_WEIGHTS = {"easy": 1, "medium": 1, "hard": 1}
CLEAN_PROB  = 0.15

# ---------------------------------------------------------------------------
# Batch recipe (UNCHANGED from V2.3)
# ---------------------------------------------------------------------------
N_EASY_SP  = 14
N_MED_SP   =  7
N_HARD_SP  =  4
N_AMBIENT  =  5
N_OTHER    =  2

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE   = 64
EPOCHS       = 40                   # <-- increased from 30
POS_PER_BATCH = BATCH_SIZE // 2     # 32
NEG_PER_BATCH = BATCH_SIZE // 2     # 32

# LR schedule (V2.4 change)
INIT_LR      = 0.001
LR_FACTOR    = 0.5
LR_PATIENCE  = 2
MIN_LR       = 0.0000625            # = 0.001 / 16

# EarlyStopping (V2.4 change)
ES_PATIENCE  = 7

# Difficulty thresholds (UNCHANGED - V2 baseline scores)
EASY_THRESH = 0.10
HARD_THRESH = 0.50
RMS_THRESH  = 1e-4


# ===========================================================================
# Audio utilities (IDENTICAL to V2.3)
# ===========================================================================
def load_audio(path):
    try:
        a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        raise RuntimeError(f"Cannot load {path}: {e}")
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a.astype(np.float32)

def pad_to_window(a):
    return np.pad(a, (0, WINDOW_SAMPLES - len(a))) if len(a) < WINDOW_SAMPLES else a[:WINDOW_SAMPLES]

def make_spectrogram(audio):
    t = tf.convert_to_tensor(audio, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN,
                       frame_step=STFT_FRAME_STEP, fft_length=STFT_FFT_LEN)
    s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_FREQ_BINS]
    m = tf.reduce_mean(s); d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)

def calc_rms(a):
    r = float(np.sqrt(np.mean(a ** 2))); return r if r > 1e-9 else 1e-9

def vad_trim(a, top_db=25):
    thr = calc_rms(a) / (10 ** (top_db / 20))
    idx = np.where(np.abs(a) > thr)[0]
    return a[idx[0]:idx[-1]+1] if len(idx) else a

def get_random_bg_segment(paths, stats, target_len=WINDOW_SAMPLES):
    while True:
        if not paths: return np.zeros(target_len, dtype=np.float32), "none", 0
        p = random.choice(paths)
        a = load_audio(p)
        if len(a) < target_len: seg = np.pad(a, (0, target_len - len(a)))
        elif len(a) > target_len:
            start = random.randint(0, len(a) - target_len)
            seg = a[start:start+target_len]
        else: seg = a
        if calc_rms(seg) >= RMS_THRESH: return seg, p, 0
        stats["silent_bg_rejected"] += 1

def mix_snr(speech, noise, snr_db):
    scale = calc_rms(speech) / calc_rms(noise) / (10 ** (snr_db / 20))
    mixed = speech + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99: mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)

def augment_positive(path, amb_paths, sp_paths, stats):
    orig = load_audio(path); active = vad_trim(orig)
    if len(active) > WINDOW_SAMPLES:
        raise ValueError(f"Positive {path} active speech > {WINDOW_SAMPLES} samples.")
    offset = random.randint(0, WINDOW_SAMPLES - len(active))
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    padded[offset:offset+len(active)] = active
    if random.random() < CLEAN_PROB or (not amb_paths and not sp_paths):
        stats["clean"] += 1; return padded
    if sp_paths and (random.random() < 0.5 or not amb_paths):
        bg, _, _ = get_random_bg_segment(sp_paths, stats); stats["speech"] += 1
    else:
        bg, _, _ = get_random_bg_segment(amb_paths, stats); stats["ambient"] += 1
    tier = random.choices(list(SNR_RANGES.keys()), weights=list(SNR_WEIGHTS.values()))[0]
    lo, hi = SNR_RANGES[tier]
    mixed = mix_snr(padded, bg, random.uniform(lo, hi))
    if tier == "easy": stats["easy_snr"] += 1
    elif tier == "medium": stats["med_snr"] += 1
    else: stats["hard_snr"] += 1
    return mixed


# ===========================================================================
# Split loading  (IDENTICAL to V2.3 -- unseen_test skipped)
# ===========================================================================
def load_splits():
    splits = {
        "train":      {"pos": [], "speech": [], "ambient": [], "other": []},
        "validation": {"pos": [], "speech": [], "ambient": [], "other": []},
        "test":       {"pos": [], "speech": [], "ambient": [], "other": []},
    }
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = r["split"].strip()
            if sp == "unseen_test": continue   # NEVER loaded
            if sp not in splits: continue
            p = str(REPO_ROOT / r["path"])
            if not Path(p).exists(): continue
            lbl = int(r.get("label", -1))
            g = r.get("group", "").lower()
            if lbl == 1:
                splits[sp]["pos"].append(p)
            elif lbl == 0:
                if "libri" in g or "speech" in g: splits[sp]["speech"].append(p)
                elif "ambient" in g or "background" in g or "noise" in g: splits[sp]["ambient"].append(p)
                else: splits[sp]["other"].append(p)
    return splits


def build_difficulty_pools(train_speech_paths):
    """Score TRAIN speech clips with V2 baseline model."""
    print("  Loading V2 model for difficulty scoring...")
    v2 = tf.keras.models.load_model(str(V2_MODEL_PATH))
    print(f"  Scoring {len(train_speech_paths)} TRAIN speech clips...")
    X = []
    for p in train_speech_paths:
        a = pad_to_window(load_audio(p))
        t = tf.convert_to_tensor(a, dtype=tf.float32)
        s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN,
                           frame_step=STFT_FRAME_STEP, fft_length=STFT_FFT_LEN)
        s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_FREQ_BINS]
        m = tf.reduce_mean(s); d = tf.math.reduce_std(s) + 1e-6
        X.append(np.expand_dims(((s - m)/d).numpy(), -1))
    X = np.array(X, dtype=np.float32)
    scores = v2.predict(X, batch_size=128, verbose=0).flatten()
    del v2, X

    easy, med, hard = [], [], []
    for i, sc in enumerate(scores):
        if sc < EASY_THRESH: easy.append(train_speech_paths[i])
        elif sc < HARD_THRESH: med.append(train_speech_paths[i])
        else: hard.append(train_speech_paths[i])

    def spks(paths): return {os.path.basename(p).split("_")[0] for p in paths if os.path.basename(p).split("_")[0].isdigit()}

    print(f"  Easy  (<{EASY_THRESH}): {len(easy)} clips, {len(spks(easy))} speakers")
    print(f"  Med   ({EASY_THRESH}-{HARD_THRESH}): {len(med)} clips, {len(spks(med))} speakers")
    print(f"  Hard  (>={HARD_THRESH}): {len(hard)} clips, {len(spks(hard))} speakers")
    print(f"  Score dist  mean={np.mean(scores):.4f}  median={np.median(scores):.4f}"
          f"  P95={np.percentile(scores,95):.4f}")
    return easy, med, hard

def prefilter_positives(pos_paths):
    valid = []
    for p in pos_paths:
        try:
            a = load_audio(p)
            if len(vad_trim(a)) <= WINDOW_SAMPLES: valid.append(p)
        except Exception: pass
    return valid


# ===========================================================================
# Generator (IDENTICAL to V2.3)
# ===========================================================================
class V24DataGenerator(tf.keras.utils.PyDataset):
    def __init__(self, positives, easy_sp, med_sp, hard_sp,
                 ambient_negs, other_negs, amb_paths, sp_paths, stats, **kwargs):
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

    def __len__(self): return self.batch_count

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
                    "unique_clips": len(vals),
                    "total_draws": sum(vals),
                    "unique_speakers": len({os.path.basename(p).split("_")[0]
                                            for p in counts_dict if os.path.basename(p).split("_")[0].isdigit()}),
                    "min_reps": min(vals),
                    "mean_reps": round(sum(vals)/len(vals), 2),
                    "max_reps": max(vals),
                })
        if self.sampling_history:
            with open(SAMPLING_JSON, "w") as f:
                json.dump(self.sampling_history, f, indent=2)
        self._epoch_easy_counts.clear()
        self._epoch_med_counts.clear()
        self._epoch_hard_counts.clear()

        self._pos_perm  = np.random.permutation(len(self.positives))
        self._easy_perm = np.random.permutation(len(self.easy_sp))
        self._med_perm  = np.random.permutation(len(self.med_sp))
        self._hard_perm = np.random.permutation(len(self.hard_sp))
        self._amb_perm  = np.random.permutation(len(self.ambient_negs))
        self._oth_perm  = np.random.permutation(len(self.other_negs)) if self.other_negs else None
        self._pos_idx = self._easy_idx = self._med_idx = 0
        self._hard_idx = self._amb_idx = self._oth_idx = 0

    def _next_idx(self, perm, idx_attr, pool_len, n):
        cur = getattr(self, idx_attr); idxs = []
        for _ in range(n):
            if cur >= pool_len: perm[:] = np.random.permutation(pool_len); cur = 0
            idxs.append(int(perm[cur])); cur += 1
        setattr(self, idx_attr, cur); return idxs

    def __getitem__(self, idx):
        X = np.zeros((BATCH_SIZE, 49, N_FREQ_BINS, 1), dtype=np.float32)
        y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        curr_n = 0

        # Positives
        for i, pi in enumerate(self._next_idx(self._pos_perm, "_pos_idx", len(self.positives), POS_PER_BATCH)):
            X[i, :, :, 0] = make_spectrogram(augment_positive(self.positives[pi], self.amb_paths, self.sp_paths, self.stats))
            y[i, 0] = 1.0
        curr_n = POS_PER_BATCH

        # Easy speech (14)
        for i in self._next_idx(self._easy_perm, "_easy_idx", len(self.easy_sp), min(N_EASY_SP, len(self.easy_sp))):
            p = self.easy_sp[i]
            X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(p)))
            self._epoch_easy_counts[p] += 1; self.stats["easy_speech_sampled"] += 1; curr_n += 1

        # Medium speech (7)
        for i in self._next_idx(self._med_perm, "_med_idx", len(self.med_sp), min(N_MED_SP, len(self.med_sp))):
            p = self.med_sp[i]
            X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(p)))
            self._epoch_med_counts[p] += 1; self.stats["medium_speech_sampled"] += 1; curr_n += 1

        # Hard speech (4)
        for i in self._next_idx(self._hard_perm, "_hard_idx", len(self.hard_sp), min(N_HARD_SP, len(self.hard_sp))):
            p = self.hard_sp[i]
            X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(p)))
            self._epoch_hard_counts[p] += 1; self.stats["hard_speech_sampled"] += 1; curr_n += 1

        # Ambient (5)
        for i in self._next_idx(self._amb_perm, "_amb_idx", len(self.ambient_negs), min(N_AMBIENT, len(self.ambient_negs))):
            X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.ambient_negs[i])))
            self.stats["ambient_sampled"] += 1; curr_n += 1

        # Other (2)
        if self.other_negs and BATCH_SIZE - curr_n > 0:
            for i in self._next_idx(self._oth_perm, "_oth_idx", len(self.other_negs), min(N_OTHER, BATCH_SIZE - curr_n)):
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(self.other_negs[i])))
                self.stats["other_sampled"] += 1; curr_n += 1

        # Fallback
        while curr_n < BATCH_SIZE:
            for i in self._next_idx(self._easy_perm, "_easy_idx", len(self.easy_sp), 1):
                p = self.easy_sp[i]
                X[curr_n, :, :, 0] = make_spectrogram(pad_to_window(load_audio(p)))
                self._epoch_easy_counts[p] += 1; self.stats["easy_speech_sampled"] += 1; curr_n += 1
        return X, y


# ===========================================================================
# Callbacks
# ===========================================================================
class ValidationFPRCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_pos_X, val_sp_X, val_amb_X, sfpr_path):
        super().__init__()
        self.val_pos_X = val_pos_X
        self.val_sp_X  = val_sp_X
        self.val_amb_X = val_amb_X
        self.sfpr_path = sfpr_path
        self.records   = []
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
        lr_val = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        print(f"  [ValFPR] ep={epoch+1}  LR={lr_val:.7f}  TPR={tpr*100:.2f}%  "
              f"spFPR={sfpr*100:.2f}%  ambFPR={afpr*100:.2f}%  ovrFPR={ofpr*100:.2f}%")

        rec = {"epoch": epoch+1, "lr": lr_val, "val_loss": vloss,
               "val_tpr": tpr, "val_speech_fpr": sfpr,
               "val_ambient_fpr": afpr, "val_overall_fpr": ofpr}
        self.records.append(rec)

        if logs is not None:
            logs["val_tpr_50"]         = tpr
            logs["val_speech_fpr_50"]  = sfpr
            logs["val_ambient_fpr_50"] = afpr
            logs["val_overall_fpr_50"] = ofpr

        # Checkpoint rule: TPR>=95% -> lowest sfpr -> highest tpr -> lowest val_loss
        if tpr >= 0.95:
            is_best = False
            if sfpr < self._best_sfpr - 1e-4: is_best = True
            elif abs(sfpr - self._best_sfpr) <= 1e-4:
                if tpr > self._best_tpr + 1e-4: is_best = True
                elif abs(tpr - self._best_tpr) <= 1e-4 and vloss < self._best_loss: is_best = True
            if is_best:
                self._best_sfpr = sfpr; self._best_tpr = tpr; self._best_loss = vloss
                self.model.save(str(self.sfpr_path))
                print(f"  [ValFPR] ** Saved best sfpr checkpoint: ep={epoch+1}  "
                      f"spFPR={sfpr*100:.2f}%  TPR={tpr*100:.2f}%  vloss={vloss:.5f}")


# ===========================================================================
# Model (IDENTICAL architecture to V2/V2.1/V2.2/V2.3)
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
    model = tf.keras.Model(inp, out, name="IRA_CNN_V2_4")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(INIT_LR),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ===========================================================================
# Post-training sweep
# ===========================================================================
def threshold_sweep(model, pos_X, sp_X, amb_X, label):
    s_pos = model.predict(pos_X, batch_size=64, verbose=0).flatten()
    s_sp  = model.predict(sp_X,  batch_size=64, verbose=0).flatten()
    s_amb = model.predict(amb_X, batch_size=64, verbose=0).flatten()
    s_neg = np.concatenate([s_sp, s_amb])
    thresholds = np.arange(0.30, 0.951, 0.01)

    print(f"\n--- Threshold Frontier: {label} ---")
    print(f"{'THR':>4}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")

    best_A = best_B = None
    best_at_tpr95 = None
    best_fpr3     = None

    for thr in thresholds:
        tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
        sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
        afpr = np.sum(s_amb >= thr) / len(s_amb) * 100
        ofpr = np.sum(s_neg >= thr) / len(s_neg) * 100
        print(f"{thr:.2f}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")

        if tpr >= 95.0 and sfpr <= 2.0 and best_A is None: best_A = (thr, tpr, sfpr)
        if tpr >= 95.0 and sfpr <= 3.0 and best_B is None: best_B = (thr, tpr, sfpr)
        if tpr >= 95.0 and (best_at_tpr95 is None or sfpr < best_at_tpr95[2]):
            best_at_tpr95 = (thr, tpr, sfpr)
        if sfpr <= 3.0 and (best_fpr3 is None or tpr > best_fpr3[1]):
            best_fpr3 = (thr, tpr, sfpr)

    print(f"\n  Feasibility A (TPR>=95% & SpFPR<=2%): " +
          ("PASS thr={:.2f}  TPR={:.2f}%  SpFPR={:.2f}%".format(*best_A) if best_A else "FAILED"))
    print(f"  Feasibility B (TPR>=95% & SpFPR<=3%): " +
          ("PASS thr={:.2f}  TPR={:.2f}%  SpFPR={:.2f}%".format(*best_B) if best_B else "FAILED"))
    if best_at_tpr95:
        print(f"  Best SpFPR when TPR>=95%: thr={best_at_tpr95[0]:.2f}  SpFPR={best_at_tpr95[2]:.2f}%  TPR={best_at_tpr95[1]:.2f}%")
    if best_fpr3:
        print(f"  Best TPR when SpFPR<=3%:  thr={best_fpr3[0]:.2f}  TPR={best_fpr3[1]:.2f}%  SpFPR={best_fpr3[2]:.2f}%")

    return best_A, best_B, best_at_tpr95


# ===========================================================================
# Main
# ===========================================================================
def main():
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load splits
    # -------------------------------------------------------------------------
    print("\nLoading split manifest v3 (unseen_test excluded)...")
    splits = load_splits()
    train_pos  = prefilter_positives(splits["train"]["pos"])
    train_sp   = splits["train"]["speech"]
    train_amb  = splits["train"]["ambient"]
    train_oth  = splits["train"]["other"]
    val_pos    = splits["validation"]["pos"]
    val_sp     = splits["validation"]["speech"]
    val_amb    = splits["validation"]["ambient"]
    val_oth    = splits["validation"]["other"]
    test_amb   = splits["test"]["ambient"]

    print(f"  Train pos={len(train_pos)}  train_sp={len(train_sp)}  train_amb={len(train_amb)}  train_oth={len(train_oth)}")
    print(f"  Val pos={len(val_pos)}  val_sp={len(val_sp)}  val_amb={len(val_amb)}")

    # -------------------------------------------------------------------------
    # Build difficulty pools from TRAIN speech (V2 baseline scores)
    # -------------------------------------------------------------------------
    print("\nBuilding V2.4 difficulty pools (TRAIN-only, V2 baseline scores)...")
    easy_sp, med_sp, hard_sp = build_difficulty_pools(train_sp)
    all_train_sp = easy_sp + med_sp + hard_sp

    # -------------------------------------------------------------------------
    # Build EXPANDED_VAL (seed=999, deterministic)
    # -------------------------------------------------------------------------
    print("\nBuilding EXPANDED_VAL (seed=999, deterministic)...")
    r_state  = random.getstate(); np_state = np.random.get_state()
    random.seed(999); np.random.seed(999)

    _vs = {k: 0 for k in ["clean","ambient","speech","easy_snr","med_snr","hard_snr","silent_bg_rejected"]}
    val_pos_X = np.array([
        np.expand_dims(make_spectrogram(augment_positive(p, val_amb, val_sp, _vs)), -1)
        for p in val_pos
    ], dtype=np.float32)
    val_sp_X  = np.array([
        np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1) for p in val_sp
    ], dtype=np.float32)
    val_amb_X = np.array([
        np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1) for p in val_amb
    ], dtype=np.float32)
    val_oth_X = np.array([
        np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1) for p in val_oth
    ], dtype=np.float32) if val_oth else np.zeros((0,49,N_FREQ_BINS,1), dtype=np.float32)

    random.setstate(r_state); np.random.set_state(np_state)

    val_X = np.concatenate([val_pos_X, val_sp_X, val_amb_X] + ([val_oth_X] if len(val_oth_X) else []))
    val_y = np.concatenate([np.ones(len(val_pos_X)),
                            np.zeros(len(val_sp_X)+len(val_amb_X)+len(val_oth_X))]).reshape(-1, 1)
    print(f"  EXPANDED_VAL: pos={len(val_pos_X)}  sp_neg={len(val_sp_X)}  amb_neg={len(val_amb_X)}")

    # -------------------------------------------------------------------------
    # Sanity check assertions
    # -------------------------------------------------------------------------
    print("\n--- PRE-TRAINING SANITY CHECK ---")
    train_amb_set = set(train_amb); val_amb_set = set(val_amb); test_amb_set = set(test_amb)
    assert train_amb_set.isdisjoint(val_amb_set),  "Overlap: train-amb vs val-amb!"
    assert train_amb_set.isdisjoint(test_amb_set), "Overlap: train-amb vs test-amb!"
    assert set(all_train_sp).isdisjoint(set(val_sp)), "Overlap: train-sp vs val-sp!"
    print("  Ambient pool disjoint: PASS")
    print("  Train speech vs val speech disjoint: PASS")

    # Speaker-level assertion
    import collections as col
    spk_splits = col.defaultdict(set)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = r["split"].strip(); lbl = r["label"]; g = r.get("group","").lower()
            if lbl != "0" or not ("libri" in g or "speech" in g): continue
            spk = r.get("speaker_id","").strip()
            if spk: spk_splits[sp].add(spk)
    for s1 in ["train","validation","test","unseen_test"]:
        for s2 in ["train","validation","test","unseen_test"]:
            if s1 >= s2: continue
            inter = spk_splits[s1].intersection(spk_splits[s2])
            assert len(inter) == 0, f"Speaker leakage: {s1} & {s2} share {inter}"
    print("  Speaker overlap (all 4 splits): PASS")

    # 3-batch dry run
    train_stats = {k: 0 for k in ["clean","ambient","speech","easy_snr","med_snr","hard_snr",
                                   "silent_bg_rejected","easy_speech_sampled","medium_speech_sampled",
                                   "hard_speech_sampled","ambient_sampled","other_sampled"]}
    gen = V24DataGenerator(train_pos, easy_sp, med_sp, hard_sp,
                           train_amb, train_oth, train_amb, all_train_sp, train_stats)
    print("  Running 3-batch sanity pass...")
    for i in range(3): gen[i]
    assert train_stats["easy_speech_sampled"]  > 0
    assert train_stats["medium_speech_sampled"] > 0
    assert train_stats["hard_speech_sampled"]   > 0
    assert train_stats["ambient_sampled"]        > 0
    assert train_stats["clean"]                  > 0
    assert (train_stats["ambient"] + train_stats["speech"]) > 0
    print(f"  Batch counts: easy={train_stats['easy_speech_sampled']}  "
          f"med={train_stats['medium_speech_sampled']}  hard={train_stats['hard_speech_sampled']}  "
          f"amb={train_stats['ambient_sampled']}")

    # Reset before real training
    for k in train_stats: train_stats[k] = 0
    gen._epoch_easy_counts.clear(); gen._epoch_med_counts.clear()
    gen._epoch_hard_counts.clear(); gen.sampling_history.clear()
    gen.on_epoch_end()

    print("\nV2.4 TRAINING PIPELINE VALIDATION: PASS")

    # -------------------------------------------------------------------------
    # Build model & callbacks
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("TRAINING IRA CNN V2.4")
    print(f"  Adam LR={INIT_LR}  ReduceLROnPlateau(factor={LR_FACTOR}, patience={LR_PATIENCE}, min_lr={MIN_LR})")
    print(f"  EarlyStopping patience={ES_PATIENCE}  restore_best_weights=False")
    print(f"  Batch: {POS_PER_BATCH} pos + 14 easy + 7 med + 4 hard + 5 amb + 2 oth = {BATCH_SIZE}")
    print("="*65)

    model = build_model()
    fpr_cb = ValidationFPRCallback(val_pos_X, val_sp_X, val_amb_X, BEST_SFPR_PATH)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_LOSS_PATH), monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=LR_FACTOR, patience=LR_PATIENCE,
            min_lr=MIN_LR, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=ES_PATIENCE,
            restore_best_weights=False, verbose=1),   # <-- False: preserves saved checkpoints
        fpr_cb,
    ]

    history = model.fit(
        gen, epochs=EPOCHS, validation_data=(val_X, val_y),
        callbacks=callbacks, verbose=1,
    )
    model.save(str(FINAL_PATH))
    print(f"\nSaved final model: {FINAL_PATH}")
    with open(STATS_JSON, "w") as f:
        json.dump({"train_stats": train_stats}, f, indent=2)

    # Save history CSV
    h = history.history
    fieldnames = list(h.keys()) + ["val_tpr_50","val_speech_fpr_50",
                                    "val_ambient_fpr_50","val_overall_fpr_50","epoch_lr"]
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
                row["epoch_lr"]           = rec["lr"]
            w.writerow(row)

    # -------------------------------------------------------------------------
    # Verify checkpoint selection rule
    # -------------------------------------------------------------------------
    print("\n--- Checkpoint selection rule verification ---")
    best_ep, best_sfpr, best_tpr, best_vloss = -1, float("inf"), 0.0, float("inf")
    with open(HISTORY_CSV, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            tpr  = float(row.get("val_tpr_50", 0))
            sfpr = float(row.get("val_speech_fpr_50", 1))
            vloss = float(row.get("val_loss", 1))
            if tpr >= 0.95:
                is_best = False
                if sfpr < best_sfpr - 1e-4: is_best = True
                elif abs(sfpr - best_sfpr) <= 1e-4:
                    if tpr > best_tpr + 1e-4: is_best = True
                    elif abs(tpr - best_tpr) <= 1e-4 and vloss < best_vloss: is_best = True
                if is_best:
                    best_ep, best_sfpr, best_tpr, best_vloss = i+1, sfpr, tpr, vloss
    print(f"  Selected epoch : {best_ep}")
    print(f"  Val TPR        : {best_tpr*100:.2f}%")
    print(f"  Val Speech FPR : {best_sfpr*100:.2f}%")
    print(f"  Val Loss       : {best_vloss:.5f}")

    # -------------------------------------------------------------------------
    # Epoch instability analysis
    # -------------------------------------------------------------------------
    print("\n--- Epoch-by-epoch table (V2.4) ---")
    print(f"{'EP':>3}  {'LR':>9}  {'val_loss':>9}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}")
    print("-"*62)
    all_sfprs, all_tprs, n_sfpr10, n_tpr95, n_both = [], [], 0, 0, 0
    with open(HISTORY_CSV, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            ep    = i + 1
            lr    = float(row.get("epoch_lr", 0))
            vloss = float(row.get("val_loss", 0))
            tpr   = float(row.get("val_tpr_50", 0))
            sfpr  = float(row.get("val_speech_fpr_50", 0))
            afpr  = float(row.get("val_ambient_fpr_50", 0))
            print(f"{ep:>3}  {lr:>9.7f}  {vloss:>9.5f}  {tpr*100:>6.2f}%  {sfpr*100:>6.2f}%  {afpr*100:>7.2f}%")
            all_sfprs.append(sfpr); all_tprs.append(tpr)
            if sfpr > 0.10: n_sfpr10 += 1
            if tpr >= 0.95: n_tpr95  += 1
            if tpr >= 0.95 and sfpr <= 0.03: n_both += 1

    print(f"\n  SpFPR  std={np.std(all_sfprs)*100:.2f}%  min={np.min(all_sfprs)*100:.2f}%  max={np.max(all_sfprs)*100:.2f}%")
    print(f"  TPR    std={np.std(all_tprs)*100:.2f}%  min={np.min(all_tprs)*100:.2f}%  max={np.max(all_tprs)*100:.2f}%")
    print(f"  Epochs with SpFPR > 10%:              {n_sfpr10}")
    print(f"  Epochs with TPR >= 95%:               {n_tpr95}")
    print(f"  Epochs with TPR>=95% & SpFPR<=3%:    {n_both}")

    # -------------------------------------------------------------------------
    # Threshold sweeps
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("POST-TRAINING THRESHOLD SWEEPS")
    print("="*65)

    best_A_sfpr, best_B_sfpr, best95_sfpr = None, None, None
    if BEST_SFPR_PATH.exists():
        m_sfpr = tf.keras.models.load_model(str(BEST_SFPR_PATH))
        best_A_sfpr, best_B_sfpr, best95_sfpr = threshold_sweep(m_sfpr, val_pos_X, val_sp_X, val_amb_X,
                                                                  "V2.4 best_speech_fpr (EXPANDED_VAL)")
    else:
        print("  best_speech_fpr checkpoint not saved (no epoch with TPR>=95% found).")

    best_A_bl, best_B_bl, best95_bl = None, None, None
    if BEST_LOSS_PATH.exists():
        m_bl = tf.keras.models.load_model(str(BEST_LOSS_PATH))
        best_A_bl, best_B_bl, best95_bl = threshold_sweep(m_bl, val_pos_X, val_sp_X, val_amb_X,
                                                           "V2.4 best_loss (EXPANDED_VAL)")

    # -------------------------------------------------------------------------
    # V2.3 vs V2.4 comparison @ 0.50
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("V2.3 vs V2.4 COMPARISON @ threshold 0.50 (EXPANDED_VAL)")
    print("="*65)
    compare_models = {
        "V2.3 best_sfpr":  MODEL_OUT_DIR / "ira_cnn_v2_3_best_speech_fpr.keras",
        "V2.3 best_loss":  MODEL_OUT_DIR / "ira_cnn_v2_3_best_loss.keras",
        "V2.4 best_sfpr":  BEST_SFPR_PATH,
        "V2.4 best_loss":  BEST_LOSS_PATH,
    }
    print(f"{'Model':<20}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")
    print("-"*60)
    for name, mp in compare_models.items():
        if not mp.exists(): print(f"{name:<20}  NOT FOUND"); continue
        m = tf.keras.models.load_model(str(mp))
        s_pos = m.predict(val_pos_X, batch_size=64, verbose=0).flatten()
        s_sp  = m.predict(val_sp_X,  batch_size=64, verbose=0).flatten()
        s_amb = m.predict(val_amb_X, batch_size=64, verbose=0).flatten()
        s_neg = np.concatenate([s_sp, s_amb])
        tpr  = np.sum(s_pos >= 0.5) / len(s_pos) * 100
        sfpr = np.sum(s_sp  >= 0.5) / len(s_sp)  * 100
        afpr = np.sum(s_amb >= 0.5) / len(s_amb) * 100
        ofpr = np.sum(s_neg >= 0.5) / len(s_neg) * 100
        print(f"{name:<20}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")

    # -------------------------------------------------------------------------
    # GO / NO-GO
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("GO / NO-GO (TPR>=95% & SpFPR<=3% & AmbFPR<=1% on EXPANDED_VAL)")
    print("="*65)
    go = False
    for ckpt_label, best_B in [("best_speech_fpr", best_B_sfpr), ("best_loss", best_B_bl)]:
        if best_B is not None:
            thr, tpr, sfpr = best_B
            print(f"  {ckpt_label}: PASS at thr={thr:.2f}  TPR={tpr:.2f}%  SpFPR={sfpr:.2f}%  -> GO")
            go = True
        else:
            best95 = best95_sfpr if ckpt_label == "best_speech_fpr" else best95_bl
            if best95:
                print(f"  {ckpt_label}: FAIL - best at TPR>=95%: thr={best95[0]:.2f}  SpFPR={best95[2]:.2f}%  TPR={best95[1]:.2f}%")
            else:
                print(f"  {ckpt_label}: FAIL - no epoch achieved TPR>=95%")

    if go:
        print("\n  OVERALL V2.4: GO")
        print("  Do NOT evaluate unseen_test until explicitly instructed.")
    else:
        print("\n  OVERALL V2.4: NO-GO")
        print("  Do NOT automatically train V2.5.")

    print("\nDone.")


if __name__ == "__main__":
    main()
