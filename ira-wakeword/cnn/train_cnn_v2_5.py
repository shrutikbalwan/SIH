# -*- coding: utf-8 -*-
"""
train_cnn_v2_5.py
=================
IRA CNN V2.5 -- Controlled Positive Augmentation Experiment

ONLY change from V2.3:
  Composition of positive backgrounds changed from 50/50 to:
  75% ambient background
  25% unrelated speech background
  (within the 85% mixed positives).

Everything else is IDENTICAL to V2.3:
  - split_manifest_v3.csv
  - Same difficulty pools (V2 scores, TRAIN-only)
  - Same batch recipe: 32 pos + 14 easy + 7 med + 4 hard + 5 amb + 2 oth
  - Constant LR = 0.001, Max epochs = 30
  - Same CNN architecture
  - Same preprocessing
  - Same EXPANDED_VAL (seed=999)
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
BEST_LOSS_PATH = MODEL_OUT_DIR / "ira_cnn_v2_5_best_loss.keras"
BEST_SFPR_PATH = MODEL_OUT_DIR / "ira_cnn_v2_5_best_speech_fpr.keras"
FINAL_PATH     = MODEL_OUT_DIR / "ira_cnn_v2_5_final.keras"

HISTORY_CSV        = REPO_ROOT / "v2_5_training_history.csv"
STATS_JSON         = REPO_ROOT / "v2_5_augmentation_stats.json"
SAMPLING_STATS_JSON= REPO_ROOT / "v2_5_sampling_stats.json"

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
SPEECH_BG_PROB = 0.25  # <-- V2.5 CHANGE: 25% speech, 75% ambient

# V2.5 batch recipe (unchanged)
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

    # V2.5 CHANGE: 25% speech background, 75% ambient background
    if sp_paths and (random.random() < SPEECH_BG_PROB or not amb_paths):
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
class V25DataGenerator(tf.keras.utils.PyDataset):
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
# Model architecture (identical to V2/V2.1/V2.2/V2.3)
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
    model = tf.keras.Model(inp, out, name="IRA_CNN_V2_5")
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

    print(f"\n--- Threshold Frontier: {label} ---")
    print(f"{'THR':>4}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")

    best_strict = None
    best_target = None
    best_at_tpr95 = None

    for thr in thresholds:
        tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
        sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
        afpr = np.sum(s_amb >= thr) / len(s_amb) * 100
        ofpr = np.sum(s_neg >= thr) / len(s_neg) * 100
        print(f"{thr:.2f}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")

        if tpr >= 95.0 and sfpr <= 2.0 and afpr <= 1.0 and best_strict is None:
            best_strict = (thr, tpr, sfpr, afpr)
        if tpr >= 95.0 and sfpr <= 3.0 and afpr <= 1.0 and best_target is None:
            best_target = (thr, tpr, sfpr, afpr)
        if tpr >= 95.0 and (best_at_tpr95 is None or sfpr < best_at_tpr95[2]):
            best_at_tpr95 = (thr, tpr, sfpr, afpr)

    print(f"\n  Strict Target (TPR>=95% & SpFPR<=2% & AmbFPR<=1%): " +
          ("PASS thr={:.2f}  TPR={:.2f}%  SpFPR={:.2f}%  AmbFPR={:.2f}%".format(*best_strict) if best_strict else "FAILED"))
    print(f"  Goal Target (TPR>=95% & SpFPR<=3% & AmbFPR<=1%): " +
          ("PASS thr={:.2f}  TPR={:.2f}%  SpFPR={:.2f}%  AmbFPR={:.2f}%".format(*best_target) if best_target else "FAILED"))
    if best_at_tpr95:
        print(f"  Best SpFPR when TPR>=95%: thr={best_at_tpr95[0]:.2f}  TPR={best_at_tpr95[1]:.2f}%  SpFPR={best_at_tpr95[2]:.2f}%  AmbFPR={best_at_tpr95[3]:.2f}%")

    return best_target


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
    val_pos    = splits["validation"]["pos"]
    val_sp     = splits["validation"]["speech"]
    val_amb    = splits["validation"]["ambient"]
    val_oth    = splits["validation"]["other"]
    test_amb   = splits["test"]["ambient"]

    print("\nBuilding V2.5 difficulty pools (TRAIN speech only)...")
    easy_sp, med_sp, hard_sp, spk_easy, spk_med, spk_hard = \
        build_difficulty_pools_from_manifest(train_sp)

    all_train_sp = easy_sp + med_sp + hard_sp

    # -------------------------------------------------------------------------
    # SANITY CHECK
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("PRE-TRAINING SANITY CHECK (V2.5)")
    print("="*65)

    # 3-batch generator dry run
    train_stats = {k: 0 for k in [
        "clean","ambient","speech","easy_snr","med_snr","hard_snr","silent_bg_rejected",
        "easy_speech_sampled","medium_speech_sampled","hard_speech_sampled",
        "ambient_sampled","other_sampled"
    ]}
    
    # Run a quick 10-batch dry run to verify the 75/25 ratio
    gen = V25DataGenerator(train_pos, easy_sp, med_sp, hard_sp, train_amb, train_oth,
                           train_amb, all_train_sp, train_stats)

    print("  Running 10-batch sanity pass to verify 75/25 ratio...")
    for i in range(10): gen[i]
    
    assert train_stats["easy_speech_sampled"]   > 0, "No easy speech in batches!"
    assert train_stats["medium_speech_sampled"]  > 0, "No medium speech in batches!"
    assert train_stats["hard_speech_sampled"]    > 0, "No hard speech in batches!"
    assert train_stats["ambient_sampled"]        > 0, "No ambient in batches!"
    assert train_stats["clean"]                  > 0, "No clean positives in batches!"
    assert train_stats["ambient"]                > 0, "No ambient-mixed positives!"
    assert train_stats["speech"]                 > 0, "No speech-mixed positives!"
    
    mixed_total = train_stats["ambient"] + train_stats["speech"]
    amb_pct = train_stats["ambient"] / mixed_total * 100
    sp_pct = train_stats["speech"] / mixed_total * 100
    
    print(f"  Positive augmentation mix: clean={train_stats['clean']}  "
          f"amb_mix={train_stats['ambient']} ({amb_pct:.1f}%)  "
          f"sp_mix={train_stats['speech']} ({sp_pct:.1f}%)")
    print(f"  Target was ~75% ambient, ~25% speech (within the mixed pool).")

    # Reset before real training
    for k in train_stats: train_stats[k] = 0
    gen._epoch_easy_counts.clear()
    gen._epoch_med_counts.clear()
    gen._epoch_hard_counts.clear()
    gen.sampling_history.clear()
    gen.on_epoch_end()

    print("\nV2.5 TRAINING PIPELINE VALIDATION: PASS")

    # Build deterministic validation arrays
    print("\nBuilding fixed deterministic validation set (seed=999)...")
    r_state  = random.getstate(); np_state = np.random.get_state()
    random.seed(999); np.random.seed(999)

    _vstats = {k: 0 for k in ["clean","ambient","speech","easy_snr","med_snr","hard_snr","silent_bg_rejected"]}
    val_pos_X = np.array([np.expand_dims(make_spectrogram(augment_positive(p, val_amb, val_sp, _vstats)), -1)
                          for p in val_pos], dtype=np.float32)
    val_sp_X  = np.array([np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1)
                          for p in val_sp], dtype=np.float32)
    val_amb_X = np.array([np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1)
                          for p in val_amb], dtype=np.float32)
    val_oth_X = np.array([np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1)
                          for p in val_oth], dtype=np.float32) if val_oth else np.zeros((0,49,N_FREQ_BINS,1), dtype=np.float32)

    random.setstate(r_state); np.random.set_state(np_state)

    val_X = np.concatenate([val_pos_X, val_sp_X, val_amb_X] + ([val_oth_X] if len(val_oth_X) else []))
    val_y = np.concatenate([np.ones(len(val_pos_X)),
                            np.zeros(len(val_sp_X)+len(val_amb_X)+len(val_oth_X))]).reshape(-1,1)
    
    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("TRAINING IRA CNN V2.5")
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
    print("POST-TRAINING THRESHOLD SWEEPS (EXPANDED_VAL)")
    print("="*65)
    
    go_target_sfpr = None
    if BEST_SFPR_PATH.exists():
        m_sfpr = tf.keras.models.load_model(str(BEST_SFPR_PATH))
        go_target_sfpr = threshold_sweep(m_sfpr, val_pos_X, val_sp_X, val_amb_X, "V2.5 best_speech_fpr")
    else:
        print("  best_speech_fpr model not found (no eligible epoch).")

    go_target_bl = None
    if BEST_LOSS_PATH.exists():
        m_bl = tf.keras.models.load_model(str(BEST_LOSS_PATH))
        go_target_bl = threshold_sweep(m_bl, val_pos_X, val_sp_X, val_amb_X, "V2.5 best_loss")
    else:
        print("  best_loss model not found.")

    # -------------------------------------------------------------------------
    # COMPARE V2.3 vs V2.5
    # -------------------------------------------------------------------------
    print("\n" + "="*65)
    print("V2.3 vs V2.5 COMPARISON (EXPANDED_VAL)")
    print("="*65)
    
    # We will compute the best operating threshold for each model (TPR >= 95, lowest SpFPR)
    compare_models = {
        "V2.3 best_loss": MODEL_OUT_DIR / "ira_cnn_v2_3_best_loss.keras",
        "V2.3 best_sfpr": MODEL_OUT_DIR / "ira_cnn_v2_3_best_speech_fpr.keras",
        "V2.5 best_loss": BEST_LOSS_PATH,
        "V2.5 best_sfpr": BEST_SFPR_PATH,
    }
    
    print(f"{'Model':<20}  {'BestThr':>7}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")
    print("-"*65)
    
    for name, mp in compare_models.items():
        if not mp.exists(): print(f"{name:<20}  NOT FOUND"); continue
        m = tf.keras.models.load_model(str(mp))
        s_pos = m.predict(val_pos_X, batch_size=64, verbose=0).flatten()
        s_sp  = m.predict(val_sp_X,  batch_size=64, verbose=0).flatten()
        s_amb = m.predict(val_amb_X, batch_size=64, verbose=0).flatten()
        s_neg = np.concatenate([s_sp, s_amb])
        
        # Find best thr
        best_thr = 0.50
        best_sfpr = float('inf')
        for thr in np.arange(0.30, 0.951, 0.01):
            tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
            sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
            if tpr >= 95.0 and sfpr < best_sfpr:
                best_sfpr = sfpr
                best_thr = thr
                
        # Calculate stats at best_thr
        tpr  = np.sum(s_pos >= best_thr) / len(s_pos) * 100
        sfpr = np.sum(s_sp  >= best_thr) / len(s_sp)  * 100
        afpr = np.sum(s_amb >= best_thr) / len(s_amb) * 100
        ofpr = np.sum(s_neg >= best_thr) / len(s_neg) * 100
        
        print(f"{name:<20}  {best_thr:>7.2f}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")

    print("\n" + "="*65)
    print("GO / NO-GO")
    print("="*65)
    if go_target_sfpr is not None or go_target_bl is not None:
        print("  OVERALL V2.5: GO")
    else:
        print("  OVERALL V2.5: NO-GO")

    print("\nDone. V2.5 complete.")

if __name__ == "__main__":
    main()
