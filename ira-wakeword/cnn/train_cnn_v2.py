# -*- coding: utf-8 -*-
"""
train_cnn_v2.py
===============
Trains IRA CNN V2 with dynamic background augmentation and hard negatives.
"""

import os
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
from scipy.signal import resample_poly

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ===========================================================================
# Configuration
# ===========================================================================
REPO_ROOT   = Path(__file__).parent.parent
MODEL_DIR   = REPO_ROOT / "cnn" / "models"
MANIFEST    = REPO_ROOT / "dataset" / "split_manifest.csv"
HARD_NEG_DIR= REPO_ROOT / "streaming_false_triggers"

os.makedirs(MODEL_DIR, exist_ok=True)

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
STFT_FRAME_LENGTH = 480
STFT_FRAME_STEP = 320
STFT_FFT_LENGTH = 512
NUM_FREQ_BINS = 40

BATCH_SIZE = 64
EPOCHS = 30
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===========================================================================
# Preprocessing
# ===========================================================================
def load_audio(path: str) -> np.ndarray:
    try:
        a, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {path}: {e}")
        
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a

def vad_trim(audio: np.ndarray, top_db=25):
    yt, _ = librosa.effects.trim(audio, top_db=top_db)
    return yt

def calc_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))) + 1e-9)

def make_spectrogram(audio: np.ndarray) -> np.ndarray:
    audio_t = tf.convert_to_tensor(audio, dtype=tf.float32)
    spec = tf.signal.stft(
        audio_t,
        frame_length=STFT_FRAME_LENGTH,
        frame_step=STFT_FRAME_STEP,
        fft_length=STFT_FFT_LENGTH,
    )
    spec = tf.abs(spec)
    spec = tf.math.log(spec + 1e-6)
    spec = spec[:, :NUM_FREQ_BINS]
    mean = tf.reduce_mean(spec)
    std = tf.math.reduce_std(spec) + 1e-6
    spec = (spec - mean) / std
    return spec.numpy().astype(np.float32)

# ===========================================================================
# Data Loading & Split Manifest parsing
# ===========================================================================
def parse_dataset():
    splits = {
        'train': {'pos': [], 'neg': [], 'amb': [], 'sp': []},
        'validation': {'pos': [], 'neg': [], 'amb': [], 'sp': []},
        'test': {'pos': [], 'neg': [], 'amb': [], 'sp': []}
    }
    
    with open(MANIFEST, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            s = r['split']
            if s not in splits: continue
            
            p = str(REPO_ROOT / r['path'])
            if not os.path.exists(p): continue
            
            group = r.get('group', '').lower()
            path_str = r['path'].lower()
            label = int(r.get('label', -1))
            
            if label == 1:
                splits[s]['pos'].append(p)
            elif label == 0:
                splits[s]['neg'].append(p)
                if 'background' in group or 'ambient' in path_str or 'background' in path_str:
                    splits[s]['amb'].append(p)
                elif 'speech' in group or 'librispeech' in group or 'librispeech' in path_str:
                    splits[s]['sp'].append(p)
                    
    hard_negs = []
    if HARD_NEG_DIR.exists():
        for f in HARD_NEG_DIR.rglob("*.wav"):
            hard_negs.append(str(f))
            
    return splits, hard_negs

# ===========================================================================
# Augmentation Logic
# ===========================================================================

def get_empty_stats():
    return {
        "num_positives_used": 0,
        "clean": 0,
        "ambient": 0,
        "speech": 0,
        "easy_snr": 0,
        "med_snr": 0,
        "hard_snr": 0,
        "skipped_too_long": 0,
        "silent_bg_rejected": 0,
        "hard_negatives_sampled": 0,
        "ordinary_other_sampled": 0,
        "ordinary_ambient_sampled": 0,
        "ordinary_speech_sampled": 0
    }

def get_random_bg_segment(bg_files: list, stats: dict, target_len: int = WINDOW_SAMPLES):
    while True:
        if not bg_files:
            return np.zeros(target_len, dtype=np.float32)
        path = random.choice(bg_files)
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
            return seg
        stats["silent_bg_rejected"] += 1

def augment_positive(path, amb_files, sp_files, stats):
    orig = load_audio(path)
    active = vad_trim(orig)
    
    # We should have pre-filtered positives that are too long, so raise exception if violated
    if len(active) > WINDOW_SAMPLES:
        raise ValueError(f"Positive {path} active speech exceeds {WINDOW_SAMPLES} samples. Pre-filtering failed.")
        
    pos_rms = calc_rms(active)
    max_shift = WINDOW_SAMPLES - len(active)
    start_idx = random.randint(0, max_shift) if max_shift > 0 else 0
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    window[start_idx:start_idx+len(active)] = active
    
    # 15% clean, 85% mixed
    if random.random() < 0.15 or (not amb_files and not sp_files):
        stats["clean"] += 1
        return window
        
    # 50/50 ambient/speech
    if random.random() < 0.5 and amb_files:
        bg_list = amb_files
        stats["ambient"] += 1
    elif sp_files:
        bg_list = sp_files
        stats["speech"] += 1
    else:
        bg_list = amb_files
        stats["ambient"] += 1
        
    bg_seg = get_random_bg_segment(bg_list, stats)
    raw_bg_rms = calc_rms(bg_seg)
    
    snr_band = random.random()
    if snr_band < 0.333:
        snr = random.uniform(15, 25)
        stats["easy_snr"] += 1
    elif snr_band < 0.666:
        snr = random.uniform(5, 15)
        stats["med_snr"] += 1
    else:
        snr = random.uniform(-5, 5)
        stats["hard_snr"] += 1
        
    target_bg_rms = pos_rms / (10 ** (snr / 20.0))
    scale_factor = target_bg_rms / raw_bg_rms
    scaled_bg = bg_seg * scale_factor
    
    mix = window + scaled_bg
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix * (0.99 / peak)
    return mix

def pad_to_window(audio):
    if len(audio) < WINDOW_SAMPLES:
        return np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    return audio[:WINDOW_SAMPLES]

# ===========================================================================
# Custom Data Generator
# ===========================================================================
class IraV2Generator(tf.keras.utils.Sequence):
    def __init__(self, positives, neg_other, neg_amb, neg_sp, hard_negatives, amb_bg, sp_bg, stats, batch_size=64):
        self.positives = positives
        self.neg_other = neg_other
        self.neg_amb = neg_amb
        self.neg_sp = neg_sp
        self.hard_negatives = hard_negatives
        self.amb_bg = amb_bg
        self.sp_bg = sp_bg
        self.stats = stats
        self.batch_size = batch_size
        
        self.pos_per_batch = batch_size // 2
        self.neg_per_batch = batch_size - self.pos_per_batch
        
        # Drop remainder to keep batches strictly balanced
        self.batch_count = len(self.positives) // self.pos_per_batch
        
        # Class distribution config: Hard negs should be ~15% of the negative portion
        self.hard_neg_per_batch = int(self.neg_per_batch * 0.15) if self.hard_negatives else 0
        self.ord_neg_per_batch = self.neg_per_batch - self.hard_neg_per_batch
        
        # We perform stratified category-balanced sampling for ordinary negatives:
        # We select equally among OTHER / AMBIENT / SPEECH pools (if available).
        self.on_epoch_end()

    def __len__(self):
        return self.batch_count

    def _sample_ordinary_negatives(self, count):
        sampled = []
        pools = [
            (self.neg_other, "ordinary_other_sampled"),
            (self.neg_amb, "ordinary_ambient_sampled"),
            (self.neg_sp, "ordinary_speech_sampled")
        ]
        
        valid_pools = [(p, s) for p, s in pools if p]
        
        for _ in range(count):
            if not valid_pools: break
            pool, stat_key = random.choice(valid_pools)
            sampled.append(random.choice(pool))
            self.stats[stat_key] += 1
            
        return sampled

    def __getitem__(self, index):
        start_pos = index * self.pos_per_batch
        end_pos = start_pos + self.pos_per_batch
        batch_pos = self.positives[start_pos:end_pos]
        
        batch_neg = self._sample_ordinary_negatives(self.ord_neg_per_batch)
        batch_hard_neg = random.sample(self.hard_negatives, self.hard_neg_per_batch) if self.hard_negatives else []
        self.stats["hard_negatives_sampled"] += len(batch_hard_neg)
        
        X, y = [], []
        
        # Positives
        for p in batch_pos:
            audio = augment_positive(p, self.amb_bg, self.sp_bg, self.stats)
            spec = make_spectrogram(audio)
            X.append(spec)
            y.append(1.0)
            self.stats["num_positives_used"] += 1
            
        # Negatives
        for p in batch_neg + batch_hard_neg:
            audio = pad_to_window(load_audio(p))
            spec = make_spectrogram(audio)
            X.append(spec)
            y.append(0.0)
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X = X[..., np.newaxis]
        
        # Shuffle within batch
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    def on_epoch_end(self):
        random.shuffle(self.positives)

# ===========================================================================
# Validation Generation (Fixed)
# ===========================================================================
def generate_fixed_validation(pos_list, neg_list, amb_bg, sp_bg, stats):
    random.seed(999)
    np.random.seed(999)
    
    X, y = [], []
    for p in pos_list:
        audio = augment_positive(p, amb_bg, sp_bg, stats)
        if np.any(audio): # Exclude skipped
            X.append(make_spectrogram(audio))
            y.append(1.0)
            stats["num_positives_used"] += 1
            
    for p in neg_list:
        audio = pad_to_window(load_audio(p))
        X.append(make_spectrogram(audio))
        y.append(0.0)
        
    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.float32)
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    return X, y

# ===========================================================================
# Custom LR History Callback
# ===========================================================================
class LrHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.keys = ["loss", "val_loss", "accuracy", "val_accuracy", "precision", "val_precision", "recall", "val_recall", "learning_rate"]
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch"] + self.keys)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        row = [epoch + 1]
        for k in self.keys:
            if k == "learning_rate": row.append(lr)
            else: row.append(logs.get(k, 0))
            
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

# ===========================================================================
# Main Training Routine
# ===========================================================================
def main():
    splits, hard_negs = parse_dataset()
    
    # ------------------------------------------------------------
    # Pre-filter Positives
    # ------------------------------------------------------------
    print("Pre-filtering positives > 16000 samples...")
    def filter_positives(path_list):
        valid = []
        for p in path_list:
            audio = load_audio(p)
            active = vad_trim(audio)
            if len(active) <= WINDOW_SAMPLES:
                valid.append(p)
        return valid
        
    train_pos = filter_positives(splits['train']['pos'])
    val_pos = filter_positives(splits['validation']['pos'])
    
    skipped_train = len(splits['train']['pos']) - len(train_pos)
    skipped_val = len(splits['validation']['pos']) - len(val_pos)
    print(f"Skipped Positives: Train={skipped_train}, Validation={skipped_val}")
    
    # ------------------------------------------------------------
    # Prepare Negative Sources
    # ------------------------------------------------------------
    train_amb = splits['train']['amb']
    train_sp = splits['train']['sp']
    train_neg_other = list(set(splits['train']['neg']) - set(train_amb) - set(train_sp))
    
    val_amb = splits['validation']['amb']
    val_sp = splits['validation']['sp']
    val_neg_other = list(set(splits['validation']['neg']) - set(val_amb) - set(val_sp))
    
    val_neg_all = val_neg_other + val_amb + val_sp
    
    te_pos = splits['test']['pos']
    te_neg_all = splits['test']['neg'] + splits['test']['amb'] + splits['test']['sp']
    
    print("\n--- TRAIN SPLIT COUNTS ---")
    print(f"Positives: {len(train_pos)}")
    print(f"Ordinary Negative OTHER: {len(train_neg_other)}")
    print(f"Ordinary Negative AMBIENT: {len(train_amb)}")
    print(f"Ordinary Negative SPEECH: {len(train_sp)}")
    print(f"Hard Negatives: {len(hard_negs)}")
    
    print("\n--- VALIDATION SPLIT COUNTS ---")
    print(f"Positives: {len(val_pos)}")
    print(f"Ordinary Negative OTHER: {len(val_neg_other)}")
    print(f"Ordinary Negative AMBIENT: {len(val_amb)}")
    print(f"Ordinary Negative SPEECH: {len(val_sp)}")
    print(f"Validation Negatives ALL: {len(val_neg_all)}")
    
    print("\n--- TEST SPLIT COUNTS ---")
    print(f"Positives: {len(te_pos)}")
    print(f"Negatives ALL: {len(te_neg_all)}")
    
    # ------------------------------------------------------------
    # Strict Isolation Asserts
    # ------------------------------------------------------------
    def get_all(split_name):
        return set(splits[split_name]['pos'] + splits[split_name]['neg'] + splits[split_name]['amb'] + splits[split_name]['sp'])
        
    tr_set = get_all('train').union(set(hard_negs))
    va_set = get_all('validation')
    te_set = get_all('test')
    
    assert not tr_set.intersection(va_set), "TRAIN and VALIDATION overlap!"
    assert not tr_set.intersection(te_set), "TRAIN and TEST overlap!"
    assert not va_set.intersection(te_set), "VALIDATION and TEST overlap!"
    print("\nSPLIT ISOLATION ASSERTS PASSED.")
    
    # ------------------------------------------------------------
    # Validation Setup
    # ------------------------------------------------------------
    val_stats = get_empty_stats()
    print("Pre-generating fixed deterministic validation set...")
    X_val, y_val = generate_fixed_validation(val_pos, val_neg_all, val_amb, val_sp, val_stats)
    print(f"Validation shape: {X_val.shape}")
    
    # ------------------------------------------------------------
    # Pipeline Sanity Check before Fit
    # ------------------------------------------------------------
    train_stats = get_empty_stats()
    train_gen = IraV2Generator(
        train_pos, train_neg_other, train_amb, train_sp, hard_negs,
        train_amb, train_sp, train_stats, batch_size=BATCH_SIZE
    )
    
    print("\nRunning Pipeline Sanity Check on 3 batches...")
    all_ok = True
    for i in range(3):
        X_batch, y_batch = train_gen[i]
        if X_batch.shape[1:] != (49, 40, 1):
            print(f"FAIL: Batch {i} shape {X_batch.shape} != (_, 49, 40, 1)")
            all_ok = False
        if np.any(np.isnan(X_batch)) or np.any(np.isinf(X_batch)):
            print(f"FAIL: Batch {i} has NaNs/Infs")
            all_ok = False
            
    required_stats = [
        "clean", "ambient", "speech", "easy_snr", "med_snr", "hard_snr",
        "hard_negatives_sampled", "ordinary_ambient_sampled", "ordinary_speech_sampled"
    ]
    for s in required_stats:
        if train_stats[s] <= 0:
            print(f"FAIL: Cumulative stats missing {s}")
            all_ok = False
            
    if not all_ok:
        print("PIPELINE VALIDATION FAILED. STOPPING.")
        return
        
    print("\nV2 TRAINING PIPELINE VALIDATION: PASS\n")
    
    # Reset train_stats before fit so json contains actual training statistics only
    for k in train_stats:
        train_stats[k] = 0
    
    # ------------------------------------------------------------
    # Model Architecture
    # ------------------------------------------------------------
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(49, 40, 1)),
        tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    
    model.summary()
    print("\nClass Weights: None (Generator explicitly balances 50/50 batches)")
    
    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    hist_path = REPO_ROOT / "v2_training_history.csv"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "ira_cnn_v2_best_recall.keras"),
            monitor="val_recall", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "ira_cnn_v2_best_loss.keras"),
            monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        LrHistoryCallback(hist_path)
    ]
    
    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    print("========================================")
    print("Training IRA CNN V2")
    print("========================================\n")
    
    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        class_weight=None,
        callbacks=callbacks,
        verbose=1
    )
    
    final_path = MODEL_DIR / "ira_cnn_v2_final.keras"
    model.save(str(final_path))
    print(f"\nSaved final model to {final_path}")
    
    stats_path = REPO_ROOT / "v2_augmentation_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"train_stats": train_stats, "val_stats": val_stats}, f, indent=4)
        
if __name__ == "__main__":
    main()
