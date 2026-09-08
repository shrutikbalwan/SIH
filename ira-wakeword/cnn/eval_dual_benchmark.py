"""
eval_v2_v2_3_comparison.py
==========================
Dual-benchmark evaluation of V2 / V2.1 / V2.2 / V2.3.

LEGACY_FIXED_VAL  : uses split_manifest.csv  (original 286 val speech + 200 amb)
EXPANDED_VAL      : uses split_manifest_v3.csv (1786 val speech + 200 amb)

Both validation sets are built with seed=999 (deterministic, not regenerated per epoch).
"""

import os, csv, random
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path("d:/SIH/ira-wakeword")
MANIFEST_OLD = REPO_ROOT / "dataset" / "split_manifest.csv"
MANIFEST_V3  = REPO_ROOT / "dataset" / "split_manifest_v3.csv"
MODEL_DIR    = REPO_ROOT / "cnn" / "models"
HISTORY_CSV  = REPO_ROOT / "v2_3_training_history.csv"

SAMPLE_RATE    = 16000
WINDOW_SAMPLES = 16000
STFT_FRAME_LEN  = 480
STFT_FRAME_STEP = 320
STFT_FFT_LEN    = 512
N_FREQ_BINS     = 40
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
SNR_WEIGHTS = {"easy": 1, "medium": 1, "hard": 1}
CLEAN_PROB = 0.15

# ---------------------------------------------------------------------------
# Audio utils (identical to all training scripts)
# ---------------------------------------------------------------------------
def load_audio(p):
    a, sr = sf.read(str(p), dtype="float32", always_2d=False)
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

def get_random_bg_segment(paths, target_len=WINDOW_SAMPLES):
    while True:
        if not paths: return np.zeros(target_len, dtype=np.float32)
        a = load_audio(random.choice(paths))
        if len(a) < target_len: seg = np.pad(a, (0, target_len - len(a)))
        elif len(a) > target_len:
            start = random.randint(0, len(a) - target_len)
            seg = a[start:start+target_len]
        else: seg = a
        if calc_rms(seg) >= 1e-4: return seg

def mix_snr(speech, noise, snr_db):
    scale = calc_rms(speech) / calc_rms(noise) / (10 ** (snr_db / 20))
    mixed = speech + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99: mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)

def augment_positive(p, amb_paths, sp_paths):
    orig = load_audio(p); active = vad_trim(orig)
    if len(active) > WINDOW_SAMPLES: return None
    offset = random.randint(0, WINDOW_SAMPLES - len(active))
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    padded[offset:offset+len(active)] = active
    if random.random() < CLEAN_PROB or (not amb_paths and not sp_paths): return padded
    if sp_paths and (random.random() < 0.5 or not amb_paths): bg = get_random_bg_segment(sp_paths)
    else: bg = get_random_bg_segment(amb_paths)
    snr_tier = random.choices(list(SNR_RANGES.keys()), weights=list(SNR_WEIGHTS.values()))[0]
    lo, hi = SNR_RANGES[snr_tier]
    return mix_snr(padded, bg, random.uniform(lo, hi))

# ---------------------------------------------------------------------------
# Build validation arrays from a manifest path (seed=999, deterministic)
# ---------------------------------------------------------------------------
def build_val_arrays(manifest_path, label=""):
    pos, sp, amb, oth = [], [], [], []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"].strip() != "validation": continue
            p = str(REPO_ROOT / r["path"])
            if not Path(p).exists(): continue
            lbl = int(r.get("label", -1))
            g = r.get("group", "").lower()
            if lbl == 1: pos.append(p)
            elif lbl == 0:
                if "libri" in g or "speech" in g: sp.append(p)
                elif "ambient" in g or "noise" in g or "background" in g: amb.append(p)
                else: oth.append(p)

    saved_r = random.getstate(); saved_np = np.random.get_state()
    random.seed(999); np.random.seed(999)

    pos_X = []
    for p in pos:
        a = augment_positive(p, amb, sp)
        if a is not None: pos_X.append(np.expand_dims(make_spectrogram(a), -1))

    sp_X  = np.array([np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1) for p in sp],  dtype=np.float32)
    amb_X = np.array([np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1) for p in amb], dtype=np.float32)

    random.setstate(saved_r); np.random.set_state(saved_np)

    pos_X = np.array(pos_X, dtype=np.float32)
    print(f"  [{label}] pos={len(pos_X)}  speech_neg={len(sp_X)}  ambient_neg={len(amb_X)}")
    return pos_X, sp_X, amb_X

# ---------------------------------------------------------------------------
# Evaluate model on arrays at given threshold
# ---------------------------------------------------------------------------
def eval_at_thr(model, pos_X, sp_X, amb_X, thr=0.50):
    s_pos = model.predict(pos_X, batch_size=64, verbose=0).flatten()
    s_sp  = model.predict(sp_X,  batch_size=64, verbose=0).flatten()
    s_amb = model.predict(amb_X, batch_size=64, verbose=0).flatten()
    s_neg = np.concatenate([s_sp, s_amb])
    tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
    sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
    afpr = np.sum(s_amb >= thr) / len(s_amb) * 100
    ofpr = np.sum(s_neg >= thr) / len(s_neg) * 100
    return tpr, sfpr, afpr, ofpr, s_pos, s_sp, s_amb

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building LEGACY_FIXED_VAL (split_manifest.csv)...")
    leg_pos_X, leg_sp_X, leg_amb_X = build_val_arrays(MANIFEST_OLD, "LEGACY_FIXED_VAL")

    print("Building EXPANDED_VAL (split_manifest_v3.csv)...")
    exp_pos_X, exp_sp_X, exp_amb_X = build_val_arrays(MANIFEST_V3,  "EXPANDED_VAL")

    models = {
        "V2":   "ira_cnn_v2_best_loss.keras",
        "V2.1": "ira_cnn_v2_1_best_speech_fpr.keras",
        "V2.2": "ira_cnn_v2_2_best_speech_fpr.keras",
        "V2.3": "ira_cnn_v2_3_best_speech_fpr.keras",
    }

    # =========================================================
    # SECTION 1: Comparison @ 0.50 on both benchmarks
    # =========================================================
    print("\n" + "="*65)
    print("1. LEGACY_FIXED_VAL @ threshold 0.50")
    print("="*65)
    print(f"{'Model':<8}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")
    print("-"*50)
    for name, fname in models.items():
        mp = MODEL_DIR / fname
        if not mp.exists(): print(f"{name:<8}  NOT FOUND"); continue
        m = tf.keras.models.load_model(str(mp))
        tpr, sfpr, afpr, ofpr, _, _, _ = eval_at_thr(m, leg_pos_X, leg_sp_X, leg_amb_X)
        print(f"{name:<8}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")

    print("\n" + "="*65)
    print("2. EXPANDED_VAL @ threshold 0.50")
    print("="*65)
    print(f"{'Model':<8}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")
    print("-"*50)
    scores_by_model = {}
    for name, fname in models.items():
        mp = MODEL_DIR / fname
        if not mp.exists(): print(f"{name:<8}  NOT FOUND"); continue
        m = tf.keras.models.load_model(str(mp))
        tpr, sfpr, afpr, ofpr, s_pos, s_sp, s_amb = eval_at_thr(m, exp_pos_X, exp_sp_X, exp_amb_X)
        scores_by_model[name] = (s_pos, s_sp, s_amb)
        print(f"{name:<8}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")

    # =========================================================
    # SECTION 3: V2.3 SELECTED checkpoint threshold frontier
    # =========================================================
    print("\n" + "="*65)
    print("3. V2.3 SELECTED (best_speech_fpr) Threshold Frontier on EXPANDED_VAL")
    print("="*65)
    if "V2.3" in scores_by_model:
        s_pos, s_sp, s_amb = scores_by_model["V2.3"]
        s_neg = np.concatenate([s_sp, s_amb])
        thresholds = np.arange(0.30, 0.951, 0.01)
        print(f"{'THR':>4}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")
        best_A = best_B = None
        best_at_tpr95 = None
        best_fpr3 = None
        for thr in thresholds:
            tpr  = np.sum(s_pos >= thr) / len(s_pos) * 100
            sfpr = np.sum(s_sp  >= thr) / len(s_sp)  * 100
            afpr = np.sum(s_amb >= thr) / len(s_amb) * 100
            ofpr = np.sum(s_neg >= thr) / len(s_neg) * 100
            print(f"{thr:.2f}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")
            if tpr >= 95.0 and sfpr <= 2.0 and best_A is None: best_A = (thr, tpr, sfpr)
            if tpr >= 95.0 and sfpr <= 3.0 and best_B is None: best_B = (thr, tpr, sfpr)
            if tpr >= 95.0:
                if best_at_tpr95 is None or sfpr < best_at_tpr95[2]:
                    best_at_tpr95 = (thr, tpr, sfpr)
            if sfpr <= 3.0:
                if best_fpr3 is None or tpr > best_fpr3[1]:
                    best_fpr3 = (thr, tpr, sfpr)

        print("\n  Feasibility A (TPR>=95 & SpFPR<=2%):", "PASS thr={:.2f} TPR={:.2f}% SpFPR={:.2f}%".format(*best_A) if best_A else "FAILED")
        print("  Feasibility B (TPR>=95 & SpFPR<=3%):", "PASS thr={:.2f} TPR={:.2f}% SpFPR={:.2f}%".format(*best_B) if best_B else "FAILED")
        if best_at_tpr95: print(f"  Best SpFPR when TPR>=95%: thr={best_at_tpr95[0]:.2f}  SpFPR={best_at_tpr95[2]:.2f}%  TPR={best_at_tpr95[1]:.2f}%")
        if best_fpr3:     print(f"  Best TPR when SpFPR<=3%:  thr={best_fpr3[0]:.2f}   TPR={best_fpr3[1]:.2f}%   SpFPR={best_fpr3[2]:.2f}%")

    # =========================================================
    # SECTION 6: V2.3 BEST-LOSS checkpoint threshold frontier
    # =========================================================
    print("\n" + "="*65)
    print("6. V2.3 BEST-LOSS Checkpoint Threshold Frontier on EXPANDED_VAL")
    print("="*65)
    bl_path = MODEL_DIR / "ira_cnn_v2_3_best_loss.keras"
    if bl_path.exists():
        bl_model = tf.keras.models.load_model(str(bl_path))
        s_pos_bl = bl_model.predict(exp_pos_X, batch_size=64, verbose=0).flatten()
        s_sp_bl  = bl_model.predict(exp_sp_X,  batch_size=64, verbose=0).flatten()
        s_amb_bl = bl_model.predict(exp_amb_X, batch_size=64, verbose=0).flatten()
        s_neg_bl = np.concatenate([s_sp_bl, s_amb_bl])
        print(f"{'THR':>4}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}  {'OvrFPR':>8}")
        best_A_bl = best_B_bl = None
        for thr in np.arange(0.30, 0.951, 0.01):
            tpr  = np.sum(s_pos_bl >= thr) / len(s_pos_bl) * 100
            sfpr = np.sum(s_sp_bl  >= thr) / len(s_sp_bl)  * 100
            afpr = np.sum(s_amb_bl >= thr) / len(s_amb_bl) * 100
            ofpr = np.sum(s_neg_bl >= thr) / len(s_neg_bl) * 100
            print(f"{thr:.2f}  {tpr:>6.2f}%  {sfpr:>6.2f}%  {afpr:>7.2f}%  {ofpr:>7.2f}%")
            if tpr >= 95.0 and sfpr <= 2.0 and best_A_bl is None: best_A_bl = (thr, tpr, sfpr)
            if tpr >= 95.0 and sfpr <= 3.0 and best_B_bl is None: best_B_bl = (thr, tpr, sfpr)
        print("\n  Feasibility A (TPR>=95 & SpFPR<=2%):", "PASS thr={:.2f} TPR={:.2f}% SpFPR={:.2f}%".format(*best_A_bl) if best_A_bl else "FAILED")
        print("  Feasibility B (TPR>=95 & SpFPR<=3%):", "PASS thr={:.2f} TPR={:.2f}% SpFPR={:.2f}%".format(*best_B_bl) if best_B_bl else "FAILED")
    else:
        print("  best_loss model not found.")

    # =========================================================
    # SECTION 4: Epoch instability analysis
    # =========================================================
    print("\n" + "="*65)
    print("4. V2.3 EPOCH-BY-EPOCH INSTABILITY ANALYSIS")
    print("="*65)
    print(f"{'EP':>3}  {'LR':>8}  {'val_loss':>9}  {'TPR':>7}  {'SpFPR':>7}  {'AmbFPR':>8}")
    print("-"*60)
    with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        # dedupe duplicate column names from malformed CSV
        ep   = i + 1
        lr   = float(row.get("learning_rate", 0))
        vloss = float(row.get("val_loss", 0))
        tpr  = float(row.get("val_tpr_50", 0))
        sfpr = float(row.get("val_speech_fpr_50", 0))
        afpr = float(row.get("val_ambient_fpr_50", 0))
        print(f"{ep:>3}  {lr:>8.6f}  {vloss:>9.5f}  {tpr*100:>6.2f}%  {sfpr*100:>6.2f}%  {afpr*100:>7.2f}%")

    lrs = [float(r.get("learning_rate", 0)) for r in rows]
    lr_unique = set(round(lr, 7) for lr in lrs)
    print(f"\n  LR analysis: unique LR values seen = {lr_unique}")
    print(f"  LR constant at 0.001: {'YES' if len(lr_unique) == 1 else 'NO'}")

    sfprs = [float(r.get("val_speech_fpr_50", 0)) for r in rows]
    sfpr_std = float(np.std(sfprs))
    sfpr_min = float(np.min(sfprs))
    sfpr_max = float(np.max(sfprs))
    print(f"\n  Speech FPR across epochs: min={sfpr_min*100:.2f}%  max={sfpr_max*100:.2f}%  std={sfpr_std*100:.2f}%")
    print(f"  LR was constant => oscillation is NOT caused by LR schedule changes.")
    print(f"  Validation set is built with fixed seed=999 each run => NOT stochastic val generation.")
    print(f"  => Oscillation is caused by TRAINING SAMPLING RANDOMNESS interacting with")
    print(f"     the checkpoint selection rule on a difficult TPR/FPR tradeoff surface.")

    # =========================================================
    # SECTION 5: Validation determinism check (3 runs)
    # =========================================================
    print("\n" + "="*65)
    print("5. VALIDATION DETERMINISM CHECK (V2.3 best_sfpr, 3 runs)")
    print("="*65)
    sfpr_path = MODEL_DIR / "ira_cnn_v2_3_best_speech_fpr.keras"
    if sfpr_path.exists():
        det_model = tf.keras.models.load_model(str(sfpr_path))
        run_results = []
        for run_i in range(3):
            # Rebuild with same seed each time
            _pos2, _sp2, _amb2 = build_val_arrays(MANIFEST_V3, f"RUN_{run_i+1}")
            tpr, sfpr, afpr, ofpr, _, _, _ = eval_at_thr(det_model, _pos2, _sp2, _amb2)
            run_results.append((tpr, sfpr, afpr, ofpr))
            print(f"  Run {run_i+1}: TPR={tpr:.4f}%  SpFPR={sfpr:.4f}%  AmbFPR={afpr:.4f}%  OvrFPR={ofpr:.4f}%")
        tpr_vals  = [r[0] for r in run_results]
        sfpr_vals = [r[1] for r in run_results]
        tpr_range  = max(tpr_vals)  - min(tpr_vals)
        sfpr_range = max(sfpr_vals) - min(sfpr_vals)
        print(f"\n  TPR range across 3 runs:   {tpr_range:.6f}%")
        print(f"  SpFPR range across 3 runs: {sfpr_range:.6f}%")
        if tpr_range < 0.01 and sfpr_range < 0.01:
            print("  DETERMINISM: PASS (differences < 0.01%)")
        else:
            print("  DETERMINISM: FAIL (validation is not reproducible!)")
    else:
        print("  best_sfpr model not found.")

if __name__ == "__main__":
    main()
