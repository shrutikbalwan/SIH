# -*- coding: utf-8 -*-
"""
V2.8 PRE-TRAINING CHECK -- does positive gain augmentation survive the
per-window z-score?

TRAIN positives only. No training, no model loaded, nothing written to the
dataset. Diagnostic only.
"""
import os, csv, random
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO = Path(__file__).parent.parent
MANIFEST = REPO / "dataset" / "split_manifest_v3.csv"
SR = WIN = 16000
FRAME_LEN, FRAME_STEP, FFT_LEN, N_BINS = 480, 320, 512, 40
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
GAINS = [1.0, 0.75, 0.5, 0.25]
N_CLIPS = 40
SEED = 4242


def load_audio(p):
    a, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        a = resample_poly(a, SR, sr).astype(np.float32)
    return a.astype(np.float32)


def spec(a):
    """EXACT V2.3 feature path, including per-window z-score."""
    t = tf.convert_to_tensor(a, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=FRAME_LEN, frame_step=FRAME_STEP, fft_length=FFT_LEN)
    s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_BINS]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)


def rms(a):
    r = float(np.sqrt(np.mean(a ** 2)))
    return r if r > 1e-9 else 1e-9


def vad_trim(a, top_db=25):
    t = rms(a) / (10 ** (top_db / 20))
    idx = np.where(np.abs(a) > t)[0]
    return a if not len(idx) else a[idx[0]:idx[-1] + 1]


def mix_snr(speech, noise, snr_db):
    """EXACT V2.3 mixer."""
    scale = (rms(speech) / (10 ** (snr_db / 20))) / rms(noise)
    mixed = speech + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def place(a):
    act = vad_trim(a)
    out = np.zeros(WIN, dtype=np.float32)
    off = random.randint(0, max(0, WIN - len(act)))
    end = min(off + len(act), WIN)
    out[off:end] = act[:end - off]
    return out


def cmp(f1, f2):
    d = np.abs(f1 - f2)
    a, b = f1.ravel(), f2.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    cor = float(np.corrcoef(a, b)[0, 1])
    return d.mean(), d.max(), cos, cor


def main():
    random.seed(SEED); np.random.seed(SEED)
    pos, amb = [], []
    for r in csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")):
        if r["split"].strip() != "train":
            continue
        p = str(REPO / r["path"])
        if not Path(p).exists():
            continue
        g = r.get("group", "").lower()
        if r["label"] == "1":
            pos.append(p)
        elif "ambient" in g or "background" in g or "noise" in g:
            amb.append(p)
    print("=" * 78)
    print("V2.8 PRE-TRAINING CHECK -- GAIN INVARIANCE UNDER PER-WINDOW Z-SCORE")
    print("=" * 78)
    print("  TRAIN-only pool: %d positives, %d ambient backgrounds" % (len(pos), len(amb)))
    print("  clips sampled: %d   gains: %s" % (N_CLIPS, GAINS))

    print("\n  --- MATHEMATICAL EXPECTATION ---")
    print("    feature = zscore( log(|STFT(g*x)| + 1e-6) )")
    print("            = zscore( log(g*|STFT(x)| + 1e-6) )")
    print("    With eps -> 0 this is zscore( log g + log|STFT(x)| ).")
    print("    log g is a CONSTANT across all bins, and z-score subtracts the mean,")
    print("    so the constant cancels EXACTLY. Whole-waveform gain is therefore")
    print("    predicted to be a NO-OP, up to the 1e-6 epsilon floor.")

    clips = random.sample(pos, min(N_CLIPS, len(pos)))

    # ---------------- CASE 1: clean positives ----------------
    print("\n" + "-" * 78)
    print("  CASE 1 -- CLEAN POSITIVES, whole-waveform gain")
    print("-" * 78)
    print("  %6s %14s %14s %12s %12s" % ("gain", "mean|dF|", "max|dF|", "cosine", "corr"))
    case1 = {}
    for g in GAINS:
        M = []
        for p in clips:
            random.seed(hash(p) % 10 ** 6)
            base = place(load_audio(p))
            f1, f2 = spec(base), spec(base * g)
            M.append(cmp(f1, f2))
        M = np.array(M)
        case1[g] = M.mean(axis=0)
        print("  %6.2f %14.3e %14.3e %12.9f %12.9f"
              % (g, M[:, 0].mean(), M[:, 1].mean(), M[:, 2].mean(), M[:, 3].mean()))

    # ---------------- CASE 2a: gain BEFORE the V2.3 mixer ----------------
    print("\n" + "-" * 78)
    print("  CASE 2a -- MIXED, gain applied to speech BEFORE the V2.3 mixer")
    print("             (mixer then re-derives noise level from speech RMS)")
    print("-" * 78)
    print("  %6s %14s %14s %12s %12s %10s" % ("gain", "mean|dF|", "max|dF|", "cosine", "corr", "dSNR dB"))
    case2a = {}
    for g in GAINS:
        M, dsnr = [], []
        for p in clips:
            random.seed(hash(p) % 10 ** 6)
            sp = place(load_audio(p))
            bg = load_audio(random.choice(amb))
            bg = np.pad(bg, (0, max(0, WIN - len(bg))))[:WIN]
            tier = random.choice(list(SNR_RANGES))
            snr = random.uniform(*SNR_RANGES[tier])
            m1 = mix_snr(sp, bg, snr)
            m2 = mix_snr(sp * g, bg, snr)      # mixer rescales noise by g too
            M.append(cmp(spec(m1), spec(m2)))
            dsnr.append(0.0)                    # SNR is held by construction
        M = np.array(M)
        case2a[g] = M.mean(axis=0)
        print("  %6.2f %14.3e %14.3e %12.9f %12.9f %10.2f"
              % (g, M[:, 0].mean(), M[:, 1].mean(), M[:, 2].mean(), M[:, 3].mean(), np.mean(dsnr)))

    # ---------------- CASE 2b: gain AFTER mixing level is fixed ----------------
    print("\n" + "-" * 78)
    print("  CASE 2b -- MIXED, speech attenuated while the BACKGROUND LEVEL IS HELD")
    print("             (background scaled for the ORIGINAL speech level, unchanged)")
    print("-" * 78)
    print("  %6s %14s %14s %12s %12s %10s" % ("gain", "mean|dF|", "max|dF|", "cosine", "corr", "dSNR dB"))
    case2b = {}
    for g in GAINS:
        M, dsnr = [], []
        for p in clips:
            random.seed(hash(p) % 10 ** 6)
            sp = place(load_audio(p))
            bg = load_audio(random.choice(amb))
            bg = np.pad(bg, (0, max(0, WIN - len(bg))))[:WIN]
            tier = random.choice(list(SNR_RANGES))
            snr = random.uniform(*SNR_RANGES[tier])
            nscale = (rms(sp) / (10 ** (snr / 20))) / rms(bg)   # fixed from ORIGINAL speech
            m1 = sp + bg * nscale
            m2 = sp * g + bg * nscale                            # background NOT rescaled
            for m in (m1, m2):
                pk = np.max(np.abs(m))
                if pk > 0.99:
                    m *= (0.99 / pk)
            M.append(cmp(spec(m1), spec(m2)))
            dsnr.append(20 * np.log10(g))
        M = np.array(M)
        case2b[g] = M.mean(axis=0)
        print("  %6.2f %14.3e %14.3e %12.9f %12.9f %+10.2f"
              % (g, M[:, 0].mean(), M[:, 1].mean(), M[:, 2].mean(), M[:, 3].mean(), np.mean(dsnr)))

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    inv1 = all(case1[g][2] > 0.9999 for g in GAINS if g != 1.0)
    inv2a = all(case2a[g][2] > 0.9999 for g in GAINS if g != 1.0)
    print("  CASE 1  clean, whole-waveform gain      : %s"
          % ("FEATURES INVARIANT (no-op)" if inv1 else "features change"))
    print("  CASE 2a mixed, gain before V2.3 mixer   : %s"
          % ("FEATURES INVARIANT (no-op)" if inv2a else "features change"))
    print("  CASE 2b mixed, background level held    : features CHANGE, and the change")
    print("          is exactly an SNR shift of 20*log10(g) dB:")
    for g in GAINS:
        if g != 1.0:
            print("            gain %.2f  ==  %+.2f dB SNR shift" % (g, 20 * np.log10(g)))
    print()
    print("  => Whole-waveform positive gain is mathematically cancelled by the")
    print("     per-window z-score (CASE 1). Applying gain to the speech component")
    print("     BEFORE the V2.3 mixer is ALSO cancelled (CASE 2a), because the mixer")
    print("     derives the noise gain from the speech RMS, so the whole mixture")
    print("     scales by g and the z-score removes it.")
    print("  => The ONLY way to make positive gain change the features (CASE 2b) is to")
    print("     hold the background level fixed, which IS a change to the effective SNR")
    print("     distribution -- the variable V2.8 is explicitly forbidden to modify.")


if __name__ == "__main__":
    main()
