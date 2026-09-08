# -*- coding: utf-8 -*-
"""
Stage 7 -- Correct pool membership (the real pools are defined by V2 BASELINE
scores, not V2.3) and establish the TRAIN base rate, so we can say whether the
phonetic pattern actually enriches for hard negatives.

READ-ONLY.
"""
import os, csv, json, collections, random
import numpy as np
import soundfile as sf
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO = Path(__file__).parent.parent
DIAG = REPO / "diagnostics"
MANIFEST = REPO / "dataset" / "split_manifest_v3.csv"
V2 = REPO / "cnn" / "models" / "ira_cnn_v2_best_loss.keras"
V23 = REPO / "cnn" / "models" / "ira_cnn_v2_3_best_loss.keras"

SR = CLIP = 16000
EASY_T, HARD_T = 0.10, 0.50
STFT_FRAME_LEN, STFT_FRAME_STEP, STFT_FFT_LEN, N_FREQ_BINS = 480, 320, 512, 40
random.seed(99)


def spec(a):
    t = tf.convert_to_tensor(a, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN, frame_step=STFT_FRAME_STEP,
                       fft_length=STFT_FFT_LEN)
    s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_FREQ_BINS]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)


def load(p):
    wav = Path(p)
    if not wav.is_absolute():
        wav = REPO / wav
    a, _ = sf.read(str(wav), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if len(a) < CLIP:
        a = np.pad(a, (0, CLIP - len(a)))
    return np.expand_dims(spec(a[:CLIP]), -1)


def pool_of(s):
    return "easy" if s < EASY_T else ("medium" if s < HARD_T else "hard")


def main():
    cands = list(csv.DictReader(open(DIAG / "train_only_confuser_candidates.csv",
                                     encoding="utf-8")))
    print("=" * 74)
    print("STAGE 7 -- POOL MEMBERSHIP (V2 BASELINE SCORES) AND TRAIN BASE RATE")
    print("=" * 74)

    train = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"].strip() != "train" or r["label"] != "0":
                continue
            g = r.get("group", "").lower()
            if "libri" in g or "speech" in g:
                train.append(r["path"])
    cand_paths = {c["path"] for c in cands}
    control = random.sample([p for p in train if p not in cand_paths], 1000)

    m2 = tf.keras.models.load_model(str(V2))
    m23 = tf.keras.models.load_model(str(V23))

    def score(paths, tag):
        X = np.array([load(p) for p in paths], dtype=np.float32)
        return (m2.predict(X, batch_size=128, verbose=0).flatten(),
                m23.predict(X, batch_size=128, verbose=0).flatten())

    print("\n  Scoring %d pattern candidates and %d random TRAIN controls..."
          % (len(cands), len(control)))
    c_v2, c_v23 = score([c["path"] for c in cands], "cand")
    k_v2, k_v23 = score(control, "control")

    for c, a, b in zip(cands, c_v2, c_v23):
        c["v2_baseline_score"] = round(float(a), 6)
        c["pool_v2_baseline"] = pool_of(a)
        c["v2_3_score"] = round(float(b), 6)

    print("\n  --- SECTION 8: pool membership by the ACTUAL (V2 baseline) pool rule ---")
    pc = collections.Counter(c["pool_v2_baseline"] for c in cands)
    kc = collections.Counter(pool_of(s) for s in k_v2)
    print("    %-8s %22s %26s %10s" % ("pool", "pattern candidates", "random TRAIN control", "enrich"))
    for k in ["easy", "medium", "hard"]:
        pa = pc.get(k, 0) / len(cands) * 100
        ka = kc.get(k, 0) / len(control) * 100
        print("    %-8s %8d  (%6.1f%%)   %10d  (%6.1f%%)   %8.2fx"
              % (k, pc.get(k, 0), pa, kc.get(k, 0), ka, (pa / ka) if ka else float("nan")))

    print("\n    candidates      : V2 mean=%.4f median=%.4f  |  V2.3 mean=%.4f median=%.4f"
          % (c_v2.mean(), np.median(c_v2), c_v23.mean(), np.median(c_v23)))
    print("    random control  : V2 mean=%.4f median=%.4f  |  V2.3 mean=%.4f median=%.4f"
          % (k_v2.mean(), np.median(k_v2), k_v23.mean(), np.median(k_v23)))

    # does the pattern find anything V2.3 still gets wrong that pools miss?
    hi = [(c, b) for c, b in zip(cands, c_v23) if b >= 0.43]
    hi_ctrl = int((k_v23 >= 0.43).sum())
    print("\n    V2.3 scores >=0.43 (would be an FP if unseen):")
    print("      pattern candidates : %d/%d = %.1f%%" % (len(hi), len(cands), len(hi) / len(cands) * 100))
    print("      random TRAIN       : %d/%d = %.1f%%" % (hi_ctrl, len(control), hi_ctrl / len(control) * 100))
    if hi_ctrl:
        print("      enrichment         : %.2fx"
              % ((len(hi) / len(cands)) / (hi_ctrl / len(control))))

    fields = list(cands[0].keys())
    with open(DIAG / "train_only_confuser_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(cands, key=lambda c: -float(c["v2_3_score"])))
    print("\n    updated train_only_confuser_candidates.csv with V2 baseline pool labels")

    json.dump({
        "candidates": len(cands),
        "pool_candidates": dict(pc),
        "pool_control": dict(kc),
        "cand_v2_mean": float(c_v2.mean()), "ctrl_v2_mean": float(k_v2.mean()),
        "cand_v23_ge_043": len(hi), "ctrl_v23_ge_043": hi_ctrl,
    }, open(DIAG / "train_confuser_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
