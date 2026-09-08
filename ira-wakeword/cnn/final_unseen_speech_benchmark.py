# -*- coding: utf-8 -*-
"""
ONE-SHOT final unseen speech-negative benchmark for the INT8 deployment
candidate.

Scope limit: unseen_test contains ONLY speech negatives. This measures speech
false-positive generalization and NOTHING else -- not TPR, not ambient FPR,
not streaming FAPH, not real-world recall.

No training. No threshold tuning. No sweep. The dataset is not modified.
"""
import os, csv, json, hashlib, collections, math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO = Path(__file__).parent.parent
MANIFEST = REPO / "dataset" / "split_manifest_v3.csv"
TFLITE = REPO / "cnn" / "models" / "ira_cnn_v2_3_int8.tflite"
FREEZE_JSON = REPO / "cnn" / "ira_cnn_v2_3_int8_candidate_freeze.json"
REPORT_JSON = REPO / "cnn" / "final_unseen_speech_report.json"
PER_SPK_CSV = REPO / "cnn" / "final_unseen_per_speaker.csv"
PER_CHAP_CSV = REPO / "cnn" / "final_unseen_per_chapter.csv"

SEMANTIC_THRESHOLD = 0.40
DEV_REFERENCE_FPR = 3.30          # frozen EXPANDED_VAL INT8 @ 0.40
ENGINEERING_TARGET = 3.0          # Speech FPR <= 3%

SAMPLE_RATE = WINDOW_SAMPLES = 16000
STFT_FRAME_LEN, STFT_FRAME_STEP, STFT_FFT_LEN, N_FREQ_BINS = 480, 320, 512, 40


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


# ------------------------------------------------- deployment pipeline (exact)
def load_audio(path):
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE:
        a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a.astype(np.float32)


def pad_to_window(a):
    if len(a) < WINDOW_SAMPLES:
        return np.pad(a, (0, WINDOW_SAMPLES - len(a)))
    return a[:WINDOW_SAMPLES]


def make_spectrogram(a):
    t = tf.convert_to_tensor(a, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN,
                       frame_step=STFT_FRAME_STEP, fft_length=STFT_FFT_LEN)
    s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_FREQ_BINS]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI (Clopper-Pearson), via the beta quantile identity."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def main():
    # =====================================================================
    print("=" * 74)
    print("SECTION 1 -- FREEZE THE FINAL INT8 CANDIDATE")
    print("=" * 74)
    if not TFLITE.exists():
        raise SystemExit("STOP: %s not found" % TFLITE)

    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = float(inp["quantization"][0]), int(inp["quantization"][1])
    out_scale, out_zp = float(out["quantization"][0]), int(out["quantization"][1])

    print("  path        : %s" % TFLITE.resolve())
    print("  sha256      : %s" % sha256(TFLITE))
    print("  size        : %d bytes (%.2f KB)" % (TFLITE.stat().st_size,
                                                  TFLITE.stat().st_size / 1024))
    print("\n  INPUT   dtype=%s  scale=%.10f  zero_point=%d"
          % (inp["dtype"].__name__, in_scale, in_zp))
    print("  OUTPUT  dtype=%s  scale=%.10f  zero_point=%d"
          % (out["dtype"].__name__, out_scale, out_zp))
    print("\n  semantic threshold : %.2f" % SEMANTIC_THRESHOLD)

    # ---- derive the raw INT8 boundary MATHEMATICALLY (not from observed codes)
    print("\n  --- Firmware decision boundary, derived from scale/zero_point ---")
    print("    dequantisation:  probability = (q - zero_point) * scale")
    print("    require:         (q - (%d)) * %.10f >= %.2f" % (out_zp, out_scale, SEMANTIC_THRESHOLD))
    q_real = SEMANTIC_THRESHOLD / out_scale + out_zp
    q_min = int(math.ceil(q_real))
    print("    =>  q >= %.4f / %.10f + (%d) = %.6f"
          % (SEMANTIC_THRESHOLD, out_scale, out_zp, q_real))
    print("    =>  q >= ceil(%.6f) = %d" % (q_real, q_min))
    # int8 domain check
    if q_min < -128 or q_min > 127:
        raise SystemExit("STOP: derived boundary %d outside int8 range" % q_min)
    p_at = (q_min - out_zp) * out_scale
    p_below = (q_min - 1 - out_zp) * out_scale
    print("\n    RAW INT8 BOUNDARY: fire iff  q_out >= %d" % q_min)
    print("      q = %d  -> p = %.10f  (>= %.2f  : fires)" % (q_min, p_at, SEMANTIC_THRESHOLD))
    print("      q = %d  -> p = %.10f  (<  %.2f  : does not fire)"
          % (q_min - 1, p_below, SEMANTIC_THRESHOLD))
    print("    effective semantic threshold actually enforced: %.10f" % p_at)
    print("    (int8 cannot represent %.2f exactly; %.10f is the smallest"
          % (SEMANTIC_THRESHOLD, p_at))
    print("     representable probability satisfying p >= %.2f)" % SEMANTIC_THRESHOLD)
    print("\n    Firmware implementation (no float needed):")
    print("      int8_t q = output_tensor[0];")
    print("      bool wake = (q >= %d);" % q_min)

    freeze = {
        "tflite_path": str(TFLITE.resolve()),
        "tflite_sha256": sha256(TFLITE),
        "tflite_size_bytes": TFLITE.stat().st_size,
        "input": {"dtype": inp["dtype"].__name__, "scale": in_scale, "zero_point": in_zp,
                  "shape": [int(x) for x in inp["shape"]]},
        "output": {"dtype": out["dtype"].__name__, "scale": out_scale, "zero_point": out_zp,
                   "shape": [int(x) for x in out["shape"]]},
        "semantic_threshold": SEMANTIC_THRESHOLD,
        "raw_int8_boundary": {
            "rule": "wake iff q_out >= q_min",
            "q_min": q_min,
            "derivation": "q_min = ceil(threshold/out_scale + out_zero_point)",
            "q_min_real_valued": q_real,
            "probability_at_q_min": p_at,
            "probability_at_q_min_minus_1": p_below,
            "effective_semantic_threshold": p_at,
            "derived_from": "scale and zero_point only (not observed codes)",
        },
        "threshold_provenance": "selected on development validation only",
    }
    json.dump(freeze, open(FREEZE_JSON, "w"), indent=2)
    print("\n  freeze manifest -> %s" % FREEZE_JSON)

    # =====================================================================
    print("\n" + "=" * 74)
    print("SECTION 2 -- EVALUATION POLICY (SCOPE LIMIT)")
    print("=" * 74)
    print("  unseen_test = 1500 speech-negative clips, 15 unseen LibriSpeech speakers.")
    print("  It contains NO positives and NO ambient examples.")
    print("  It can measure ONLY: speech false-positive generalization.")
    print("  It CANNOT measure: TPR, ambient FPR, streaming FAPH, real-world recall.")
    print("  This is NOT a complete final test.")

    # =====================================================================
    print("\n" + "=" * 74)
    print("SECTION 3 -- OPENING unseen_test (ONCE)")
    print("=" * 74)
    splits = collections.defaultdict(list)
    spk_by_split = collections.defaultdict(set)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = r["split"].strip()
            splits[sp].append(r)
            s = r.get("speaker_id", "").strip()
            if s and r["label"] == "0":
                g = r.get("group", "").lower()
                if "libri" in g or "speech" in g:
                    spk_by_split[sp].add(s)

    ut = splits["unseen_test"]
    ut_spk = sorted({r.get("speaker_id", "").strip() for r in ut if r.get("speaker_id", "").strip()})
    print("  clips           : %d  (expected 1500)" % len(ut))
    print("  unique speakers : %d  (expected 15)" % len(ut_spk))
    assert len(ut) == 1500, "unseen_test is not 1500 clips"
    assert len(ut_spk) == 15, "unseen_test is not 15 speakers"
    labels = {r["label"] for r in ut}
    print("  labels present  : %s  (speech negatives only)" % sorted(labels))
    assert labels == {"0"}, "unseen_test contains non-negative labels"

    for other in ["train", "validation", "test"]:
        inter = set(ut_spk) & spk_by_split[other]
        print("  speaker overlap vs %-11s: %d  %s"
              % (other, len(inter), "OK" if not inter else "LEAKAGE " + str(inter)))
        assert not inter, "speaker leakage with " + other
    print("  ASSERTIONS: PASS")

    # =====================================================================
    print("\n" + "=" * 74)
    print("SECTION 4 -- SCORING VIA EXACT DEPLOYMENT PIPELINE (FULL INT8)")
    print("=" * 74)
    print("  16 kHz / 1 s / STFT 480-320-512 / first 40 bins / log(|.|+1e-6) /")
    print("  per-window z-score / [1,49,40,1] / int8 quantised input")
    print("  Keras model is NOT used. Threshold is NOT re-tuned.")

    def parse(p):
        stem = Path(p).stem
        head, _, _ = stem.rpartition("_neg_")
        spk, _, utt = head.partition("_")
        parts = utt.split("-")
        return spk, (parts[1] if len(parts) >= 2 else "")

    recs = []
    missing = 0
    for r in ut:
        p = Path(r["path"])
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            missing += 1
            continue
        a = pad_to_window(load_audio(p))
        x = np.expand_dims(make_spectrogram(a), -1)
        q_in = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q_in[None, ...])
        interp.invoke()
        q_out = int(interp.get_tensor(out["index"])[0, 0])
        spk, chap = parse(r["path"])
        recs.append({"path": str(r["path"]), "speaker": spk, "chapter": chap,
                     "q": q_out, "p": (q_out - out_zp) * out_scale,
                     "fire": q_out >= q_min})
    print("  scored %d clips (%d missing files)" % (len(recs), missing))
    assert missing == 0, "some unseen_test files were missing"

    P = np.array([r["p"] for r in recs])
    fire = np.array([r["fire"] for r in recs])
    n, k = len(recs), int(fire.sum())

    # =====================================================================
    print("\n" + "=" * 74)
    print("SECTION 5 -- PRIMARY RESULT")
    print("=" * 74)
    fpr = k / n * 100
    lo, hi = clopper_pearson(k, n)
    print("  total speech negatives : %d" % n)
    print("  false positives        : %d" % k)
    print("  true negatives         : %d" % (n - k))
    print("  Speech FPR             : %.4f%%" % fpr)
    print("  exact binomial 95%% CI   : [%.4f%%, %.4f%%]  (Clopper-Pearson)"
          % (lo * 100, hi * 100))
    print("\n  development reference (frozen EXPANDED_VAL, INT8 @0.40): %.2f%%"
          % DEV_REFERENCE_FPR)
    print("  absolute difference    : %+.4f percentage points" % (fpr - DEV_REFERENCE_FPR))
    print("  (reported only; nothing is altered on the basis of it)")

    # =====================================================================
    print("\n" + "=" * 74)
    print("SECTION 6 -- PER-SPEAKER GENERALIZATION")
    print("=" * 74)
    by_spk = collections.defaultdict(list)
    for r in recs:
        by_spk[r["speaker"]].append(r)
    spk_rows = []
    for s, rs in by_spk.items():
        pp = np.array([x["p"] for x in rs])
        f = sum(x["fire"] for x in rs)
        spk_rows.append({"speaker": s, "clips": len(rs), "fp": f,
                         "fpr": f / len(rs) * 100, "mean": pp.mean(),
                         "p95": np.percentile(pp, 95), "max": pp.max()})
    spk_rows.sort(key=lambda r: -r["fpr"])
    print("  %8s %7s %5s %9s %9s %9s %9s" %
          ("speaker", "clips", "FP", "FPR", "mean", "P95", "max"))
    for r in spk_rows:
        print("  %8s %7d %5d %8.2f%% %9.4f %9.4f %9.4f"
              % (r["speaker"], r["clips"], r["fp"], r["fpr"], r["mean"], r["p95"], r["max"]))

    zero = sum(1 for r in spk_rows if r["fp"] == 0)
    med = float(np.median([r["fpr"] for r in spk_rows]))
    print("\n  speakers with ZERO false positives : %d / %d" % (zero, len(spk_rows)))
    print("  median speaker FPR                 : %.2f%%" % med)
    print("  maximum speaker FPR                : %.2f%%" % spk_rows[0]["fpr"])
    print("\n  Top 5 speakers by FPR:")
    for r in spk_rows[:5]:
        print("    %8s  %d/%d = %.2f%%   max score %.4f" % (r["speaker"], r["fp"], r["clips"], r["fpr"], r["max"]))

    with open(PER_SPK_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["speaker", "clips", "fp", "fpr", "mean", "p95", "max"])
        w.writeheader()
        w.writerows(spk_rows)

    # =====================================================================
    print("\n" + "=" * 74)
    print("SECTION 7 -- SESSION / CHAPTER ANALYSIS")
    print("=" * 74)
    by_ch = collections.defaultdict(list)
    for r in recs:
        by_ch["%s/%s" % (r["speaker"], r["chapter"])].append(r)
    ch_rows = []
    for key, rs in by_ch.items():
        pp = np.array([x["p"] for x in rs])
        f = sum(x["fire"] for x in rs)
        s, c = key.split("/")
        ch_rows.append({"speaker": s, "chapter": c, "clips": len(rs), "fp": f,
                        "fpr": f / len(rs) * 100, "mean": pp.mean(), "max": pp.max()})
    ch_rows.sort(key=lambda r: (-r["fp"], -r["fpr"]))
    print("  %d speaker/chapter sessions" % len(ch_rows))
    print("  %8s %9s %7s %5s %9s %9s %9s" %
          ("speaker", "chapter", "clips", "FP", "FPR", "mean", "max"))
    for r in ch_rows:
        print("  %8s %9s %7d %5d %8.2f%% %9.4f %9.4f"
              % (r["speaker"], r["chapter"], r["clips"], r["fp"], r["fpr"], r["mean"], r["max"]))

    with open(PER_CHAP_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["speaker", "chapter", "clips", "fp", "fpr", "mean", "max"])
        w.writeheader()
        w.writerows(ch_rows)

    if k > 0:
        nz = [r for r in ch_rows if r["fp"] > 0]
        top = sorted(nz, key=lambda r: -r["fp"])
        cum = 0
        need = 0
        for r in top:
            cum += r["fp"]
            need += 1
            if cum >= k / 2:
                break
        share_clips = sum(r["clips"] for r in top[:need]) / n * 100
        print("\n  sessions with >=1 FP           : %d / %d" % (len(nz), len(ch_rows)))
        print("  sessions holding >=50%% of FPs  : %d (covering %.1f%% of clips)"
              % (need, share_clips))
        print("  concentration verdict          : %s"
              % ("CONCENTRATED" if need <= 2 and share_clips < 25 else "DIFFUSE"))
    else:
        print("\n  no false positives -- concentration analysis not applicable")

    # =====================================================================
    print("\n" + "=" * 74)
    print("SECTION 8 -- SCORE DISTRIBUTION (diagnostic only, no sweep)")
    print("=" * 74)
    print("  mean   : %.6f" % P.mean())
    print("  median : %.6f" % np.median(P))
    print("  P90    : %.6f" % np.percentile(P, 90))
    print("  P95    : %.6f" % np.percentile(P, 95))
    print("  P99    : %.6f" % np.percentile(P, 99))
    print("  max    : %.6f" % P.max())
    print("\n  %8s %8s %9s" % ("score >=", "count", "of 1500"))
    counts = {}
    for t in [0.20, 0.30, 0.40, 0.50, 0.70, 0.90]:
        c = int((P >= t).sum())
        counts[t] = c
        print("  %8.2f %8d %8.2f%%" % (t, c, c / n * 100))

    # =====================================================================
    print("\n" + "=" * 74)
    print("SECTION 9 -- FINAL SPEECH-NEGATIVE VERDICT")
    print("=" * 74)
    met = fpr <= ENGINEERING_TARGET
    print("  pre-established engineering target: Speech FPR <= %.0f%%" % ENGINEERING_TARGET)
    print("  observed Speech FPR               : %.4f%%  (%d/%d)" % (fpr, k, n))
    print("  exact binomial 95%% CI              : [%.4f%%, %.4f%%]" % (lo * 100, hi * 100))
    print()
    print("  >>> FINAL UNSEEN SPEECH TARGET %s <<<" % ("MET" if met else "NOT MET"))
    print()
    print("  Scope: speech false-positive generalization ONLY.")
    print("  unseen_test is now CONSUMED development evidence. Any future model")
    print("  work requires a new, untouched final speech set.")

    json.dump({
        "candidate": freeze,
        "n": n, "false_positives": k, "true_negatives": n - k,
        "speech_fpr_pct": fpr,
        "ci95_pct": [lo * 100, hi * 100],
        "dev_reference_fpr_pct": DEV_REFERENCE_FPR,
        "delta_pp": fpr - DEV_REFERENCE_FPR,
        "target_pct": ENGINEERING_TARGET,
        "target_met": bool(met),
        "per_speaker": spk_rows,
        "per_chapter": ch_rows,
        "score_distribution": {
            "mean": float(P.mean()), "median": float(np.median(P)),
            "P90": float(np.percentile(P, 90)), "P95": float(np.percentile(P, 95)),
            "P99": float(np.percentile(P, 99)), "max": float(P.max()),
            "counts_at": {str(t): int(c) for t, c in counts.items()},
        },
        "scope_limits": ["no positives", "no ambient", "no streaming",
                         "cannot measure TPR / ambient FPR / FAPH / real-world recall"],
    }, open(REPORT_JSON, "w"), indent=2)
    print("\n  report -> %s" % REPORT_JSON)


if __name__ == "__main__":
    main()
