# -*- coding: utf-8 -*-
"""
Streaming-style diagnostic continuation of the frozen-model positive evaluation.

Frozen candidate. No retraining, no fine-tuning, no model change, no threshold
change, no threshold sweep. Source audio opened read-only and never modified.

PRIMARY   : 1 s window slid at 100 ms stride over the complete recording.
DIAGNOSTIC: same at 10 ms stride (labelled DIAGNOSTIC ONLY, not the verdict).

A recording is TP if ANY evaluated window has q_out >= -25, FN only if NO
window fires.

Window coverage: strides start at 0 and step by the stride length. If the last
strided window does not reach the end of the recording, one extra window
anchored at (len - 16000) is appended so the COMPLETE recording is covered.
Recordings shorter than 1 s are zero-padded at the end and evaluated once.
"""
import os, csv, json, hashlib, collections
import numpy as np
import soundfile as sf
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

SRC = Path(r"D:\SIH\ira-wakeword\ira words")
REPO = Path(__file__).parent.parent
TFLITE = REPO / "cnn" / "models" / "ira_cnn_v2_3_int8.tflite"

EXPECT_SHA = "b9554c054bc9a57a00e874824e5a4ab9c3254b4505ca8a3ec67835d193103de0"
EXPECT_IN_SCALE, EXPECT_IN_ZP = 0.06078097224235535, 49
EXPECT_OUT_SCALE, EXPECT_OUT_ZP = 0.00390625, -128
Q_MIN = -25

SR = WIN = 16000
FRAME_LEN, FRAME_STEP, FFT_LEN, N_BINS = 480, 320, 512, 40
STRIDE_PRIMARY = 1600      # 100 ms
STRIDE_DIAG = 160          # 10 ms

PREV_TP, PREV_FN, PREV_TPR = 542, 289, 65.2226

OUT_JSON = REPO / "cnn" / "real_human_positive_streaming_evaluation.json"
OUT_CSV = REPO / "cnn" / "real_human_positive_streaming_results.csv"
OUT_FN = REPO / "cnn" / "real_human_positive_streaming_false_negatives.csv"
PREV_CSV = REPO / "cnn" / "real_human_positive_results.csv"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def offsets(n_samples, stride):
    """Window start offsets covering the COMPLETE recording."""
    if n_samples <= WIN:
        return [0]
    last = n_samples - WIN
    offs = list(range(0, last + 1, stride))
    if offs[-1] != last:
        offs.append(last)
    return offs


def main():
    print("=" * 78)
    print("STREAMING-STYLE FROZEN INT8 POSITIVE EVALUATION")
    print("=" * 78)

    got = sha256(TFLITE)
    print("  model  : %s" % TFLITE)
    print("  sha256 : %s" % got)
    if got != EXPECT_SHA:
        raise SystemExit("ERROR: SHA256 MISMATCH\n  expected %s\n  got      %s" % (EXPECT_SHA, got))
    print("  SHA256 MATCHES FROZEN CANDIDATE.")

    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_s, in_z = float(inp["quantization"][0]), int(inp["quantization"][1])
    out_s, out_z = float(out["quantization"][0]), int(out["quantization"][1])
    if abs(in_s - EXPECT_IN_SCALE) > 1e-12 or in_z != EXPECT_IN_ZP:
        raise SystemExit("ERROR: input quantization mismatch")
    if abs(out_s - EXPECT_OUT_SCALE) > 1e-12 or out_z != EXPECT_OUT_ZP:
        raise SystemExit("ERROR: output quantization mismatch")
    print("  input  scale=%.17g zp=%d   output scale=%.17g zp=%d  VERIFIED"
          % (in_s, in_z, out_s, out_z))
    print("  decision rule (frozen): WAKE iff q_out >= %d" % Q_MIN)
    print("  primary stride 100 ms | diagnostic stride 10 ms (DIAGNOSTIC ONLY)")

    in_idx, out_idx = inp["index"], out["index"]

    def scan(a, stride):
        """Return (q_max, t_max, t_first_fire, n_fire, n_windows)."""
        offs = offsets(len(a), stride)
        q_max, t_max, t_first, n_fire = -128, 0.0, None, 0
        for s in offs:
            w = a[s:s + WIN]
            t = tf.convert_to_tensor(w, dtype=tf.float32)
            sp = tf.signal.stft(t, frame_length=FRAME_LEN, frame_step=FRAME_STEP,
                                fft_length=FFT_LEN)
            sp = tf.math.log(tf.abs(sp) + 1e-6)[:, :N_BINS]
            m = tf.reduce_mean(sp)
            d = tf.math.reduce_std(sp) + 1e-6
            x = ((sp - m) / d).numpy().astype(np.float32)[..., None]
            q = np.clip(np.round(x / in_s + in_z), -128, 127).astype(np.int8)
            interp.set_tensor(in_idx, q[None, ...])
            interp.invoke()
            qo = int(interp.get_tensor(out_idx)[0, 0])
            if qo > q_max:
                q_max, t_max = qo, s / SR
            if qo >= Q_MIN:
                n_fire += 1
                if t_first is None:
                    t_first = s / SR
        return q_max, t_max, t_first, n_fire, len(offs)

    files = sorted([p for p in SRC.rglob("*.wav") if p.is_file()])
    print("\n  recordings found: %d" % len(files))

    rows = []
    for p in files:
        a, sr = sf.read(str(p), dtype="float64", always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)
        assert sr == SR, "unexpected sample rate in %s" % p.name
        a = a.astype(np.float32)
        dur = len(a) / SR
        pad = a if len(a) >= WIN else np.pad(a, (0, WIN - len(a)))

        q1, t1, f1, nf1, nw1 = scan(pad, STRIDE_PRIMARY)
        q2, t2, f2, nf2, nw2 = scan(pad, STRIDE_DIAG)

        rows.append({
            "filename": p.name, "duration_s": round(dur, 4),
            "n_windows_100ms": nw1, "fired_100ms": q1 >= Q_MIN,
            "max_q_out_100ms": q1, "max_score_100ms": (q1 - out_z) * out_s,
            "t_max_score_100ms": round(t1, 4),
            "t_first_fire_100ms": "" if f1 is None else round(f1, 4),
            "n_firing_windows_100ms": nf1,
            "n_windows_10ms": nw2, "fired_10ms": q2 >= Q_MIN,
            "max_q_out_10ms": q2, "max_score_10ms": (q2 - out_z) * out_s,
            "t_max_score_10ms": round(t2, 4),
            "t_first_fire_10ms": "" if f2 is None else round(f2, 4),
            "n_firing_windows_10ms": nf2,
            "audio_sha1": hashlib.sha1(np.round(a * 32767).astype(np.int16).tobytes()).hexdigest(),
        })

    seen, uniq = set(), []
    for r in rows:
        if r["audio_sha1"] not in seen:
            seen.add(r["audio_sha1"])
            uniq.append(r)
    n_dup = len(rows) - len(uniq)

    def stat(rs, key):
        n = len(rs)
        k = sum(1 for r in rs if r[key])
        lo, hi = clopper_pearson(k, n)
        return n, k, n - k, k / n * 100, lo * 100, hi * 100

    print("\n" + "=" * 78)
    print("PRIMARY RESULT -- 100 ms STRIDE STREAMING-STYLE")
    print("=" * 78)
    n, tp, fn, tpr, lo, hi = stat(uniq, "fired_100ms")
    print("  files found                : %d" % len(files))
    print("  duplicates removed         : %d" % n_dup)
    print("  unique recordings evaluated: %d" % n)
    print("  TP                         : %d" % tp)
    print("  FN                         : %d" % fn)
    print("  TPR                        : %.4f%%" % tpr)
    print("  95%% CI                      : [%.4f%%, %.4f%%]  (Clopper-Pearson exact)" % (lo, hi))
    wtot = sum(r["n_windows_100ms"] for r in uniq)
    print("  windows evaluated          : %d total, %.1f per recording (mean)"
          % (wtot, wtot / n))

    print("\n  --- TPR by duration bin (100 ms stride) ---")
    bins = [(0, 1.0, "<1.0 s"), (1.0, 1.25, "1.0-1.25 s"), (1.25, 1.5, "1.25-1.5 s"),
            (1.5, 2.0, "1.5-2.0 s"), (2.0, 1e9, ">2.0 s")]
    print("    %-12s %6s %5s %5s %9s %22s" % ("bin", "n", "TP", "FN", "TPR", "95% CI"))
    bin_out = []
    for a0, b0, lbl in bins:
        rs = [r for r in uniq if a0 <= r["duration_s"] < b0]
        if not rs:
            continue
        bn, btp, bfn, btpr, blo, bhi = stat(rs, "fired_100ms")
        bin_out.append({"bin": lbl, "n": bn, "TP": btp, "FN": bfn, "TPR": btpr,
                        "ci": [blo, bhi]})
        print("    %-12s %6d %5d %5d %8.2f%% [%7.2f%%, %7.2f%%]" % (lbl, bn, btp, bfn, btpr, blo, bhi))

    ff = [r["t_first_fire_100ms"] for r in uniq if r["t_first_fire_100ms"] != ""]
    ff = np.array([float(x) for x in ff])
    print("\n  --- first-fire time distribution (%d firing recordings) ---" % len(ff))
    print("    min=%.3f s  median=%.3f s  mean=%.3f s  max=%.3f s" % (ff.min(), np.median(ff), ff.mean(), ff.max()))
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print("    P%-3d = %.3f s" % (q, np.percentile(ff, q)))
    print("    histogram:")
    for lo_, hi_ in [(0, 0.001), (0.001, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 1e9)]:
        c = int(((ff >= lo_) & (ff < hi_)).sum())
        if c:
            lbl = "exactly 0.0" if hi_ == 0.001 else "%.1f-%.1f s" % (lo_, hi_)
            print("      %-12s %s %d" % (lbl, "#" * min(40, c // 15), c))

    ms = np.array([r["max_score_100ms"] for r in uniq])
    print("\n  --- maximum-score distribution (100 ms stride) ---")
    print("    mean=%.6f median=%.6f min=%.6f max=%.6f" % (ms.mean(), np.median(ms), ms.min(), ms.max()))
    for q in [1, 5, 10, 25, 50, 75, 90]:
        print("    P%-3d = %.6f" % (q, np.percentile(ms, q)))
    print("\n    %10s %8s %9s" % ("max score >=", "count", "of %d" % n))
    for t in [0.10, 0.20, 0.30, 0.4023437500, 0.50, 0.70, 0.90]:
        c = int((ms >= t).sum())
        tag = "  <- frozen boundary" if abs(t - 0.40234375) < 1e-9 else ""
        print("    %10.4f %8d %8.2f%%%s" % (t, c, c / n * 100, tag))

    never = sorted([r for r in uniq if not r["fired_100ms"]], key=lambda r: -r["max_q_out_100ms"])
    print("\n  --- RECORDINGS THAT NEVER FIRE (100 ms stride): %d ---" % len(never))
    print("    %6s %10s %9s %9s  %s" % ("max_q", "max_score", "dur_s", "t_max", "filename"))
    for r in never:
        print("    %6d %10.6f %9.3f %9.3f  %s"
              % (r["max_q_out_100ms"], r["max_score_100ms"], r["duration_s"],
                 r["t_max_score_100ms"], r["filename"]))

    # ---------------- comparison with max-energy single window ----------------
    print("\n" + "=" * 78)
    print("COMPARISON vs PREVIOUS MAX-ENERGY SINGLE-WINDOW RESULT")
    print("=" * 78)
    prev = {}
    if PREV_CSV.exists():
        for r in csv.DictReader(open(PREV_CSV, encoding="utf-8")):
            prev[r["audio_sha1"]] = r["detected"] == "True"
    rescued = still = was_tp_now_fn = 0
    for r in uniq:
        p_det = prev.get(r["audio_sha1"])
        if p_det is None:
            continue
        if not p_det and r["fired_100ms"]:
            rescued += 1
        elif not p_det and not r["fired_100ms"]:
            still += 1
        elif p_det and not r["fired_100ms"]:
            was_tp_now_fn += 1
    print("  previous (max-energy single window): TP=%d  FN=%d  TPR=%.4f%%" % (PREV_TP, PREV_FN, PREV_TPR))
    print("  streaming 100 ms stride            : TP=%d  FN=%d  TPR=%.4f%%" % (tp, fn, tpr))
    print("  delta                              : TPR %+.4f pp" % (tpr - PREV_TPR))
    print()
    print("  previous FNs that become TP under streaming : %d / %d (%.1f%%)"
          % (rescued, PREV_FN, rescued / PREV_FN * 100))
    print("  previous FNs that remain FN                 : %d" % still)
    print("  previous TPs that become FN                 : %d" % was_tp_now_fn)

    # ---------------- diagnostic ----------------
    dn, dtp, dfn, dtpr, dlo, dhi = stat(uniq, "fired_10ms")
    print("\n" + "=" * 78)
    print("DIAGNOSTIC ONLY -- 10 ms STRIDE (NOT THE VERDICT)")
    print("=" * 78)
    print("  TP=%d  FN=%d  TPR=%.4f%%  95%% CI [%.4f%%, %.4f%%]" % (dtp, dfn, dtpr, dlo, dhi))
    print("  windows evaluated: %d total, %.1f per recording (mean)"
          % (sum(r["n_windows_10ms"] for r in uniq), sum(r["n_windows_10ms"] for r in uniq) / n))
    print("  extra recordings rescued by the finer stride: %d" % (dtp - tp))
    print("  (10x the windows = 10x the chances to fire; sensitivity check only)")

    if tpr >= 97:
        verdict = "STRONG PASS"
    elif tpr >= 95:
        verdict = "PASS"
    elif tpr >= 90:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("  PRIMARY CLASSIFICATION (100 ms stride): %s" % verdict)
    print("  TPR = %.4f%%  95%% CI [%.4f%%, %.4f%%]" % (tpr, lo, hi))
    print("=" * 78)

    fields = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(OUT_FN, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(never)
    json.dump({
        "model": {"path": str(TFLITE), "sha256": got, "sha256_verified": True,
                  "input_scale": in_s, "input_zero_point": in_z,
                  "output_scale": out_s, "output_zero_point": out_z,
                  "decision_rule": "WAKE iff q_out >= %d" % Q_MIN},
        "source_folder": str(SRC),
        "method": {
            "primary": "1 s window, 100 ms stride, complete-coverage tail window, any-fire",
            "diagnostic": "same at 10 ms stride (DIAGNOSTIC ONLY)",
            "short_files": "zero-padded to 1 s, evaluated once",
        },
        "counts": {"files_found": len(files), "duplicates_removed": n_dup, "evaluated_unique": n},
        "primary_100ms": {"n": n, "TP": tp, "FN": fn, "TPR_pct": tpr, "ci95_pct": [lo, hi],
                          "windows_total": wtot},
        "tpr_by_duration_100ms": bin_out,
        "first_fire_seconds": {"min": float(ff.min()), "median": float(np.median(ff)),
                               "mean": float(ff.mean()), "max": float(ff.max()),
                               "P90": float(np.percentile(ff, 90)),
                               "P99": float(np.percentile(ff, 99))},
        "max_score_distribution": {"mean": float(ms.mean()), "median": float(np.median(ms)),
                                   "min": float(ms.min()), "max": float(ms.max()),
                                   "P1": float(np.percentile(ms, 1)),
                                   "P5": float(np.percentile(ms, 5)),
                                   "P10": float(np.percentile(ms, 10))},
        "never_fire": [{"filename": r["filename"], "max_q_out": r["max_q_out_100ms"],
                        "max_score": r["max_score_100ms"], "duration_s": r["duration_s"]}
                       for r in never],
        "comparison_vs_max_energy_single_window": {
            "previous_TP": PREV_TP, "previous_FN": PREV_FN, "previous_TPR_pct": PREV_TPR,
            "previous_FN_rescued": rescued, "previous_FN_remaining": still,
            "previous_TP_now_FN": was_tp_now_fn, "delta_pp": tpr - PREV_TPR},
        "diagnostic_10ms": {"TP": dtp, "FN": dfn, "TPR_pct": dtpr, "ci95_pct": [dlo, dhi],
                            "label": "DIAGNOSTIC ONLY - not the verdict"},
        "classification_primary": verdict,
    }, open(OUT_JSON, "w"), indent=2)
    print("\n  wrote %s" % OUT_JSON)
    print("  wrote %s" % OUT_CSV)
    print("  wrote %s" % OUT_FN)
    print("  source audio unmodified.")


if __name__ == "__main__":
    main()
