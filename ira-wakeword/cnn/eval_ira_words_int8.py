# -*- coding: utf-8 -*-
"""
STEP 2 -- Frozen INT8 positive evaluation on the new real-human 'Ira' recordings.

Frozen candidate. No training, no fine-tuning, no model change, no threshold
selection. The source folder is opened read-only and never modified.

WINDOWING METHOD (documented before results; chosen from structure, not score)
-----------------------------------------------------------------------------
Recordings are 0.673-3.313 s (median 1.234 s); the model input is exactly 1 s.
A naive first-second crop would truncate the VAD-active span in 685/836 files
(82%), so it is disqualified.

Primary method -- MAX-ENERGY 1-SECOND WINDOW:
  For each recording, choose the 1-second window whose sum of squares is
  maximal, evaluated at EVERY sample offset (computed exactly via a cumulative
  sum of squares, not a subsampled search). Ties resolve to the earliest
  offset. Files shorter than 1 s are zero-padded at the end.
  This is deterministic, identical for every file, independent of the model,
  and yields exactly ONE decision per recording.

Secondary DIAGNOSTIC ONLY -- sliding window (10 ms hop, fire if any window
fires). Reported for deployment context. It does NOT set the verdict, because
multiple windows per file give multiple chances to fire.
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
Q_MIN = -25                      # frozen firmware rule: WAKE iff q_out >= -25

SR = WIN = 16000
FRAME_LEN, FRAME_STEP, FFT_LEN, N_BINS = 480, 320, 512, 40
HOP_DIAG = 160                   # 10 ms, diagnostic sliding window only

OUT_JSON = REPO / "cnn" / "real_human_positive_evaluation.json"
OUT_CSV = REPO / "cnn" / "real_human_positive_results.csv"
OUT_FN = REPO / "cnn" / "real_human_false_negatives.csv"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def spec(a):
    t = tf.convert_to_tensor(a, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=FRAME_LEN, frame_step=FRAME_STEP, fft_length=FFT_LEN)
    s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_BINS]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)


def max_energy_offset(a):
    """Exact argmax over every sample offset of the 1-second window energy."""
    if len(a) <= WIN:
        return 0
    cs = np.concatenate([[0.0], np.cumsum(a.astype(np.float64) ** 2)])
    e = cs[WIN:] - cs[:-WIN]          # energy of window starting at each offset
    return int(np.argmax(e))          # np.argmax -> earliest on ties


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def main():
    print("=" * 76)
    print("STEP 2 -- FROZEN INT8 POSITIVE EVALUATION")
    print("=" * 76)

    # ---------------- integrity gate ----------------
    got = sha256(TFLITE)
    print("  model  : %s" % TFLITE)
    print("  sha256 : %s" % got)
    if got != EXPECT_SHA:
        raise SystemExit("ERROR: SHA256 MISMATCH\n  expected %s\n  got      %s\n"
                         "Refusing to evaluate." % (EXPECT_SHA, got))
    print("  SHA256 MATCHES FROZEN CANDIDATE.")

    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_s, in_z = float(inp["quantization"][0]), int(inp["quantization"][1])
    out_s, out_z = float(out["quantization"][0]), int(out["quantization"][1])
    print("\n  input  scale=%.17g zero_point=%d  (expected %.17g / %d)"
          % (in_s, in_z, EXPECT_IN_SCALE, EXPECT_IN_ZP))
    print("  output scale=%.17g zero_point=%d  (expected %.17g / %d)"
          % (out_s, out_z, EXPECT_OUT_SCALE, EXPECT_OUT_ZP))
    bad = []
    if abs(in_s - EXPECT_IN_SCALE) > 1e-12 or in_z != EXPECT_IN_ZP:
        bad.append("input quantization")
    if abs(out_s - EXPECT_OUT_SCALE) > 1e-12 or out_z != EXPECT_OUT_ZP:
        bad.append("output quantization")
    if bad:
        raise SystemExit("ERROR: %s does not match expected values. Refusing." % ", ".join(bad))
    print("  QUANTIZATION PARAMETERS VERIFIED.")
    print("\n  decision rule (frozen, not re-derived): WAKE iff q_out >= %d" % Q_MIN)
    print("    q=%d -> p=%.10f fires;  q=%d -> p=%.10f does not fire"
          % (Q_MIN, (Q_MIN - out_z) * out_s, Q_MIN - 1, (Q_MIN - 1 - out_z) * out_s))

    def run(x1s):
        q = np.clip(np.round(spec(x1s)[..., None] / in_s + in_z), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None, ...])
        interp.invoke()
        return int(interp.get_tensor(out["index"])[0, 0])

    # ---------------- evaluate ----------------
    files = sorted([p for p in SRC.rglob("*.wav") if p.is_file()])
    print("\n  recordings found: %d" % len(files))
    print("  windowing (primary): exact max-energy 1-second window, all offsets")
    print("  windowing (diagnostic): sliding 1 s window, 10 ms hop, any-fire")

    rows = []
    for p in files:
        a, sr = sf.read(str(p), dtype="float64", always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)
        assert sr == SR, "unexpected sample rate %d in %s" % (sr, p.name)
        a = a.astype(np.float32)
        dur = len(a) / SR
        pad = a if len(a) >= WIN else np.pad(a, (0, WIN - len(a)))

        off = max_energy_offset(pad)
        q_main = run(pad[off:off + WIN])

        # diagnostic sliding window
        q_best, off_best = q_main, off
        n_win = 1
        if len(pad) > WIN:
            n_win = 0
            for s in range(0, len(pad) - WIN + 1, HOP_DIAG):
                n_win += 1
                qq = run(pad[s:s + WIN])
                if qq > q_best:
                    q_best, off_best = qq, s
            if (len(pad) - WIN) % HOP_DIAG:
                n_win += 1
                qq = run(pad[len(pad) - WIN:])
                if qq > q_best:
                    q_best, off_best = qq, len(pad) - WIN

        rows.append({
            "name": p.name, "path": str(p), "duration_s": round(dur, 4),
            "window_offset_s": round(off / SR, 4),
            "q_out": q_main, "score": (q_main - out_z) * out_s,
            "detected": q_main >= Q_MIN,
            "diag_best_q": q_best, "diag_best_score": (q_best - out_z) * out_s,
            "diag_best_offset_s": round(off_best / SR, 4),
            "diag_any_fire": q_best >= Q_MIN, "diag_n_windows": n_win,
            "audio_sha1": hashlib.sha1(np.round(a * 32767).astype(np.int16).tobytes()).hexdigest(),
        })

    # ---------------- de-duplicate for the headline statistic ----------------
    seen, unique = set(), []
    for r in rows:
        if r["audio_sha1"] not in seen:
            seen.add(r["audio_sha1"])
            unique.append(r)
    n_dup = len(rows) - len(unique)

    def stats(rs, key="detected"):
        n = len(rs)
        k = sum(1 for r in rs if r[key])
        lo, hi = clopper_pearson(k, n)
        return n, k, n - k, k / n * 100, lo * 100, hi * 100

    print("\n" + "=" * 76)
    print("STEP 3 -- RESULTS")
    print("=" * 76)
    print("  total recordings found     : %d" % len(files))
    print("  readable / valid positives : %d" % len(rows))
    print("  byte/audio-identical dups  : %d" % n_dup)
    print("  UNIQUE recordings evaluated: %d   <- headline denominator" % len(unique))

    n, tp, fn, tpr, lo, hi = stats(unique)
    print("\n  --- PRIMARY (max-energy 1-second window, one decision per file) ---")
    print("    evaluated : %d" % n)
    print("    TP        : %d" % tp)
    print("    FN        : %d" % fn)
    print("    TPR       : %.4f%%" % tpr)
    print("    95%% CI     : [%.4f%%, %.4f%%]  (Clopper-Pearson exact)" % (lo, hi))

    an, atp, afn, atpr, alo, ahi = stats(rows)
    print("\n    including the %d duplicates (all %d files): TPR=%.4f%% [%.4f%%, %.4f%%]"
          % (n_dup, an, atpr, alo, ahi))

    dn, dtp, dfn, dtpr, dlo, dhi = stats(unique, "diag_any_fire")
    print("\n  --- DIAGNOSTIC ONLY (sliding window, any-fire) -- NOT the verdict ---")
    print("    TP=%d  FN=%d  TPR=%.4f%%  [%.4f%%, %.4f%%]" % (dtp, dfn, dtpr, dlo, dhi))
    print("    (multiple windows per file = multiple chances to fire; shown for")
    print("     deployment context only, it does not set the classification)")

    P = np.array([r["score"] for r in unique])
    Q = np.array([r["q_out"] for r in unique])
    print("\n  --- score distribution (primary window, %d unique) ---" % len(unique))
    print("    mean=%.6f median=%.6f  min=%.6f max=%.6f" % (P.mean(), np.median(P), P.min(), P.max()))
    for q in [1, 5, 10, 25, 50, 75, 90]:
        print("    P%-3d = %.6f  (q_out=%d)" % (q, np.percentile(P, q), int(np.percentile(Q, q))))
    print("\n    %10s %8s %9s" % ("score >=", "count", "of %d" % len(unique)))
    for t in [0.10, 0.20, 0.30, 0.4023437500, 0.50, 0.70, 0.90]:
        c = int((P >= t).sum())
        tag = "  <- frozen decision boundary" if abs(t - 0.40234375) < 1e-9 else ""
        print("    %10.4f %8d %8.2f%%%s" % (t, c, c / len(unique) * 100, tag))

    fns = sorted([r for r in unique if not r["detected"]], key=lambda r: -r["q_out"])
    print("\n  --- FALSE NEGATIVES (%d) ---" % len(fns))
    if fns:
        print("    %6s %10s %9s %9s  %s" % ("q_out", "score", "dur_s", "win_off", "filename"))
        for r in fns:
            print("    %6d %10.6f %9.3f %9.3f  %s"
                  % (r["q_out"], r["score"], r["duration_s"], r["window_offset_s"], r["name"]))
    else:
        print("    none")

    # ---------------- speaker / condition ----------------
    print("\n  --- per-speaker ---")
    print("    Speaker IDs are NOT determinable: filenames are WhatsApp export")
    print("    timestamps with no speaker field, no per-speaker folders, no sidecar")
    print("    metadata. Per-speaker breakdown cannot be produced.")

    print("\n  --- by available metadata (WhatsApp export date / message kind) ---")
    print("    NOTE: these are transport artefacts, not recording conditions.")
    groups = collections.defaultdict(list)
    for r in unique:
        day = r["name"].split(" at ")[0].replace("WhatsApp Audio ", "").replace("WhatsApp Ptt ", "") \
            if " at " in r["name"] else "unknown"
        kind = "Ptt" if "Ptt" in r["name"] else "Audio"
        groups[(day, kind)].append(r)
    print("    %-14s %-6s %7s %5s %5s %9s" % ("export date", "kind", "clips", "TP", "FN", "TPR"))
    for k in sorted(groups):
        rs = groups[k]
        t = sum(1 for r in rs if r["detected"])
        print("    %-14s %-6s %7d %5d %5d %8.2f%%" % (k[0], k[1], len(rs), t, len(rs) - t, t / len(rs) * 100))

    # ---------------- verdict ----------------
    if tpr >= 97:
        verdict = "STRONG PASS"
    elif tpr >= 95:
        verdict = "PASS"
    elif tpr >= 90:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 76)
    print("  CLASSIFICATION: %s   (TPR = %.4f%%, 95%% CI [%.4f%%, %.4f%%])"
          % (verdict, tpr, lo, hi))
    print("=" * 76)

    # ---------------- save ----------------
    fields = ["name", "path", "duration_s", "window_offset_s", "q_out", "score", "detected",
              "diag_best_q", "diag_best_score", "diag_best_offset_s", "diag_any_fire",
              "diag_n_windows", "audio_sha1"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(OUT_FN, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(fns)
    json.dump({
        "model": {"path": str(TFLITE), "sha256": got, "sha256_verified": True,
                  "input_scale": in_s, "input_zero_point": in_z,
                  "output_scale": out_s, "output_zero_point": out_z,
                  "decision_rule": "WAKE iff q_out >= %d" % Q_MIN,
                  "effective_threshold": (Q_MIN - out_z) * out_s},
        "source_folder": str(SRC),
        "windowing": {
            "primary": "exact max-energy 1-second window over all sample offsets",
            "diagnostic": "sliding 1s window, 10ms hop, any-fire (not the verdict)",
            "naive_first_second_rejected_because":
                "would truncate the VAD-active span in 685/836 files (82%)"},
        "counts": {"files_found": len(files), "valid": len(rows),
                   "duplicates_removed": n_dup, "evaluated_unique": len(unique)},
        "primary": {"n": n, "TP": tp, "FN": fn, "TPR_pct": tpr, "ci95_pct": [lo, hi]},
        "including_duplicates": {"n": an, "TP": atp, "FN": afn, "TPR_pct": atpr,
                                 "ci95_pct": [alo, ahi]},
        "diagnostic_any_fire": {"TP": dtp, "FN": dfn, "TPR_pct": dtpr, "ci95_pct": [dlo, dhi]},
        "score_distribution": {"mean": float(P.mean()), "median": float(np.median(P)),
                               "min": float(P.min()), "max": float(P.max()),
                               "P1": float(np.percentile(P, 1)), "P5": float(np.percentile(P, 5)),
                               "P10": float(np.percentile(P, 10)), "P25": float(np.percentile(P, 25))},
        "false_negatives": [{"name": r["name"], "q_out": r["q_out"], "score": r["score"],
                             "duration_s": r["duration_s"]} for r in fns],
        "speaker_ids_determinable": False,
        "classification": verdict,
    }, open(OUT_JSON, "w"), indent=2)
    print("\n  wrote %s" % OUT_JSON)
    print("  wrote %s" % OUT_CSV)
    print("  wrote %s" % OUT_FN)
    print("  source folder unmodified.")


if __name__ == "__main__":
    main()
