# -*- coding: utf-8 -*-
"""
Evaluate the frozen V2.3 INT8 deployment model on the 66 NEW real-human "Ira"
recordings in dataset/new data set/.

FROZEN. No retraining, no model modification, no requantization, no threshold
change, no stride change, no tuning of any kind against these results. The WAVs
are opened read-only: never trimmed, normalized, resampled, or rewritten.

Primary   : 100 ms stride  (deployment policy)
Diagnostic: 10 ms stride   (run separately, never the headline)
"""
import os, csv, sys, json, hashlib, argparse
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO = Path(__file__).parent.parent
SRC = REPO / "dataset" / "new data set"
TFLITE = REPO / "cnn" / "models" / "ira_cnn_v2_3_int8.tflite"
AUDIT_CSV = SRC / "audit_new_ira_dataset.csv"

OUT_CSV = SRC / "v23_int8_100ms_results.csv"
OUT_MD = SRC / "V23_INT8_100MS_REPORT.md"
OUT_DIAG_CSV = SRC / "v23_int8_10ms_diagnostic.csv"
OUT_DIAG_MD = SRC / "V23_INT8_10MS_DIAGNOSTIC.md"

EXPECT_SHA = "b9554c054bc9a57a00e874824e5a4ab9c3254b4505ca8a3ec67835d193103de0"
EXPECT_IN_SCALE, EXPECT_IN_ZP = 0.06078097224235535, 49
EXPECT_OUT_SCALE, EXPECT_OUT_ZP = 0.00390625, -128
EXPECT_IN_SHAPE = [1, 49, 40, 1]
EXPECT_OUT_SHAPE = [1, 1]

Q_MIN = -25                 # frozen firmware decision
SR = 16000
WIN = 16000
STRIDE_PRIMARY = 1600       # 100 ms
STRIDE_DIAG = 160           # 10 ms
FRAME_LEN, FRAME_STEP, FFT_LEN, N_BINS = 480, 320, 512, 40


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo * 100, hi * 100


def offsets(n_samples, stride):
    """Window starts covering the COMPLETE recording, incl. an end-anchored tail."""
    if n_samples <= WIN:
        return [0]
    last = n_samples - WIN
    offs = list(range(0, last + 1, stride))
    if offs[-1] != last:
        offs.append(last)
    return offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic-10ms", action="store_true",
                    help="also run the 10 ms diagnostic pass (never the headline)")
    args = ap.parse_args()

    print("=" * 78)
    print("FROZEN V2.3 INT8 EVALUATION -- 66 NEW REAL-HUMAN 'IRA' RECORDINGS")
    print("=" * 78)

    # ---------------------------------------------------------- model gate
    got = sha256_file(TFLITE)
    print("  model  : %s" % TFLITE)
    print("  sha256 : %s" % got)
    if got != EXPECT_SHA:
        raise SystemExit("STOP: SHA256 MISMATCH\n  expected %s\n  got      %s" % (EXPECT_SHA, got))
    print("  SHA256 MATCHES FROZEN MODEL.")

    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_s, in_z = float(inp["quantization"][0]), int(inp["quantization"][1])
    out_s, out_z = float(out["quantization"][0]), int(out["quantization"][1])
    in_shape, out_shape = [int(x) for x in inp["shape"]], [int(x) for x in out["shape"]]

    print("\n  INPUT  dtype=%s shape=%s scale=%.17g zp=%d"
          % (inp["dtype"].__name__, in_shape, in_s, in_z))
    print("  OUTPUT dtype=%s shape=%s scale=%.17g zp=%d"
          % (out["dtype"].__name__, out_shape, out_s, out_z))
    bad = []
    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        bad.append("tensors are not INT8")
    if in_shape != EXPECT_IN_SHAPE:
        bad.append("input shape %s != %s" % (in_shape, EXPECT_IN_SHAPE))
    if out_shape != EXPECT_OUT_SHAPE:
        bad.append("output shape %s != %s" % (out_shape, EXPECT_OUT_SHAPE))
    if abs(in_s - EXPECT_IN_SCALE) > 1e-12 or in_z != EXPECT_IN_ZP:
        bad.append("input quantization mismatch")
    if abs(out_s - EXPECT_OUT_SCALE) > 1e-12 or out_z != EXPECT_OUT_ZP:
        bad.append("output quantization mismatch")
    if bad:
        raise SystemExit("STOP: " + "; ".join(bad))
    print("  MODEL SPEC VERIFIED (shapes, dtypes, quantization).")
    print("  decision rule (frozen): WAKE iff q_out >= %d   (p >= %.10f)"
          % (Q_MIN, (Q_MIN - out_z) * out_s))

    in_i, out_i = inp["index"], out["index"]

    def run_window(w):
        """Exact V2.3 preprocessing for one 16000-sample window -> q_out."""
        t = tf.convert_to_tensor(w, dtype=tf.float32)
        st = tf.signal.stft(t, frame_length=FRAME_LEN, frame_step=FRAME_STEP,
                            fft_length=FFT_LEN, pad_end=False)
        mag = tf.abs(st)
        lg = tf.math.log(mag + 1e-6)[:, :N_BINS]
        m = tf.reduce_mean(lg)
        d = tf.math.reduce_std(lg) + 1e-6
        feat = ((lg - m) / d).numpy().astype(np.float32)
        if feat.shape != (49, N_BINS):
            raise SystemExit("STOP: feature shape %s != (49, %d)" % (feat.shape, N_BINS))
        q = np.clip(np.round(feat[..., None] / in_s + in_z), -128, 127).astype(np.int8)
        interp.set_tensor(in_i, q[None, ...])
        interp.invoke()
        return int(interp.get_tensor(out_i)[0, 0])

    # ---------------------------------------------------------- audit join
    clipped_by_file = {}
    if AUDIT_CSV.exists():
        for r in csv.DictReader(open(AUDIT_CSV, encoding="utf-8")):
            clipped_by_file[r["filename"]] = (r.get("is_clipped", "").strip().lower() == "true")
        print("\n  audit CSV loaded: clipped flags for %d files" % len(clipped_by_file))
    else:
        print("\n  audit CSV not found -- clipped flag will be blank")

    # ---------------------------------------------------------- load + scan
    wavs = sorted((p for p in SRC.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"),
                  key=lambda p: str(p).lower())
    print("  recordings found: %d" % len(wavs))
    print("\n  windowing: 1 s window; files <16000 samples zero-padded ONCE;")
    print("             files >=16000 scanned every %d samples (100 ms) plus an" % STRIDE_PRIMARY)
    print("             end-anchored final window when the stride misses the tail.")
    print("  no VAD trim, no normalization, no resampling.")

    rows = []
    for p in wavs:
        a, sr = sf.read(str(p), dtype="float64", always_2d=False)
        if sr != SR:
            raise SystemExit("STOP: %s has sample rate %d, expected %d "
                             "(audit said all files were 16 kHz)" % (p.name, sr, SR))
        if a.ndim != 1:
            raise SystemExit("STOP: %s is not mono (audit said all files were mono)" % p.name)
        a = a.astype(np.float32)
        n0 = len(a)
        dur = n0 / SR
        padded = a if n0 >= WIN else np.pad(a, (0, WIN - n0))

        offs = offsets(len(padded), STRIDE_PRIMARY)
        q_max, t_max, t_first, n_fire = -128, 0.0, None, 0
        for s in offs:
            q = run_window(padded[s:s + WIN])
            if q > q_max:
                q_max, t_max = q, s / SR
            if q >= Q_MIN:
                n_fire += 1
                if t_first is None:
                    t_first = s / SR
        rows.append({
            "filename": p.name,
            "duration_s": round(dur, 4),
            "n_samples": n0,
            "zero_padded": n0 < WIN,
            "clipped": clipped_by_file.get(p.name, ""),
            "n_windows_100ms": len(offs),
            "max_q_out_100ms": q_max,
            "max_score_100ms": (q_max - out_z) * out_s,
            "t_max_score_100ms": round(t_max, 4),
            "detected_100ms": q_max >= Q_MIN,
            "first_fire_offset_100ms": "" if t_first is None else round(t_first, 4),
            "n_firing_windows_100ms": n_fire,
        })

    n = len(rows)
    tp = sum(1 for r in rows if r["detected_100ms"])
    fn = n - tp
    tpr = tp / n * 100
    lo, hi = clopper_pearson(tp, n)
    wins = np.array([r["n_windows_100ms"] for r in rows])

    # ---------------------------------------------------------- report
    print("\n" + "=" * 78)
    print("  PRIMARY RESULT -- 100 ms STRIDE")
    print("=" * 78)
    print("  total unique recordings : %d" % n)
    print("  TP                      : %d" % tp)
    print("  FN                      : %d" % fn)
    print("  TPR                     : %.4f%%" % tpr)
    print("  95%% CI (Clopper-Pearson): [%.4f%%, %.4f%%]" % (lo, hi))
    print("  total windows evaluated : %d" % int(wins.sum()))
    print("  windows per recording   : mean=%.2f median=%.1f max=%d"
          % (wins.mean(), float(np.median(wins)), int(wins.max())))

    print("\n  --- per recording ---")
    print("  %-44s %7s %4s %4s %7s %9s %4s %8s"
          % ("filename", "dur_s", "clip", "win", "max_q", "max_score", "det", "first"))
    for r in sorted(rows, key=lambda r: r["filename"]):
        print("  %-44s %7.3f %4s %4d %7d %9.6f %4s %8s"
              % (r["filename"][:44], r["duration_s"],
                 "Y" if r["clipped"] is True else ("N" if r["clipped"] is False else "?"),
                 r["n_windows_100ms"], r["max_q_out_100ms"], r["max_score_100ms"],
                 "YES" if r["detected_100ms"] else "NO",
                 r["first_fire_offset_100ms"] if r["first_fire_offset_100ms"] != "" else "-"))

    fns = [r for r in rows if not r["detected_100ms"]]
    if fns:
        print("\n  --- FALSE NEGATIVES (%d) ---" % len(fns))
        for r in sorted(fns, key=lambda r: -r["max_q_out_100ms"]):
            print("    q=%4d  score=%.6f  dur=%.3f s  %s"
                  % (r["max_q_out_100ms"], r["max_score_100ms"], r["duration_s"], r["filename"]))

    # ---------------------------------------------------------- breakdowns
    def grp(sel, label):
        g = [r for r in rows if sel(r)]
        if not g:
            return None
        t = sum(1 for r in g if r["detected_100ms"])
        glo, ghi = clopper_pearson(t, len(g))
        print("    %-16s %3d/%3d = %6.2f%%   95%% CI [%5.2f%%, %6.2f%%]"
              % (label, t, len(g), t / len(g) * 100, glo, ghi))
        return {"label": label, "TP": t, "n": len(g), "TPR": t / len(g) * 100, "ci": [glo, ghi]}

    print("\n  --- breakdown by clipping (descriptive only; no causal claim) ---")
    bd_clip = [grp(lambda r: r["clipped"] is True, "clipped"),
               grp(lambda r: r["clipped"] is False, "not clipped")]

    print("\n  --- breakdown by duration (descriptive only) ---")
    bd_dur = [grp(lambda r: r["duration_s"] < 1.0, "<1.0 s"),
              grp(lambda r: 1.0 <= r["duration_s"] < 1.3, "1.0-1.3 s"),
              grp(lambda r: 1.3 <= r["duration_s"] < 1.5, "1.3-1.5 s"),
              grp(lambda r: r["duration_s"] >= 1.5, ">=1.5 s")]

    # ---------------------------------------------------------- save
    fields = ["filename", "duration_s", "n_samples", "zero_padded", "clipped",
              "n_windows_100ms", "max_q_out_100ms", "max_score_100ms", "t_max_score_100ms",
              "detected_100ms", "first_fire_offset_100ms", "n_firing_windows_100ms"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["filename"]))

    L = []
    A = L.append
    A("# Frozen V2.3 INT8 — New Real-Human \"Ira\" Recordings (100 ms stride)\n")
    A("**Model:** `cnn/models/ira_cnn_v2_3_int8.tflite`  ")
    A("**SHA-256:** `%s` (verified)  " % got)
    A("**Decision rule:** `q_out >= %d` (p >= %.10f)  " % (Q_MIN, (Q_MIN - out_z) * out_s))
    A("**Primary stride:** 100 ms  ")
    A("**Source:** `dataset/new data set/` — %d recordings\n" % n)
    A("> Frozen evaluation. No retraining, model change, requantization, threshold")
    A("> change, or stride change. WAVs opened read-only — never trimmed, normalized,")
    A("> or resampled. Nothing tuned against these results.\n")
    A("## Primary result\n")
    A("| metric | value |")
    A("|---|---|")
    A("| Total unique recordings | %d |" % n)
    A("| TP | %d |" % tp)
    A("| FN | %d |" % fn)
    A("| **TPR** | **%.4f%%** |" % tpr)
    A("| 95%% CI (Clopper–Pearson) | [%.4f%%, %.4f%%] |" % (lo, hi))
    A("| Total windows evaluated | %d |" % int(wins.sum()))
    A("| Windows per recording (mean / median / max) | %.2f / %.1f / %d |"
      % (wins.mean(), float(np.median(wins)), int(wins.max())))
    A("")
    A("## Breakdown by clipping\n")
    A("*Descriptive only — no causal claim is made about clipping.*\n")
    A("| group | TP / total | TPR | 95% CI |")
    A("|---|---|---|---|")
    for b in bd_clip:
        if b:
            A("| %s | %d / %d | %.2f%% | [%.2f%%, %.2f%%] |"
              % (b["label"], b["TP"], b["n"], b["TPR"], b["ci"][0], b["ci"][1]))
    A("")
    A("## Breakdown by duration\n")
    A("| group | TP / total | TPR | 95% CI |")
    A("|---|---|---|---|")
    for b in bd_dur:
        if b:
            A("| %s | %d / %d | %.2f%% | [%.2f%%, %.2f%%] |"
              % (b["label"], b["TP"], b["n"], b["TPR"], b["ci"][0], b["ci"][1]))
    A("")
    if fns:
        A("## False negatives (%d)\n" % len(fns))
        A("| max q_out | max score | duration | filename |")
        A("|---|---|---|---|")
        for r in sorted(fns, key=lambda r: -r["max_q_out_100ms"]):
            A("| %d | %.6f | %.3f s | `%s` |"
              % (r["max_q_out_100ms"], r["max_score_100ms"], r["duration_s"], r["filename"]))
        A("")
    else:
        A("## False negatives\n\nNone — every recording fired.\n")
    A("## Per-recording detail\n")
    A("| filename | dur_s | clipped | windows | max q_out | max score | detected | first fire |")
    A("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["filename"]):
        A("| `%s` | %.3f | %s | %d | %d | %.6f | %s | %s |"
          % (r["filename"], r["duration_s"],
             "yes" if r["clipped"] is True else ("no" if r["clipped"] is False else "?"),
             r["n_windows_100ms"], r["max_q_out_100ms"], r["max_score_100ms"],
             "yes" if r["detected_100ms"] else "**no**",
             r["first_fire_offset_100ms"] if r["first_fire_offset_100ms"] != "" else "—"))
    A("")
    A("---\n")
    A("**Scope limit.** This set carries no speaker, device, distance, room, or session")
    A("metadata, and every file is a WhatsApp PTT capture. The only supported conclusion")
    A("is how well frozen V2.3 detects these %d recordings. It does NOT identify speaker," % n)
    A("distance, room, device, or session effects. Development/pilot data — not a")
    A("benchmark, not training data.")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n  wrote %s" % OUT_CSV)
    print("  wrote %s" % OUT_MD)

    # ---------------------------------------------------------- 10 ms diagnostic
    if args.diagnostic_10ms:
        print("\n" + "=" * 78)
        print("  DIAGNOSTIC ONLY -- 10 ms STRIDE (does NOT replace the 100 ms headline)")
        print("=" * 78)
        drows = []
        for p in wavs:
            a, sr = sf.read(str(p), dtype="float64", always_2d=False)
            a = a.astype(np.float32)
            padded = a if len(a) >= WIN else np.pad(a, (0, WIN - len(a)))
            offs = offsets(len(padded), STRIDE_DIAG)
            q_max, t_first = -128, None
            for s in offs:
                q = run_window(padded[s:s + WIN])
                if q > q_max:
                    q_max = q
                if q >= Q_MIN and t_first is None:
                    t_first = s / SR
            base = next(r for r in rows if r["filename"] == p.name)
            drows.append({"filename": p.name,
                          "detected_100ms": base["detected_100ms"],
                          "max_q_out_100ms": base["max_q_out_100ms"],
                          "n_windows_10ms": len(offs),
                          "max_q_out_10ms": q_max,
                          "max_score_10ms": (q_max - out_z) * out_s,
                          "detected_10ms": q_max >= Q_MIN,
                          "first_fire_offset_10ms": "" if t_first is None else round(t_first, 4)})
        d_tp = sum(1 for r in drows if r["detected_10ms"])
        d_tpr = d_tp / n * 100
        dlo, dhi = clopper_pearson(d_tp, n)
        rescued = [r for r in drows if not r["detected_100ms"] and r["detected_10ms"]]
        still = [r for r in drows if not r["detected_100ms"] and not r["detected_10ms"]]
        print("  10 ms TPR              : %d/%d = %.4f%%  95%% CI [%.4f%%, %.4f%%]"
              % (d_tp, n, d_tpr, dlo, dhi))
        print("  100 ms FNs recovered   : %d of %d" % (len(rescued), fn))
        print("  still FN at 10 ms      : %d" % len(still))
        print("  windows evaluated      : %d (%.1fx the 100 ms pass)"
              % (sum(r["n_windows_10ms"] for r in drows),
                 sum(r["n_windows_10ms"] for r in drows) / max(1, int(wins.sum()))))
        for r in rescued:
            print("    recovered: %s (100ms q=%d -> 10ms q=%d)"
                  % (r["filename"], r["max_q_out_100ms"], r["max_q_out_10ms"]))
        with open(OUT_DIAG_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(drows[0].keys()))
            w.writeheader()
            w.writerows(drows)
        D = ["# 10 ms Stride — DIAGNOSTIC ONLY\n",
             "**This does NOT replace the 100 ms headline result "
             "(%d/%d = %.4f%%).**\n" % (tp, n, tpr),
             "| metric | value |", "|---|---|",
             "| 10 ms TPR | %d / %d = %.4f%% |" % (d_tp, n, d_tpr),
             "| 95%% CI | [%.4f%%, %.4f%%] |" % (dlo, dhi),
             "| 100 ms FNs recovered at 10 ms | %d of %d |" % (len(rescued), fn),
             "| Still FN at 10 ms | %d |" % len(still),
             "| Windows evaluated | %d |" % sum(r["n_windows_10ms"] for r in drows), "",
             "A finer stride gives more windows and therefore more chances to fire, at",
             "proportionally higher inference cost and false-accept exposure. Deployment",
             "stride remains 100 ms; nothing here changes it.\n"]
        OUT_DIAG_MD.write_text("\n".join(D), encoding="utf-8")
        print("\n  wrote %s" % OUT_DIAG_CSV)
        print("  wrote %s" % OUT_DIAG_MD)

    print("\n  No audio modified. Model unchanged. Threshold and stride unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
