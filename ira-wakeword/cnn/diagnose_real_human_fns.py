# -*- coding: utf-8 -*-
"""
DIAGNOSTIC ONLY -- why do 90 real-human positives fail the frozen INT8 model?

No training, no fine-tuning, no threshold change, no threshold sweep, no model
modification. Source audio opened read-only. Evaluation recordings are NOT
added to training.
"""
import os, csv, json, hashlib, collections, math
import numpy as np
import soundfile as sf
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO = Path(__file__).parent.parent
CNN = REPO / "cnn"
SRC = Path(r"D:\SIH\ira-wakeword\ira words")
TFLITE = CNN / "models" / "ira_cnn_v2_3_int8.tflite"
EXPECT_SHA = "b9554c054bc9a57a00e874824e5a4ab9c3254b4505ca8a3ec67835d193103de0"

AUDIT_NEW = CNN / "new_real_positive_audit_irawords.csv"   # ira words
AUDIT_OLD = CNN / "new_real_positive_audit.csv"            # today ira wav (WhatsApp names)
STREAM = CNN / "real_human_positive_streaming_results.csv"

OUT_JSON = CNN / "real_human_fn_diagnostic.json"
OUT_CSV = CNN / "real_human_fn_diagnostic.csv"
OUT_REVIEW = CNN / "real_human_hard_fn_review.csv"

SR = WIN = 16000
FRAME_LEN, FRAME_STEP, FFT_LEN, N_BINS = 480, 320, 512, 40
Q_MIN = -25
CLIP_T = 0.999


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def mannwhitney(x, y):
    """U test + rank-biserial effect size + Cliff's delta. Returns (p, rb)."""
    from scipy.stats import mannwhitneyu
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan")
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    rb = 2 * u / (len(x) * len(y)) - 1     # rank-biserial == Cliff's delta
    return float(p), float(rb)


def describe(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "P10": float(np.percentile(v, 10)), "P25": float(np.percentile(v, 25)),
            "P75": float(np.percentile(v, 75)), "P90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max())}


def vad_span(a, top_db=25):
    r = float(np.sqrt(np.mean(a ** 2))) or 1e-9
    t = r / (10 ** (top_db / 20))
    idx = np.where(np.abs(a) > t)[0]
    return (int(idx[0]), int(idx[-1])) if len(idx) else (0, len(a) - 1)


def props(a):
    n = len(a)
    peak = float(np.max(np.abs(a))) if n else 0.0
    rms = float(np.sqrt(np.mean(a ** 2))) if n else 0.0
    nclip = int(np.sum(np.abs(a) >= CLIP_T))
    s, e = vad_span(a)
    act = a[s:e + 1]
    a_rms = float(np.sqrt(np.mean(act ** 2))) if len(act) else 0.0
    S = np.abs(np.fft.rfft(a * np.hanning(n))) ** 2 + 1e-20
    f = np.fft.rfftfreq(n, 1 / SR)
    cent = float((f * S).sum() / S.sum())
    flat = float(np.exp(np.mean(np.log(S))) / np.mean(S))
    zcr = float(np.mean(np.abs(np.diff(np.sign(a))) > 0))
    return {
        "duration_s": n / SR, "rms": rms, "peak": peak,
        "clipped_samples": nclip, "clipped_pct": nclip / n * 100 if n else 0.0,
        "is_clipped": peak >= CLIP_T,
        "silence_before_s": s / SR, "silence_after_s": (n - 1 - e) / SR,
        "active_dur_s": (e - s + 1) / SR, "active_rms": a_rms,
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "spectral_centroid": cent, "spectral_flatness": flat, "zcr": zcr,
    }


def main():
    got = sha256(TFLITE)
    print("=" * 78)
    print("REAL-HUMAN FALSE-NEGATIVE DIAGNOSTIC (frozen model, diagnostic only)")
    print("=" * 78)
    print("  model sha256: %s  %s" % (got, "MATCH" if got == EXPECT_SHA else "MISMATCH"))
    if got != EXPECT_SHA:
        raise SystemExit("ERROR: model SHA256 mismatch. Refusing to proceed.")

    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_s, in_z = float(inp["quantization"][0]), int(inp["quantization"][1])
    out_s, out_z = float(out["quantization"][0]), int(out["quantization"][1])
    in_i, out_i = inp["index"], out["index"]

    def qat(a, s):
        w = a[s:s + WIN]
        t = tf.convert_to_tensor(w, dtype=tf.float32)
        sp = tf.signal.stft(t, frame_length=FRAME_LEN, frame_step=FRAME_STEP, fft_length=FFT_LEN)
        sp = tf.math.log(tf.abs(sp) + 1e-6)[:, :N_BINS]
        m = tf.reduce_mean(sp); d = tf.math.reduce_std(sp) + 1e-6
        x = ((sp - m) / d).numpy().astype(np.float32)[..., None]
        q = np.clip(np.round(x / in_s + in_z), -128, 127).astype(np.int8)
        interp.set_tensor(in_i, q[None, ...]); interp.invoke()
        return int(interp.get_tensor(out_i)[0, 0])

    # ---------------------------------------------------------------- STEP 1
    print("\n" + "=" * 78)
    print("STEP 1 -- RECOVER RECORDING GROUPS VIA AUDIO-HASH MAPPING")
    print("=" * 78)
    # Both audit CSVs hash float64-decoded audio, so they join to each other.
    # The streaming CSV hashed float32 audio, so join THAT by filename instead.
    old_by_hash = collections.defaultdict(list)
    for r in csv.DictReader(open(AUDIT_OLD, encoding="utf-8")):
        old_by_hash[r["audio_sha1"]].append(r["name"])
    orig_by_file = {}
    for r in csv.DictReader(open(AUDIT_NEW, encoding="utf-8")):
        names = old_by_hash.get(r["audio_sha1"], [])
        if names:
            orig_by_file[r["name"]] = sorted(names)[0]

    stream, seen = [], set()
    for r in csv.DictReader(open(STREAM, encoding="utf-8")):
        if r["audio_sha1"] in seen:
            continue
        seen.add(r["audio_sha1"])
        stream.append(r)
    print("  unique recordings in streaming results: %d" % len(stream))

    recovered = 0
    for r in stream:
        r["original_name"] = orig_by_file.get(r["filename"], "")
        if r["original_name"]:
            recovered += 1
    print("  original WhatsApp names recovered      : %d / %d (%.1f%%)"
          % (recovered, len(stream), recovered / len(stream) * 100))
    print("  NOTE: original names encode export timestamp + message kind (Ptt/Audio)")
    print("        ONLY. No speaker identity is encoded anywhere, so speakers are")
    print("        NOT inferred. Grouping below is by capture/export session.")

    # session grouping: consecutive exports within a 120 s gap
    def ts_of(name):
        # 'WhatsApp Audio 2026-09-06 at 11.07.35 PM (1).wav'
        try:
            body = name.replace("WhatsApp Audio ", "").replace("WhatsApp Ptt ", "")
            date, _, rest = body.partition(" at ")
            clock = rest.split(" (")[0].replace(".wav", "").strip()
            hms, _, ampm = clock.rpartition(" ")
            h, m, s = [int(x) for x in hms.split(".")]
            if ampm == "PM" and h != 12:
                h += 12
            if ampm == "AM" and h == 12:
                h = 0
            d = [int(x) for x in date.split("-")]
            return (d[0] * 366 + d[1] * 31 + d[2]) * 86400 + h * 3600 + m * 60 + s
        except Exception:
            return None

    for r in stream:
        r["ts"] = ts_of(r["original_name"]) if r["original_name"] else None
        r["kind"] = ("Ptt" if "Ptt" in r["original_name"] else
                     ("Audio" if r["original_name"] else ""))

    timed = sorted([r for r in stream if r["ts"] is not None], key=lambda r: r["ts"])
    GAP = 120
    sess, sid = [], 0
    prev = None
    for r in timed:
        if prev is None or r["ts"] - prev > GAP:
            sid += 1
        r["session"] = "S%02d" % sid
        prev = r["ts"]
    for r in stream:
        r.setdefault("session", "unknown")

    groups = collections.defaultdict(list)
    for r in stream:
        groups[r["session"]].append(r)
    print("\n  sessions found (>%ds export gap = new session): %d" % (GAP, len(groups)))
    print("  %-8s %6s %5s %5s %9s %-22s" % ("session", "n", "TP", "FN", "TPR", "ira_NNNN range"))
    sess_rows = []
    for k in sorted(groups):
        rs = groups[k]
        tp = sum(1 for r in rs if r["fired_100ms"] == "True")
        nums = sorted(int(r["filename"][4:8]) for r in rs)
        rng = "ira_%04d-%04d" % (nums[0], nums[-1])
        kinds = collections.Counter(r["kind"] for r in rs)
        row = {"session": k, "n": len(rs), "TP": tp, "FN": len(rs) - tp,
               "TPR": tp / len(rs) * 100, "range": rng,
               "kind": max(kinds, key=kinds.get) if kinds else ""}
        sess_rows.append(row)
        print("  %-8s %6d %5d %5d %8.2f%% %-22s" % (k, len(rs), tp, len(rs) - tp, row["TPR"], rng))

    base_fn = sum(1 for r in stream if r["fired_100ms"] == "False") / len(stream)
    print("\n  overall FN rate: %.2f%%" % (base_fn * 100))
    print("  sessions with FN rate >= 2x overall:")
    hot = [s for s in sess_rows if s["n"] >= 10 and (1 - s["TPR"] / 100) >= 2 * base_fn]
    for s in sorted(hot, key=lambda s: s["TPR"]):
        print("    %-8s %s  n=%d  FN=%d  FNrate=%.1f%% (%.1fx)"
              % (s["session"], s["range"], s["n"], s["FN"], 100 - s["TPR"],
                 (1 - s["TPR"] / 100) / base_fn))

    # concentration of FNs by session
    fn_by_sess = collections.Counter(r["session"] for r in stream if r["fired_100ms"] == "False")
    tot_fn = sum(fn_by_sess.values())
    top = fn_by_sess.most_common()
    cum, need = 0, 0
    for _, c in top:
        cum += c; need += 1
        if cum >= tot_fn / 2:
            break
    clips_in = sum(len(groups[k]) for k, _ in top[:need])
    print("\n  FN concentration: %d of %d sessions hold >=50%% of FNs (%d clips, %.1f%% of set)"
          % (need, len(groups), clips_in, clips_in / len(stream) * 100))

    # the flagged contiguous runs
    print("\n  --- flagged contiguous runs ---")
    for lo, hi in [(180, 218), (682, 695)]:
        rs = [r for r in stream if lo <= int(r["filename"][4:8]) <= hi]
        tp = sum(1 for r in rs if r["fired_100ms"] == "True")
        ss = collections.Counter(r["session"] for r in rs)
        print("    ira_%04d-%04d: n=%d  TP=%d  FN=%d  TPR=%.1f%%  sessions=%s"
              % (lo, hi, len(rs), tp, len(rs) - tp, tp / len(rs) * 100 if rs else 0, dict(ss)))

    # ---------------------------------------------------------------- STEP 2
    print("\n" + "=" * 78)
    print("STEP 2 -- AUDIO-PROPERTY COMPARISON: 90 FN vs 741 TP")
    print("=" * 78)
    for r in stream:
        p = SRC / "ira words" / r["filename"]
        if not p.exists():
            p = next(SRC.rglob(r["filename"]))
        a, _ = sf.read(str(p), dtype="float64")
        if a.ndim > 1:
            a = a.mean(1)
        r.update(props(a.astype(np.float64)))

    FN = [r for r in stream if r["fired_100ms"] == "False"]
    TP = [r for r in stream if r["fired_100ms"] == "True"]
    print("  FN=%d  TP=%d" % (len(FN), len(TP)))

    metrics = ["duration_s", "rms", "peak", "clipped_pct", "silence_before_s",
               "silence_after_s", "active_dur_s", "active_rms", "crest_factor",
               "spectral_centroid", "spectral_flatness", "zcr"]
    print("\n  %-18s %20s %20s %9s %8s" % ("metric", "FN median [P25-P75]", "TP median [P25-P75]", "Cliff d", "p"))
    prop_rows = {}
    for m in metrics:
        x = np.array([r[m] for r in FN], float)
        y = np.array([r[m] for r in TP], float)
        p, rb = mannwhitney(x, y)
        prop_rows[m] = {"FN": describe(x), "TP": describe(y), "cliffs_delta": rb, "p": p}
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print("  %-18s %8.3f [%6.3f-%6.3f] %8.3f [%6.3f-%6.3f] %9.3f %8.2g%s"
              % (m, np.median(x), np.percentile(x, 25), np.percentile(x, 75),
                 np.median(y), np.percentile(y, 25), np.percentile(y, 75), rb, p, star))
    print("\n  Cliff's delta: |d|<0.147 negligible, <0.33 small, <0.474 medium, else large")

    # clipping association
    from scipy.stats import fisher_exact
    fn_c = sum(1 for r in FN if r["is_clipped"])
    tp_c = sum(1 for r in TP if r["is_clipped"])
    orr, pf = fisher_exact([[fn_c, len(FN) - fn_c], [tp_c, len(TP) - tp_c]], alternative="two-sided")
    print("\n  --- CLIPPING vs FAILURE (peak >= %.3f) ---" % CLIP_T)
    print("    FN clipped: %d/%d = %.1f%%" % (fn_c, len(FN), fn_c / len(FN) * 100))
    print("    TP clipped: %d/%d = %.1f%%" % (tp_c, len(TP), tp_c / len(TP) * 100))
    print("    Fisher exact: odds ratio=%.3f  p=%.4f  -> %s"
          % (orr, pf, "ASSOCIATED" if pf < 0.05 else "NOT associated"))
    fnr_c = fn_c / (fn_c + tp_c) * 100
    fnr_n = (len(FN) - fn_c) / ((len(FN) - fn_c) + (len(TP) - tp_c)) * 100
    print("    FN rate | clipped   : %.2f%%" % fnr_c)
    print("    FN rate | not clipped: %.2f%%" % fnr_n)

    # ---------------------------------------------------------------- STEP 3
    print("\n" + "=" * 78)
    print("STEP 3 -- ALIGNMENT FAILURES vs HARD MODEL FAILURES")
    print("=" * 78)
    A = [r for r in FN if r["fired_10ms"] == "True"]
    B = [r for r in FN if r["fired_10ms"] == "False"]
    print("  Group A -- STRIDE/ALIGNMENT (fail@100ms, fire@10ms): %d  (expected ~40)" % len(A))
    print("  Group B -- HARD MODEL      (fail at both strides)  : %d  (expected ~50)" % len(B))
    print("  A + B = %d = total FN %s" % (len(A) + len(B), "OK" if len(A) + len(B) == len(FN) else "MISMATCH"))

    print("\n  --- Group A: width of the firing region (1 ms probe) ---")
    widths = []
    for r in A:
        p = next(SRC.rglob(r["filename"]))
        a, _ = sf.read(str(p), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        a = a if len(a) >= WIN else np.pad(a, (0, WIN - len(a)))
        last = len(a) - WIN
        fire = [s for s in range(0, last + 1, 16) if qat(a, s) >= Q_MIN]   # 1 ms probe
        if not fire:
            widths.append({"filename": r["filename"], "total_ms": 0.0, "max_run_ms": 0.0, "runs": 0})
            continue
        runs, cur = [], [fire[0]]
        for s in fire[1:]:
            if s - cur[-1] <= 16:
                cur.append(s)
            else:
                runs.append(cur); cur = [s]
        runs.append(cur)
        widths.append({"filename": r["filename"],
                       "total_ms": len(fire) * 1.0,
                       "max_run_ms": max((c[-1] - c[0]) / 16 + 1 for c in runs),
                       "runs": len(runs)})
    mw = np.array([w["max_run_ms"] for w in widths])
    print("    widest contiguous firing window per recording (ms):")
    print("      min=%.0f  P25=%.0f  median=%.0f  P75=%.0f  max=%.0f"
          % (mw.min(), np.percentile(mw, 25), np.median(mw), np.percentile(mw, 75), mw.max()))
    for lo, hi, lbl in [(0, 10, "<10 ms"), (10, 25, "10-25 ms"), (25, 50, "25-50 ms"),
                        (50, 100, "50-100 ms"), (100, 1e9, ">=100 ms")]:
        c = int(((mw >= lo) & (mw < hi)).sum())
        if c:
            print("      %-10s %s %d" % (lbl, "#" * c, c))
    narrow = int((mw < 50).sum())
    print("    firing region narrower than the 100 ms stride: %d/%d (%.0f%%)"
          % (int((mw < 100).sum()), len(A), (mw < 100).mean() * 100))

    print("\n  --- DIAGNOSTIC ONLY: would a 50 ms stride catch them? ---")
    caught = 0
    for r in A:
        p = next(SRC.rglob(r["filename"]))
        a, _ = sf.read(str(p), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        a = a if len(a) >= WIN else np.pad(a, (0, WIN - len(a)))
        last = len(a) - WIN
        offs = list(range(0, last + 1, 800))
        if offs[-1] != last:
            offs.append(last)
        if any(qat(a, s) >= Q_MIN for s in offs):
            caught += 1
    print("    %d/%d Group-A recordings would fire at 50 ms stride" % (caught, len(A)))
    print("    (DIAGNOSTIC ONLY -- 50 ms is NOT adopted as a deployment setting)")

    print("\n  --- Group B: hard failures ---")
    bq = np.array([int(r["max_q_out_10ms"]) for r in B])
    bs = np.array([float(r["max_score_10ms"]) for r in B])
    print("    max score @10ms: median=%.4f  P25=%.4f  P75=%.4f  min=%.4f  max=%.4f"
          % (np.median(bs), np.percentile(bs, 25), np.percentile(bs, 75), bs.min(), bs.max()))
    print("    within 2 int8 steps of firing (q>=-33): %d/%d" % (int((bq >= -33).sum()), len(B)))
    print("    deep failures (score <= 0.15)         : %d/%d" % (int((bs <= 0.15).sum()), len(B)))
    for m in ["duration_s", "active_dur_s", "clipped_pct", "rms", "crest_factor"]:
        xb = np.array([r[m] for r in B], float)
        xa = np.array([r[m] for r in A], float)
        xt = np.array([r[m] for r in TP], float)
        print("    %-14s  B median=%8.3f | A median=%8.3f | TP median=%8.3f"
              % (m, np.median(xb), np.median(xa), np.median(xt)))

    fn_sess_B = collections.Counter(r["session"] for r in B)
    print("    Group B session spread: %d sessions, top: %s"
          % (len(fn_sess_B), fn_sess_B.most_common(5)))

    # ---------------------------------------------------------------- STEP 4
    print("\n" + "=" * 78)
    print("STEP 4 -- HUMAN REVIEW MANIFEST FOR THE %d HARD FAILURES" % len(B))
    print("=" * 78)
    rev_fields = ["filename", "full_path", "original_name", "session", "duration_s",
                  "max_q_out_10ms", "max_score_10ms", "t_max_score_10ms",
                  "max_q_out_100ms", "max_score_100ms", "rms", "peak",
                  "clipped_samples", "clipped_pct", "silence_before_s", "silence_after_s",
                  "active_dur_s", "active_rms", "crest_factor",
                  "LABEL_correct_ira", "LABEL_pronunciation_variation", "LABEL_unclear_mumbled",
                  "LABEL_corrupted_clipped", "LABEL_wrong_word", "LABEL_multiple_words_speech",
                  "LABEL_other", "REVIEWER_NOTES"]
    with open(OUT_REVIEW, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rev_fields)
        w.writeheader()
        for r in sorted(B, key=lambda r: -float(r["max_score_10ms"])):
            row = {k: r.get(k, "") for k in rev_fields}
            row["full_path"] = str(next(SRC.rglob(r["filename"])))
            for k in rev_fields:
                if k.startswith("LABEL_"):
                    row[k] = ""
            row["REVIEWER_NOTES"] = ""
            w.writerow(row)
    print("  wrote %s" % OUT_REVIEW)
    print("  %d rows, sorted by max score descending, with blank LABEL_* columns." % len(B))
    print("  Every recording is retained -- none discarded.")

    # ---------------------------------------------------------------- save
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        keys = ["filename", "original_name", "session", "kind", "fired_100ms", "fired_10ms",
                "max_q_out_100ms", "max_score_100ms", "max_q_out_10ms", "max_score_10ms",
                "t_max_score_10ms", "n_windows_100ms"] + metrics + ["is_clipped", "clipped_samples"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(stream, key=lambda r: r["filename"]):
            w.writerow(r)

    json.dump({
        "model_sha256": got, "decision_rule": "q_out >= %d" % Q_MIN,
        "step1": {"unique": len(stream), "original_names_recovered": recovered,
                  "sessions": len(groups), "session_gap_seconds": GAP,
                  "session_table": sess_rows,
                  "fn_sessions_holding_half": need,
                  "speaker_identity_encoded": False},
        "step2": {"properties": prop_rows,
                  "clipping": {"FN_clipped": fn_c, "FN_total": len(FN),
                               "TP_clipped": tp_c, "TP_total": len(TP),
                               "odds_ratio": float(orr), "fisher_p": float(pf),
                               "fn_rate_clipped_pct": fnr_c, "fn_rate_unclipped_pct": fnr_n}},
        "step3": {"group_A_alignment": len(A), "group_B_hard": len(B),
                  "A_firing_width_ms": {"median": float(np.median(mw)),
                                        "P25": float(np.percentile(mw, 25)),
                                        "P75": float(np.percentile(mw, 75)),
                                        "min": float(mw.min()), "max": float(mw.max())},
                  "A_narrower_than_100ms": int((mw < 100).sum()),
                  "A_caught_by_50ms_stride_DIAGNOSTIC_ONLY": caught,
                  "B_max_score_10ms": describe(bs)},
        "outputs": {"diagnostic_csv": str(OUT_CSV), "review_csv": str(OUT_REVIEW)},
    }, open(OUT_JSON, "w"), indent=2)
    print("\n  wrote %s" % OUT_JSON)
    print("  wrote %s" % OUT_CSV)
    print("  source audio unmodified. No training, no threshold change.")


if __name__ == "__main__":
    main()
