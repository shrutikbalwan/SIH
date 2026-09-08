# -*- coding: utf-8 -*-
"""
Human adjudication of the 50 hard false negatives. ANALYSIS ONLY.

No training, no fine-tuning, no V2.8, no model change, no threshold change,
no stride change, no threshold sweep. The human-label file is opened
READ-ONLY and never written.
"""
import csv, json, collections
import numpy as np
from pathlib import Path

CNN = Path(__file__).parent
LABELS = CNN / "real_human_hard_fn_review - real_human_hard_fn_review.csv.csv"
REVIEW = CNN / "real_human_hard_fn_review.csv"
DIAG = CNN / "real_human_fn_diagnostic.csv"
OUT = CNN / "real_human_hard_fn_adjudication.json"

LABEL_COLS = ["LABEL_correct_ira", "LABEL_pronunciation_variation", "LABEL_unclear_mumbled",
              "LABEL_corrupted_clipped", "LABEL_wrong_word", "LABEL_multiple_words_speech",
              "LABEL_other"]

BENCH = {"n": 831, "TP": 741, "FN": 90, "TPR": 89.1697, "ci": [86.8560, 91.2008]}


def parse_flag(v):
    """Checkbox cell. Accepts '1' or verbose 'LABEL_x = 1' / 'LABEL_x =1'. Blank = no."""
    s = (v or "").strip()
    if not s:
        return False, None
    t = s.replace(" ", "").lower()
    if t in ("1", "yes", "y", "true", "x"):
        return True, None
    if t.endswith("=1"):
        return True, None
    if t.endswith("=0") or t in ("0", "no", "n", "false"):
        return False, None
    return False, s          # malformed


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo * 100, hi * 100


def desc(v):
    v = np.asarray(v, float)
    if not len(v):
        return {}
    return {"n": int(len(v)), "median": float(np.median(v)),
            "P10": float(np.percentile(v, 10)), "P25": float(np.percentile(v, 25)),
            "P75": float(np.percentile(v, 75)), "P90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max())}


def main():
    rep = {}
    print("=" * 78)
    print("STEP 0 -- VERIFY THE HUMAN LABEL FILE")
    print("=" * 78)
    print("  file: %s" % LABELS.name)
    if not LABELS.exists():
        raise SystemExit("STOP: human label file not found.")
    print("  exists: YES   size: %d bytes" % LABELS.stat().st_size)

    rows = list(csv.DictReader(open(LABELS, encoding="utf-8-sig")))
    print("  data rows: %d (expected 50)" % len(rows))
    if len(rows) != 50:
        raise SystemExit("STOP: expected 50 data rows, found %d." % len(rows))

    flags, malformed = {}, []
    for r in rows:
        f = r["filename"]
        d = {}
        for c in LABEL_COLS:
            ok, bad = parse_flag(r.get(c))
            d[c] = ok
            if bad:
                malformed.append((f, c, bad))
        flags[f] = d

    unlabeled = [f for f, d in flags.items() if not any(d.values())]
    dupes = [k for k, v in collections.Counter(r["filename"] for r in rows).items() if v > 1]
    multi = {f: [c for c in LABEL_COLS if d[c]] for f, d in flags.items()
             if sum(d.values()) > 1}

    print("  rows with >=1 label   : %d" % (len(rows) - len(unlabeled)))
    print("  completely unlabeled  : %d %s" % (len(unlabeled), unlabeled or ""))
    print("  malformed label values: %d %s" % (len(malformed), malformed or ""))
    print("  duplicate filenames   : %d %s" % (len(dupes), dupes or ""))
    print("  rows with >1 label    : %d %s" % (len(multi), multi or ""))
    print("  NOTE: cells are written verbosely as 'LABEL_x = 1'; parsed as checkbox=1.")
    if unlabeled:
        raise SystemExit("STOP: not all 50 rows are labeled.")
    print("  VERIFICATION: PASS -- all 50 rows carry at least one human label.")

    rep["step0"] = {"file": str(LABELS), "rows": len(rows),
                    "rows_with_label": len(rows) - len(unlabeled),
                    "unlabeled": unlabeled, "malformed": malformed,
                    "duplicate_filenames": dupes, "multi_label_rows": multi,
                    "verification": "PASS"}

    # ------------------------------------------------------------- STEP 1
    print("\n" + "=" * 78)
    print("STEP 1 -- HUMAN LABEL COUNTS (n=50)")
    print("=" * 78)
    counts = {c: sum(1 for d in flags.values() if d[c]) for c in LABEL_COLS}
    for c in LABEL_COLS:
        print("  %-32s %d" % (c.replace("LABEL_", ""), counts[c]))
    print("  ---")
    print("  sum of category counts: %d (labels may overlap)" % sum(counts.values()))
    both = [f for f, d in flags.items() if d["LABEL_correct_ira"] and d["LABEL_pronunciation_variation"]]
    print("  overlap 'correct_ira + pronunciation_variation': %d %s" % (len(both), both or ""))
    print("  no row carries more than one label, so there are no other overlaps.")
    rep["step1"] = {"counts": {c.replace("LABEL_", ""): counts[c] for c in LABEL_COLS},
                    "overlap_correct_plus_pronvar": len(both)}

    # ------------------------------------------------------------- STEP 2
    print("\n" + "=" * 78)
    print("STEP 2 -- VALID / INVALID / AMBIGUOUS")
    print("=" * 78)
    print("  Rule:")
    print("    VALID     = correct_ira OR pronunciation_variation")
    print("                (reasonable pronunciation variation stays VALID)")
    print("    INVALID   = wrong_word OR corrupted_clipped OR multiple_words_speech")
    print("                (human review establishes it is not a usable positive)")
    print("    AMBIGUOUS = unclear_mumbled or other, with no VALID label")
    valid, invalid, ambiguous = [], [], []
    for f, d in flags.items():
        if d["LABEL_correct_ira"] or d["LABEL_pronunciation_variation"]:
            valid.append(f)
        elif d["LABEL_wrong_word"] or d["LABEL_corrupted_clipped"] or d["LABEL_multiple_words_speech"]:
            invalid.append(f)
        else:
            ambiguous.append(f)
    print("\n  VALID     = %d" % len(valid))
    print("  INVALID   = %d  %s" % (len(invalid), invalid))
    print("  AMBIGUOUS = %d  %s" % (len(ambiguous), ambiguous))
    rep["step2"] = {"rule": "VALID=correct_ira|pronunciation_variation; "
                            "INVALID=wrong_word|corrupted_clipped|multiple_words_speech; "
                            "AMBIGUOUS=unclear_mumbled|other with no valid label",
                    "valid": len(valid), "invalid": len(invalid), "ambiguous": len(ambiguous),
                    "invalid_files": invalid, "ambiguous_files": ambiguous}

    # metrics join
    diag = {r["filename"]: r for r in csv.DictReader(open(DIAG, encoding="utf-8"))}
    rev = {r["filename"]: r for r in csv.DictReader(open(REVIEW, encoding="utf-8"))}

    def num(f, k, src=None):
        s = (src or diag).get(f, {}).get(k, "")
        try:
            return float(s)
        except Exception:
            return float("nan")

    # ------------------------------------------------------------- STEP 3
    print("\n" + "=" * 78)
    print("STEP 3 -- GENUINE HARD MODEL MISSES ON VALID 'IRA'")
    print("=" * 78)
    gq = np.array([num(f, "max_q_out_10ms") for f in valid])
    gs = np.array([num(f, "max_score_10ms") for f in valid])
    print("  genuine hard misses : %d" % len(valid))
    print("  as %% of the 50      : %.1f%%" % (len(valid) / 50 * 100))
    print("  median max q_out    : %.1f" % np.median(gq))
    print("  median max score    : %.6f" % np.median(gs))
    print("  score P10/P25/P50/P75/P90: %.4f / %.4f / %.4f / %.4f / %.4f"
          % tuple(np.percentile(gs, [10, 25, 50, 75, 90])))
    print("  min / max score     : %.6f / %.6f" % (gs.min(), gs.max()))
    deep = int((gs <= 0.15).sum())
    near = int((gq >= -33).sum())
    print("  deeply rejected (score <= 0.15) : %d (%.0f%%)" % (deep, deep / len(valid) * 100))
    print("  close to boundary (q >= -33)    : %d (%.0f%%)" % (near, near / len(valid) * 100))
    print("  (proximity to the boundary is NOT a reason to change the threshold)")
    rep["step3"] = {"genuine_hard_misses": len(valid), "pct_of_50": len(valid) / 50 * 100,
                    "median_max_q_out": float(np.median(gq)),
                    "median_max_score": float(np.median(gs)),
                    "score_distribution": desc(gs),
                    "deeply_rejected_le_0.15": deep, "near_boundary_q_ge_-33": near}

    # ------------------------------------------------------------- STEP 4
    print("\n" + "=" * 78)
    print("STEP 4 -- HUMAN-ADJUDICATED DIAGNOSTIC TPR")
    print("=" * 78)
    print("  ORIGINAL BENCHMARK (unchanged, remains the result of record):")
    print("    %d / %d = %.4f%%   95%% CI [%.4f%%, %.4f%%]"
          % (BENCH["TP"], BENCH["n"], BENCH["TPR"], *BENCH["ci"]))
    n_inv = len(invalid)
    if n_inv == 0:
        print("\n  No hard-FN recording is confirmed invalid as a positive example.")
        print("  There is no human-adjudicated improvement to the benchmark.")
        adj = None
    else:
        num_, den = BENCH["TP"], BENCH["n"] - n_inv
        tpr = num_ / den * 100
        lo, hi = clopper_pearson(num_, den)
        print("\n  Removing ONLY the %d confirmed-invalid positive(s): %s" % (n_inv, invalid))
        print("  Ambiguous recordings are NOT removed (no rule justifies it).")
        print("\n  HUMAN-ADJUDICATED DIAGNOSTIC TPR:")
        print("    %d / %d = %.4f%%   95%% CI [%.4f%%, %.4f%%]" % (num_, den, tpr, lo, hi))
        print("    change vs benchmark: %+.4f pp" % (tpr - BENCH["TPR"]))
        adj = {"numerator": num_, "denominator": den, "TPR": tpr, "ci": [lo, hi],
               "removed": invalid, "delta_pp": tpr - BENCH["TPR"]}
    print("\n  *** THIS DOES NOT REPLACE THE ORIGINAL BENCHMARK. ***")
    rep["step4"] = {"original_benchmark": BENCH, "human_adjudicated": adj,
                    "note": "does not replace the original benchmark"}

    # ------------------------------------------------------------- STEP 5
    print("\n" + "=" * 78)
    print("STEP 5 -- SESSION ANALYSIS OF GENUINE MISSES")
    print("=" * 78)
    sess_all = collections.Counter(r["session"] for r in diag.values())
    sess_fn = collections.Counter(r["session"] for r in diag.values() if r["fired_100ms"] == "False")
    sess_gen = collections.Counter(diag[f]["session"] for f in valid if f in diag)
    print("  %-8s %7s %8s %10s %12s %10s" %
          ("session", "clips", "all FN", "genuine", "genuine/clip", "share"))
    tot = len(valid)
    rows_s = []
    for s in sorted(sess_all, key=lambda s: -sess_gen[s]):
        if sess_gen[s] == 0 and sess_fn[s] == 0:
            continue
        r = {"session": s, "clips": sess_all[s], "all_fn": sess_fn[s],
             "genuine": sess_gen[s], "rate": sess_gen[s] / sess_all[s] * 100,
             "share": sess_gen[s] / tot * 100}
        rows_s.append(r)
        print("  %-8s %7d %8d %10d %11.2f%% %9.1f%%"
              % (s, r["clips"], r["all_fn"], r["genuine"], r["rate"], r["share"]))
    print("\n  S07 genuine hard misses: %d" % sess_gen.get("S07", 0))
    print("  S16 genuine hard misses: %d" % sess_gen.get("S16", 0))
    others = {s: c for s, c in sess_gen.items() if s not in ("S07", "S16") and c}
    print("  other contributing sessions: %s" % others)
    top2 = sess_gen.get("S07", 0) + sess_gen.get("S16", 0)
    print("  S07+S16 hold %d/%d = %.1f%% of genuine misses, from %d/%d = %.1f%% of clips"
          % (top2, tot, top2 / tot * 100,
             sess_all.get("S07", 0) + sess_all.get("S16", 0), sum(sess_all.values()),
             (sess_all.get("S07", 0) + sess_all.get("S16", 0)) / sum(sess_all.values()) * 100))
    print("  'session' = export/capture grouping, NOT verified speaker identity.")
    rep["step5"] = {"per_session": rows_s, "S07": sess_gen.get("S07", 0),
                    "S16": sess_gen.get("S16", 0), "others": others,
                    "caveat": "session is export grouping, not verified speaker"}

    # ------------------------------------------------------------- STEP 6
    print("\n" + "=" * 78)
    print("STEP 6 -- AUDIO CHARACTERISTICS OF GENUINE MISSES")
    print("=" * 78)
    tp_rows = [r for r in diag.values() if r["fired_100ms"] == "True"]
    mets = ["rms", "peak", "active_rms", "duration_s", "active_dur_s",
            "clipped_pct", "silence_before_s", "silence_after_s", "crest_factor"]
    print("  %-18s %14s %14s" % ("metric", "genuine (n=%d)" % len(valid), "TP (n=%d)" % len(tp_rows)))
    audio = {}
    for m in mets:
        g = np.array([num(f, m) for f in valid])
        t = np.array([float(r[m]) for r in tp_rows])
        audio[m] = {"genuine_median": float(np.median(g)), "tp_median": float(np.median(t))}
        print("  %-18s %14.4f %14.4f" % (m, np.median(g), np.median(t)))
    gclip = sum(1 for f in valid if diag[f]["is_clipped"] == "True")
    tclip = sum(1 for r in tp_rows if r["is_clipped"] == "True")
    print("\n  clipped: genuine %d/%d = %.1f%%   TP %d/%d = %.1f%%"
          % (gclip, len(valid), gclip / len(valid) * 100,
             tclip, len(tp_rows), tclip / len(tp_rows) * 100))
    print("\n  Prior conclusions re-checked after adjudication:")
    print("    quieter than TPs                : %s" %
          ("HOLDS" if audio["rms"]["genuine_median"] < audio["rms"]["tp_median"] else "does not hold"))
    print("    clipping not causing failure    : %s" %
          ("HOLDS" if gclip / len(valid) < tclip / len(tp_rows) else "does not hold"))
    print("    active speech NOT unusually long: %s" %
          ("HOLDS" if audio["active_dur_s"]["genuine_median"] <= audio["active_dur_s"]["tp_median"]
           else "does not hold"))
    audio["clipped_genuine_pct"] = gclip / len(valid) * 100
    audio["clipped_tp_pct"] = tclip / len(tp_rows) * 100
    rep["step6"] = audio

    # ------------------------------------------------------------- STEP 7
    print("\n" + "=" * 78)
    print("STEP 7 -- PRONUNCIATION VARIATION")
    print("=" * 78)
    pv = [f for f in valid if flags[f]["LABEL_pronunciation_variation"]]
    print("  genuine misses labeled pronunciation_variation: %d" % len(pv))
    print("  as %% of genuine misses: %.1f%%" % (len(pv) / len(valid) * 100 if valid else 0))
    if not pv:
        print("  The reviewer labeled ZERO recordings as pronunciation variation.")
        print("  => Pronunciation variation is NOT enriched among hard failures.")
        print("  => Hypothesis B (pronunciation variation) is NOT supported by this review.")
    rep["step7"] = {"pronunciation_variation_among_genuine": len(pv),
                    "pct": len(pv) / len(valid) * 100 if valid else 0,
                    "supported": bool(pv)}

    json.dump(rep, open(OUT, "w"), indent=2)
    print("\n  wrote %s" % OUT)
    print("  human-label file NOT modified.")


if __name__ == "__main__":
    main()
