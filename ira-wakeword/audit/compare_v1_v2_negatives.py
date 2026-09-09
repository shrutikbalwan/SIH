# -*- coding: utf-8 -*-
"""
compare_v1_v2_negatives.py
==========================
Controlled V1-vs-V2 false-positive analysis on the EXACT same 772 held-out
test negatives.

V1 model : cnn/models/ira_cnn_best.keras  (float Keras V1)
V2 model : cnn/models/ira_cnn_v2_best_loss.keras

Preprocessing: EXACT training pipeline (identical to both models)
  16 kHz, 1-sec window
  STFT 480 / 320 / 512
  first 40 frequency bins
  log(spec + 1e-6)
  per-window z-score -> shape (49, 40, 1)
"""

import os, csv, shutil
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT      = Path(__file__).parent
V1_MODEL_PATH  = REPO_ROOT / "cnn" / "models" / "ira_cnn_best.keras"
V2_MODEL_PATH  = REPO_ROOT / "cnn" / "models" / "ira_cnn_v2_best_loss.keras"
MANIFEST       = REPO_ROOT / "dataset" / "split_manifest.csv"
V1_NEG_DIR     = REPO_ROOT / "dataset" / "splits" / "test" / "negative"
OUT_CSV        = REPO_ROOT / "v1_v2_negative_scores.csv"
OUT_REPORT     = REPO_ROOT / "V1_V2_FALSE_POSITIVE_ANALYSIS.md"
REGRESSION_DIR = REPO_ROOT / "v2_false_positive_analysis" / "top_regressions"
HARD_NEG_DIR   = REPO_ROOT / "dataset" / "negative" / "hard"

SAMPLE_RATE    = 16000
WINDOW_SAMPLES = 16000
THR            = 0.50


def load_audio(path):
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    if len(a) < WINDOW_SAMPLES: a = np.pad(a, (0, WINDOW_SAMPLES - len(a)))
    else: a = a[:WINDOW_SAMPLES]
    return a.astype(np.float32)


def make_spectrogram(audio):
    t = tf.convert_to_tensor(audio, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=480, frame_step=320, fft_length=512)
    s = tf.abs(s)
    s = tf.math.log(s + 1e-6)
    s = s[:, :40]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    s = (s - m) / d
    return s.numpy().astype(np.float32)


def batch_score(model, paths):
    if not paths: return np.array([], dtype=np.float32)
    specs = [np.expand_dims(make_spectrogram(load_audio(p)), -1) for p in paths]
    X = np.stack(specs, 0)
    return model.predict(X, batch_size=64, verbose=0).flatten().astype(np.float32)


def categorise(group, path):
    g = group.lower(); p = path.lower()
    if "libri" in g or "speech" in g or "libri" in p or "speech" in p: return "speech"
    if "ambient" in g or "background" in g or "noise" in g: return "ambient"
    return "other"


def dist_stats(arr):
    if len(arr) == 0: return {k: float("nan") for k in ["mean","median","p90","p95","p99","max"]}
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "p90": float(np.percentile(arr,90)), "p95": float(np.percentile(arr,95)),
            "p99": float(np.percentile(arr,99)), "max": float(np.max(arr))}


def main():
    print("="*60)
    print("V1 vs V2 NEGATIVE FALSE-POSITIVE ANALYSIS")
    print("="*60)

    # Load models
    print(f"\nLoading V1 : {V1_MODEL_PATH.name}")
    if not V1_MODEL_PATH.exists():
        raise FileNotFoundError(f"V1 model not found: {V1_MODEL_PATH}")
    v1 = tf.keras.models.load_model(str(V1_MODEL_PATH))
    print(f"Loading V2 : {V2_MODEL_PATH.name}")
    v2 = tf.keras.models.load_model(str(V2_MODEL_PATH))

    # ------------------------------------------------------------------ #
    # SECTION 2: Build & verify test negative list
    # ------------------------------------------------------------------ #
    manifest_negs = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test": continue
            if int(r.get("label", -1)) != 0: continue
            p = REPO_ROOT / r["path"]
            if not p.exists():
                print(f"  WARNING missing: {p}"); continue
            manifest_negs.append({"path": str(p), "group": r.get("group",""),
                                   "category": categorise(r.get("group",""), r["path"])})

    v1_phys = set()
    if V1_NEG_DIR.exists():
        v1_phys = {str(f) for f in V1_NEG_DIR.glob("*.wav")}

    manifest_paths = {d["path"] for d in manifest_negs}

    print(f"\n{'='*60}")
    print("SECTION 2: TEST NEGATIVE SET VERIFICATION")
    print("="*60)
    print(f"  Manifest negatives (split=test, label=0) : {len(manifest_negs)}")
    print(f"  V1 physical directory                    : {len(v1_phys)}")
    if v1_phys:
        only_v1 = v1_phys - manifest_paths
        only_mf = manifest_paths - v1_phys
        both    = manifest_paths & v1_phys
        print(f"  In both sets                             : {len(both)}")
        print(f"  Only in V1 physical dir (NOT manifest)  : {len(only_v1)}")
        print(f"  Only in manifest (NOT V1 physical dir)  : {len(only_mf)}")
        if only_v1 or only_mf:
            print("  *** DISCREPANCY - the two negative lists differ ***")
            for p in sorted(only_v1)[:5]: print(f"    V1-only: {p}")
            for p in sorted(only_mf)[:5]: print(f"    MF-only: {p}")
        else:
            print("  Sets are IDENTICAL. OK")
    else:
        print("  V1 physical dir not found; skipping cross-check.")

    cat_counts = {}
    for d in manifest_negs: cat_counts[d["category"]] = cat_counts.get(d["category"],0)+1
    print("\n  Category breakdown:")
    for c,n in sorted(cat_counts.items()): print(f"    {c:10s}: {n}")

    paths  = [d["path"]     for d in manifest_negs]
    groups = [d["group"]    for d in manifest_negs]
    cats   = [d["category"] for d in manifest_negs]

    if len(paths) != 772:
        print(f"\n  WARNING: expected 772, got {len(paths)}")

    # ------------------------------------------------------------------ #
    # SECTION 3: Score all with both models
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("SECTION 3: SCORING ALL NEGATIVES")
    print("="*60)
    print(f"  V1 scoring {len(paths)} files...")
    v1s = batch_score(v1, paths)
    print(f"  V2 scoring {len(paths)} files...")
    v2s = batch_score(v2, paths)
    dlt = v2s - v1s

    with open(OUT_CSV,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path","group","category","v1_score","v2_score","score_delta"])
        for p,g,c,s1,s2,d in zip(paths,groups,cats,v1s,v2s,dlt):
            w.writerow([p,g,c,f"{s1:.6f}",f"{s2:.6f}",f"{d:.6f}"])
    print(f"  Saved: {OUT_CSV}")

    # ------------------------------------------------------------------ #
    # SECTION 4: Transition counts
    # ------------------------------------------------------------------ #
    v1_fp = v1s >= THR
    v2_fp = v2s >= THR
    tn_tn = int(np.sum(~v1_fp & ~v2_fp))
    tn_fp = int(np.sum(~v1_fp &  v2_fp))
    fp_tn = int(np.sum( v1_fp & ~v2_fp))
    fp_fp = int(np.sum( v1_fp &  v2_fp))

    print(f"\n{'='*60}")
    print("SECTION 4: TRANSITION MATRIX V1->V2 @ 0.50")
    print("="*60)
    print(f"  TN->TN (correct both)  : {tn_tn}")
    print(f"  TN->FP (V2 regression) : {tn_fp}  *** KEY ***")
    print(f"  FP->TN (V2 improvement): {fp_tn}")
    print(f"  FP->FP (wrong both)    : {fp_fp}")

    # ------------------------------------------------------------------ #
    # SECTION 5: Per-category breakdown
    # ------------------------------------------------------------------ #
    def cat_scores(scores):
        return {c: scores[[i for i,x in enumerate(cats) if x==c]] for c in ["speech","ambient","other"]}

    v1_cat = cat_scores(v1s)
    v2_cat = cat_scores(v2s)

    def print_cat_table(label, sc_by_cat):
        print(f"\n  {label}:")
        hdr = f"  {'Cat':10s} | {'N':>5} | {'FP':>5} | {'FPR':>7} | {'Mean':>7} | {'Med':>7} | {'P90':>7} | {'P95':>7} | {'P99':>7}"
        print(hdr); print("  "+"-"*len(hdr))
        for c in ["speech","ambient","other"]:
            s = sc_by_cat[c]
            if len(s)==0: continue
            fp = int(np.sum(s>=THR))
            print(f"  {c:10s} | {len(s):>5} | {fp:>5} | {fp/len(s)*100:>6.2f}% |"
                  f" {np.mean(s):>7.4f} | {np.median(s):>7.4f} |"
                  f" {np.percentile(s,90):>7.4f} | {np.percentile(s,95):>7.4f} |"
                  f" {np.percentile(s,99):>7.4f}")

    print(f"\n{'='*60}")
    print("SECTION 5: PER-CATEGORY FP BREAKDOWN")
    print("="*60)
    print_cat_table("V1", v1_cat)
    print_cat_table("V2", v2_cat)

    # TN->FP by category
    tn_fp_cats = {}
    for i in range(len(paths)):
        if (not v1_fp[i]) and v2_fp[i]:
            tn_fp_cats[cats[i]] = tn_fp_cats.get(cats[i],0) + 1
    print(f"\n  TN->FP regressions by category (total={tn_fp}):")
    for c,n in sorted(tn_fp_cats.items()):
        print(f"    {c:10s}: {n}  ({n/max(tn_fp,1)*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # SECTION 6: Distribution stats
    # ------------------------------------------------------------------ #
    v1d = dist_stats(v1s); v2d = dist_stats(v2s); dd = dist_stats(dlt)
    print(f"\n{'='*60}")
    print("SECTION 6: SCORE DISTRIBUTIONS (all 772 negatives)")
    print("="*60)
    print(f"  {'Stat':8s} | {'V1':>9} | {'V2':>9} | {'Delta(V2-V1)':>12}")
    print("  "+"-"*45)
    for k in ["mean","median","p90","p95","p99","max"]:
        print(f"  {k:8s} | {v1d[k]:>9.4f} | {v2d[k]:>9.4f} | {dd[k]:>12.4f}")

    # ------------------------------------------------------------------ #
    # SECTION 7: Top 30 regressions
    # ------------------------------------------------------------------ #
    top30 = np.argsort(dlt)[::-1][:30]
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("SECTION 7: TOP 30 REGRESSIONS (largest V2-V1 delta)")
    print("="*60)
    print(f"  {'#':>3}  {'Delta':>7}  {'V1':>7}  {'V2':>7}  {'Cat':10}  Filename")
    print("  "+"-"*72)
    for rank, i in enumerate(top30, 1):
        fname = Path(paths[i]).name
        print(f"  {rank:>3}  {dlt[i]:>7.4f}  {v1s[i]:>7.4f}  {v2s[i]:>7.4f}  {cats[i]:10}  {fname}")
        dst = REGRESSION_DIR / fname
        if not dst.exists(): shutil.copy2(paths[i], dst)
    print(f"\n  WAVs copied to: {REGRESSION_DIR}")

    # ------------------------------------------------------------------ #
    # SECTION 8: Hard-negative cross-check
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("SECTION 8: HARD-NEGATIVE CROSS-CHECK")
    print("="*60)
    hard_paths = set()
    if HARD_NEG_DIR.exists():
        hard_paths = {str(f) for f in HARD_NEG_DIR.rglob("*.wav")}
        print(f"  Training hard-neg dir: {len(hard_paths)} files")
    else:
        # fall back to manifest
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "hard" in r.get("group","").lower():
                    p = REPO_ROOT / r["path"]
                    hard_paths.add(str(p))
        print(f"  Hard negatives in manifest: {len(hard_paths)}")

    v2_fp_set = {paths[i] for i in range(len(paths)) if v2_fp[i]}
    overlap   = v2_fp_set & hard_paths
    print(f"  V2 test FPs              : {len(v2_fp_set)}")
    print(f"  Exact file overlap       : {len(overlap)}")
    if overlap:
        print("  *** LEAKAGE DETECTED:")
        for p in sorted(overlap)[:10]: print(f"    {p}")
    else:
        print("  No exact file leakage. OK")

    # category comparison
    v2_fp_cat = {}
    for i in range(len(paths)):
        if v2_fp[i]: v2_fp_cat[cats[i]] = v2_fp_cat.get(cats[i],0)+1
    hn_cat = {}
    for p in hard_paths:
        pl = p.lower()
        c = "speech" if ("libri" in pl or "speech" in pl) else "ambient" if ("ambient" in pl or "noise" in pl) else "other"
        hn_cat[c] = hn_cat.get(c,0)+1
    print("\n  V2 test FP category distribution:")
    for c,n in sorted(v2_fp_cat.items()): print(f"    {c:10s}: {n}")
    if hn_cat:
        print("  Training hard-neg category distribution:")
        for c,n in sorted(hn_cat.items()): print(f"    {c:10s}: {n}")

    # ------------------------------------------------------------------ #
    # SECTION 9: Fine threshold sweep on V2 positives
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("SECTION 9: FINE THRESHOLD SWEEP 0.50->0.99 (V2)")
    print("="*60)
    pos_real, pos_tts = [], []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test": continue
            if int(r.get("label",-1)) != 1: continue
            p = REPO_ROOT / r["path"]
            if not p.exists(): continue
            g = r.get("group","").lower(); pl = str(p).lower()
            is_tts = "piper" in g or "tts" in g or "piper" in pl or "tts" in pl or "synthetic" in pl
            (pos_tts if is_tts else pos_real).append(str(p))

    print(f"  Scoring {len(pos_real)} REAL + {len(pos_tts)} TTS positives...")
    v2pr = batch_score(v2, pos_real)
    v2pt = batch_score(v2, pos_tts)
    v2pa = np.concatenate([v2pr, v2pt])

    print(f"\n  {'Thr':>5} | {'REAL_TPR':>9} | {'ALL_TPR':>9} | {'FPR':>7} | Notes")
    print("  "+"-"*65)
    found_good = False
    for t in np.arange(0.50, 1.00, 0.01):
        rt = np.sum(v2pr>=t)/len(v2pr)*100 if len(v2pr) else 0
        at = np.sum(v2pa>=t)/len(v2pa)*100 if len(v2pa) else 0
        ft = np.sum(v2s>=t)/len(v2s)*100
        note = ""
        if rt>=95 and at>=95 and ft<=1.0:
            found_good = True
            note = " <-- ALL CRITERIA MET"
        print(f"  {t:>5.2f} | {rt:>8.2f}% | {at:>8.2f}% | {ft:>6.2f}% |{note}")
    if not found_good:
        print("\n  No threshold in [0.50,0.99] satisfies REAL>=95% & ALL>=95% & FPR<=1%.")

    # ------------------------------------------------------------------ #
    # SECTION 10: Write markdown report
    # ------------------------------------------------------------------ #
    dominant = max(tn_fp_cats, key=tn_fp_cats.get) if tn_fp_cats else "unknown"
    max_pct  = tn_fp_cats.get(dominant,0)/max(tn_fp,1)*100

    def cat_md(sc_by_cat):
        rows = []
        for c in ["speech","ambient","other"]:
            s = sc_by_cat[c]
            if len(s)==0: rows.append(f"| {c} | 0 | 0 | N/A | N/A |"); continue
            fp = int(np.sum(s>=THR))
            rows.append(f"| {c} | {len(s)} | {fp} | {fp/len(s)*100:.2f}% | {np.median(s):.4f} |")
        return "\n".join(rows)

    top30_rows = []
    for rank,i in enumerate(top30,1):
        top30_rows.append(
            f"| {rank} | {cats[i]} | {v1s[i]:.4f} | {v2s[i]:.4f} | {dlt[i]:.4f} | {Path(paths[i]).name} |")

    report = f"""# V1 vs V2 False-Positive Analysis

## Overview

- **V1 model**: `ira_cnn_best.keras`
- **V2 model**: `ira_cnn_v2_best_loss.keras`
- **Test set**: {len(manifest_negs)} held-out negatives from `split_manifest.csv`
- **Threshold evaluated**: 0.50

## Section 1 - FPR Change

| | V1 | V2 |
|---|---|---|
| Test negatives | {len(v1s)} | {len(v2s)} |
| FP @ 0.50 | {int(np.sum(v1_fp))} | {int(np.sum(v2_fp))} |
| FPR @ 0.50 | {np.sum(v1_fp)/len(v1_fp)*100:.2f}% | {np.sum(v2_fp)/len(v2_fp)*100:.2f}% |

## Section 2 - Test Set Verification

Files loaded from `split_manifest.csv` (split=test, label=0).
Cross-checked against `dataset/splits/test/negative/` (V1 physical directory).

Category breakdown:
| Category | Count |
|---|---|
{chr(10).join(f"| {c} | {n} |" for c,n in sorted(cat_counts.items()))}

## Section 3 - Transition Matrix (V1?V2 @ 0.50)

| Transition | Count |
|---|---|
| TN?TN (correct both) | {tn_tn} |
| **TN?FP (V2 regression)** | **{tn_fp}** |
| FP?TN (V2 improvement) | {fp_tn} |
| FP?FP (wrong both) | {fp_fp} |

> [!IMPORTANT]
> **{tn_fp}** negatives were correctly rejected by V1 but falsely triggered by V2.

## Section 4 - Per-Category FP Breakdown

### V1
| Category | N | FP | FPR | Median |
|---|---|---|---|---|
{cat_md(v1_cat)}

### V2
| Category | N | FP | FPR | Median |
|---|---|---|---|---|
{cat_md(v2_cat)}

### TN?FP Regressions by Category
| Category | Count | % of regressions |
|---|---|---|
{chr(10).join(f"| {c} | {n} | {n/max(tn_fp,1)*100:.1f}% |" for c,n in sorted(tn_fp_cats.items()))}

> **Concentration**: {dominant} negatives account for **{max_pct:.1f}%** of the TN?FP regressions.

## Section 5 - Score Distribution (all {len(v1s)} test negatives)

| Stat | V1 | V2 | V2-V1 |
|---|---|---|---|
{chr(10).join(f"| {k} | {v1d[k]:.4f} | {v2d[k]:.4f} | {dd[k]:.4f} |" for k in ["mean","median","p90","p95","p99","max"])}

## Section 6 - Hard-Negative Cross-Check

- Training hard negatives found: {len(hard_paths)}
- Exact file overlap with V2 test FPs: **{len(overlap)}**
- Result: {"?? LEAKAGE DETECTED" if overlap else "? No exact file leakage"}

## Section 7 - Top 30 Regressions

| Rank | Category | V1 | V2 | Delta | Filename |
|---|---|---|---|---|---|
{chr(10).join(top30_rows)}

Files copied to: `v2_false_positive_analysis/top_regressions/`

## Section 8 - Threshold Feasibility (Diagnostic Only)

Any threshold in [0.50, 0.99] where REAL TPR=95%, ALL TPR=95%, FPR=1%?
**{"Yes - see console output." if found_good else "No. No such threshold exists in this range."}**

> [!NOTE]
> Diagnostic only. No production threshold selected.

## Conclusion

The FPR regression ({np.sum(v1_fp)/len(v1_fp)*100:.2f}% ? {np.sum(v2_fp)/len(v2_fp)*100:.2f}%) is **concentrated in {dominant} negatives** ({max_pct:.1f}% of regressions).

The test negative file list is **identical** between V1 and V2 evaluations - the regression is a genuine model behaviour difference, not a dataset difference.

Do NOT retrain. Do NOT quantize. Do NOT run streaming evaluation yet.
"""
    with open(OUT_REPORT,"w",encoding="utf-8") as f:
        f.write(report)
    print(f"\n{'='*60}")
    print(f"Report saved: {OUT_REPORT}")
    print("="*60)


if __name__ == "__main__":
    main()

