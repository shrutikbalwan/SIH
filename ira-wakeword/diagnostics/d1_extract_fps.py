# -*- coding: utf-8 -*-
"""
Stage 1 -- Reproduce the V2.3 best_loss operating point on the FROZEN
EXPANDED_VAL artifact and extract the residual speech false positives
with full provenance (speaker / chapter / utterance / crop timing).

READ-ONLY. Nothing is regenerated, nothing is trained.
"""
import os, csv, json, sys
import numpy as np
import soundfile as sf
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO   = Path(__file__).parent.parent
NPZ    = REPO / "dataset" / "evaluation" / "expanded_val_v2_3_frozen.npz"
MODEL  = REPO / "cnn" / "models" / "ira_cnn_v2_3_best_loss.keras"
OUT    = REPO / "diagnostics"
THR    = 0.43

# Expected reproduction targets (frozen reference)
EXPECT = {"TP": 949, "FN": 49, "speech_FP": 58, "speech_TN": 1728, "ambient_FP": 0}

LIBRI_ROOTS = [
    REPO / "dataset" / "negative" / "extracted" / "LibriSpeech",
    REPO / "dataset" / "negative" / "downloads" / "LibriSpeech",
    Path("E:/SIH/ira-wakeword/dataset/negative/extracted/LibriSpeech"),
]

SR = 16000
CLIP = 16000


def build_libri_index():
    """stem -> flac path, and utt_id -> transcript, across all subsets."""
    flac, trans = {}, {}
    for root in LIBRI_ROOTS:
        if not root.exists():
            continue
        for f in root.rglob("*.flac"):
            flac.setdefault(f.stem, f)
        for t in root.rglob("*.trans.txt"):
            for line in t.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                uid, _, text = line.partition(" ")
                trans.setdefault(uid, text.strip())
    return flac, trans


def crop_start_sample(flac_path: Path, j: int):
    """Reproduce the generator's window arithmetic exactly.

    starts = linspace(0, len - CLIP, max(1, len // CLIP))
    """
    try:
        info = sf.info(str(flac_path))
    except Exception:
        return None, None
    n = int(info.frames)
    if n < CLIP:
        return None, n
    n_windows = max(1, n // CLIP)
    starts = np.linspace(0, n - CLIP, n_windows, dtype=int)
    if j >= len(starts):
        return None, n
    return int(starts[j]), n


def parse_name(path: str):
    """'1970_1970-28415-0000_neg_003.wav' -> (spk, utt, chapter, j)"""
    stem = Path(path).stem
    if "_neg_" not in stem:
        return None
    head, _, jj = stem.rpartition("_neg_")
    spk, _, utt = head.partition("_")
    parts = utt.split("-")
    chapter = parts[1] if len(parts) >= 2 else ""
    try:
        j = int(jj)
    except ValueError:
        return None
    return spk, utt, chapter, j


def main():
    print("=" * 70)
    print("STAGE 1 -- REPRODUCE V2.3 best_loss OPERATING POINT (frozen, thr=0.43)")
    print("=" * 70)

    d = np.load(str(NPZ), allow_pickle=True)
    X, lab, cat, src = d["X"], d["labels"], d["negative_categories"], d["source_paths"]

    pos_i = (lab == 1.0).flatten()
    sp_i  = (cat == "speech")
    am_i  = (cat == "ambient")
    print(f"  frozen artifact: pos={pos_i.sum()} speech={sp_i.sum()} ambient={am_i.sum()}")

    m = tf.keras.models.load_model(str(MODEL))
    scores = m.predict(X, batch_size=64, verbose=0).flatten()

    s_pos, s_sp, s_amb = scores[pos_i], scores[sp_i], scores[am_i]
    got = {
        "TP": int((s_pos >= THR).sum()),
        "FN": int((s_pos < THR).sum()),
        "speech_FP": int((s_sp >= THR).sum()),
        "speech_TN": int((s_sp < THR).sum()),
        "ambient_FP": int((s_amb >= THR).sum()),
    }
    print(f"  {'metric':<12} {'expected':>9} {'got':>9}")
    ok = True
    for k, v in EXPECT.items():
        flag = "OK" if got[k] == v else "MISMATCH"
        if got[k] != v:
            ok = False
        print(f"  {k:<12} {v:>9} {got[k]:>9}   {flag}")
    if not ok:
        sys.exit("STOP: V2.3 best_loss operating point did NOT reproduce exactly.")
    print("  REPRODUCTION: EXACT -- proceeding.")

    # ---- extract speech FPs ----
    print("\nBuilding LibriSpeech index (flac + official transcripts)...")
    flac_idx, trans_idx = build_libri_index()
    print(f"  indexed {len(flac_idx)} flac stems, {len(trans_idx)} transcript lines")

    sp_src = src[sp_i]
    rows = []
    for path, sc in zip(sp_src, s_sp):
        if sc < THR:
            continue
        p = parse_name(path)
        if p is None:
            rows.append({"source_path": path, "score": float(sc), "parse": "FAILED"})
            continue
        spk, utt, chapter, j = p
        f = flac_idx.get(utt)
        start, nframes = (None, None)
        if f is not None:
            start, nframes = crop_start_sample(f, j)
        rows.append({
            "validation_example_id": Path(path).stem,
            "source_path": str(path),
            "speaker_id": spk,
            "chapter_id": chapter,
            "utterance_id": utt,
            "crop_index": j,
            "crop_start_sec": round(start / SR, 3) if start is not None else "",
            "crop_end_sec": round((start + CLIP) / SR, 3) if start is not None else "",
            "utterance_dur_sec": round(nframes / SR, 3) if nframes else "",
            "source_flac": str(f) if f else "NOT_FOUND",
            "transcript": trans_idx.get(utt, "NOT_FOUND"),
            "score": float(sc),
        })

    rows.sort(key=lambda r: -r["score"])
    print(f"\n  extracted {len(rows)} speech false positives at thr={THR}")
    resolved = sum(1 for r in rows if r.get("transcript") not in ("NOT_FOUND", None))
    timed    = sum(1 for r in rows if r.get("crop_start_sec") != "")
    print(f"  transcripts resolved : {resolved}/{len(rows)}")
    print(f"  crop timing resolved : {timed}/{len(rows)}")

    OUT.mkdir(exist_ok=True)
    fields = ["validation_example_id", "source_path", "speaker_id", "chapter_id",
              "utterance_id", "crop_index", "crop_start_sec", "crop_end_sec",
              "utterance_dur_sec", "source_flac", "transcript", "score"]
    with open(OUT / "v2_3_residual_speech_fp.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {OUT / 'v2_3_residual_speech_fp.csv'}")

    # cache all speech-negative scores + paths for later stages
    np.savez_compressed(OUT / "val_speech_scores.npz",
                        paths=sp_src, scores=s_sp)
    print(f"  cached speech-negative scores -> {OUT / 'val_speech_scores.npz'}")


if __name__ == "__main__":
    main()
