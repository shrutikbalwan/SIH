# -*- coding: utf-8 -*-
"""
Stage 6 -- TRAIN-ONLY confuser search (spec section 7) and difficulty-pool
membership (spec section 8).

Searches ONLY TRAIN LibriSpeech speakers for the ira-like phone pattern, scores
the candidates with V2.3 best_loss, and reports where they fall in the existing
score-based easy/medium/hard pools.

No validation clip is ever proposed as training data. READ-ONLY.
"""
import os, csv, json, collections, sys
import numpy as np
import soundfile as sf
from pathlib import Path
import cmudict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO = Path(__file__).parent.parent
DIAG = REPO / "diagnostics"
MANIFEST = REPO / "dataset" / "split_manifest_v3.csv"
MODEL = REPO / "cnn" / "models" / "ira_cnn_v2_3_best_loss.keras"

SR = CLIP = 16000
EASY_T, HARD_T = 0.10, 0.50
CMU = cmudict.dict()

FRONT_HIGH = {"IY", "IH"}
REDUCED = {"AH", "AA", "ER", "AO", "IH", "OW", "AE", "EH"}

LIBRI_ROOTS = [
    REPO / "dataset" / "negative" / "extracted" / "LibriSpeech",
    REPO / "dataset" / "negative" / "downloads" / "LibriSpeech",
    Path("E:/SIH/ira-wakeword/dataset/negative/extracted/LibriSpeech"),
]

STFT_FRAME_LEN, STFT_FRAME_STEP, STFT_FFT_LEN, N_FREQ_BINS = 480, 320, 512, 40


def phones(word):
    pr = CMU.get(word.lower())
    return [p.rstrip("0123456789") for p in pr[0]] if pr else None


def ira_hits(words):
    seq, oov = [], []
    for w in words:
        ph = phones(w)
        if ph is None:
            oov.append(w.lower())
            continue
        seq.extend(ph)
    hits = []
    for i in range(len(seq) - 2):
        if seq[i] in FRONT_HIGH and seq[i + 1] == "R" and seq[i + 2] in REDUCED:
            hits.append(" ".join(seq[i:i + 3]))
    return hits, oov


def word_hits(transcript):
    """Which individual words carry the pattern (for the 'reason selected')."""
    out = []
    for w in transcript.split():
        ph = phones(w)
        if not ph:
            continue
        for i in range(len(ph) - 2):
            if ph[i] in FRONT_HIGH and ph[i + 1] == "R" and ph[i + 2] in REDUCED:
                out.append(w.lower())
                break
    return out


def build_index():
    flac, trans = {}, {}
    for root in LIBRI_ROOTS:
        if not root.exists():
            continue
        for f in root.rglob("*.flac"):
            flac.setdefault(f.stem, f)
        for t in root.rglob("*.trans.txt"):
            for line in t.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    uid, _, txt = line.partition(" ")
                    trans.setdefault(uid, txt.strip())
    return flac, trans


def locate(clip, src):
    if len(src) < len(clip):
        return None, 0.0
    c = clip - clip.mean()
    cn = np.linalg.norm(c)
    if cn == 0:
        return None, 0.0
    nfft = 1 << ((len(src) + len(clip)) - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(src, nfft) * np.fft.rfft(c[::-1], nfft),
                        nfft)[len(clip) - 1: len(src)]
    L = len(clip)
    cs1 = np.concatenate([[0.0], np.cumsum(src)])
    cs2 = np.concatenate([[0.0], np.cumsum(src ** 2)])
    idx = np.arange(0, len(src) - L + 1)
    wsum = cs1[idx + L] - cs1[idx]
    wsq = cs2[idx + L] - cs2[idx]
    wn = np.sqrt(np.maximum(wsq - wsum ** 2 / L, 1e-12))
    ncc = corr[: len(idx)] / (wn * cn)
    k = int(np.argmax(ncc))
    return int(idx[k]), float(ncc[k])


def window_words(transcript, s, e, dur):
    words = transcript.split()
    if not words or dur <= 0:
        return []
    lens = np.array([len(w) + 1 for w in words], dtype=float)
    ed = np.concatenate([[0.0], np.cumsum(lens)])
    ed = ed / ed[-1] * dur
    return [w for i, w in enumerate(words) if ed[i + 1] > s and ed[i] < e]


def spec(a):
    t = tf.convert_to_tensor(a, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN, frame_step=STFT_FRAME_STEP,
                       fft_length=STFT_FFT_LEN)
    s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_FREQ_BINS]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)


def load_manifest():
    rows = collections.defaultdict(list)
    spk_by_split = collections.defaultdict(set)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = r["split"].strip()
            if r["label"] != "0":
                continue
            g = r.get("group", "").lower()
            if "libri" not in g and "speech" not in g:
                continue
            p = r["path"]
            rows[sp].append(p)
            s = r.get("speaker_id", "").strip()
            if s:
                spk_by_split[sp].add(s)
    return rows, spk_by_split


def main():
    flac_idx, trans_idx = build_index()
    rows, spk_by_split = load_manifest()
    train = rows["train"]
    print("=" * 74)
    print("STAGE 6 -- TRAIN-ONLY CONFUSER SEARCH")
    print("=" * 74)
    print("  TRAIN speech negatives: %d" % len(train))

    # --- speaker disjointness assertion (spec section 7) ---
    tr_s = spk_by_split["train"]
    for other in ["validation", "test", "unseen_test"]:
        inter = tr_s & spk_by_split[other]
        print("  TRAIN vs %-12s speaker overlap: %d %s"
              % (other, len(inter), "OK" if not inter else "LEAKAGE " + str(inter)))
        if inter:
            sys.exit("STOP: speaker leakage between train and " + other)

    # --- utterance-level prefilter, then exact crop-window check ---
    def parse(p):
        stem = Path(p).stem
        head, _, _ = stem.rpartition("_neg_")
        spk, _, utt = head.partition("_")
        return spk, utt

    prefilter = []
    for p in train:
        spk, utt = parse(p)
        tr = trans_idx.get(utt)
        if tr and word_hits(tr):
            prefilter.append((p, spk, utt, tr))
    print("\n  TRAIN clips whose SOURCE UTTERANCE carries the pattern: %d" % len(prefilter))

    cands = []
    for p, spk, utt, tr in prefilter:
        wav = Path(p)
        if not wav.is_absolute():
            wav = REPO / wav
        fl = flac_idx.get(utt)
        if not wav.exists() or not fl:
            continue
        clip, _ = sf.read(str(wav), dtype="float64")
        src, _ = sf.read(str(fl), dtype="float64")
        if clip.ndim > 1:
            clip = clip.mean(1)
        if src.ndim > 1:
            src = src.mean(1)
        st, ncc = locate(clip, src)
        if st is None or ncc < 0.98:
            continue
        ws = window_words(tr, st / SR, (st + CLIP) / SR, len(src) / SR)
        hits, _ = ira_hits(ws)
        if not hits:
            continue
        cands.append({"path": p, "speaker_id": spk, "utterance_id": utt,
                      "crop_start_sec": round(st / SR, 3),
                      "crop_end_sec": round((st + CLIP) / SR, 3),
                      "transcript": tr,
                      "crop_words_estimated": " ".join(ws),
                      "trigger_words": ",".join(sorted(set(word_hits(" ".join(ws))))),
                      "phone_pattern": ";".join(sorted(set(hits))),
                      "reason_selected": "estimated crop window contains <IY|IH> R <vowel>"})
    print("  TRAIN clips whose ESTIMATED CROP WINDOW carries the pattern: %d" % len(cands))

    if not cands:
        print("  No candidates -- nothing to score.")
        return

    # --- score candidates with V2.3 best_loss ---
    print("\n  Scoring %d candidates with V2.3 best_loss..." % len(cands))
    m = tf.keras.models.load_model(str(MODEL))
    X = []
    for c in cands:
        wav = Path(c["path"])
        if not wav.is_absolute():
            wav = REPO / wav
        a, sr = sf.read(str(wav), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if len(a) < CLIP:
            a = np.pad(a, (0, CLIP - len(a)))
        X.append(np.expand_dims(spec(a[:CLIP]), -1))
    sc = m.predict(np.array(X, dtype=np.float32), batch_size=128, verbose=0).flatten()
    for c, s in zip(cands, sc):
        c["v2_3_score"] = round(float(s), 6)
        c["pool"] = "easy" if s < EASY_T else ("medium" if s < HARD_T else "hard")

    cands.sort(key=lambda c: -c["v2_3_score"])

    # --- section 8: pool membership ---
    print("\n" + "=" * 74)
    print("SECTION 8 -- WHERE DO THE CONFUSER CANDIDATES FALL IN THE EXISTING POOLS?")
    print("=" * 74)
    pc = collections.Counter(c["pool"] for c in cands)
    print("  Candidate pool membership (V2 pools are score-based on the V2 baseline;")
    print("  scores here are V2.3 best_loss, reported as requested):")
    for k in ["easy", "medium", "hard"]:
        print("    %-7s %4d  (%.1f%%)" % (k, pc.get(k, 0), pc.get(k, 0) / len(cands) * 100))
    print("\n  Candidate score distribution: mean=%.4f median=%.4f P90=%.4f max=%.4f"
          % (np.mean(sc), np.median(sc), np.percentile(sc, 90), np.max(sc)))

    print("\n  Top 15 TRAIN-only candidates by V2.3 score:")
    print("    %8s %7s %-7s %-22s %s" % ("score", "spk", "pool", "trigger", "crop words (est.)"))
    for c in cands[:15]:
        print("    %8.4f %7s %-7s %-22s %s"
              % (c["v2_3_score"], c["speaker_id"], c["pool"],
                 c["trigger_words"][:22], c["crop_words_estimated"][:52]))

    fields = ["path", "speaker_id", "utterance_id", "crop_start_sec", "crop_end_sec",
              "v2_3_score", "pool", "trigger_words", "phone_pattern",
              "crop_words_estimated", "transcript", "reason_selected"]
    with open(DIAG / "train_only_confuser_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(cands)
    print("\n  wrote train_only_confuser_candidates.csv (%d rows, TRAIN speakers only)"
          % len(cands))

    # --- base-rate contrast: do pattern clips score higher than TRAIN at large? ---
    json.dump({"n_candidates": len(cands), "pool_counts": dict(pc),
               "score_mean": float(np.mean(sc)), "score_median": float(np.median(sc))},
              open(DIAG / "train_confuser_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
