# -*- coding: utf-8 -*-
"""
Stage 3 -- Score-distribution context, speaker concentration, and
transcript/phonetic analysis of the residual V2.3 speech false positives.

READ-ONLY. No training, no dataset modification.
"""
import csv, json, collections, random
import numpy as np
import soundfile as sf
from pathlib import Path
import cmudict

REPO = Path(__file__).parent.parent
DIAG = REPO / "diagnostics"
SR, CLIP, THR = 16000, 16000, 0.43
random.seed(1234)

CMU = cmudict.dict()

LIBRI_ROOTS = [
    REPO / "dataset" / "negative" / "extracted" / "LibriSpeech",
    REPO / "dataset" / "negative" / "downloads" / "LibriSpeech",
    Path("E:/SIH/ira-wakeword/dataset/negative/extracted/LibriSpeech"),
]

# Wakeword "ira" ~= EE-rah.  CMUdict lists IRA as AY1 R AH0 (eye-rah), which is
# NOT the target pronunciation, so the target is specified explicitly by phones.
FRONT_HIGH = {"IY", "IH"}                                      # the "EE"
REDUCED    = {"AH", "AA", "ER", "AO", "IH", "OW", "AE", "EH"}  # the "-ah"


def strip_stress(ph):
    return ph.rstrip("0123456789")


def phones(word):
    prons = CMU.get(word.lower())
    if not prons:
        return None
    return [strip_stress(p) for p in prons[0]]


def ira_like(seq):
    """Does the phone sequence contain <front-high vowel> R <vowel>?"""
    hits = []
    for i in range(len(seq) - 2):
        if seq[i] in FRONT_HIGH and seq[i + 1] == "R" and seq[i + 2] in REDUCED:
            hits.append(" ".join(seq[i:i + 3]))
    return hits


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


def parse_name(path):
    stem = Path(path).stem
    head, _, _ = stem.rpartition("_neg_")
    spk, _, utt = head.partition("_")
    return spk, utt


def window_words(transcript, start_s, end_s, dur_s):
    """Approximate the words overlapping the crop.

    LibriSpeech ships NO official word-level time alignments, so this is a
    UNIFORM PROPORTIONAL ESTIMATE (words allocated across the utterance in
    proportion to character length). It is an estimate, not an alignment.
    """
    words = transcript.split()
    if not words or dur_s <= 0:
        return []
    lens = np.array([len(w) + 1 for w in words], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(lens)])
    edges = edges / edges[-1] * dur_s
    return [w for i, w in enumerate(words)
            if edges[i + 1] > start_s and edges[i] < end_s]


def group_stats(name, scores, speakers):
    per_spk = collections.Counter(speakers)
    vals = list(per_spk.values())
    return {
        "group": name,
        "n": len(scores),
        "score_mean": round(float(np.mean(scores)), 4),
        "score_median": round(float(np.median(scores)), 4),
        "score_P10": round(float(np.percentile(scores, 10)), 4),
        "score_P90": round(float(np.percentile(scores, 90)), 4),
        "speaker_count": len(per_spk),
        "clips_per_speaker_min": min(vals) if vals else 0,
        "clips_per_speaker_median": float(np.median(vals)) if vals else 0,
        "clips_per_speaker_max": max(vals) if vals else 0,
    }


def main():
    flac_idx, trans_idx = build_index()
    d = np.load(DIAG / "val_speech_scores.npz", allow_pickle=True)
    paths, scores = d["paths"], d["scores"]

    spk_all = [parse_name(p)[0] for p in paths]
    total_per_spk = collections.Counter(spk_all)

    order = np.argsort(-scores)
    fp_idx = [i for i in order if scores[i] >= THR]
    tn_sorted = [i for i in order if scores[i] < THR]
    B_idx = tn_sorted[:100]
    low = [i for i in range(len(scores)) if scores[i] < 0.10]
    C_idx = random.sample(low, 100)

    print("=" * 74)
    print("SECTION 4 -- SCORE-DISTRIBUTION CONTEXT (frozen EXPANDED_VAL speech negatives)")
    print("=" * 74)
    groups = {}
    for nm, idxs in [("A: false positives (>=0.43)", fp_idx),
                     ("B: near-miss TN (top 100 <0.43)", B_idx),
                     ("C: low-score control (100 random <0.10)", C_idx)]:
        g = group_stats(nm, scores[idxs], [spk_all[i] for i in idxs])
        groups[nm] = g
        print("\n  " + nm)
        print("    n=%d  mean=%s  median=%s  P10=%s  P90=%s"
              % (g["n"], g["score_mean"], g["score_median"], g["score_P10"], g["score_P90"]))
        print("    speakers=%d  clips/speaker min=%d med=%s max=%d"
              % (g["speaker_count"], g["clips_per_speaker_min"],
                 g["clips_per_speaker_median"], g["clips_per_speaker_max"]))
    print("\n  (total speech negatives with score<0.10: %d/%d)" % (len(low), len(scores)))

    # ---------------- Section 5: speaker concentration ----------------
    print("\n" + "=" * 74)
    print("SECTION 5 -- SPEAKER CONCENTRATION OF THE 58 FALSE POSITIVES")
    print("=" * 74)
    fp_spk = collections.Counter(spk_all[i] for i in fp_idx)
    print("  unique speakers with >=1 FP : %d of %d validation speakers"
          % (len(fp_spk), len(total_per_spk)))
    print("  overall speech FPR          : %d/%d = %.2f%%"
          % (len(fp_idx), len(scores), len(fp_idx) / len(scores) * 100))

    rows = [{"speaker": s, "fp": c, "total": total_per_spk[s],
             "rate": c / total_per_spk[s] * 100} for s, c in fp_spk.items()]

    print("\n  Top 10 speakers by FP COUNT")
    print("    %6s %4s %6s %9s" % ("spk", "FP", "clips", "FP rate"))
    for r in sorted(rows, key=lambda x: (-x["fp"], -x["rate"]))[:10]:
        print("    %6s %4d %6d %8.2f%%" % (r["speaker"], r["fp"], r["total"], r["rate"]))

    print("\n  Top 10 speakers by FP RATE")
    print("    %6s %4s %6s %9s" % ("spk", "FP", "clips", "FP rate"))
    for r in sorted(rows, key=lambda x: (-x["rate"], -x["fp"]))[:10]:
        print("    %6s %4d %6d %8.2f%%" % (r["speaker"], r["fp"], r["total"], r["rate"]))

    top5 = sorted(rows, key=lambda x: -x["fp"])[:5]
    print("\n  Concentration: top 5 speakers hold %d/58 FPs (%.1f%%) from %d/%d clips (%.1f%% of the set)"
          % (sum(r["fp"] for r in top5), sum(r["fp"] for r in top5) / 58 * 100,
             sum(r["total"] for r in top5), len(scores),
             sum(r["total"] for r in top5) / len(scores) * 100))

    with open(DIAG / "v2_3_fp_speaker_concentration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["speaker", "fp", "total", "rate"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: -x["fp"]))

    # ---------------- Section 6: transcript / phonetic ----------------
    print("\n" + "=" * 74)
    print("SECTION 6 -- TRANSCRIPT / PHONETIC CONFUSER ANALYSIS")
    print("=" * 74)
    print('  Target: "ira" ~= EE-rah. CMUdict lists IRA as AY1 R AH0 (eye-rah),')
    print("  which is NOT the target, so the pattern is specified by phones:")
    print("    <IY|IH> R <AH|AA|ER|AO|IH|OW|AE|EH>   (stress stripped)")
    print("  NOTE: LibriSpeech has NO official word-level time alignments.")
    print("  Overlapping words are a UNIFORM PROPORTIONAL ESTIMATE, not an alignment.")

    def analyse(idxs, label):
        words_ctr = collections.Counter()
        bigrams_ctr = collections.Counter()
        patt_ctr = collections.Counter()
        oov = collections.Counter()
        n_hit = n_done = 0
        detail = []
        for i in idxs:
            p = Path(str(paths[i]))
            wav = p if p.is_absolute() else REPO / p
            spk, utt = parse_name(paths[i])
            tr = trans_idx.get(utt, "")
            fl = flac_idx.get(utt)
            if not (wav.exists() and fl and tr):
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
            n_done += 1
            s_s, e_s, dur = st / SR, (st + CLIP) / SR, len(src) / SR
            ws = window_words(tr, s_s, e_s, dur)
            words_ctr.update(w.lower() for w in ws)
            bigrams_ctr.update("%s %s" % (a.lower(), b.lower()) for a, b in zip(ws, ws[1:]))

            seq = []
            for w in ws:
                ph = phones(w)
                if ph is None:
                    oov[w.lower()] += 1
                    continue
                seq.extend(ph)
            hits = ira_like(seq)
            if hits:
                n_hit += 1
                patt_ctr.update(hits)
            detail.append({"idx": int(i), "score": float(scores[i]), "spk": spk,
                           "utt": utt, "start": round(s_s, 3), "end": round(e_s, 3),
                           "words": " ".join(ws), "ira_like": ";".join(hits)})
        return dict(words=words_ctr, bigrams=bigrams_ctr, patterns=patt_ctr,
                    n_hit=n_hit, n=n_done, oov=oov, detail=detail, label=label)

    A = analyse(fp_idx, "A: FP")
    B = analyse(B_idx, "B: near-miss TN")
    C = analyse(C_idx, "C: low-score control")

    print("\n  --- ira-like phone pattern <IY|IH> R <vowel> in the estimated crop window ---")
    print("    %-26s %6s %13s %8s" % ("group", "clips", "with pattern", "rate"))
    for g in (A, B, C):
        print("    %-26s %6d %13d %7.1f%%"
              % (g["label"], g["n"], g["n_hit"], g["n_hit"] / max(1, g["n"]) * 100))

    print("\n  Pattern instances in group A (FPs):")
    for k, v in A["patterns"].most_common(10):
        print("    %-16s %d" % (k, v))
    if not A["patterns"]:
        print("    (none)")

    print("\n  Most frequent words in FP crop windows (top 15):")
    for k, v in A["words"].most_common(15):
        print("    %-16s %d" % (k, v))

    print("\n  Repeated bigrams in FP crop windows:")
    shown = False
    for k, v in A["bigrams"].most_common(20):
        if v > 1:
            print("    %-24s %d" % (k, v))
            shown = True
    if not shown:
        print("    (no bigram occurs more than once)")

    if A["oov"]:
        print("\n  Words absent from CMUdict (no pronunciation invented): %d distinct, e.g. %s"
              % (len(A["oov"]), list(A["oov"])[:8]))

    json.dump({
        "groups": groups,
        "ira_like_rate": {g["label"]: {"n": g["n"], "hits": g["n_hit"]} for g in (A, B, C)},
        "fp_patterns": A["patterns"].most_common(),
        "fp_top_words": A["words"].most_common(40),
        "fp_top_bigrams": A["bigrams"].most_common(40),
        "speaker_concentration": sorted(rows, key=lambda x: -x["fp"]),
    }, open(DIAG / "v2_3_fp_analysis.json", "w"), indent=2)

    with open(DIAG / "v2_3_fp_crop_words.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["idx", "score", "spk", "utt", "start", "end",
                                           "words", "ira_like"])
        w.writeheader()
        w.writerows(sorted(A["detail"], key=lambda r: -r["score"]))
    print("\n  wrote v2_3_fp_analysis.json, v2_3_fp_crop_words.csv, "
          "v2_3_fp_speaker_concentration.csv")


if __name__ == "__main__":
    main()
