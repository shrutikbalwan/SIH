# -*- coding: utf-8 -*-
"""
Stage 5 -- Chapter (recording-session) concentration and simple acoustic
descriptors, to separate 'this speaker's voice' from 'this recording session'.

READ-ONLY.
"""
import collections
import numpy as np
import soundfile as sf
from pathlib import Path

REPO = Path(__file__).parent.parent
DIAG = REPO / "diagnostics"
THR = 0.43
rng = np.random.default_rng(11)


def parse(path):
    stem = Path(str(path)).stem
    head, _, _ = stem.rpartition("_neg_")
    spk, _, utt = head.partition("_")
    parts = utt.split("-")
    return spk, (parts[1] if len(parts) >= 2 else "")


def descriptors(wav_paths, limit=40):
    """Crude recording-condition descriptors averaged over a sample of clips."""
    rms, cent, flat, hf = [], [], [], []
    for p in wav_paths[:limit]:
        pp = Path(str(p))
        if not pp.is_absolute():
            pp = REPO / pp
        if not pp.exists():
            continue
        a, sr = sf.read(str(pp), dtype="float64")
        if a.ndim > 1:
            a = a.mean(1)
        rms.append(float(np.sqrt(np.mean(a ** 2))))
        S = np.abs(np.fft.rfft(a * np.hanning(len(a))))
        f = np.fft.rfftfreq(len(a), 1 / sr)
        p_ = S ** 2 + 1e-20
        cent.append(float((f * p_).sum() / p_.sum()))
        flat.append(float(np.exp(np.mean(np.log(p_))) / np.mean(p_)))
        hf.append(float(p_[f > 4000].sum() / p_.sum()))
    if not rms:
        return None
    return dict(rms=np.mean(rms), centroid=np.mean(cent),
                flatness=np.mean(flat), hf_frac=np.mean(hf))


def main():
    d = np.load(DIAG / "val_speech_scores.npz", allow_pickle=True)
    paths, scores = d["paths"], d["scores"]
    spk = np.array([parse(p)[0] for p in paths])
    chap = np.array([parse(p)[1] for p in paths])
    key = np.array(["%s/%s" % (s, c) for s, c in zip(spk, chap)])

    is_fp = scores >= THR
    n, k = len(scores), int(is_fp.sum())
    base = k / n

    print("=" * 74)
    print("STAGE 5 -- CHAPTER (RECORDING SESSION) CONCENTRATION")
    print("=" * 74)
    tot = collections.Counter(key)
    fpc = collections.Counter(key[is_fp])
    print("  %d speech negatives across %d speaker/chapter sessions, base rate %.2f%%"
          % (n, len(tot), base * 100))

    rows = [(kk, fpc.get(kk, 0), tot[kk], fpc.get(kk, 0) / tot[kk]) for kk in tot]
    rows.sort(key=lambda r: (-r[1], -r[3]))
    print("\n  Top 12 sessions by FP count:")
    print("    %-16s %5s %7s %9s %8s" % ("spk/chapter", "FP", "clips", "rate", "x base"))
    for kk, c, t, r in rows[:12]:
        print("    %-16s %5d %7d %8.2f%% %7.2fx" % (kk, c, t, r * 100, r / base))

    top2 = rows[:2]
    m2 = sum(r[1] for r in top2)
    c2 = sum(r[2] for r in top2)
    print("\n  Top 2 sessions: %d/%d FPs (%.1f%%) from %d/%d clips (%.1f%% of the set)"
          % (m2, k, m2 / k * 100, c2, n, c2 / n * 100))

    # permutation test at session level
    obs_top2 = m2
    idx = np.arange(n)
    perm = []
    for _ in range(20000):
        pick = rng.choice(idx, size=k, replace=False)
        c = collections.Counter(key[pick])
        perm.append(sum(v for _, v in c.most_common(2)))
    perm = np.array(perm)
    print("  Permutation test: observed %d, null mean %.1f, p = %.5f"
          % (obs_top2, perm.mean(), float((perm >= obs_top2).mean())))

    # sessions of the same speakers, for contrast
    print("\n  Same speakers, other sessions (is it the voice or the session?):")
    for s in ["652", "1988"]:
        for kk, c, t, r in sorted([x for x in rows if x[0].startswith(s + "/")],
                                  key=lambda x: -x[3]):
            v = scores[key == kk]
            print("    %-16s FP %2d/%-3d  rate %6.2f%%   mean=%.4f median=%.4f P90=%.4f"
                  % (kk, c, t, r * 100, v.mean(), np.median(v), np.percentile(v, 90)))

    # acoustic descriptors for the worst sessions vs a clean contrast
    print("\n  Recording-condition descriptors (mean over <=40 clips per session):")
    print("    %-16s %10s %10s %10s %10s" % ("spk/chapter", "rms", "centroid", "flatness", "hf_frac"))
    interesting = [r[0] for r in rows[:4]] + [r[0] for r in rows if r[1] == 0][:3]
    for kk in interesting:
        sel = [p for p, kx in zip(paths, key) if kx == kk]
        de = descriptors(sel)
        if de:
            print("    %-16s %10.4f %10.1f %10.5f %10.4f"
                  % (kk, de["rms"], de["centroid"], de["flatness"], de["hf_frac"]))


if __name__ == "__main__":
    main()
