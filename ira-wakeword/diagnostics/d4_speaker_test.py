# -*- coding: utf-8 -*-
"""
Stage 4 -- Is the speaker concentration of the residual FPs real, and does it
extend to the near-miss band? Also checks chapter (recording-session) effects.

READ-ONLY.
"""
import collections
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent.parent
DIAG = REPO / "diagnostics"
THR = 0.43
rng = np.random.default_rng(7)


def parse(path):
    stem = Path(str(path)).stem
    head, _, _ = stem.rpartition("_neg_")
    spk, _, utt = head.partition("_")
    parts = utt.split("-")
    chap = parts[1] if len(parts) >= 2 else ""
    return spk, chap, utt


def main():
    d = np.load(DIAG / "val_speech_scores.npz", allow_pickle=True)
    paths, scores = d["paths"], d["scores"]
    spk = np.array([parse(p)[0] for p in paths])
    chap = np.array([parse(p)[1] for p in paths])

    is_fp = scores >= THR
    n, k = len(scores), int(is_fp.sum())
    base = k / n

    print("=" * 74)
    print("STAGE 4 -- IS THE SPEAKER CONCENTRATION REAL?")
    print("=" * 74)
    print("  %d speech negatives, %d FPs, base rate %.2f%%" % (n, k, base * 100))

    # --- permutation test on the concentration statistic ---
    obs_counts = collections.Counter(spk[is_fp])
    obs_top5 = sum(c for _, c in obs_counts.most_common(5))
    obs_max = obs_counts.most_common(1)[0][1]

    perm_top5, perm_max = [], []
    idx = np.arange(n)
    for _ in range(20000):
        pick = rng.choice(idx, size=k, replace=False)
        c = collections.Counter(spk[pick])
        perm_top5.append(sum(v for _, v in c.most_common(5)))
        perm_max.append(c.most_common(1)[0][1])
    perm_top5 = np.array(perm_top5)
    perm_max = np.array(perm_max)

    p_top5 = float((perm_top5 >= obs_top5).mean())
    p_max = float((perm_max >= obs_max).mean())
    print("\n  Permutation test (20000 draws, FPs reassigned at random):")
    print("    top-5-speaker FP mass : observed %d, null mean %.1f, p = %.5f"
          % (obs_top5, perm_top5.mean(), p_top5))
    print("    single-speaker max    : observed %d, null mean %.1f, p = %.5f"
          % (obs_max, perm_max.mean(), p_max))

    # --- per-speaker binomial-style view ---
    print("\n  Per-speaker FP rate vs base rate %.2f%%:" % (base * 100))
    print("    %6s %6s %5s %8s %8s" % ("spk", "clips", "FP", "rate", "x base"))
    tot = collections.Counter(spk)
    for s in sorted(tot, key=lambda s: -(obs_counts.get(s, 0) / tot[s])):
        c = obs_counts.get(s, 0)
        r = c / tot[s]
        print("    %6s %6d %5d %7.2f%% %7.2fx" % (s, tot[s], c, r * 100, r / base))

    # --- does the same skew appear in the near-miss band? ---
    print("\n  Near-miss band (top 100 TNs below 0.43) -- speaker mix:")
    order = np.argsort(-scores)
    nm = [i for i in order if scores[i] < THR][:100]
    nm_c = collections.Counter(spk[nm])
    for s, c in nm_c.most_common(6):
        print("    spk %-6s %3d/100 near-miss   (FPs: %d)" % (s, c, obs_counts.get(s, 0)))

    # --- mean score per speaker: a whole-distribution shift, or just a tail? ---
    print("\n  Whole-distribution view (is the speaker shifted, or only its tail?):")
    print("    %6s %9s %9s %9s %9s" % ("spk", "mean", "median", "P90", "P99"))
    for s in sorted(tot, key=lambda s: -(obs_counts.get(s, 0) / tot[s]))[:8]:
        v = scores[spk == s]
        print("    %6s %9.4f %9.4f %9.4f %9.4f"
              % (s, v.mean(), np.median(v), np.percentile(v, 90), np.percentile(v, 99)))
    print("    %6s %9.4f %9.4f %9.4f %9.4f"
          % ("ALL", scores.mean(), np.median(scores),
             np.percentile(scores, 90), np.percentile(scores, 99)))

    # --- chapter concentration within the worst speakers ---
    print("\n  Chapter (recording session) spread of FPs, worst speakers:")
    for s, _ in obs_counts.most_common(4):
        m = (spk == s)
        ch_tot = collections.Counter(chap[m])
        ch_fp = collections.Counter(chap[m & is_fp])
        parts = ["%s:%d/%d" % (c, ch_fp.get(c, 0), ch_tot[c]) for c in sorted(ch_tot)]
        print("    spk %-6s %s" % (s, "  ".join(parts)))


if __name__ == "__main__":
    main()
