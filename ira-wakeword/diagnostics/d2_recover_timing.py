# -*- coding: utf-8 -*-
"""
Stage 2 -- Recover exact crop start times for the residual FPs by locating
each 1-second crop inside its source LibriSpeech utterance.

The speech_expanded generator numbered clips with a per-speaker counter
rather than a window index, so the offset cannot be derived arithmetically.
The crop is a verbatim slice (possibly peak-rescaled by a scalar and
quantised to PCM_16), so normalised cross-correlation recovers it exactly.
"""
import csv
import numpy as np
import soundfile as sf
from pathlib import Path

REPO = Path(__file__).parent.parent
CSV  = REPO / "diagnostics" / "v2_3_residual_speech_fp.csv"
SR, CLIP = 16000, 16000


def locate(clip: np.ndarray, src: np.ndarray):
    """Return (start_sample, normalised correlation) of clip within src."""
    if len(src) < len(clip):
        return None, 0.0
    c = clip - clip.mean()
    cn = np.linalg.norm(c)
    if cn == 0:
        return None, 0.0

    n = len(src) + len(clip)
    nfft = 1 << (n - 1).bit_length()
    S = np.fft.rfft(src, nfft)
    C = np.fft.rfft(c[::-1], nfft)
    corr = np.fft.irfft(S * C, nfft)[len(clip) - 1: len(src)]

    # local norms of src windows, via cumulative sums
    cs1 = np.concatenate([[0.0], np.cumsum(src)])
    cs2 = np.concatenate([[0.0], np.cumsum(src ** 2)])
    L = len(clip)
    idx = np.arange(0, len(src) - L + 1)
    win_sum = cs1[idx + L] - cs1[idx]
    win_sq = cs2[idx + L] - cs2[idx]
    win_norm = np.sqrt(np.maximum(win_sq - win_sum ** 2 / L, 1e-12))

    ncc = corr[: len(idx)] / (win_norm * cn)
    k = int(np.argmax(ncc))
    return int(idx[k]), float(ncc[k])


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"Recovering crop timing for {len(rows)} residual FPs...")

    ok = fail = 0
    for r in rows:
        wav = Path(r["source_path"])
        if not wav.is_absolute():
            wav = REPO / wav
        flac = Path(r["source_flac"])
        if not wav.exists() or not flac.exists():
            r["timing_method"] = "source_missing"
            fail += 1
            continue

        clip, _ = sf.read(str(wav), dtype="float64", always_2d=False)
        src, _ = sf.read(str(flac), dtype="float64", always_2d=False)
        if clip.ndim > 1:
            clip = clip.mean(1)
        if src.ndim > 1:
            src = src.mean(1)

        start, ncc = locate(clip, src)
        if start is None or ncc < 0.98:
            r["timing_method"] = f"unmatched(ncc={ncc:.3f})"
            fail += 1
            continue

        r["crop_start_sec"] = round(start / SR, 3)
        r["crop_end_sec"] = round((start + CLIP) / SR, 3)
        r["utterance_dur_sec"] = round(len(src) / SR, 3)
        r["timing_method"] = f"xcorr(ncc={ncc:.4f})"
        ok += 1

    print(f"  recovered: {ok}/{len(rows)}   failed: {fail}")

    fields = list(rows[0].keys())
    with open(CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  updated {CSV}")


if __name__ == "__main__":
    main()
