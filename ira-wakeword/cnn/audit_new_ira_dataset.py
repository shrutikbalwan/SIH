# -*- coding: utf-8 -*-
"""
Audit the NEW real-human "Ira" recordings in dataset/new data set/.

READ-ONLY. This script does not resample, normalize, trim, convert, delete,
rename, move, or otherwise modify any recording. It loads no wake-word model
and runs no inference.

Outputs:
    dataset/new data set/audit_new_ira_dataset.csv
    dataset/new data set/AUDIT_REPORT.md
"""
import os, csv, sys, json, hashlib, argparse, collections
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).parent.parent
DEFAULT_SRC = REPO / "dataset" / "new data set"

EXPECTED_SR = 16000
EXPECTED_CH = 1
CLIP_T = 0.999           # sample magnitude counted as clipped
CLIP_MIN_SAMPLES = 1     # >=1 clipped sample flags the file
NEAR_SILENT_RMS = 1e-4
SILENT_RMS = 1e-6


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_audio(mono):
    """Hash of decoded PCM16 samples: catches duplicates across containers."""
    return hashlib.sha256(
        np.round(np.asarray(mono, dtype=np.float64) * 32767).astype(np.int16).tobytes()
    ).hexdigest()


def fmt(v, spec="%.6f"):
    return "" if v is None else (spec % v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    args = ap.parse_args()
    src = Path(args.src)

    print("=" * 78)
    print("NEW REAL-HUMAN 'IRA' DATASET AUDIT")
    print("=" * 78)
    print("  source : %s" % src)
    print("  READ-ONLY: no resampling, normalization, trimming, conversion,")
    print("             renaming, deletion, or any other modification.")
    print("  No model is loaded. No inference is run.")

    if not src.exists():
        print("\n  ERROR: source directory not found.")
        return 2

    wavs = sorted((p for p in src.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"),
                  key=lambda p: str(p).lower())
    others = sorted(p for p in src.rglob("*")
                    if p.is_file() and p.suffix.lower() not in (".wav", ".csv", ".md"))
    print("\n  WAV files found     : %d" % len(wavs))
    if others:
        print("  non-WAV files found : %d  %s" % (len(others), [p.name for p in others[:5]]))

    rows, unreadable = [], []
    for p in wavs:
        rel = str(p.relative_to(src)).replace("\\", "/")
        rec = {"filename": p.name, "relative_path": rel, "bytes": p.stat().st_size}
        try:
            info = sf.info(str(p))
            data, sr = sf.read(str(p), dtype="float64", always_2d=False)
            mono = data if data.ndim == 1 else data.mean(axis=1)
            n = int(len(mono))
            peak = float(np.max(np.abs(mono))) if n else 0.0
            rms = float(np.sqrt(np.mean(mono ** 2))) if n else 0.0
            nclip = int(np.sum(np.abs(mono) >= CLIP_T))
            rec.update(
                readable=True,
                sample_rate=info.samplerate,
                channels=info.channels,
                n_samples=int(info.frames),
                duration_s=float(info.duration),
                format=info.format,
                subtype=info.subtype,
                peak_amplitude=peak,
                rms=rms,
                clipped_samples=nclip,
                clipped_pct=(nclip / n * 100.0) if n else 0.0,
                is_clipped=bool(nclip >= CLIP_MIN_SAMPLES),
                is_silent=bool(rms < SILENT_RMS),
                is_near_silent=bool(rms < NEAR_SILENT_RMS),
                sha256_file=sha256_file(p),
                sha256_audio=sha256_audio(mono),
                error="",
            )
        except Exception as e:
            rec.update(readable=False, error="%s: %s" % (type(e).__name__, e))
            for k in ("sample_rate", "channels", "n_samples", "duration_s", "format",
                      "subtype", "peak_amplitude", "rms", "clipped_samples", "clipped_pct",
                      "is_clipped", "is_silent", "is_near_silent", "sha256_audio"):
                rec.setdefault(k, "")
            try:
                rec["sha256_file"] = sha256_file(p)
            except Exception:
                rec["sha256_file"] = ""
            unreadable.append(rec)
        rows.append(rec)

    ok = [r for r in rows if r.get("readable")]

    # ------------------------------------------------------------ per-file
    print("\n" + "-" * 78)
    print("  PER-FILE DETAIL")
    print("-" * 78)
    print("  %-42s %6s %3s %8s %9s %9s %5s" %
          ("filename", "SR", "ch", "dur_s", "peak", "rms", "clip"))
    for r in rows:
        if not r.get("readable"):
            print("  %-42s  UNREADABLE: %s" % (r["filename"][:42], r["error"]))
            continue
        print("  %-42s %6d %3d %8.3f %9.5f %9.5f %5s" %
              (r["filename"][:42], r["sample_rate"], r["channels"], r["duration_s"],
               r["peak_amplitude"], r["rms"], "YES" if r["is_clipped"] else "-"))

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    issues, notes = [], []

    print("  total WAV files : %d" % len(wavs))
    print("  readable        : %d" % len(ok))
    print("  unreadable      : %d" % len(unreadable))
    if unreadable:
        issues.append("%d unreadable file(s)" % len(unreadable))
        for r in unreadable:
            print("      %s -> %s" % (r["filename"], r["error"]))

    if not ok:
        print("\n  Nothing readable to summarise.")
        return 2

    sr_dist = collections.Counter(r["sample_rate"] for r in ok)
    ch_dist = collections.Counter(r["channels"] for r in ok)
    fmt_dist = collections.Counter("%s/%s" % (r["format"], r["subtype"]) for r in ok)

    print("\n  sample-rate distribution:")
    for k, v in sorted(sr_dist.items()):
        flag = "  <-- EXPECTED" if k == EXPECTED_SR else "  <-- NOT %d Hz" % EXPECTED_SR
        print("      %6d Hz : %3d%s" % (k, v, flag))
    if set(sr_dist) != {EXPECTED_SR}:
        issues.append("sample rates other than %d Hz present: %s"
                      % (EXPECTED_SR, dict(sr_dist)))

    print("\n  channel distribution:")
    for k, v in sorted(ch_dist.items()):
        lbl = {1: "mono", 2: "stereo"}.get(k, "%d-channel" % k)
        flag = "  <-- EXPECTED" if k == EXPECTED_CH else "  <-- NOT mono"
        print("      %-8s : %3d%s" % (lbl, v, flag))
    if set(ch_dist) != {EXPECTED_CH}:
        issues.append("non-mono files present: %s" % dict(ch_dist))

    print("\n  format/subtype distribution:")
    for k, v in fmt_dist.most_common():
        print("      %-16s : %3d" % (k, v))

    d = np.array([r["duration_s"] for r in ok])
    print("\n  duration (s): min=%.3f  max=%.3f  mean=%.3f  median=%.3f"
          % (d.min(), d.max(), d.mean(), float(np.median(d))))
    lt1 = int((d < 1.0).sum())
    gt1 = int((d > 1.0).sum())
    eq1 = int((np.abs(d - 1.0) <= 1e-9).sum())
    gt3 = int((d > 3.0).sum())
    print("      shorter than 1 s : %d" % lt1)
    print("      longer  than 1 s : %d" % gt1)
    print("      exactly     1 s  : %d" % eq1)
    print("      longer  than 3 s : %d" % gt3)
    if eq1 > len(ok) * 0.5:
        issues.append("most files are exactly 1.000 s -- recordings appear PRE-CUT")
    if lt1:
        notes.append("%d recording(s) shorter than the 1 s model window "
                     "(would need zero-padding at evaluation time)" % lt1)
    if gt3:
        notes.append("%d recording(s) longer than 3 s" % gt3)

    n_clip = sum(1 for r in ok if r["is_clipped"])
    clip_pct = n_clip / len(ok) * 100
    print("\n  clipping (any sample |x| >= %.3f):" % CLIP_T)
    print("      clipped files : %d (%.1f%%)" % (n_clip, clip_pct))
    if n_clip:
        worst = sorted((r for r in ok if r["is_clipped"]),
                       key=lambda r: -r["clipped_pct"])[:5]
        for r in worst:
            print("        %-44s %6d samples (%.3f%%)"
                  % (r["filename"][:44], r["clipped_samples"], r["clipped_pct"]))
    if clip_pct > 10:
        issues.append("%.1f%% of recordings are clipped -- check input gain/AGC" % clip_pct)
    elif n_clip:
        notes.append("%d clipped recording(s) (%.1f%%)" % (n_clip, clip_pct))

    n_silent = sum(1 for r in ok if r["is_silent"])
    n_near = sum(1 for r in ok if r["is_near_silent"])
    rms = np.array([r["rms"] for r in ok])
    peak = np.array([r["peak_amplitude"] for r in ok])
    print("\n  level:")
    print("      RMS  min=%.6f median=%.6f max=%.6f" % (rms.min(), float(np.median(rms)), rms.max()))
    print("      peak min=%.6f median=%.6f max=%.6f" % (peak.min(), float(np.median(peak)), peak.max()))
    print("      silent      (RMS < %.0e) : %d" % (SILENT_RMS, n_silent))
    print("      near-silent (RMS < %.0e) : %d" % (NEAR_SILENT_RMS, n_near))
    if n_silent:
        issues.append("%d silent file(s)" % n_silent)
    if n_near - n_silent:
        notes.append("%d near-silent file(s) (reported, not rejected)" % (n_near - n_silent))

    name_dup = {k: v for k, v in collections.Counter(r["filename"] for r in rows).items() if v > 1}
    fhash = collections.defaultdict(list)
    ahash = collections.defaultdict(list)
    for r in ok:
        fhash[r["sha256_file"]].append(r["filename"])
        ahash[r["sha256_audio"]].append(r["filename"])
    fdup = {k: v for k, v in fhash.items() if len(v) > 1}
    adup = {k: v for k, v in ahash.items() if len(v) > 1}
    n_dup_files = sum(len(v) - 1 for v in adup.values())

    print("\n  duplicates:")
    print("      duplicate filenames            : %d %s" % (len(name_dup), list(name_dup)[:5]))
    print("      identical file hashes (groups) : %d" % len(fdup))
    print("      identical audio hashes (groups): %d" % len(adup))
    print("      redundant copies (extra files) : %d" % n_dup_files)
    for h, v in list(adup.items())[:5]:
        print("        x%d: %s" % (len(v), v[:3]))
    if name_dup:
        issues.append("duplicate filenames: %s" % list(name_dup)[:5])
    if adup:
        issues.append("%d group(s) of identical audio (%d redundant copies)"
                      % (len(adup), n_dup_files))

    unique_audio = len(ahash)
    print("\n  unique recordings by audio hash : %d of %d" % (unique_audio, len(ok)))

    # ------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    if issues:
        print("  AUDIT NEEDS REVIEW")
        print("=" * 78)
        for i in issues:
            print("    ISSUE : %s" % i)
    else:
        print("  AUDIT PASS")
        print("=" * 78)
    for n in notes:
        print("    note  : %s" % n)
    status = "AUDIT NEEDS REVIEW" if issues else "AUDIT PASS"

    # ------------------------------------------------------------ outputs
    out_csv = src / "audit_new_ira_dataset.csv"
    out_md = src / "AUDIT_REPORT.md"
    fields = ["filename", "relative_path", "bytes", "readable", "sample_rate", "channels",
              "n_samples", "duration_s", "format", "subtype", "peak_amplitude", "rms",
              "clipped_samples", "clipped_pct", "is_clipped", "is_silent", "is_near_silent",
              "sha256_file", "sha256_audio", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    L = []
    A = L.append
    A("# New Real-Human \"Ira\" Dataset — Audit Report\n")
    A("**Source:** `%s`  " % src)
    A("**Status:** **%s**\n" % status)
    A("> Read-only audit. No recording was resampled, normalized, trimmed, converted,")
    A("> renamed, moved, or deleted. No model was loaded and no inference was run.\n")
    A("## Summary\n")
    A("| metric | value |")
    A("|---|---|")
    A("| Total WAV files | %d |" % len(wavs))
    A("| Readable | %d |" % len(ok))
    A("| Unreadable | %d |" % len(unreadable))
    A("| Unique recordings (audio hash) | %d |" % unique_audio)
    A("| Sample rates | %s |" % ", ".join("%d Hz x%d" % (k, v) for k, v in sorted(sr_dist.items())))
    A("| Channels | %s |" % ", ".join("%s x%d" % ({1: "mono", 2: "stereo"}.get(k, str(k)), v)
                                      for k, v in sorted(ch_dist.items())))
    A("| Format | %s |" % ", ".join("%s x%d" % (k, v) for k, v in fmt_dist.most_common()))
    A("| Duration min / mean / max | %.3f / %.3f / %.3f s |" % (d.min(), d.mean(), d.max()))
    A("| Shorter than 1 s | %d |" % lt1)
    A("| Longer than 1 s | %d |" % gt1)
    A("| Longer than 3 s | %d |" % gt3)
    A("| Clipped recordings | %d (%.1f%%) |" % (n_clip, clip_pct))
    A("| Silent | %d |" % n_silent)
    A("| Near-silent | %d |" % n_near)
    A("| Duplicate filenames | %d |" % len(name_dup))
    A("| Duplicate audio groups | %d (%d redundant copies) |" % (len(adup), n_dup_files))
    A("")
    A("**Deployment expectation:** %d Hz, mono — %s\n"
      % (EXPECTED_SR, "MET" if set(sr_dist) == {EXPECTED_SR} and set(ch_dist) == {EXPECTED_CH}
         else "NOT MET"))
    if issues:
        A("## Issues\n")
        for i in issues:
            A("- %s" % i)
        A("")
    if notes:
        A("## Notes\n")
        for n in notes:
            A("- %s" % n)
        A("")
    A("## Per-file detail\n")
    A("| filename | SR | ch | dur_s | samples | subtype | peak | RMS | clipped | sha256 (file, first 16) |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r.get("readable"):
            A("| `%s` | — | — | — | — | — | — | — | — | UNREADABLE: %s |" % (r["filename"], r["error"]))
            continue
        A("| `%s` | %d | %d | %.3f | %d | %s | %.5f | %.5f | %s | `%s` |"
          % (r["filename"], r["sample_rate"], r["channels"], r["duration_s"], r["n_samples"],
             r["subtype"], r["peak_amplitude"], r["rms"],
             "yes" if r["is_clipped"] else "no", r["sha256_file"][:16]))
    A("")
    A("Full hashes and clipping counts: `audit_new_ira_dataset.csv`.\n")
    A("---\n")
    A("**Data hygiene:** these recordings are NEW development/diagnostic data. They")
    A("have not been added to any training, validation, or existing manifest, and were")
    A("not used for model training, selection, or thresholding.")
    out_md.write_text("\n".join(L), encoding="utf-8")

    print("\n  wrote %s" % out_csv)
    print("  wrote %s" % out_md)
    print("  no audio was modified.")
    print("\n%s" % status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
