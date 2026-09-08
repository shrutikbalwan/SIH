# -*- coding: utf-8 -*-
"""
STEP 1 -- Audit the new real-human 'Ira' recordings.

READ-ONLY. Nothing in the source folder is modified, renamed, moved or deleted.
"""
import os, csv, json, hashlib, collections
import numpy as np
import soundfile as sf
from pathlib import Path

SRC = Path(r"D:\SIH\ira-wakeword\today ira wav")
OUT = Path(r"D:\SIH\ira-wakeword\cnn")
AUDIT_CSV = OUT / "new_real_positive_audit.csv"
AUDIT_JSON = OUT / "new_real_positive_audit.json"


def main():
    files = sorted([p for p in SRC.rglob("*") if p.is_file()])
    print("=" * 74)
    print("STEP 1 -- AUDIT: %s" % SRC)
    print("=" * 74)
    print("  files found (recursive, all types): %d" % len(files))
    ext = collections.Counter(p.suffix.lower() for p in files)
    print("  extensions: %s" % dict(ext))

    rows, unreadable = [], []
    for p in files:
        rec = {"path": str(p), "name": p.name, "bytes": p.stat().st_size}
        try:
            info = sf.info(str(p))
            rec.update(format=info.format, subtype=info.subtype,
                       samplerate=info.samplerate, channels=info.channels,
                       frames=info.frames, duration=info.duration)
            a, sr = sf.read(str(p), dtype="float64", always_2d=False)
            if a.ndim > 1:
                a = a.mean(axis=1)
            rec["rms"] = float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0
            rec["peak"] = float(np.max(np.abs(a))) if len(a) else 0.0
            rec["readable"] = True
            # content hash of decoded audio (catches re-encoded duplicates too)
            rec["audio_sha1"] = hashlib.sha1(
                np.round(a * 32767).astype(np.int16).tobytes()).hexdigest()
            rec["byte_sha1"] = hashlib.sha1(p.read_bytes()).hexdigest()
        except Exception as e:
            rec["readable"] = False
            rec["error"] = "%s: %s" % (type(e).__name__, e)
            unreadable.append(rec)
        rows.append(rec)

    ok = [r for r in rows if r.get("readable")]
    print("\n  readable   : %d" % len(ok))
    print("  unreadable : %d" % len(unreadable))
    for r in unreadable[:10]:
        print("     %s -> %s" % (r["name"], r["error"]))

    if not ok:
        print("  nothing readable; stopping audit.")
        return

    print("\n  --- formats ---")
    for k, v in collections.Counter((r["format"], r["subtype"]) for r in ok).most_common():
        print("    %-22s %d" % ("%s / %s" % k, v))
    print("\n  --- sample rates ---")
    for k, v in collections.Counter(r["samplerate"] for r in ok).most_common():
        print("    %6d Hz  %d" % (k, v))
    print("\n  --- channels ---")
    for k, v in collections.Counter(r["channels"] for r in ok).most_common():
        print("    %s  %d" % ({1: "mono  ", 2: "stereo"}.get(k, str(k)), v))

    d = np.array([r["duration"] for r in ok])
    print("\n  --- duration (s) ---")
    print("    min=%.3f  mean=%.3f  median=%.3f  max=%.3f" % (d.min(), d.mean(), np.median(d), d.max()))
    for q in [1, 5, 25, 75, 95, 99]:
        print("    P%-3d = %.3f" % (q, np.percentile(d, q)))
    print("    histogram:")
    hist, edges = np.histogram(d, bins=[0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9])
    lbl = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-5", "5-10", ">10"]
    for l, h in zip(lbl, hist):
        if h:
            print("      %-8s %s %d" % (l + "s", "#" * min(40, int(h / 8)), h))

    print("\n  --- duplicates ---")
    bd = collections.Counter(r["byte_sha1"] for r in ok)
    ad = collections.Counter(r["audio_sha1"] for r in ok)
    bdup = {k: v for k, v in bd.items() if v > 1}
    adup = {k: v for k, v in ad.items() if v > 1}
    print("    exact byte-identical duplicate files : %d (in %d groups)"
          % (sum(v - 1 for v in bdup.values()), len(bdup)))
    print("    identical decoded audio (any container): %d (in %d groups)"
          % (sum(v - 1 for v in adup.values()), len(adup)))
    if adup:
        for h, v in list(adup.items())[:5]:
            names = [r["name"] for r in ok if r["audio_sha1"] == h]
            print("      x%d: %s" % (v, names[:3]))

    print("\n  --- near-silent / abnormally short ---")
    silent = [r for r in ok if r["rms"] < 1e-4]
    quiet = [r for r in ok if 1e-4 <= r["rms"] < 1e-3]
    short = [r for r in ok if r["duration"] < 0.5]
    vshort = [r for r in ok if r["duration"] < 0.25]
    print("    RMS < 1e-4 (near-silent) : %d" % len(silent))
    print("    RMS < 1e-3 (very quiet)  : %d" % len(quiet))
    print("    duration < 0.50 s        : %d" % len(short))
    print("    duration < 0.25 s        : %d" % len(vshort))
    print("    clipped (peak >= 0.999)  : %d" % sum(1 for r in ok if r["peak"] >= 0.999))
    for r in sorted(short, key=lambda r: r["duration"])[:5]:
        print("      %.3fs rms=%.5f  %s" % (r["duration"], r["rms"], r["name"]))

    print("\n  --- speaker identification ---")
    print("    Filenames are WhatsApp export names (timestamp only).")
    print("    No speaker ID, no folder-per-speaker structure, no sidecar metadata.")
    print("    => speaker count is NOT determinable from filenames/folders.")
    ts = collections.Counter(r["name"][:30] for r in ok)
    print("    distinct WhatsApp date-stamps present:")
    days = collections.Counter()
    for r in ok:
        n = r["name"]
        if " at " in n:
            days[n.split(" at ")[0].replace("WhatsApp Audio ", "").replace("WhatsApp Ptt ", "")] += 1
    for k, v in sorted(days.items()):
        print("      %-14s %d" % (k, v))
    kinds = collections.Counter("Ptt" if "Ptt" in r["name"] else "Audio" for r in ok)
    print("    WhatsApp message kind: %s" % dict(kinds))

    fields = ["name", "path", "bytes", "readable", "format", "subtype", "samplerate",
              "channels", "frames", "duration", "rms", "peak", "audio_sha1", "byte_sha1", "error"]
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    json.dump({
        "source_folder": str(SRC),
        "files_found": len(files),
        "readable": len(ok), "unreadable": len(unreadable),
        "extensions": {k: v for k, v in ext.items()},
        "formats": {"%s/%s" % k: v for k, v in collections.Counter((r["format"], r["subtype"]) for r in ok).items()},
        "sample_rates": {str(k): v for k, v in collections.Counter(r["samplerate"] for r in ok).items()},
        "channels": {str(k): v for k, v in collections.Counter(r["channels"] for r in ok).items()},
        "duration": {"min": float(d.min()), "mean": float(d.mean()),
                     "median": float(np.median(d)), "max": float(d.max())},
        "duplicate_files_byte": sum(v - 1 for v in bdup.values()),
        "duplicate_files_audio": sum(v - 1 for v in adup.values()),
        "near_silent": len(silent), "short_lt_0.5s": len(short),
        "speaker_ids_determinable": False,
    }, open(AUDIT_JSON, "w"), indent=2)
    print("\n  wrote %s" % AUDIT_CSV)
    print("  wrote %s" % AUDIT_JSON)


if __name__ == "__main__":
    main()
