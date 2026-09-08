# -*- coding: utf-8 -*-
"""
Quality-control audit for the NEW real-human DEVELOPMENT set.

Audits WAV files and metadata ONLY. Loads no wake-word model, computes no
TPR, and never modifies, renames, moves or deletes audio.

Quiet recordings are REPORTED, never rejected -- low level is a variable
under study, not a defect.

Usage:
    python cnn/audit_real_human_dev_dataset.py
    python cnn/audit_real_human_dev_dataset.py --root <path> --strict
"""
import os, csv, json, hashlib, argparse, collections
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).parent.parent
DEFAULT_ROOT = REPO / "dataset" / "real_human_dev"
OUT_JSON = REPO / "cnn" / "real_human_dev_dataset_audit.json"

REQUIRED_FIELDS = ["recording_id", "filename", "speaker_id", "session_id", "device_id",
                   "distance_m", "room_id", "room_condition", "noise_condition",
                   "repetition_index", "timestamp", "sample_rate", "duration_s", "notes"]
OPTIONAL_FIELDS = ["speaker_sex_or_voice_range", "orientation", "device_gain_mode",
                   "agc_known", "background_type", "estimated_noise_level",
                   "review_label", "review_notes"]
# 'notes' and 'review_*' may legitimately be blank
BLANK_OK = {"notes", "review_label", "review_notes"}

CLIP_T = 0.999
NEAR_SILENT_RMS = 1e-4
SHORT_S = 0.5
LONG_S = 5.0
EXPECTED_SR = {16000, 48000}


def sha1_audio(a):
    return hashlib.sha1(np.round(np.asarray(a, dtype=np.float64) * 32767)
                        .astype(np.int16).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any ERROR-level problem is found")
    args = ap.parse_args()

    root = Path(args.root)
    meta_path = root / "metadata.csv"
    rec_dir = root / "recordings"

    print("=" * 78)
    print("REAL-HUMAN DEVELOPMENT SET -- QUALITY-CONTROL AUDIT")
    print("=" * 78)
    print("  root      : %s" % root)
    print("  audio is READ-ONLY; no model is loaded; nothing is rejected for being quiet.")

    errors, warnings = [], []
    report = {"root": str(root)}

    # ---------------------------------------------------------------- inputs
    if not rec_dir.exists():
        print("\n  recordings/ directory not found -- nothing collected yet.")
        print("  Expected layout: %s/<SPKxx>/<SESSIONxxx>/*.wav" % rec_dir)
        json.dump({"status": "NO_RECORDINGS", "root": str(root)}, open(OUT_JSON, "w"), indent=2)
        print("  wrote %s" % OUT_JSON)
        return 0

    wavs = sorted(p for p in rec_dir.rglob("*.wav") if p.is_file())
    print("\n  WAV files found      : %d" % len(wavs))

    if not meta_path.exists():
        print("  metadata.csv         : MISSING")
        errors.append("metadata.csv not found at %s" % meta_path)
        rows = []
    else:
        rows = list(csv.DictReader(open(meta_path, encoding="utf-8-sig")))
        print("  metadata rows        : %d" % len(rows))

    if not wavs and not rows:
        print("\n  Nothing to audit yet.")
        json.dump({"status": "EMPTY", "root": str(root)}, open(OUT_JSON, "w"), indent=2)
        return 0

    # ---------------------------------------------------- schema completeness
    print("\n" + "-" * 78)
    print("  METADATA SCHEMA")
    print("-" * 78)
    if rows:
        cols = list(rows[0].keys())
        missing_cols = [c for c in REQUIRED_FIELDS if c not in cols]
        extra = [c for c in cols if c not in REQUIRED_FIELDS + OPTIONAL_FIELDS]
        print("  columns present : %d" % len(cols))
        print("  required missing: %s" % (missing_cols or "none"))
        print("  unrecognised    : %s" % (extra or "none"))
        if missing_cols:
            errors.append("metadata missing required columns: %s" % missing_cols)

        blank_counts = collections.Counter()
        for r in rows:
            for c in REQUIRED_FIELDS:
                if c in r and c not in BLANK_OK and not (r.get(c) or "").strip():
                    blank_counts[c] += 1
        if blank_counts:
            print("  blank required values:")
            for c, n in blank_counts.most_common():
                print("    %-22s %d rows" % (c, n))
                errors.append("%d rows have blank required field '%s'" % (n, c))
        else:
            print("  blank required values: none")

    # ---------------------------------------------------- cross-reference
    print("\n" + "-" * 78)
    print("  METADATA <-> FILE CROSS-REFERENCE")
    print("-" * 78)
    wav_by_name = {}
    dup_names = collections.Counter()
    for p in wavs:
        dup_names[p.name] += 1
        wav_by_name.setdefault(p.name, p)
    meta_names = [(r.get("filename") or "").strip() for r in rows]
    meta_set = set(n for n in meta_names if n)

    no_meta = sorted(set(wav_by_name) - meta_set)
    no_file = sorted(meta_set - set(wav_by_name))
    print("  WAVs with no metadata row : %d" % len(no_meta))
    for n in no_meta[:10]:
        print("      %s" % n)
    print("  metadata rows with no WAV : %d" % len(no_file))
    for n in no_file[:10]:
        print("      %s" % n)
    if no_meta:
        errors.append("%d WAV files have no metadata row" % len(no_meta))
    if no_file:
        errors.append("%d metadata rows have no WAV file" % len(no_file))

    dupf = {n: c for n, c in dup_names.items() if c > 1}
    dupm = {n: c for n, c in collections.Counter(meta_names).items() if c > 1 and n}
    dupid = {i: c for i, c in collections.Counter(
        (r.get("recording_id") or "").strip() for r in rows).items() if c > 1 and i}
    print("  duplicate filenames on disk    : %d %s" % (len(dupf), list(dupf)[:5]))
    print("  duplicate filenames in metadata: %d %s" % (len(dupm), list(dupm)[:5]))
    print("  duplicate recording_ids        : %d %s" % (len(dupid), list(dupid)[:5]))
    for d, lbl in ((dupf, "filenames on disk"), (dupm, "filenames in metadata"),
                   (dupid, "recording_ids")):
        if d:
            errors.append("duplicate %s: %s" % (lbl, list(d)[:5]))

    # ---------------------------------------------------- audio properties
    print("\n" + "-" * 78)
    print("  AUDIO PROPERTIES")
    print("-" * 78)
    audio, unreadable = [], []
    hashes = collections.defaultdict(list)
    for p in wavs:
        rec = {"filename": p.name, "path": str(p),
               "rel": str(p.relative_to(rec_dir)).replace("\\", "/")}
        try:
            info = sf.info(str(p))
            a, sr = sf.read(str(p), dtype="float64", always_2d=False)
            mono = a if a.ndim == 1 else a.mean(axis=1)
            n = len(mono)
            rec.update(readable=True, format=info.format, subtype=info.subtype,
                       samplerate=info.samplerate, channels=info.channels,
                       duration_s=info.duration,
                       rms=float(np.sqrt(np.mean(mono ** 2))) if n else 0.0,
                       peak=float(np.max(np.abs(mono))) if n else 0.0,
                       clipped_samples=int(np.sum(np.abs(mono) >= CLIP_T)),
                       clipped_pct=(float(np.sum(np.abs(mono) >= CLIP_T)) / n * 100) if n else 0.0)
            rec["audio_sha1"] = sha1_audio(mono)
            hashes[rec["audio_sha1"]].append(p.name)
        except Exception as e:
            rec.update(readable=False, error="%s: %s" % (type(e).__name__, e))
            unreadable.append(rec)
        audio.append(rec)

    ok = [r for r in audio if r.get("readable")]
    print("  readable / unreadable : %d / %d" % (len(ok), len(unreadable)))
    for r in unreadable[:10]:
        print("      %s -> %s" % (r["filename"], r["error"]))
        errors.append("unreadable file %s" % r["filename"])

    if ok:
        for label, key in (("formats", lambda r: "%s/%s" % (r["format"], r["subtype"])),
                           ("sample rates", lambda r: r["samplerate"]),
                           ("channels", lambda r: r["channels"])):
            c = collections.Counter(key(r) for r in ok)
            print("  %-13s: %s" % (label, dict(c)))
        bad_sr = [r["filename"] for r in ok if r["samplerate"] not in EXPECTED_SR]
        bad_ch = [r["filename"] for r in ok if r["channels"] != 1]
        if bad_sr:
            warnings.append("%d files at unexpected sample rate (expect 16k or 48k)" % len(bad_sr))
        if bad_ch:
            errors.append("%d files are not mono" % len(bad_ch))

        d = np.array([r["duration_s"] for r in ok])
        print("\n  duration s : min=%.3f mean=%.3f median=%.3f max=%.3f"
              % (d.min(), d.mean(), np.median(d), d.max()))
        rms = np.array([r["rms"] for r in ok])
        pk = np.array([r["peak"] for r in ok])
        print("  RMS        : min=%.5f median=%.5f max=%.5f" % (rms.min(), np.median(rms), rms.max()))
        print("  peak       : min=%.5f median=%.5f max=%.5f" % (pk.min(), np.median(pk), pk.max()))

        n_clip = sum(1 for r in ok if r["peak"] >= CLIP_T)
        n_quiet = sum(1 for r in ok if r["rms"] < NEAR_SILENT_RMS)
        n_short = sum(1 for r in ok if r["duration_s"] < SHORT_S)
        n_long = sum(1 for r in ok if r["duration_s"] > LONG_S)
        n_1s = sum(1 for r in ok if abs(r["duration_s"] - 1.0) < 0.02)
        print("\n  clipped (peak>=%.3f)      : %d (%.1f%%)" % (CLIP_T, n_clip, n_clip / len(ok) * 100))
        print("  near-silent (RMS<%.0e)   : %d   [REPORTED, NOT REJECTED]" % (NEAR_SILENT_RMS, n_quiet))
        print("  shorter than %.1f s        : %d" % (SHORT_S, n_short))
        print("  longer than %.1f s         : %d" % (LONG_S, n_long))
        print("  suspiciously exactly 1.00 s: %d" % n_1s)
        if n_clip / len(ok) > 0.10:
            warnings.append("%.1f%% of files clipped -- check input gain/AGC" % (n_clip / len(ok) * 100))
        if n_quiet:
            warnings.append("%d near-silent files (reported only, kept)" % n_quiet)
        if n_1s > len(ok) * 0.5:
            errors.append("most files are exactly 1.00 s -- recordings appear PRE-CUT, "
                          "which breaks the alignment diagnostic")
        if n_long:
            warnings.append("%d files longer than %.1f s" % (n_long, LONG_S))

        dup_audio = {h: v for h, v in hashes.items() if len(v) > 1}
        print("  identical audio hashes    : %d group(s)" % len(dup_audio))
        for h, v in list(dup_audio.items())[:5]:
            print("      x%d: %s" % (len(v), v[:3]))
        if dup_audio:
            errors.append("%d groups of byte-identical audio" % len(dup_audio))

    # ---------------------------------------------------- design balance
    print("\n" + "-" * 78)
    print("  DESIGN COVERAGE AND BALANCE")
    print("-" * 78)
    balance = {}
    if rows:
        for f in ["speaker_id", "session_id", "device_id", "room_id",
                  "distance_m", "room_condition", "noise_condition"]:
            c = collections.Counter((r.get(f) or "").strip() for r in rows)
            balance[f] = dict(c)
            print("  %-16s %d distinct: %s" % (f, len(c), dict(c) if len(c) <= 8 else "(many)"))

        spk = sorted({(r.get("speaker_id") or "").strip() for r in rows})
        rooms = sorted({(r.get("room_id") or "").strip() for r in rows})
        dists = sorted({(r.get("distance_m") or "").strip() for r in rows})

        print("\n  speaker x room x distance cell counts (blank = MISSING CELL):")
        header = "  %-8s" % "spk"
        for rm in rooms:
            for ds in dists:
                header += " %10s" % ("%s/%s" % (rm[-3:], ds))
        print(header)
        holes = 0
        for s in spk:
            line = "  %-8s" % s
            for rm in rooms:
                for ds in dists:
                    n = sum(1 for r in rows
                            if (r.get("speaker_id") or "").strip() == s
                            and (r.get("room_id") or "").strip() == rm
                            and (r.get("distance_m") or "").strip() == ds)
                    line += " %10s" % (n if n else "-")
                    if not n:
                        holes += 1
            print(line)
        print("  empty cells: %d" % holes)
        if holes:
            warnings.append("%d empty speaker x room x distance cells -- "
                            "unbalanced design weakens factor separation" % holes)

        # crossing / repeated-measures checks -- the whole point of this design
        spk_rooms = collections.defaultdict(set)
        spk_room_sessions = collections.defaultdict(set)
        for r in rows:
            s = (r.get("speaker_id") or "").strip()
            rm = (r.get("room_id") or "").strip()
            se = (r.get("session_id") or "").strip()
            spk_rooms[s].add(rm)
            spk_room_sessions[(s, rm)].add(se)
        multi_room = [s for s, v in spk_rooms.items() if len(v) >= 2]
        repeats = [k for k, v in spk_room_sessions.items() if len(v) >= 2]
        print("\n  speakers recorded in >=2 rooms (crosses speaker x room): %d/%d"
              % (len(multi_room), len(spk_rooms)))
        print("  speaker+room combos repeated in >=2 sessions            : %d %s"
              % (len(repeats), repeats[:4]))
        if len(multi_room) < max(2, len(spk_rooms) // 2):
            warnings.append("few speakers span multiple rooms -- speaker and room "
                            "effects will remain partly confounded")
        if len(repeats) < 2:
            warnings.append("fewer than 2 session-repeats of the same speaker+room -- "
                            "session variance cannot be separated from room variance")

        # session integrity: one session should not mix speaker/room/device
        bad_sessions = []
        by_sess = collections.defaultdict(lambda: collections.defaultdict(set))
        for r in rows:
            se = (r.get("session_id") or "").strip()
            for f in ["speaker_id", "room_id", "device_id"]:
                by_sess[se][f].add((r.get(f) or "").strip())
        for se, d in by_sess.items():
            mixed = [f for f, v in d.items() if len(v) > 1]
            if mixed:
                bad_sessions.append((se, mixed))
        print("  sessions mixing speaker/room/device: %d %s" % (len(bad_sessions), bad_sessions[:3]))
        if bad_sessions:
            errors.append("sessions mix speaker/room/device: %s" % bad_sessions[:3])

        # metadata vs measured duration / sample rate
        mism = []
        for r in rows:
            fn = (r.get("filename") or "").strip()
            m = next((x for x in ok if x["filename"] == fn), None)
            if not m:
                continue
            try:
                if abs(float(r.get("duration_s") or 0) - m["duration_s"]) > 0.05:
                    mism.append((fn, "duration"))
                if int(float(r.get("sample_rate") or 0)) != m["samplerate"]:
                    mism.append((fn, "sample_rate"))
            except ValueError:
                mism.append((fn, "unparseable numeric"))
        print("  metadata/file mismatches (duration or sample_rate): %d %s" % (len(mism), mism[:5]))
        if mism:
            warnings.append("%d metadata rows disagree with the actual file" % len(mism))

    # ---------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("  AUDIT SUMMARY")
    print("=" * 78)
    print("  ERRORS   : %d" % len(errors))
    for e in errors:
        print("    ERROR  : %s" % e)
    print("  WARNINGS : %d" % len(warnings))
    for w in warnings:
        print("    WARN   : %s" % w)
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    print("\n  STATUS: %s" % status)
    if status != "FAIL":
        print("  Ready for the stride/alignment diagnostic "
              "(see cnn/real_human_dev_analysis_plan.md).")
    print("  Reminder: this is DEVELOPMENT data. It can never become the final "
          "untouched positive benchmark.")

    json.dump({"status": status, "root": str(root),
               "n_wav": len(wavs), "n_metadata": len(rows),
               "readable": len(ok), "unreadable": len(unreadable),
               "errors": errors, "warnings": warnings,
               "balance": balance,
               "files": audio}, open(OUT_JSON, "w"), indent=2, default=float)
    print("\n  wrote %s" % OUT_JSON)
    print("  no audio was modified.")
    return 1 if (errors and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
