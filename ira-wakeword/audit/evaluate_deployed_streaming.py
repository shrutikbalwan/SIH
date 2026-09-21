#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_deployed_streaming.py
==============================
Streaming evaluation of the frozen IRA INT8 model at the DEPLOYED operating
points, using the CORRECT feature frontend (the tf.signal.stft path, which is
bit-identical to esp32/main/ira_features.cpp).

This script deliberately does NOT use pymicro_features / MicroFrontend. That
frontend produced the retracted 65.5% TPR / 864 FAPH figures; see
reports/CORRECTED_EVALUATION_REPORT.md.

WHAT IT MEASURES
----------------
Continuous streaming over concatenated audio at a 100 ms stride (the device
stride), NOT per-clip classification. A detection EVENT is a threshold crossing
that survives the smoothing rule, followed by a refractory period so that one
utterance is counted once.

OPERATING POINTS
----------------
  OP-committed : q_out >= -25   (p >= 0.40234375), no smoothing
                 This is what the firmware does TODAY
                 (esp32/main/ira_inference.h: IRA_Q_THRESHOLD).
  OP-deployed  : p >= 0.95, 3 consecutive detections required.
  OP-legacy    : p >= 0.50, no smoothing -- for comparison with the old README.

NOTE ON p = 0.95: the output scale is 0.00390625 (= 1/256) with zero-point -128,
so probability is quantised in steps of 1/256 and 0.95 is NOT exactly
representable. The nearest achievable operating point is q_out >= 116, i.e.
p >= 0.953125. This script uses q_out >= 116 and reports the effective
probability, rather than silently pretending 0.95 was achieved.

SPEAKER LEAKAGE
---------------
Before reporting any TPR the script verifies that no speaker in the positive
test set appears in the training split. If the manifest cannot prove separation
(e.g. speaker_id is blank), the script FAILS LOUDLY and refuses to report TPR
unless --allow-leakage is passed, in which case every TPR figure is stamped
CONTAMINATED.

Read-only with respect to audio and the model; writes only --json-out.
"""

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# --------------------------------------------------------------------------
# Frozen device contract -- must match esp32/main/ira_features.h and
# esp32/main/ira_inference.h exactly. Do not "tidy" these numbers.
# --------------------------------------------------------------------------
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000          # 1.0 s analysis window
FRAME_LENGTH = 480
FRAME_STEP = 320
FFT_LENGTH = 512
NUM_FRAMES = 49                 # (16000 - 480) / 320 + 1
NUM_BINS = 40                   # first 40 of the 257 rfft bins
LOG_EPSILON = 1e-6
STD_EPSILON = 1e-6

IN_SCALE = 0.06078097224235535
IN_ZP = 49
OUT_SCALE = 0.00390625
OUT_ZP = -128

Q_THRESHOLD_COMMITTED = -25     # IRA_Q_THRESHOLD in ira_inference.h

STRIDE_MS = 100
STRIDE_SAMPLES = SAMPLE_RATE * STRIDE_MS // 1000   # 1600

# --------------------------------------------------------------------------
# Hardware-observed operating-point constants
# Source: results/screenshots/serial_log_detection.png (IDF_v6.0.3 serial)
# These are DERIVED FROM LOG TIMESTAMPS / inferred from log behaviour,
# not directly measured by instrumented firmware. Update when firmware
# exposes explicit timing APIs.
# --------------------------------------------------------------------------

# Inference time: read directly from "inference XX.XX ms" column in the log.
# Observed range 30.91–30.94 ms across all visible readings.
HW_INFERENCE_TIME_MS = 30.9          # mean; DERIVED FROM LOG

# Inference interval: delta between consecutive IDF timestamps (ms).
# e.g. (107003 - 106853) = 150, (107133 - 107003) = 130, mean ≈ 136 ms.
HW_INFERENCE_INTERVAL_MS = 136       # DERIVED FROM LOG TIMESTAMPS

# CPU budget fraction on one core; across both cores (symmetric SMP assumed).
# Formula: HW_INFERENCE_TIME_MS / HW_INFERENCE_INTERVAL_MS
HW_CPU_ONE_CORE  = HW_INFERENCE_TIME_MS / HW_INFERENCE_INTERVAL_MS   # ~0.228
HW_CPU_TWO_CORES = HW_CPU_ONE_CORE / 2                               # ~0.114

# Smoothing rule: N consecutive above-threshold readings must be seen
# before a detection is announced ("candidate 1/3, 2/3, 3/3" in log).
HW_SMOOTHING_N = 3                   # DERIVED FROM LOG ("candidate X/3")

# Detection threshold: the exact firmware constant is not printed in the log.
# 0.9766 never became a candidate; 0.9805 always did → threshold lies between.
# Use 0.98 as the working approximation; CONFIRM FROM LIVE FIRMWARE SOURCE.
HW_THRESHOLD_INFERRED = 0.98         # INFERRED from log, not confirmed

# Confirmed detection probability observed in the captured log.
HW_CONFIRMED_PROB = 0.9961           # DERIVED FROM LOG (>>> IRA DETECTED <<<)

# Cooldown / refractory: 0.9922 reading immediately after detection was
# ignored (not counted as a second detection).
HW_COOLDOWN_OBSERVED = True          # DERIVED FROM LOG


def q_for_probability(p):
    """Smallest INT8 q_out whose dequantised probability is >= p."""
    q = int(np.ceil(p / OUT_SCALE + OUT_ZP))
    return max(-128, min(127, q))


def p_for_q(q):
    return (q - OUT_ZP) * OUT_SCALE


# ==========================================================================
# Feature frontend -- numpy reimplementation of the frozen V2.3 contract
# ==========================================================================
_HANN = None


def _hann():
    """PERIODIC Hann, matching tf.signal.hann_window(periodic=True)."""
    global _HANN
    if _HANN is None:
        n = np.arange(FRAME_LENGTH, dtype=np.float64)
        _HANN = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / FRAME_LENGTH)).astype(np.float32)
    return _HANN


def features_float(samples):
    """float32 samples in [-1,1), length 16000 -> normalised (49, 40) float32.

    Mirrors train_cnn_v2_3.py::make_spectrogram and ira_features.cpp.
    """
    x = np.asarray(samples, dtype=np.float32)
    if len(x) != WINDOW_SAMPLES:
        raise ValueError("expected %d samples, got %d" % (WINDOW_SAMPLES, len(x)))
    idx = (np.arange(NUM_FRAMES)[:, None] * FRAME_STEP
           + np.arange(FRAME_LENGTH)[None, :])
    frames = x[idx] * _hann()
    spec = np.fft.rfft(frames, n=FFT_LENGTH, axis=1)
    mag = np.abs(spec)[:, :NUM_BINS]
    lg = np.log(mag + LOG_EPSILON)
    # ONE global mean/std over all 49*40 values; population std (ddof=0),
    # accumulated in float64 to match the C++ double accumulation.
    m = lg.astype(np.float64).mean()
    s = lg.astype(np.float64).std()
    return ((lg - m) / (s + STD_EPSILON)).astype(np.float32)


def quantize(f):
    """clamp(rint(v / scale) + zp, -128, 127); np.round is half-to-even == rintf."""
    return np.clip(np.round(f / IN_SCALE + IN_ZP), -128, 127).astype(np.int8)


def self_check_against_tensorflow(probe_wavs=(), float_tol=1e-2):
    """Abort if the numpy frontend drifts from the tf.signal.stft reference.

    The binding criterion is INT8 exactness: the model consumes quantised
    features, so 0 INT8 mismatches is what "identical frontend" means. The
    float tolerance is a loose sanity bound only.

    Synthetic noise/tones are deliberately NOT the primary probe. Where the
    log-magnitude of a bin is near zero the float32 path is ill-conditioned and
    the two implementations legitimately differ in the last bits without any
    INT8 consequence -- see esp32/ESP32_FEATURE_PARITY_REPORT.md section 5.1.
    Real audio is used whenever it is available.
    """
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
    except Exception as e:
        print("  [self-check] SKIPPED -- TensorFlow unavailable (%s)"
              % type(e).__name__)
        print("  [self-check] numpy frontend is UNVERIFIED in this run.")
        return None

    probes = []
    for p in list(probe_wavs)[:3]:
        try:
            a = load_wav_mono16k(p)
            a = (np.pad(a, (0, WINDOW_SAMPLES - len(a)))
                 if len(a) < WINDOW_SAMPLES else a[:WINDOW_SAMPLES])
            probes.append(("real:" + Path(p).name, a))
        except Exception:
            pass
    rng = np.random.default_rng(0xC0FFEE)
    while len(probes) < 3:
        probes.append(("synthetic-noise",
                       rng.uniform(-0.5, 0.5, WINDOW_SAMPLES).astype(np.float32)))

    worst, worst_q = 0.0, 0
    for label, x in probes:
        t = tf.convert_to_tensor(x)
        s = tf.signal.stft(t, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP,
                           fft_length=FFT_LENGTH, pad_end=False)
        lg = tf.math.log(tf.abs(s) + LOG_EPSILON)[:, :NUM_BINS]
        ref = ((lg - tf.reduce_mean(lg))
               / (tf.math.reduce_std(lg) + STD_EPSILON)).numpy()
        mine = features_float(x)
        d = float(np.abs(ref - mine).max())
        nq = int((quantize(ref) != quantize(mine)).sum())
        worst = max(worst, d)
        worst_q = max(worst_q, nq)
        print("    %-24s max abs diff = %.3e   INT8 mismatches = %d/%d"
              % (label, d, nq, NUM_FRAMES * NUM_BINS))
    print("  [self-check] worst: max abs diff = %.3e, INT8 mismatches = %d/%d"
          % (worst, worst_q, NUM_FRAMES * NUM_BINS))
    if worst_q > 0:
        sys.exit("STOP: numpy frontend produces different INT8 features from the "
                 "TensorFlow reference. Refusing to report numbers from an "
                 "unverified frontend.")
    if worst > float_tol:
        sys.exit("STOP: numpy frontend float drift %.3e exceeds %.3e. Refusing "
                 "to report numbers from an unverified frontend."
                 % (worst, float_tol))
    return worst


# ==========================================================================
# Model
# ==========================================================================
class Model:
    def __init__(self, path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
        self.itp = Interpreter(model_path=str(path))
        self.itp.allocate_tensors()
        self.inp = self.itp.get_input_details()[0]
        self.out = self.itp.get_output_details()[0]
        si, zi = self.inp["quantization"]
        so, zo = self.out["quantization"]
        # Verify the model actually carries the frozen constants.
        for name, got, want in (("input scale", si, IN_SCALE),
                                ("input zero-point", zi, IN_ZP),
                                ("output scale", so, OUT_SCALE),
                                ("output zero-point", zo, OUT_ZP)):
            if not np.isclose(got, want, rtol=0, atol=1e-12):
                sys.exit("STOP: %s is %s, expected %s. This is not the frozen "
                         "model this script describes." % (name, got, want))

    def q_out(self, q_feat):
        self.itp.set_tensor(self.inp["index"],
                            q_feat.reshape(1, NUM_FRAMES, NUM_BINS, 1))
        self.itp.invoke()
        return int(self.itp.get_tensor(self.out["index"])[0, 0])


# ==========================================================================
# Streaming
# ==========================================================================
def load_wav_mono16k(path):
    sr, x = wavfile.read(str(path))
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        x = resample_poly(x, SAMPLE_RATE, sr).astype(np.float32)
    return x


def embed_in_context(clip, ambient_pool, context_ms, rng):
    """Place a positive clip inside continuous context audio.

    Why this is required: a 1.0 s clip with a 1.0 s analysis window yields
    exactly ONE window, so a "3 consecutive detections" rule could never fire
    and would report 0% TPR as a pure artefact of the harness. On the device
    the wake word arrives inside a continuous stream and several overlapping
    windows contain it. This reproduces that.

    Context is real ambient audio when available. Digital silence is avoided
    deliberately: ira_audio_guard rejects all-zero windows, so zero padding
    would model something the device never sees.
    """
    pad = int(SAMPLE_RATE * context_ms / 1000)
    if pad <= 0:
        return clip

    def context(n):
        if ambient_pool:
            a = ambient_pool[rng.integers(len(ambient_pool))]
            if len(a) >= n:
                s = rng.integers(0, len(a) - n + 1)
                return a[s:s + n].copy()
            return np.tile(a, int(np.ceil(n / len(a))))[:n].copy()
        # Fallback: very low-level noise, never digital silence.
        return (rng.standard_normal(n) * 1e-4).astype(np.float32)

    return np.concatenate([context(pad), clip, context(pad)]).astype(np.float32)


def stream_scores(audio, model):
    """Slide a 16000-sample window at 100 ms stride; return q_out per window."""
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    out = []
    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        w = audio[start:start + WINDOW_SAMPLES]
        out.append(model.q_out(quantize(features_float(w))))
    return np.asarray(out, dtype=np.int16)


def count_events(scores, q_thresh, consecutive, refractory_windows):
    """Detection events under a consecutive-N rule plus a refractory period.

    Returns (n_events, event_window_indices).
    """
    events = []
    run = 0
    blocked_until = -1
    for i, q in enumerate(scores):
        if q >= q_thresh:
            run += 1
        else:
            run = 0
            continue
        if run >= consecutive and i >= blocked_until:
            events.append(i)
            blocked_until = i + refractory_windows
            run = 0
    return len(events), events


# ==========================================================================
# Speaker leakage
# ==========================================================================
def check_speaker_leakage(manifest_path, allow):
    """Verify positive test speakers are disjoint from training speakers.

    Fails loudly when separation cannot be PROVEN.
    """
    print("\n" + "=" * 78)
    print("SPEAKER LEAKAGE CHECK")
    print("=" * 78)
    if not manifest_path or not Path(manifest_path).exists():
        msg = "manifest not found: %s" % manifest_path
        if not allow:
            sys.exit("STOP: %s\n  Cannot prove train/test speaker separation.\n"
                     "  Re-run with --allow-leakage to report CONTAMINATED "
                     "numbers." % msg)
        print("  [!] %s -- cannot verify." % msg)
        return False

    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8-sig")))
    pos = [r for r in rows if str(r.get("label", "")).strip() == "1"]
    by_split = collections.defaultdict(list)
    for r in pos:
        by_split[r.get("split", "").strip()].append(r)

    print("  manifest       : %s" % manifest_path)
    print("  positive rows  : %d" % len(pos))
    for sp in sorted(by_split):
        print("    split=%-12s n=%d" % (sp, len(by_split[sp])))

    train = by_split.get("train", [])
    test = by_split.get("test", [])

    blank_test = [r for r in test if not r.get("speaker_id", "").strip()]
    blank_train = [r for r in train if not r.get("speaker_id", "").strip()]

    clean = True
    if blank_test or blank_train:
        clean = False
        print("\n  [X] speaker_id is BLANK on %d/%d train and %d/%d test "
              "positive rows." % (len(blank_train), len(train),
                                  len(blank_test), len(test)))
        print("      Speaker separation CANNOT BE VERIFIED from this manifest.")
        srcs = sorted({r.get("source", "?") for r in blank_train + blank_test})
        print("      affected sources: %s" % ", ".join(srcs))
        # Group-level overlap is a strong leakage signal when ids are missing.
        gtr = {r.get("group", "") for r in train}
        gte = {r.get("group", "") for r in test}
        shared = sorted(g for g in (gtr & gte) if g)
        if shared:
            print("      [X] recording GROUPS appear in BOTH train and test: %s"
                  % ", ".join(shared))
            print("          Same recording session on both sides of the split.")
    else:
        s_tr = {r["speaker_id"].strip() for r in train}
        s_te = {r["speaker_id"].strip() for r in test}
        overlap = sorted(s_tr & s_te)
        if overlap:
            clean = False
            print("\n  [X] %d speaker(s) in BOTH train and test: %s"
                  % (len(overlap), ", ".join(overlap[:10])))
        else:
            print("\n  [OK] %d test speakers, none seen in training "
                  "(%d train speakers)." % (len(s_te), len(s_tr)))

    if not clean:
        print("\n  " + "!" * 70)
        print("  !! TEST LEAKAGE: TPR measured on this data is OPTIMISTIC and")
        print("  !! MUST NOT be reported as real-world accuracy.")
        print("  " + "!" * 70)
        if not allow:
            sys.exit("\nSTOP: refusing to report TPR on a contaminated split.\n"
                     "  Fix the split (hold out whole speakers), or re-run with\n"
                     "  --allow-leakage to print numbers stamped CONTAMINATED.")
    return clean


# ==========================================================================
def gather(d, limit=None):
    if not d:
        return []
    p = Path(d)
    if not p.exists():
        return []
    fs = sorted(p.rglob("*.wav"))
    return fs[:limit] if limit else fs


def main():
    ap = argparse.ArgumentParser(
        description="Streaming evaluation at the deployed operating points.")
    ap.add_argument("data_dir", help="root data directory (e.g. ira-wakeword/dataset)")
    ap.add_argument("--model", default="cnn/models/ira_cnn_v2_3_int8.tflite")
    ap.add_argument("--manifest", default=None,
                    help="split manifest CSV for the speaker-leakage check "
                         "(default: <data_dir>/split_manifest_v3.csv)")
    ap.add_argument("--positives", default=None,
                    help="dir of positive wavs "
                         "(default: <data_dir>/splits/test/positive)")
    ap.add_argument("--negatives", default=None,
                    help="dir of negative wavs, concatenated into continuous "
                         "audio (default: <data_dir>/negative/speech)")
    ap.add_argument("--hard-negatives", default=None,
                    help="dir with era/either/Ida recordings (optional)")
    ap.add_argument("--ambient", default=None,
                    help="ambient audio used as context around positive clips "
                         "(default: <data_dir>/negative/background)")
    ap.add_argument("--positive-context-ms", type=int, default=1000,
                    help="ambient context added before and after each positive "
                         "and hard-negative clip so that multiple overlapping "
                         "windows contain it, as on device. 0 disables, which "
                         "makes any consecutive-N rule unmeasurable on 1 s clips.")
    ap.add_argument("--min-negative-hours", type=float, default=1.0)
    ap.add_argument("--refractory-ms", type=int, default=1000,
                    help="suppress further events for this long after one fires")
    ap.add_argument("--max-positives", type=int, default=None)
    ap.add_argument("--allow-leakage", action="store_true",
                    help="report CONTAMINATED numbers instead of refusing")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        sys.exit("STOP: data_dir does not exist: %s" % root)

    manifest = args.manifest or (root / "split_manifest_v3.csv")
    pos_dir = args.positives or (root / "splits" / "test" / "positive")
    neg_dir = args.negatives or (root / "negative" / "speech")
    refractory = max(1, args.refractory_ms // STRIDE_MS)

    print("=" * 78)
    print("IRA -- STREAMING EVALUATION AT DEPLOYED OPERATING POINTS")
    print("=" * 78)
    print("  model     : %s" % args.model)
    print("  data_dir  : %s" % root)
    print("  stride    : %d ms (%d samples)" % (STRIDE_MS, STRIDE_SAMPLES))
    print("  window    : %d samples (1.0 s)" % WINDOW_SAMPLES)
    print("  refractory: %d ms (%d windows)" % (args.refractory_ms, refractory))
    print("\n  frontend self-check:")
    self_check_against_tensorflow(probe_wavs=gather(pos_dir, 3))

    clean = check_speaker_leakage(manifest, args.allow_leakage)
    stamp = "" if clean else "   [CONTAMINATED -- train/test speaker overlap]"

    model = Model(args.model)
    rng = np.random.default_rng(42)

    amb_dir = args.ambient or (root / "negative" / "background")
    ambient_pool = []
    if args.positive_context_ms > 0:
        for f in gather(amb_dir, 200):
            try:
                ambient_pool.append(load_wav_mono16k(f))
            except Exception:
                pass
        print("\n  positive context: %d ms of ambient before/after each clip"
              % args.positive_context_ms)
        if ambient_pool:
            print("                    from %d ambient clips in %s"
                  % (len(ambient_pool), amb_dir))
        else:
            print("                    [!] no ambient audio at %s -- falling "
                  "back to low-level noise" % amb_dir)
    else:
        print("\n  [!] --positive-context-ms 0: each 1 s clip yields ONE window,")
        print("      so consecutive-N operating points cannot fire. TPR for")
        print("      those rows will be 0%% as a harness artefact, not a result.")

    q95 = q_for_probability(0.95)
    q50 = q_for_probability(0.50)
    ops = [
        ("committed  q>=-25, no smoothing", Q_THRESHOLD_COMMITTED, 1),
        ("deployed   p>=0.95, 3-of-3", q95, 3),
        ("legacy     p>=0.50, no smoothing", q50, 1),
    ]
    print("\n" + "=" * 78)
    print("OPERATING POINTS")
    print("=" * 78)
    for name, q, n in ops:
        print("  %-34s -> q_out >= %4d  (effective p >= %.10f)"
              % (name, q, p_for_q(q)))
    if not np.isclose(p_for_q(q95), 0.95):
        print("\n  NOTE: p=0.95 is not representable at output scale %s."
              % OUT_SCALE)
        print("        Using q>=%d, i.e. p>=%.10f." % (q95, p_for_q(q95)))

    results = {"operating_points": {n: {"q": q, "p": p_for_q(q),
                                        "consecutive": c}
                                    for n, q, c in ops},
               "leakage_clean": clean}

    # ---------------------------------------------------------------- TPR
    pos_files = gather(pos_dir, args.max_positives)
    print("\n" + "=" * 78)
    print("TPR -- positives%s" % stamp)
    print("=" * 78)
    print("  dir: %s" % pos_dir)
    if not pos_files:
        print("  [X] no positive wavs found -- TPR not measured.")
    else:
        print("  clips: %d" % len(pos_files))
        hits = {n: 0 for n, _, _ in ops}
        for i, f in enumerate(pos_files):
            clip = embed_in_context(load_wav_mono16k(f), ambient_pool,
                                    args.positive_context_ms, rng)
            sc = stream_scores(clip, model)
            for n, q, c in ops:
                if count_events(sc, q, c, refractory)[0] > 0:
                    hits[n] += 1
            if (i + 1) % 200 == 0:
                print("    ... %d/%d" % (i + 1, len(pos_files)))
        print()
        results["tpr"] = {}
        for n, _, _ in ops:
            tpr = 100.0 * hits[n] / len(pos_files)
            print("  %-34s TPR = %7.3f%%  (%d/%d)%s"
                  % (n, tpr, hits[n], len(pos_files), stamp))
            results["tpr"][n] = {"tpr_pct": tpr, "hits": hits[n],
                                 "n": len(pos_files), "contaminated": not clean}

    # ---------------------------------------------------------------- FAPH
    neg_files = gather(neg_dir)
    print("\n" + "=" * 78)
    print("FAPH -- continuous negative audio")
    print("=" * 78)
    print("  dir: %s" % neg_dir)
    if not neg_files:
        print("  [X] no negative wavs found -- FAPH not measured.")
    else:
        need = int(args.min_negative_hours * 3600 * SAMPLE_RATE)
        chunks, total = [], 0
        for f in neg_files:
            a = load_wav_mono16k(f)
            chunks.append(a)
            total += len(a)
            if total >= need:
                break
        stream = np.concatenate(chunks)
        hours = len(stream) / SAMPLE_RATE / 3600.0
        print("  concatenated %d clips -> %.4f h (%d samples)"
              % (len(chunks), hours, len(stream)))
        if hours < args.min_negative_hours:
            print("  [!] only %.4f h available, %.2f h requested -- FAPH is "
                  "noisier." % (hours, args.min_negative_hours))
        print("  NOTE: clips are concatenated, so clip boundaries are artificial")
        print("        discontinuities that do not occur in real continuous audio.")
        sc = stream_scores(stream, model)
        print("  windows scored: %d" % len(sc))
        results["faph"] = {"hours": hours, "n_clips": len(chunks)}
        print()
        for n, q, c in ops:
            ne, _ = count_events(sc, q, c, refractory)
            faph = ne / hours if hours > 0 else float("nan")
            print("  %-34s events = %5d   FAPH = %9.2f/h" % (n, ne, faph))
            results["faph"][n] = {"events": ne, "faph": faph}

    # ------------------------------------------------------- hard negatives
    print("\n" + "=" * 78)
    print('HARD NEGATIVES -- "era", "either", "Ida"')
    print("=" * 78)
    hn_files = gather(args.hard_negatives)
    if not hn_files:
        where = (" at %s" % args.hard_negatives if args.hard_negatives
                 else " (--hard-negatives not given)")
        print("  [X] no hard-negative recordings found%s." % where)
        print("      Not measured. Record era/either/Ida clips to populate this.")
    else:
        buckets = collections.defaultdict(list)
        for f in hn_files:
            nm = f.name.lower()
            for w in ("era", "either", "ida"):
                if w in nm:
                    buckets[w].append(f)
                    break
            else:
                buckets["other"].append(f)
        results["hard_negatives"] = {}
        for w in sorted(buckets):
            fs = buckets[w]
            counts = {n: 0 for n, _, _ in ops}
            for f in fs:
                clip = embed_in_context(load_wav_mono16k(f), ambient_pool,
                                        args.positive_context_ms, rng)
                sc = stream_scores(clip, model)
                for n, q, c in ops:
                    if count_events(sc, q, c, refractory)[0] > 0:
                        counts[n] += 1
            print('\n  "%s"  (%d clips)' % (w, len(fs)))
            results["hard_negatives"][w] = {"n": len(fs), "detections": {}}
            for n, _, _ in ops:
                rate = 100.0 * counts[n] / len(fs)
                print("    %-34s detected %4d/%d  (%6.2f%%)"
                      % (n, counts[n], len(fs), rate))
                results["hard_negatives"][w]["detections"][n] = counts[n]

    # ---------------------------------------------------------------- model
    mp = Path(args.model)
    print("\n" + "=" * 78)
    print("MODEL")
    print("=" * 78)
    sz = mp.stat().st_size
    print("  file            : %s" % mp)
    print("  size            : %d bytes (%.2f KB)" % (sz, sz / 1024.0))
    print("  tensor arena    : configured 40960 bytes (IRA_TENSOR_ARENA_SIZE);")
    print("                    ACTUAL USAGE UNMEASURED -- requires hardware.")
    results["model"] = {"path": str(mp), "size_bytes": sz,
                        "arena_configured_bytes": 40960,
                        "arena_used_bytes": None}

    if not clean:
        print("\n" + "!" * 78)
        print("!! REMINDER: every TPR above is CONTAMINATED by train/test speaker")
        print("!! overlap and overstates real-world performance.")
        print("!" * 78)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print("\n  wrote %s" % args.json_out)


# ==========================================================================
# Serial-log fixture & parser
# Source: results/screenshots/serial_log_detection.png
# Raw IDF_v6.0.3 output captured from live ESP32-S3 firmware.
# ==========================================================================

# SAMPLE_LOG — verbatim lines from the serial screenshot.
# Timestamp column is the IDF millisecond uptime printed by ESP_LOGx.
# Columns: I/W  (timestamp)  IRA_WAKEWORD:  <payload>
SAMPLE_LOG = """\
I (104823) IRA_WAKEWORD: RMS 0.014144 | IRA 0.7227 | inference 30.92 ms
I (104963) IRA_WAKEWORD: RMS 0.013914 | IRA 0.7578 | inference 30.92 ms
I (105093) IRA_WAKEWORD: RMS 0.013907 | IRA 0.6055 | inference 30.92 ms
I (105233) IRA_WAKEWORD: RMS 0.014215 | IRA 0.5000 | inference 30.91 ms
I (105363) IRA_WAKEWORD: RMS 0.013755 | IRA 1.0160 | inference 30.93 ms
I (105503) IRA_WAKEWORD: RMS 0.010837 | IRA 0.0391 | inference 30.92 ms
I (105633) IRA_WAKEWORD: RMS 0.009488 | IRA 0.0547 | inference 30.92 ms
I (105773) IRA_WAKEWORD: RMS 0.008332 | IRA 0.0859 | inference 30.92 ms
I (105903) IRA_WAKEWORD: RMS 0.007897 | IRA 0.1836 | inference 30.92 ms
I (106043) IRA_WAKEWORD: RMS 0.007655 | IRA 0.4570 | inference 30.91 ms
I (106173) IRA_WAKEWORD: RMS 0.006623 | IRA 0.3125 | inference 30.92 ms
I (106313) IRA_WAKEWORD: RMS 0.044493 | IRA 0.5859 | inference 30.92 ms
I (106443) IRA_WAKEWORD: RMS 0.044609 | IRA 0.9766 | inference 30.93 ms
I (106583) IRA_WAKEWORD: RMS 0.044530 | IRA 0.9805 | inference 30.92 ms
I (106583) IRA_WAKEWORD: IRA candidate 1/3
I (106713) IRA_WAKEWORD: RMS 0.044421 | IRA 0.9883 | inference 30.92 ms
I (106713) IRA_WAKEWORD: IRA candidate 2/3
I (106853) IRA_WAKEWORD: RMS 0.044410 | IRA 0.9961 | inference 30.91 ms
I (106853) IRA_WAKEWORD: IRA candidate 3/3
W (106853) IRA_WAKEWORD: ==============================
W (106853) IRA_WAKEWORD: >>> IRA DETECTED <<<
W (106853) IRA_WAKEWORD: Probability 0.9961
W (106863) IRA_WAKEWORD: ==============================
I (107003) IRA_WAKEWORD: RMS 0.044408 | IRA 0.9922 | inference 30.94 ms
I (107133) IRA_WAKEWORD: RMS 0.044415 | IRA 0.9766 | inference 30.92 ms
I (107273) IRA_WAKEWORD: RMS 0.044385 | IRA 0.9805 | inference 30.92 ms
I (107273) IRA_WAKEWORD: IRA candidate 1/3
I (107403) IRA_WAKEWORD: RMS 0.044387 | IRA 0.6680 | inference 30.92 ms
I (107543) IRA_WAKEWORD: RMS 0.032175 | IRA 0.0781 | inference 30.93 ms
I (107673) IRA_WAKEWORD: RMS 0.010651 | IRA 0.0625 | inference 30.92 ms
I (107813) IRA_WAKEWORD: RMS 0.010683 | IRA 0.0312 | inference 30.94 ms
I (107943) IRA_WAKEWORD: RMS 0.011108 | IRA 0.0938 | inference 30.92 ms
I (108083) IRA_WAKEWORD: RMS 0.011168 | IRA 0.2422 | inference 30.92 ms
I (108213) IRA_WAKEWORD: RMS 0.011211 | IRA 0.5430 | inference 30.92 ms
I (108353) IRA_WAKEWORD: RMS 0.011286 | IRA 0.6055 | inference 30.92 ms
"""


import re as _re


def parse_serial_log(text):
    """Parse IDF_v6 serial output produced by ira_wakeword firmware.

    Returns a list of dicts, one per log line, with keys:

        level         : 'I' or 'W'
        timestamp_ms  : int   IDF uptime in milliseconds
        tag           : str   e.g. 'IRA_WAKEWORD'
        rms           : float or None
        probability   : float or None
        inference_ms  : float or None
        candidate_n   : int or None   (N from "candidate N/3")
        candidate_of  : int or None   (denominator, always 3 in current firmware)
        detected      : bool          True on the ">>> IRA DETECTED <<<" line
        det_prob      : float or None probability printed on the "Probability" line

    Lines that do not match any known pattern are silently skipped.

    Example
    -------
    >>> rows = parse_serial_log(SAMPLE_LOG)
    >>> detections = [r for r in rows if r['detected']]
    >>> assert len(detections) == 1
    >>> assert detections[0]['timestamp_ms'] == 106853
    """
    # Pattern for the main "RMS ... | IRA ... | inference ... ms" lines.
    _PAT_SCORE = _re.compile(
        r'^([IW])\s+\((\d+)\)\s+(\S+):\s+'
        r'RMS\s+([\d.]+)\s+\|\s+IRA\s+([\d.]+)\s+\|\s+inference\s+([\d.]+)\s+ms'
    )
    # Pattern for "IRA candidate N/M" lines.
    _PAT_CAND = _re.compile(
        r'^([IW])\s+\((\d+)\)\s+(\S+):\s+IRA candidate (\d+)/(\d+)'
    )
    # Pattern for ">>> IRA DETECTED <<<" warning lines.
    _PAT_DET = _re.compile(
        r'^W\s+\((\d+)\)\s+(\S+):\s+>>>\s+IRA DETECTED\s+<<<'
    )
    # Pattern for "Probability X.XXXX" lines that follow a detection.
    _PAT_PROB = _re.compile(
        r'^W\s+\((\d+)\)\s+(\S+):\s+Probability\s+([\d.]+)'
    )

    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = _PAT_SCORE.match(line)
        if m:
            rows.append(dict(
                level=m.group(1),
                timestamp_ms=int(m.group(2)),
                tag=m.group(3),
                rms=float(m.group(4)),
                probability=float(m.group(5)),
                inference_ms=float(m.group(6)),
                candidate_n=None,
                candidate_of=None,
                detected=False,
                det_prob=None,
            ))
            continue

        m = _PAT_CAND.match(line)
        if m:
            rows.append(dict(
                level=m.group(1),
                timestamp_ms=int(m.group(2)),
                tag=m.group(3),
                rms=None,
                probability=None,
                inference_ms=None,
                candidate_n=int(m.group(4)),
                candidate_of=int(m.group(5)),
                detected=False,
                det_prob=None,
            ))
            continue

        m = _PAT_DET.match(line)
        if m:
            rows.append(dict(
                level='W',
                timestamp_ms=int(m.group(1)),
                tag=m.group(2),
                rms=None,
                probability=None,
                inference_ms=None,
                candidate_n=None,
                candidate_of=None,
                detected=True,
                det_prob=None,
            ))
            continue

        m = _PAT_PROB.match(line)
        if m:
            rows.append(dict(
                level='W',
                timestamp_ms=int(m.group(1)),
                tag=m.group(2),
                rms=None,
                probability=None,
                inference_ms=None,
                candidate_n=None,
                candidate_of=None,
                detected=False,
                det_prob=float(m.group(3)),
            ))
            continue
        # Separator lines ("===") and other noise are silently ignored.

    return rows


if __name__ == "__main__":
    main()
