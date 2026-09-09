# -*- coding: utf-8 -*-
"""
evaluate_streaming.py
=====================
Correct continuous-audio streaming evaluation of ira_cnn_int8.tflite.

Preprocessing is IDENTICAL to cnn/train_cnn.py:
  16 kHz audio
  → 1-second analysis window (16,000 samples)
  → tf.signal.stft(frame_length=480, frame_step=320, fft_length=512)
  → tf.abs()
  → tf.math.log(x + 1e-6)
  → [:, :40]
  → per-clip z-score  (x - mean) / (std + 1e-6)
  → INT8 quantize  round(x / scale + zero_point), clip(-128,127)

Streaming mode:
  - Reads full long-form WAV/FLAC files
  - Slides a 1-second window across the audio with a configurable stride
  - Applies trigger/debounce logic (cooldown after trigger)
  - Saves false trigger audio segments + CSV logs

Usage examples:
  python evaluate_streaming.py --help
  python evaluate_streaming.py --neg-dir dataset/negative/speech_streams
  python evaluate_streaming.py --stride-ms 100 --threshold 0.50 --cooldown-ms 1500
  python evaluate_streaming.py --sweep  (runs thresholds 0.30..0.90)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).parent
MODEL_PATH  = REPO_ROOT / "cnn" / "models" / "ira_cnn_int8.tflite"
LIBRI_BASE  = REPO_ROOT / "dataset" / "negative" / "extracted" / "LibriSpeech" / "train-clean-5"
BG_DIR      = REPO_ROOT / "dataset" / "negative" / "background"
TRIGGERS_DIR = REPO_ROOT / "streaming_false_triggers"
PREDS_CSV   = REPO_ROOT / "streaming_predictions.csv"
TRIGGERS_CSV = REPO_ROOT / "streaming_false_triggers.csv"

# LibriSpeech speakers that belong ONLY to the test split (never seen in training)
TEST_SPEAKERS = ["1088", "1737", "5789", "6848"]

# ---------------------------------------------------------------------------
# Preprocessing constants — must match cnn/train_cnn.py EXACTLY
# ---------------------------------------------------------------------------
SAMPLE_RATE        = 16000
WINDOW_SAMPLES     = 16000          # 1.0 second
STFT_FRAME_LENGTH  = 480            # 30 ms
STFT_FRAME_STEP    = 320            # 20 ms
STFT_FFT_LENGTH    = 512
NUM_FREQ_BINS      = 40
# Expected output: (49, 40) — floor((16000-480)/320)+1 = 49 frames


# ===========================================================================
# Preprocessing  (verbatim from cnn/train_cnn.py)
# ===========================================================================

def make_spectrogram(window: np.ndarray) -> np.ndarray:
    """
    Convert a 16,000-sample float32 audio window into (49, 40) float32 feature.
    Exact copy of train_cnn.py::make_spectrogram().
    """
    assert len(window) == WINDOW_SAMPLES, f"Expected {WINDOW_SAMPLES} samples, got {len(window)}"
    audio_t = tf.convert_to_tensor(window, dtype=tf.float32)

    spec = tf.signal.stft(
        audio_t,
        frame_length=STFT_FRAME_LENGTH,
        frame_step=STFT_FRAME_STEP,
        fft_length=STFT_FFT_LENGTH,
    )                                        # complex (49, 257)
    spec = tf.abs(spec)                      # magnitude (49, 257)
    spec = tf.math.log(spec + 1e-6)         # log(|STFT|+1e-6)
    spec = spec[:, :NUM_FREQ_BINS]          # (49, 40)

    mean = tf.reduce_mean(spec)
    std  = tf.math.reduce_std(spec) + 1e-6
    spec = (spec - mean) / std              # per-window z-score

    return spec.numpy().astype(np.float32)  # (49, 40)


# ===========================================================================
# Model wrapper
# ===========================================================================

class IraDetector:
    """INT8 TFLite inference with correct quantize/dequantize."""

    def __init__(self, model_path: str = str(MODEL_PATH)):
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        self._interp  = interp
        self._inp     = interp.get_input_details()[0]
        self._out     = interp.get_output_details()[0]

        qp_in  = self._inp["quantization_parameters"]
        qp_out = self._out["quantization_parameters"]
        self.inp_scale = float(qp_in["scales"][0])
        self.inp_zp    = int(qp_in["zero_points"][0])
        self.out_scale = float(qp_out["scales"][0])
        self.out_zp    = int(qp_out["zero_points"][0])

        assert tuple(self._inp["shape"]) == (1, 49, 40, 1), \
            f"Unexpected input shape: {self._inp['shape']}"

    def predict(self, window: np.ndarray) -> float:
        """window: float32 (16000,). Returns probability in [0,1]."""
        spec  = make_spectrogram(window)           # (49, 40)
        x     = spec[np.newaxis, ..., np.newaxis]  # (1, 49, 40, 1) float32

        if self._inp["dtype"] == np.int8:
            x_q = np.round(x / self.inp_scale + self.inp_zp)
            x_q = np.clip(x_q, -128, 127).astype(np.int8)
        else:
            x_q = x.astype(np.float32)

        self._interp.set_tensor(self._inp["index"], x_q)
        self._interp.invoke()
        out = self._interp.get_tensor(self._out["index"])

        if self._out["dtype"] == np.int8:
            return (float(out[0, 0]) - self.out_zp) * self.out_scale
        return float(out[0, 0])

    def print_info(self):
        print(f"  Model          : {MODEL_PATH.name}")
        print(f"  Input shape    : {tuple(self._inp['shape'])}  dtype={self._inp['dtype']}")
        print(f"  Output shape   : {tuple(self._out['shape'])} dtype={self._out['dtype']}")
        print(f"  Input  quant   : scale={self.inp_scale:.8f}  zero_point={self.inp_zp}")
        print(f"  Output quant   : scale={self.out_scale:.8f}  zero_point={self.out_zp}")


# ===========================================================================
# Audio source helpers
# ===========================================================================

def load_audio_file(path: str) -> np.ndarray:
    """Load any WAV/FLAC as float32 mono at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
    return audio


def build_negative_file_list(
    use_libri: bool = True,
    use_background: bool = True,
    speakers: list = None,
) -> list:
    """
    Returns list of (path, source_tag) for negative audio files.
    Only includes test-split LibriSpeech speakers (never seen in training).
    """
    files = []
    if use_libri:
        sps = speakers or TEST_SPEAKERS
        for sp in sps:
            sp_dir = LIBRI_BASE / sp
            if sp_dir.exists():
                for f in sorted(sp_dir.rglob("*.flac")):
                    files.append((str(f), f"libri_{sp}"))
            else:
                print(f"  [WARN] Speaker directory missing: {sp_dir}")

    if use_background:
        if BG_DIR.exists():
            for f in sorted(BG_DIR.glob("*.wav")):
                files.append((str(f), "background"))
        else:
            print(f"  [WARN] Background dir missing: {BG_DIR}")

    return files


# ===========================================================================
# Streaming evaluator
# ===========================================================================

class StreamingEvaluator:
    """
    Slides a 1-second analysis window across audio with configurable stride.
    Applies trigger/cooldown logic so one acoustic event = one trigger count.
    """

    def __init__(
        self,
        detector: IraDetector,
        stride_ms: int = 100,
        threshold: float = 0.50,
        consecutive: int = 1,
        cooldown_ms: int = 1500,
    ):
        self.detector    = detector
        self.stride_ms   = stride_ms
        self.stride_samp = int(SAMPLE_RATE * stride_ms / 1000)
        self.threshold   = threshold
        self.consecutive = consecutive
        self.cooldown_samp = int(SAMPLE_RATE * cooldown_ms / 1000)

        # Accumulated results
        self.total_audio_sec   = 0.0
        self.all_predictions   = []   # (source, timestamp_sec, score)
        self.false_triggers    = []   # (source, timestamp_sec, score, audio_segment)

    def _should_trigger(self, run_length: int) -> bool:
        return run_length >= self.consecutive

    def process_file(self, audio_path: str, source_tag: str) -> dict:
        """
        Process one audio file. Returns per-file stats.
        """
        audio = load_audio_file(audio_path)
        n_samples = len(audio)
        duration_sec = n_samples / SAMPLE_RATE
        self.total_audio_sec += duration_sec

        raw_positive_frames = 0
        trigger_events      = 0

        # Streaming state
        cooldown_remaining = 0   # samples remaining in cooldown
        consecutive_run    = 0   # consecutive above-threshold windows

        pos = 0  # current window start in samples

        while pos + WINDOW_SAMPLES <= n_samples:
            window = audio[pos : pos + WINDOW_SAMPLES].copy()
            timestamp_sec = pos / SAMPLE_RATE

            score = self.detector.predict(window)
            is_positive = score >= self.threshold

            self.all_predictions.append((source_tag, timestamp_sec, score))

            if is_positive:
                raw_positive_frames += 1

            # Trigger logic
            if cooldown_remaining > 0:
                cooldown_remaining -= self.stride_samp
                cooldown_remaining = max(0, cooldown_remaining)
                consecutive_run = 0
            else:
                if is_positive:
                    consecutive_run += 1
                    if self._should_trigger(consecutive_run):
                        # FALSE TRIGGER
                        trigger_events += 1
                        # Save audio segment: 1s before + 2s after trigger point
                        seg_start = max(0, pos - SAMPLE_RATE)
                        seg_end   = min(n_samples, pos + 2 * SAMPLE_RATE)
                        segment   = audio[seg_start:seg_end].copy()
                        self.false_triggers.append({
                            "source":         source_tag,
                            "source_file":    audio_path,
                            "timestamp_sec":  timestamp_sec,
                            "score":          score,
                            "segment":        segment,
                        })
                        # Start cooldown
                        cooldown_remaining = self.cooldown_samp
                        consecutive_run    = 0
                else:
                    consecutive_run = 0

            pos += self.stride_samp

        return {
            "source":              source_tag,
            "duration_sec":        duration_sec,
            "raw_positive_frames": raw_positive_frames,
            "trigger_events":      trigger_events,
        }

    def process_all(self, file_list: list, verbose: bool = True) -> list:
        """Process a list of (path, source_tag) tuples. Returns per-file stats."""
        stats = []
        total = len(file_list)
        t0 = time.time()
        last_print = t0

        for i, (path, tag) in enumerate(file_list):
            result = self.process_file(path, tag)
            stats.append(result)

            now = time.time()
            if verbose and (now - last_print > 5.0 or i == total - 1):
                pct = (i + 1) / total * 100
                elapsed = now - t0
                audio_h = self.total_audio_sec / 3600
                n_trig  = len(self.false_triggers)
                print(f"    [{i+1:4d}/{total}] {pct:5.1f}%  "
                      f"audio={audio_h:.3f}h  triggers={n_trig}  "
                      f"elapsed={elapsed:.0f}s")
                last_print = now

        return stats

    @property
    def total_audio_hours(self) -> float:
        return self.total_audio_sec / 3600

    @property
    def n_false_triggers(self) -> int:
        return len(self.false_triggers)

    @property
    def faph(self) -> float:
        if self.total_audio_hours == 0:
            return float("nan")
        return self.n_false_triggers / self.total_audio_hours

    @property
    def n_raw_positive_frames(self) -> int:
        return sum(1 for (_, _, s) in self.all_predictions if s >= self.threshold)

    def save_triggers(self, out_dir: Path, trigger_csv: Path):
        """Save false trigger WAV clips and the CSV index."""
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, trig in enumerate(self.false_triggers, 1):
            clip_name = f"false_trigger_{idx:04d}.wav"
            clip_path = out_dir / clip_name
            sf.write(str(clip_path), trig["segment"], SAMPLE_RATE, subtype="PCM_16")
            rows.append({
                "trigger_id":      idx,
                "source_file":     trig["source_file"],
                "timestamp_sec":   f"{trig['timestamp_sec']:.3f}",
                "model_score":     f"{trig['score']:.4f}",
                "threshold":       self.threshold,
                "audio_clip_path": str(clip_path),
            })

        with trigger_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["trigger_id","source_file",
                                               "timestamp_sec","model_score",
                                               "threshold","audio_clip_path"])
            w.writeheader()
            w.writerows(rows)

    def save_predictions(self, pred_csv: Path):
        """Save per-frame prediction log."""
        trigger_times = {(t["source"], round(t["timestamp_sec"], 3))
                         for t in self.false_triggers}
        with pred_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source_file","timestamp_sec",
                                               "score","threshold",
                                               "raw_positive","final_trigger"])
            w.writeheader()
            for (src, ts, score) in self.all_predictions:
                raw_pos = 1 if score >= self.threshold else 0
                is_trig = 1 if (src, round(ts, 3)) in trigger_times else 0
                w.writerow({
                    "source_file":   src,
                    "timestamp_sec": f"{ts:.3f}",
                    "score":         f"{score:.4f}",
                    "threshold":     self.threshold,
                    "raw_positive":  raw_pos,
                    "final_trigger": is_trig,
                })

    def print_summary(self):
        h = self.total_audio_hours
        print(f"  Total negative audio : {self.total_audio_sec:.1f}s "
              f"({self.total_audio_sec/60:.2f} min = {h:.4f} h)")
        print(f"  Inference windows    : {len(self.all_predictions)}")
        print(f"  Raw positive frames  : {self.n_raw_positive_frames}")
        print(f"  False trigger events : {self.n_false_triggers}")
        print(f"  FAPH                 : {self.faph:.4f}")
        print(f"  Stride               : {self.stride_ms} ms")
        print(f"  Threshold            : {self.threshold}")
        print(f"  Consecutive req.     : {self.consecutive}")
        print(f"  Cooldown             : {self.cooldown_samp/SAMPLE_RATE*1000:.0f} ms")


# ===========================================================================
# Threshold sweep
# ===========================================================================

def threshold_sweep(
    file_list: list,
    detector: IraDetector,
    thresholds: list,
    stride_ms: int,
    consecutive: int,
    cooldown_ms: int,
    verbose: bool = True,
) -> list:
    """
    Runs streaming evaluation at multiple thresholds on the SAME file list.
    Pre-computes all scores once, then replays them at each threshold.
    Returns list of result dicts.
    """
    print("  Pre-computing scores for all files (one pass)...")
    t0 = time.time()

    stride_samp = int(SAMPLE_RATE * stride_ms / 1000)

    # Collect (source_tag, timestamp_sec, audio_segment, score) for every window
    all_windows = []
    total_audio_sec = 0.0
    total_files = len(file_list)

    for i, (path, tag) in enumerate(file_list):
        audio = load_audio_file(path)
        n     = len(audio)
        total_audio_sec += n / SAMPLE_RATE

        pos = 0
        while pos + WINDOW_SAMPLES <= n:
            window = audio[pos : pos + WINDOW_SAMPLES].copy()
            ts     = pos / SAMPLE_RATE
            score  = detector.predict(window)
            # store segment reference for potential trigger saving
            seg_start = max(0, pos - SAMPLE_RATE)
            seg_end   = min(n, pos + 2 * SAMPLE_RATE)
            all_windows.append((tag, path, ts, score, audio[seg_start:seg_end].copy()))
            pos += stride_samp

        if verbose and (i + 1) % max(1, total_files // 10) == 0:
            print(f"    scored {i+1}/{total_files} files  "
                  f"({total_audio_sec/60:.1f} min audio,  "
                  f"{len(all_windows)} windows,  "
                  f"{time.time()-t0:.0f}s elapsed)")

    total_audio_hours = total_audio_sec / 3600
    cooldown_samp     = int(SAMPLE_RATE * cooldown_ms / 1000)

    print(f"  Scoring complete: {len(all_windows)} windows, "
          f"{total_audio_sec/60:.1f} min = {total_audio_hours:.4f} h")
    print()

    sweep_results = []

    for thr in thresholds:
        # Replay with this threshold
        triggers = []
        cooldown_remaining = {}   # per (tag, path) -> remaining samples
        run_length         = {}   # per (tag, path) -> consecutive count

        for (tag, path, ts, score, segment) in all_windows:
            key = (tag, path)
            cr  = cooldown_remaining.get(key, 0)
            rl  = run_length.get(key, 0)

            if cr > 0:
                cooldown_remaining[key] = max(0, cr - stride_samp)
                run_length[key] = 0
            else:
                if score >= thr:
                    rl += 1
                    run_length[key] = rl
                    if rl >= consecutive:
                        triggers.append((tag, path, ts, score, segment))
                        cooldown_remaining[key] = cooldown_samp
                        run_length[key] = 0
                else:
                    run_length[key] = 0

        raw_pos = sum(1 for (_, _, _, s, _) in all_windows if s >= thr)
        n_trig  = len(triggers)
        faph    = n_trig / total_audio_hours if total_audio_hours > 0 else float("nan")

        sweep_results.append({
            "threshold":           thr,
            "raw_positive_frames": raw_pos,
            "trigger_events":      n_trig,
            "total_audio_hours":   total_audio_hours,
            "faph":                faph,
            "triggers":            triggers,
        })

        print(f"  thr={thr:.2f}  raw_pos={raw_pos:5d}  "
              f"triggers={n_trig:4d}  FAPH={faph:.4f}")

    return sweep_results, total_audio_sec


# ===========================================================================
# Safety checks
# ===========================================================================

def safety_checks(file_list: list, verbose: bool = True):
    """
    Verify no IRA-positive audio is in the evaluation set.
    Prints a warning and aborts if 'ira' appears in any filename.
    """
    flagged = []
    for (path, tag) in file_list:
        name = Path(path).name.lower()
        if "ira" in name and "libri" not in name and "background" not in name:
            flagged.append(path)

    if flagged:
        print("[SAFETY CHECK FAILED] The following files may contain IRA utterances:")
        for f in flagged:
            print(f"  {f}")
        print("Aborting. Do not include positive audio in negative FAPH testing.")
        sys.exit(1)

    if verbose:
        print(f"  Safety check passed: {len(file_list)} files, none flagged as IRA-positive.")

    # Verify audio durations match expected
    total_files = len(file_list)
    total_sec = 0.0
    for (path, _) in file_list:
        info = sf.info(path)
        total_sec += info.duration
    return total_sec


# ===========================================================================
# CLI
# ===========================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Streaming FAPH evaluation for IRA wake-word model."
    )
    p.add_argument("--model",        default=str(MODEL_PATH))
    p.add_argument("--stride-ms",    type=int,   default=100,
                   help="Inference stride in ms (default 100)")
    p.add_argument("--threshold",    type=float, default=0.50,
                   help="Detection threshold (default 0.50)")
    p.add_argument("--consecutive",  type=int,   default=1,
                   help="Consecutive positive windows to trigger (default 1)")
    p.add_argument("--cooldown-ms",  type=int,   default=1500,
                   help="Cooldown after trigger in ms (default 1500)")
    p.add_argument("--sweep",        action="store_true",
                   help="Run threshold sweep 0.30..0.90")
    p.add_argument("--no-background",action="store_true",
                   help="Exclude background noise clips")
    p.add_argument("--no-libri",     action="store_true",
                   help="Exclude LibriSpeech FLACs")
    p.add_argument("--save-triggers",action="store_true", default=True,
                   help="Save false trigger audio clips (default True)")
    p.add_argument("--out-dir",      default=str(TRIGGERS_DIR))
    return p


def main():
    args = build_parser().parse_args()

    print()
    print("=" * 68)
    print("  IRA STREAMING EVALUATION — Correct Training Pipeline")
    print("=" * 68)

    # Load model
    print("\nLoading model...")
    detector = IraDetector(args.model)
    detector.print_info()

    print(f"\nPreprocessing parameters (verbatim from cnn/train_cnn.py):")
    print(f"  SAMPLE_RATE       = {SAMPLE_RATE}")
    print(f"  WINDOW_SAMPLES    = {WINDOW_SAMPLES}  (1.0 s)")
    print(f"  STFT frame_length = {STFT_FRAME_LENGTH}  (30 ms)")
    print(f"  STFT frame_step   = {STFT_FRAME_STEP}    (20 ms)")
    print(f"  STFT fft_length   = {STFT_FFT_LENGTH}")
    print(f"  freq_bins_kept    = {NUM_FREQ_BINS}")
    print(f"  log_epsilon       = 1e-6")
    print(f"  normalization     = per-window z-score  (x-mean)/(std+1e-6)")
    print()
    print(f"Streaming parameters:")
    print(f"  stride_ms         = {args.stride_ms} ms")
    print(f"  threshold         = {args.threshold}")
    print(f"  consecutive       = {args.consecutive}")
    print(f"  cooldown_ms       = {args.cooldown_ms}")

    # Build file list
    print("\nBuilding negative file list...")
    file_list = build_negative_file_list(
        use_libri=not args.no_libri,
        use_background=not args.no_background,
    )
    print(f"  {len(file_list)} files queued")

    # Safety checks
    print("\nRunning safety checks...")
    total_sec_measured = safety_checks(file_list, verbose=True)
    print(f"  Total duration (measured): {total_sec_measured:.1f}s  "
          f"({total_sec_measured/60:.2f} min = {total_sec_measured/3600:.4f} h)")

    out_dir = Path(args.out_dir)

    # -----------------------------------------------------------------------
    # Sweep or single threshold
    # -----------------------------------------------------------------------

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90] if args.sweep else [args.threshold]

    print(f"\nRunning {'threshold sweep' if args.sweep else 'single threshold'}: {thresholds}")
    print()

    sweep_results, total_audio_sec = threshold_sweep(
        file_list=file_list,
        detector=detector,
        thresholds=thresholds,
        stride_ms=args.stride_ms,
        consecutive=args.consecutive,
        cooldown_ms=args.cooldown_ms,
        verbose=True,
    )

    total_h = total_audio_sec / 3600

    # -----------------------------------------------------------------------
    # Print sweep table
    # -----------------------------------------------------------------------

    print()
    print("=" * 68)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 68)
    print(f"  Total negative audio : {total_audio_sec:.1f}s  "
          f"({total_audio_sec/60:.2f} min = {total_h:.4f} h)")
    print(f"  Stride               : {args.stride_ms} ms")
    print(f"  Consecutive req.     : {args.consecutive}")
    print(f"  Cooldown             : {args.cooldown_ms} ms")
    print()
    print(f"  {'Threshold':>10}  {'Raw Pos Frames':>15}  "
          f"{'False Triggers':>15}  {'FAPH':>10}")
    print("  " + "-" * 60)
    for r in sweep_results:
        print(f"  {r['threshold']:>10.2f}  {r['raw_positive_frames']:>15d}  "
              f"{r['trigger_events']:>15d}  {r['faph']:>10.4f}")

    # -----------------------------------------------------------------------
    # Save artifacts for the primary threshold (or 0.50 from sweep)
    # -----------------------------------------------------------------------

    primary_thr = args.threshold if not args.sweep else 0.50
    primary_result = next(
        (r for r in sweep_results if abs(r["threshold"] - primary_thr) < 1e-6),
        sweep_results[0]
    )

    if args.save_triggers:
        # Save triggers for primary threshold
        print(f"\nSaving {len(primary_result['triggers'])} false trigger clips "
              f"(threshold={primary_thr})...")

        trig_dir_thr = out_dir / f"threshold_{int(primary_thr*100):02d}"
        trig_dir_thr.mkdir(parents=True, exist_ok=True)
        trig_csv = REPO_ROOT / f"streaming_false_triggers_thr{int(primary_thr*100):02d}.csv"

        trig_rows = []
        for idx, (tag, path, ts, score, segment) in enumerate(primary_result["triggers"], 1):
            clip_name = f"false_trigger_{idx:04d}.wav"
            clip_path = trig_dir_thr / clip_name
            sf.write(str(clip_path), segment, SAMPLE_RATE, subtype="PCM_16")
            trig_rows.append({
                "trigger_id":      idx,
                "source_file":     path,
                "timestamp_sec":   f"{ts:.3f}",
                "model_score":     f"{score:.4f}",
                "threshold":       primary_thr,
                "audio_clip_path": str(clip_path),
            })

        with trig_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["trigger_id","source_file",
                                               "timestamp_sec","model_score",
                                               "threshold","audio_clip_path"])
            w.writeheader()
            w.writerows(trig_rows)
        print(f"  Trigger CSV  : {trig_csv}")
        print(f"  Trigger clips: {trig_dir_thr}")

    # Save sweep summary CSV
    sweep_csv = REPO_ROOT / "streaming_sweep_results.csv"
    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold","raw_positive_frames",
                                           "trigger_events","total_audio_hours","faph"])
        w.writeheader()
        for r in sweep_results:
            w.writerow({k: v for k, v in r.items() if k != "triggers"})
    print(f"\n  Sweep CSV saved: {sweep_csv}")

    print()
    print("=" * 68)
    print("  Done.")
    print("=" * 68)

    return sweep_results


if __name__ == "__main__":
    main()
