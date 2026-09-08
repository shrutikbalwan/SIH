#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — Standalone evaluation script for the IRA wake-word TFLite models.

Usage:
    python evaluate.py [--model PATH] [--positives DIR] [--negatives DIR]
                       [--threshold FLOAT] [--sliding-window INT]
                       [--ignore-after-accept INT] [--max-neg INT] [--all-models]

What it measures
----------------
  TPR  (True Positive Rate / Recall): fraction of real "ira" clips detected.
  FNR  (False Negative Rate):         fraction of real "ira" clips missed.
  FPR  (False Positive Rate):         fraction of negative clips mis-triggered.
  FAPH (False Accepts Per Hour):      triggers on background audio per hour —
                                      the metric microWakeWord prioritises.
  ROC AUC: area under the FAPH vs. FNR curve (lower is better).

Folder layout expected
----------------------
  cnn/models/                              <- .tflite files (or --model)
  real/positive/<condition>/*.wav          <- positive test samples
  dataset/negative/speech/*.wav            <- negative test samples

Dependencies (already in your venv):
  ai_edge_litert, pymicro_features, scipy, numpy
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile

# Force UTF-8 on Windows PowerShell/cmd so Unicode chars print safely
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Python < 3.7 fallback

# ── locate the microwakeword package bundled inside this repo ──────────────────
REPO_ROOT = Path(__file__).parent
MWW_SRC = REPO_ROOT / "micro-wake-word"
if str(MWW_SRC) not in sys.path:
    sys.path.insert(0, str(MWW_SRC))

from microwakeword.inference import Model  # noqa: E402
from microwakeword.audio.audio_utils import generate_features_for_clip  # noqa: E402
from microwakeword.test import (  # noqa: E402
    compute_false_accepts_per_hour,
    generate_roc_curve,
)

# ── default paths ──────────────────────────────────────────────────────────────
DEFAULT_MODELS_DIR = REPO_ROOT / "cnn" / "models"
DEFAULT_POSITIVES_DIR = REPO_ROOT / "real" / "positive"
DEFAULT_NEGATIVES_DIR = REPO_ROOT / "dataset" / "negative" / "speech"

# ── constants matching the CNN training config ─────────────────────────────────
STEP_MS = 20        # window step used during feature extraction
SAMPLE_RATE = 16000  # Hz


# ══════════════════════════════════════════════════════════════════════════════
# Audio helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_wav_as_int16(path: str) -> Optional[np.ndarray]:
    """Load a WAV file and return int16 samples at 16 kHz.

    Returns None (with a warning) if the file cannot be read or has the
    wrong sample rate.
    """
    try:
        rate, data = wavfile.read(path)
    except Exception as exc:
        print(f"  [WARN] Cannot read {path}: {exc}")
        return None

    if rate != SAMPLE_RATE:
        print(
            f"  [WARN] {Path(path).name}: expected {SAMPLE_RATE} Hz, "
            f"got {rate} Hz — skipping"
        )
        return None

    # Stereo -> mono
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.int16)

    # float32/float64 WAVs -> int16
    if data.dtype in (np.float32, np.float64):
        data = np.clip(data * 32768, -32768, 32767).astype(np.int16)

    return data


def collect_wav_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """Recursively collect .wav paths under *directory*."""
    paths = sorted(Path(directory).rglob("*.wav"))
    if max_files and len(paths) > max_files:
        paths = paths[:max_files]
    return [str(p) for p in paths]


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation routines
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_positives(
    model: Model,
    positive_dir: str,
    threshold: float = 0.5,
    sliding_window: int = 5,
    ignore_after_accept: int = 25,
) -> Tuple[dict, dict]:
    """Evaluate TPR / FNR on positive (wake-word) clips.

    Uses the same logic as microwakeword's tflite_streaming_model_roc:
    slide the model over each clip, apply a moving-average window, and
    take the *maximum* smoothed probability as the clip score.

    Returns
    -------
    overall_metrics : dict
    per_condition   : dict[str, dict]
    """
    wav_files = collect_wav_files(positive_dir)
    if not wav_files:
        print(f"  [WARN] No .wav files found under {positive_dir}")
        return {}, {}

    per_condition: dict = {}
    all_max_probs: List[float] = []

    for wav_path in wav_files:
        audio = load_wav_as_int16(wav_path)
        if audio is None:
            continue

        spectrogram = generate_features_for_clip(audio, step_ms=STEP_MS)
        probabilities = model.predict_spectrogram(spectrogram)

        # Need at least sliding_window predictions to form one averaged window.
        if len(probabilities) < sliding_window:
            continue  # clip genuinely too short even in streaming mode

        # For positive clips we look for the highest smoothed probability the
        # model ever outputs.  We respect ignore_after_accept as a warm-up
        # offset (skipping the first N raw predictions), but fall back to the
        # full sequence if the clip is too short for that offset — otherwise
        # every ~1 s clip would be silently discarded.
        start_idx = min(ignore_after_accept, max(0, len(probabilities) - sliding_window))
        probs_to_score = probabilities[start_idx:]

        if len(probs_to_score) < sliding_window:
            probs_to_score = probabilities  # fallback: use all predictions

        smoothed = sliding_window_view(probs_to_score, sliding_window).mean(axis=-1)
        max_prob = float(np.max(smoothed)) if len(smoothed) > 0 else 0.0

        all_max_probs.append(max_prob)

        condition = Path(wav_path).parent.name
        per_condition.setdefault(condition, []).append(max_prob)

    tp = sum(p >= threshold for p in all_max_probs)
    fn = len(all_max_probs) - tp
    total = len(all_max_probs)

    overall_metrics = {
        "true_positives": tp,
        "false_negatives": fn,
        "total": total,
        "tpr": tp / total if total else float("nan"),
        "fnr": fn / total if total else float("nan"),
        "threshold": threshold,
        "max_probs": all_max_probs,
    }

    condition_metrics: dict = {}
    for cond, probs in per_condition.items():
        c_tp = sum(p >= threshold for p in probs)
        c_fn = len(probs) - c_tp
        c_total = len(probs)
        condition_metrics[cond] = {
            "tpr": c_tp / c_total if c_total else float("nan"),
            "fnr": c_fn / c_total if c_total else float("nan"),
            "total": c_total,
        }

    return overall_metrics, condition_metrics


def evaluate_negatives(
    model: Model,
    negative_dir: str,
    sliding_window: int = 5,
    ignore_after_accept: int = 25,
    max_files: Optional[int] = None,
) -> dict:
    """Evaluate FAPH on negative (non-wake-word) clips.

    Computes across 101 thresholds (0.00 -> 1.00).

    Returns
    -------
    dict with: cutoffs, faph_curve, faph_at_0_5, total_clips, total_hours
    """
    wav_files = collect_wav_files(negative_dir, max_files=max_files)
    if not wav_files:
        print(f"  [WARN] No .wav files found under {negative_dir}")
        return {}

    print(f"  Processing {len(wav_files)} negative clips …")
    streaming_probs_list: List[np.ndarray] = []
    total_audio_s = 0.0

    for i, wav_path in enumerate(wav_files):
        if i > 0 and i % 500 == 0:
            print(f"    … {i}/{len(wav_files)} done")

        audio = load_wav_as_int16(wav_path)
        if audio is None:
            continue

        total_audio_s += len(audio) / SAMPLE_RATE
        spectrogram = generate_features_for_clip(audio, step_ms=STEP_MS)
        probabilities = model.predict_spectrogram(spectrogram)

        if len(probabilities) < sliding_window:
            continue

        smoothed = sliding_window_view(probabilities, sliding_window).mean(axis=-1)
        streaming_probs_list.append(smoothed)

    if not streaming_probs_list:
        return {}

    cutoffs = np.arange(0, 1.01, 0.01)
    faph_curve = compute_false_accepts_per_hour(
        streaming_probs_list,
        cutoffs,
        ignore_slices_after_accept=ignore_after_accept,
        stride=1,
        step_s=STEP_MS / 1000.0,
    )

    return {
        "cutoffs": cutoffs,
        "faph_curve": faph_curve,
        "streaming_probabilities_list": streaming_probs_list,
        "faph_at_0_5": float(faph_curve[50]),
        "total_clips": len(wav_files),
        "total_hours": total_audio_s / 3600.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sep(char="-", width=72):
    print(char * width)


def print_positive_report(model_name: str, pos: dict, cond: dict):
    _sep("=")
    print(f"  MODEL : {model_name}")
    _sep()
    print(f"  Positive clips tested : {pos['total']}")
    print(f"  Threshold             : {pos['threshold']:.2f}")
    print(f"  True Positive Rate    : {pos['tpr']*100:6.2f}%  (TPR / Recall)")
    print(f"  False Negative Rate   : {pos['fnr']*100:6.2f}%  (missed wakes)")

    if cond:
        print()
        print("  Per-condition breakdown:")
        for name in sorted(cond):
            m = cond[name]
            filled = int(m["tpr"] * 20)
            bar = "#" * filled + "." * (20 - filled)
            print(
                f"    {name:<25s}  TPR={m['tpr']*100:5.1f}%  "
                f"FNR={m['fnr']*100:5.1f}%  (N={m['total']:3d})  [{bar}]"
            )


def print_negative_report(model_name: str, neg: dict):
    _sep()
    total_h = neg["total_hours"]
    print(f"  Negative clips tested : {neg['total_clips']}")
    print(f"  Total audio duration  : {total_h*60:.1f} min  ({total_h:.4f} h)")
    print(f"  FAPH @ threshold 0.50 : {neg['faph_at_0_5']:.4f}  false-accepts / hour")
    print()
    print("  FAPH across thresholds:")
    print("  thr   FAPH     bar")
    print("  ---   ------   " + "-" * 30)
    for thr in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        idx = round(thr * 100)
        faph = neg["faph_curve"][idx]
        bar = "|" * min(int(faph * 8), 30)
        print(f"  {thr:.2f}  {faph:6.3f}   {bar}")

    print()
    print("  Reference: released microWakeWord models ≈ 0.2 – 1.0 FAPH.")
    my_faph = neg["faph_at_0_5"]
    if my_faph < 0.5:
        verdict = "[OK]  Excellent -- better than the typical range."
    elif my_faph < 1.0:
        verdict = "[OK]  Good -- within the typical range."
    elif my_faph < 2.0:
        verdict = "[!!]  Elevated -- consider more hard-negative training data."
    else:
        verdict = "[XX]  High -- model likely needs retraining with more negatives."
    print(f"  Verdict @ 0.50 : {verdict}")


def print_roc_summary(pos: dict, neg: dict):
    """Print AUC and top operating points from the ROC curve."""
    if not pos or not neg:
        return

    max_probs = pos["max_probs"]
    cutoffs = neg["cutoffs"]
    faph_curve = neg["faph_curve"]

    fnr_at_cutoffs = np.array([
        1.0 - sum(p >= c for p in max_probs) / len(max_probs)
        for c in cutoffs
    ])

    x_coords, y_coords, cutoffs_at_pts = generate_roc_curve(
        false_accepts_per_hour=faph_curve,
        false_rejections=fnr_at_cutoffs,
        cutoffs=cutoffs,
        max_faph=2.0,
    )

    if len(x_coords) == 0:
        return

    auc = float(np.trapezoid(y_coords, x_coords))
    _sep()
    print(f"  ROC AUC (FAPH vs FNR, max_faph=2.0) : {auc:.5f}  (lower is better)")
    print()
    print("  Selected operating points:")
    print("    FAPH      FNR       threshold")
    indices = np.linspace(0, len(x_coords) - 1, min(8, len(x_coords)), dtype=int)
    for i in indices:
        print(
            f"    {x_coords[i]:<8.3f}  {y_coords[i]*100:5.1f}%     "
            f"{cutoffs_at_pts[i]:.2f}"
        )
    _sep("═")


def save_report(output_path: str, model_name: str, pos: dict, cond: dict, neg: dict):
    """Append a concise text summary to *output_path*."""
    lines = [
        "",
        "=" * 72,
        f"Model: {model_name}",
        f"Positive clips: {pos.get('total', 'N/A')}",
        f"TPR @ {pos.get('threshold', 0.5):.2f}: "
        f"{pos.get('tpr', float('nan'))*100:.2f}%",
        f"FNR @ {pos.get('threshold', 0.5):.2f}: "
        f"{pos.get('fnr', float('nan'))*100:.2f}%",
        f"Negative clips: {neg.get('total_clips', 'N/A')}",
        f"Total audio: {neg.get('total_hours', 0)*60:.1f} min",
        f"FAPH @ 0.50: {neg.get('faph_at_0_5', float('nan')):.4f}",
    ]

    if cond:
        lines.append("Per-condition TPR:")
        for name in sorted(cond):
            m = cond[name]
            lines.append(f"  {name}: TPR={m['tpr']*100:.1f}% (N={m['total']})")

    if neg.get("faph_curve") is not None:
        lines.append("FAPH curve (threshold -> faph):")
        for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            idx = round(thr * 100)
            lines.append(f"  {thr:.2f} -> {neg['faph_curve'][idx]:.4f}")

    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  Report appended to: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate IRA wake-word TFLite model(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", "-m",
        help="Path to a specific .tflite model.",
    )
    p.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help=f"Directory containing .tflite models. Default: {DEFAULT_MODELS_DIR}",
    )
    p.add_argument(
        "--all-models", action="store_true",
        help="Evaluate every .tflite in --models-dir "
             "(default: only the newest one).",
    )
    p.add_argument(
        "--positives", "-p",
        default=str(DEFAULT_POSITIVES_DIR),
        help=f"Root of positive WAV files. Default: {DEFAULT_POSITIVES_DIR}",
    )
    p.add_argument(
        "--negatives", "-n",
        default=str(DEFAULT_NEGATIVES_DIR),
        help=f"Root of negative WAV files. Default: {DEFAULT_NEGATIVES_DIR}",
    )
    p.add_argument(
        "--threshold", "-t", type=float, default=0.5,
        help="Decision threshold for TPR/FNR. Default: 0.5",
    )
    p.add_argument(
        "--sliding-window", "-w", type=int, default=5,
        help="Moving-average window in frames. Default: 5",
    )
    p.add_argument(
        "--ignore-after-accept", "-i", type=int, default=25,
        help="Cooldown frames after a detection. Default: 25",
    )
    p.add_argument(
        "--max-neg", type=int, default=None,
        help="Cap negative files for a quick test. Default: all",
    )
    p.add_argument(
        "--skip-negatives", action="store_true",
        help="Skip FAPH evaluation (positives only).",
    )
    p.add_argument(
        "--output", "-o",
        help="Append a text report to this file.",
    )
    return p


def find_models(args) -> List[str]:
    if args.model:
        if not Path(args.model).exists():
            print(f"ERROR: Model not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        return [args.model]

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}", file=sys.stderr)
        sys.exit(1)

    all_tflite = sorted(models_dir.glob("*.tflite"))
    if not all_tflite:
        print(f"ERROR: No .tflite files in {models_dir}", file=sys.stderr)
        sys.exit(1)

    if args.all_models:
        return [str(p) for p in all_tflite]

    newest = max(all_tflite, key=lambda p: p.stat().st_mtime)
    print(f"  Auto-selected newest model: {newest.name}")
    print(f"  (--all-models evaluates all; --model PATH for a specific one)")
    return [str(newest)]


def main():
    parser = build_parser()
    args = parser.parse_args()
    model_paths = find_models(args)

    print()
    print("=" * 76)
    print("  IRA Wake-Word Model Evaluation")
    print("=" * 76)
    print(f"  Positives     : {args.positives}")
    print(f"  Negatives     : {args.negatives}")
    print(f"  Threshold     : {args.threshold}")
    print(f"  Sliding window: {args.sliding_window} frames")
    print(f"  Cooldown      : {args.ignore_after_accept} frames")
    if args.max_neg:
        print(f"  Max negatives : {args.max_neg}")
    print()

    for model_path in model_paths:
        model_name = Path(model_path).name
        print(f"\nLoading: {model_name}")
        t0 = time.time()

        try:
            # stride=1 gives streaming (frame-by-frame) predictions — essential
            # for short (~1 s) clips so they generate enough probability outputs
            # to fill the sliding window.  stride=None would use stride=49 and
            # yield only 1 prediction per clip, causing every clip to be skipped.
            model = Model(model_path, stride=1)
        except Exception as exc:
            print(f"  [ERROR] Failed to load {model_name}: {exc}")
            continue

        print(
            f"  Loaded in {time.time()-t0:.2f}s  |  "
            f"Quantised={model.is_quantized_model}  |  "
            f"Input slices={model.input_feature_slices}  |  stride=1 (streaming)"
        )

        # Positives
        print(f"\n  [1/2] Evaluating positives …")
        t1 = time.time()
        pos_metrics, cond_metrics = evaluate_positives(
            model,
            args.positives,
            threshold=args.threshold,
            sliding_window=args.sliding_window,
            ignore_after_accept=args.ignore_after_accept,
        )
        print(f"  Done in {time.time()-t1:.1f}s  ({pos_metrics.get('total',0)} clips)")

        # Negatives
        neg_metrics: dict = {}
        if not args.skip_negatives:
            print(f"\n  [2/2] Evaluating negatives (FAPH) …")
            t2 = time.time()
            neg_metrics = evaluate_negatives(
                model,
                args.negatives,
                sliding_window=args.sliding_window,
                ignore_after_accept=args.ignore_after_accept,
                max_files=args.max_neg,
            )
            print(f"  Done in {time.time()-t2:.1f}s")

        # Report
        print()
        if pos_metrics:
            print_positive_report(model_name, pos_metrics, cond_metrics)
        if neg_metrics:
            print_negative_report(model_name, neg_metrics)
        if pos_metrics and neg_metrics:
            print_roc_summary(pos_metrics, neg_metrics)

        if args.output and (pos_metrics or neg_metrics):
            save_report(args.output, model_name, pos_metrics, cond_metrics, neg_metrics)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
