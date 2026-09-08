# -*- coding: utf-8 -*-
"""
evaluate_correct_pipeline.py
============================
Evaluates ira_cnn_int8.tflite using the EXACT same preprocessing pipeline
that was used during training (cnn/train_cnn.py).

Training pipeline (reproduced here):
  soundfile.read(dtype="float32")
  -> stereo-to-mono (mean)
  -> resample to 16 kHz if needed
  -> pad/truncate to exactly 16,000 samples
  -> tf.signal.stft(frame_length=480, frame_step=320, fft_length=512)
  -> tf.abs()
  -> tf.math.log(x + 1e-6)      <-- log with epsilon, NOT log1p
  -> keep first 40 frequency bins
  -> per-clip z-score: (x - mean) / (std + 1e-6)
  -> add channel dim  -> shape (49, 40, 1)

Test set used:
  dataset/splits/test/positive/   (1,003 clips)
  dataset/splits/test/negative/   (  772 clips)

These are the HELD-OUT clips that were never seen during training.

Usage:
  python evaluate_correct_pipeline.py [--verify-only] [--model PATH]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly

# Suppress oneDNN / TF info spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).parent
MODEL_INT8   = REPO_ROOT / "cnn" / "models" / "ira_cnn_int8.tflite"
TEST_POS_DIR = REPO_ROOT / "dataset" / "splits" / "test" / "positive"
TEST_NEG_DIR = REPO_ROOT / "dataset" / "splits" / "test" / "negative"

# ---------------------------------------------------------------------------
# Hyperparameters — copied verbatim from cnn/train_cnn.py
# ---------------------------------------------------------------------------
SAMPLE_RATE  = 16000          # Hz
CLIP_SECONDS = 1.0            # seconds
NUM_SAMPLES  = int(SAMPLE_RATE * CLIP_SECONDS)  # 16000

STFT_FRAME_LENGTH = 480       # 30 ms
STFT_FRAME_STEP   = 320       # 20 ms
STFT_FFT_LENGTH   = 512       # -> 257 raw bins
NUM_FREQ_BINS     = 40        # bins kept after log

# Expected spectrogram time frames:
#   floor((16000 - 480) / 320) + 1 = 49
EXPECTED_TIME_FRAMES = 49
EXPECTED_FEATURE_SHAPE = (EXPECTED_TIME_FRAMES, NUM_FREQ_BINS, 1)  # (49,40,1)


# ===========================================================================
# Step 1 — Audio loading  (identical to train_cnn.py::load_audio)
# ===========================================================================

def load_audio(path: str) -> np.ndarray:
    """Load a WAV file as float32 mono at 16 kHz, length exactly 16,000 samples."""
    audio, sr = sf.read(path, dtype="float32")

    # Stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)

    # Pad (trailing zeros) or truncate — EXACTLY as in training
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    else:
        audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


# ===========================================================================
# Step 2 — Feature extraction  (identical to train_cnn.py::make_spectrogram)
# ===========================================================================

def make_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio array to a (49, 40, 1) float32 spectrogram.

    Pipeline — copied verbatim from cnn/train_cnn.py::make_spectrogram():
      tf.signal.stft(frame_length=480, frame_step=320, fft_length=512)
      tf.abs()
      tf.math.log(x + 1e-6)    <-- epsilon = 1e-6 exactly
      [:, :40]                  <-- first 40 bins AFTER log
      (x - mean) / (std + 1e-6) per-clip z-score
    """
    audio_t = tf.convert_to_tensor(audio, dtype=tf.float32)

    spec = tf.signal.stft(
        audio_t,
        frame_length=STFT_FRAME_LENGTH,
        frame_step=STFT_FRAME_STEP,
        fft_length=STFT_FFT_LENGTH,
    )                                          # complex, shape (49, 257)

    spec = tf.abs(spec)                        # magnitude, shape (49, 257)
    spec = tf.math.log(spec + 1e-6)           # log(|STFT| + 1e-6)
    spec = spec[:, :NUM_FREQ_BINS]            # shape (49, 40)

    mean = tf.reduce_mean(spec)
    std  = tf.math.reduce_std(spec) + 1e-6
    spec = (spec - mean) / std                # per-clip z-score

    return spec.numpy().astype(np.float32)    # shape (49, 40)


def audio_to_input(audio: np.ndarray) -> np.ndarray:
    """Return spectrogram with channel dim: shape (1, 49, 40, 1) float32."""
    spec = make_spectrogram(audio)             # (49, 40)
    return spec[np.newaxis, ..., np.newaxis]  # (1, 49, 40, 1)


# ===========================================================================
# Step 3 — TFLite inference with correct INT8 handling
# ===========================================================================

class Int8Evaluator:
    """Wraps ira_cnn_int8.tflite with correct quantize/dequantize logic."""

    def __init__(self, model_path: str):
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()

        self._interp       = interp
        self._inp_details  = interp.get_input_details()[0]
        self._out_details  = interp.get_output_details()[0]

        # Quantization parameters (measured in MODEL_AUDIT.md)
        qp_in  = self._inp_details["quantization_parameters"]
        qp_out = self._out_details["quantization_parameters"]
        self.inp_scale  = float(qp_in["scales"][0])
        self.inp_zp     = int(qp_in["zero_points"][0])
        self.out_scale  = float(qp_out["scales"][0])
        self.out_zp     = int(qp_out["zero_points"][0])

        self.input_dtype  = self._inp_details["dtype"]
        self.output_dtype = self._out_details["dtype"]
        self.input_shape  = tuple(self._inp_details["shape"].tolist())

    def predict(self, x_float: np.ndarray) -> float:
        """
        x_float: numpy array of shape (1, 49, 40, 1), dtype float32.
        Returns: float probability in [0, 1].
        """
        if self.input_dtype == np.int8:
            # Quantize: float32 -> int8
            x_q = np.round(x_float / self.inp_scale + self.inp_zp)
            x_q = np.clip(x_q, -128, 127).astype(np.int8)
        else:
            # v3 model uses float32 input directly
            x_q = x_float.astype(np.float32)

        self._interp.set_tensor(self._inp_details["index"], x_q)
        self._interp.invoke()
        output = self._interp.get_tensor(self._out_details["index"])

        if self.output_dtype == np.int8:
            # Dequantize: int8 -> float32
            prob = (float(output[0, 0]) - self.out_zp) * self.out_scale
        else:
            prob = float(output[0, 0])

        return prob


# ===========================================================================
# Verification helper
# ===========================================================================

def verify_pipeline(model: Int8Evaluator, n: int = 5):
    """Run n positive and n negative clips through the pipeline and print diagnostics."""
    pos_files = sorted(TEST_POS_DIR.glob("*.wav"))[:n]
    neg_files = sorted(TEST_NEG_DIR.glob("*.wav"))[:n]

    print("=" * 65)
    print("PIPELINE VERIFICATION")
    print("=" * 65)
    print(f"  SAMPLE_RATE     = {SAMPLE_RATE}")
    print(f"  CLIP_SECONDS    = {CLIP_SECONDS}")
    print(f"  NUM_SAMPLES     = {NUM_SAMPLES}")
    print(f"  frame_length    = {STFT_FRAME_LENGTH}  (30 ms)")
    print(f"  frame_step      = {STFT_FRAME_STEP}    (20 ms)")
    print(f"  fft_length      = {STFT_FFT_LENGTH}")
    print(f"  freq_bins_kept  = {NUM_FREQ_BINS}")
    print(f"  log_epsilon     = 1e-6")
    print(f"  normalization   = per-clip z-score")
    print()
    print(f"  Model input  dtype  = {model.input_dtype}")
    print(f"  Model input  shape  = {model.input_shape}")
    print(f"  Model output dtype  = {model.output_dtype}")
    print(f"  Input  quant: scale={model.inp_scale:.8f}  zero_point={model.inp_zp}")
    print(f"  Output quant: scale={model.out_scale:.8f}  zero_point={model.out_zp}")
    print()

    for label_name, files, expected in [("POSITIVE", pos_files, 1), ("NEGATIVE", neg_files, 0)]:
        print(f"--- Sample {label_name} clips ---")
        for f in files:
            audio  = load_audio(str(f))
            spec   = make_spectrogram(audio)
            x_in   = audio_to_input(audio)
            prob   = model.predict(x_in)
            pred   = 1 if prob >= 0.5 else 0
            ok     = "OK" if pred == expected else "MISS"

            print(f"  {f.name:<40s}")
            print(f"    audio shape : {audio.shape}  dtype={audio.dtype}")
            print(f"    feature shp : {spec.shape}   (expected {(EXPECTED_TIME_FRAMES, NUM_FREQ_BINS)})")
            print(f"    feat min    : {spec.min():.4f}")
            print(f"    feat max    : {spec.max():.4f}")
            print(f"    feat mean   : {spec.mean():.6f}  (should be ~0 after z-score)")
            print(f"    feat std    : {spec.std():.6f}   (should be ~1 after z-score)")
            print(f"    probability : {prob:.4f}   pred={pred}  [{ok}]")
        print()


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate(model: Int8Evaluator,
             thresholds: list = None) -> dict:
    """
    Run inference on the full test split.

    Returns
    -------
    dict with keys: probabilities, labels, filenames
    """
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    pos_files = sorted(TEST_POS_DIR.glob("*.wav"))
    neg_files = sorted(TEST_NEG_DIR.glob("*.wav"))

    all_files  = [(f, 1) for f in pos_files] + [(f, 0) for f in neg_files]
    total      = len(all_files)

    print(f"  Positive clips : {len(pos_files)}")
    print(f"  Negative clips : {len(neg_files)}")
    print(f"  Total          : {total}")
    print()

    probs  = []
    labels = []
    names  = []

    t0 = time.time()
    for i, (fpath, label) in enumerate(all_files):
        audio = load_audio(str(fpath))
        x_in  = audio_to_input(audio)
        prob  = model.predict(x_in)

        probs.append(prob)
        labels.append(label)
        names.append(fpath.name)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    processed {i+1}/{total}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {total} clips in {elapsed:.1f}s")

    return {
        "probabilities": np.array(probs,  dtype=np.float32),
        "labels":        np.array(labels, dtype=np.int32),
        "filenames":     names,
    }


def compute_metrics(probs: np.ndarray,
                    labels: np.ndarray,
                    threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int32)
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    tp = int(np.sum(preds[pos_mask] == 1))
    fn = int(np.sum(preds[pos_mask] == 0))
    fp = int(np.sum(preds[neg_mask] == 1))
    tn = int(np.sum(preds[neg_mask] == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr = tn / (fp + tn) if (fp + tn) > 0 else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                tpr=tpr, fpr=fpr, fnr=fnr, tnr=tnr, acc=acc, precision=prec)


def print_results(results: dict):
    probs  = results["probabilities"]
    labels = results["labels"]

    print()
    print("=" * 65)
    print("THRESHOLD ANALYSIS")
    print("=" * 65)
    print(f"  {'Threshold':>10}  {'Accuracy':>9}  {'TPR/Recall':>11}  "
          f"{'FPR':>7}  {'FNR':>7}  {'Precision':>10}  "
          f"{'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")
    print("  " + "-" * 85)

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    all_metrics = []
    for thr in thresholds:
        m = compute_metrics(probs, labels, thr)
        all_metrics.append(m)
        print(f"  {m['threshold']:>10.2f}  {m['acc']:>9.4f}  {m['tpr']:>11.4f}  "
              f"{m['fpr']:>7.4f}  {m['fnr']:>7.4f}  {m['precision']:>10.4f}  "
              f"{m['tp']:>5d}  {m['fp']:>5d}  {m['fn']:>5d}  {m['tn']:>5d}")

    # Highlight threshold=0.50
    m50 = compute_metrics(probs, labels, 0.50)
    print()
    print("=" * 65)
    print("CONFUSION MATRIX  (threshold = 0.50)")
    print("=" * 65)
    print(f"                Predicted 0   Predicted 1")
    print(f"  Actual  1         {m50['fn']:5d}        {m50['tp']:5d}   (total pos = {m50['fn']+m50['tp']})")
    print(f"  Actual  0         {m50['tn']:5d}        {m50['fp']:5d}   (total neg = {m50['tn']+m50['fp']})")
    print()
    print(f"  True Positive Rate  (TPR/Recall) : {m50['tpr']*100:7.3f}%")
    print(f"  False Positive Rate (FPR)        : {m50['fpr']*100:7.3f}%")
    print(f"  False Negative Rate (FNR)        : {m50['fnr']*100:7.3f}%")
    print(f"  True Negative Rate  (TNR/Spec.)  : {m50['tnr']*100:7.3f}%")
    print(f"  Accuracy                         : {m50['acc']*100:7.3f}%")
    print(f"  Precision                        : {m50['precision']*100:7.3f}%")

    return all_metrics


# ===========================================================================
# CLI
# ===========================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate IRA wake-word model with correct training pipeline."
    )
    p.add_argument("--model", default=str(MODEL_INT8),
                   help=f"Path to .tflite model. Default: {MODEL_INT8.name}")
    p.add_argument("--verify-only", action="store_true",
                   help="Run pipeline verification only (5 clips each class), then exit.")
    p.add_argument("--test-pos", default=str(TEST_POS_DIR),
                   help="Directory of positive test WAV files.")
    p.add_argument("--test-neg", default=str(TEST_NEG_DIR),
                   help="Directory of negative test WAV files.")
    return p


def main():
    args = build_parser().parse_args()

    # Override paths if given
    global TEST_POS_DIR, TEST_NEG_DIR
    TEST_POS_DIR = Path(args.test_pos)
    TEST_NEG_DIR = Path(args.test_neg)

    print()
    print("=" * 65)
    print("  IRA Wake-Word Evaluation — CORRECT TRAINING PIPELINE")
    print("=" * 65)
    print(f"  Model      : {args.model}")
    print(f"  Test pos   : {TEST_POS_DIR}")
    print(f"  Test neg   : {TEST_NEG_DIR}")
    print()

    # Load model
    model = Int8Evaluator(args.model)

    # Sanity-check expected shape
    assert model.input_shape == (1, 49, 40, 1), (
        f"Unexpected model input shape: {model.input_shape}. Expected (1,49,40,1)."
    )

    # Verify pipeline on a few samples
    verify_pipeline(model, n=3)

    if args.verify_only:
        print("--verify-only set. Exiting.")
        return

    # Full evaluation
    print("=" * 65)
    print("FULL TEST-SET EVALUATION")
    print("=" * 65)
    results = evaluate(model)
    metrics = print_results(results)

    # Save raw probabilities for offline analysis
    out_npy = REPO_ROOT / "cnn" / "models" / "correct_pipeline_probabilities.npy"
    np.save(str(out_npy), results["probabilities"])
    print(f"\n  Raw probabilities saved: {out_npy}")

    print()
    print("=" * 65)
    print("  Evaluation complete.")
    print("=" * 65)

    return results, metrics


if __name__ == "__main__":
    main()
