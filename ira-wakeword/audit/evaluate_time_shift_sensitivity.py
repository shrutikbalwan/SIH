# -*- coding: utf-8 -*-
"""
evaluate_time_shift_sensitivity.py
==================================
Isolates temporal alignment sensitivity.
Removes original padding via VAD, then manually pads at precise offsets 
[0, 100, 200, 300, 400, 500, 600] ms inside a 1-second clean window.
"""

import os
import csv
import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import librosa
from scipy.signal import resample_poly

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# Paths & Settings
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).parent
MODEL_PATH  = REPO_ROOT / "cnn" / "models" / "ira_cnn_int8.tflite"
MANIFEST    = REPO_ROOT / "dataset" / "split_manifest.csv"

SCORES_CSV  = REPO_ROOT / "time_shift_scores.csv"
MATRIX_CSV  = REPO_ROOT / "time_shift_threshold_matrix.csv"
JSON_OUT    = REPO_ROOT / "time_shift_summary.json"

SAMPLE_RATE        = 16000
WINDOW_SAMPLES     = 16000
STFT_FRAME_LENGTH  = 480
STFT_FRAME_STEP    = 320
STFT_FFT_LENGTH    = 512
NUM_FREQ_BINS      = 40

OFFSETS_MS = [0, 100, 200, 300, 400, 500, 600]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# ===========================================================================
# Preprocessing & Model (Exact Match)
# ===========================================================================

def make_spectrogram(window: np.ndarray) -> np.ndarray:
    audio_t = tf.convert_to_tensor(window, dtype=tf.float32)
    spec = tf.signal.stft(
        audio_t,
        frame_length=STFT_FRAME_LENGTH,
        frame_step=STFT_FRAME_STEP,
        fft_length=STFT_FFT_LENGTH,
    )
    spec = tf.abs(spec)
    spec = tf.math.log(spec + 1e-6)
    spec = spec[:, :NUM_FREQ_BINS]
    mean = tf.reduce_mean(spec)
    std  = tf.math.reduce_std(spec) + 1e-6
    spec = (spec - mean) / std
    return spec.numpy().astype(np.float32)

class IraDetector:
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

    def predict(self, window: np.ndarray) -> float:
        spec = make_spectrogram(window)
        x = spec[np.newaxis, ..., np.newaxis]
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

# ===========================================================================
# Helpers
# ===========================================================================

def load_audio(path: str) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a

def get_files_from_manifest():
    train_pos = []
    test_pos = []
    
    with open(MANIFEST, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            path_str = r['path'].lower()
            group = r.get('group', '').lower()
            is_pos = ('positive' in path_str) or ('real' in group) or ('piper' in group)
            if 'label' in r and r['label'] == '0': is_pos = False
            
            if is_pos:
                path = REPO_ROOT / r['path']
                if not path.exists(): continue
                
                is_real = ('real' in group) or ('real' in path_str)
                is_tts = ('piper' in group) or ('piper' in path_str)
                # Fallback heuristics
                if not is_real and not is_tts:
                    if 'tts' in path_str: is_tts = True
                    else: is_real = True
                
                item = {"path": str(path), "is_real": is_real}
                
                if r['split'] == 'train':
                    train_pos.append(item)
                elif r['split'] == 'test':
                    test_pos.append(item)
                    
    return train_pos, test_pos

def vad_trim(audio: np.ndarray, top_db=25):
    """Returns active speech array and its original onset in samples."""
    yt, index = librosa.effects.trim(audio, top_db=top_db)
    onset_samples = index[0]
    return yt, onset_samples

# ===========================================================================
# Main Routine
# ===========================================================================

def main():
    print("\nLoading manifest...")
    train_pos, test_pos = get_files_from_manifest()
    print(f"  Train positives: {len(train_pos)}")
    print(f"  Test positives : {len(test_pos)}")
    
    # 1. Training Distribution Baseline
    random.seed(42)
    sample_train = random.sample(train_pos, min(500, len(train_pos)))
    train_onsets = []
    train_centers = []
    
    print("\nMeasuring training temporal distribution...")
    for item in sample_train:
        a = load_audio(item["path"])
        _, onset = vad_trim(a)
        offset = onset + len(_)
        train_onsets.append(onset / SAMPLE_RATE * 1000)
        train_centers.append(((onset + offset) / 2) / SAMPLE_RATE * 1000)
        
    train_stats = {
        "mean_onset_ms": np.mean(train_onsets),
        "median_onset_ms": np.median(train_onsets),
        "mean_center_ms": np.mean(train_centers),
        "median_center_ms": np.median(train_centers)
    }
    
    print(f"  Mean onset: {train_stats['mean_onset_ms']:.1f} ms (Median: {train_stats['median_onset_ms']:.1f} ms)")
    print(f"  Mean center: {train_stats['mean_center_ms']:.1f} ms (Median: {train_stats['median_center_ms']:.1f} ms)")
    
    # 2. Extract Test Speech & Generate Shifts
    print("\nLoading model and generating shifts...")
    detector = IraDetector()
    
    scores = [] # list of dicts
    
    for i, item in enumerate(test_pos):
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_pos)}")
            
        a = load_audio(item["path"])
        active_speech, _ = vad_trim(a)
        l = len(active_speech)
        
        if l > WINDOW_SAMPLES:
            continue # Too long to shift
            
        src_type = "REAL" if item["is_real"] else "TTS"
        
        for off_ms in OFFSETS_MS:
            off_samp = int(SAMPLE_RATE * off_ms / 1000)
            
            if off_samp + l > WINDOW_SAMPLES:
                continue # would truncate word
                
            # Create fresh padded window
            window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            window[off_samp : off_samp + l] = active_speech
            
            s = detector.predict(window)
            
            scores.append({
                "source_file": item["path"],
                "source_type": src_type,
                "offset_ms": off_ms,
                "score": s,
                "pred_0_50": 1 if s >= 0.50 else 0
            })
            
    # Save raw scores
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_file", "source_type", "offset_ms", "score", "pred_0_50"])
        w.writeheader()
        w.writerows(scores)
        
    print(f"\nSaved raw scores to {SCORES_CSV}")
    
    # 3. Analyze TPR by offset
    print("\nAnalyzing TPR by offset at threshold 0.50...")
    
    results = {}
    for r in scores:
        off = r["offset_ms"]
        if off not in results:
            results[off] = {"ALL": [], "REAL": [], "TTS": []}
            
        results[off]["ALL"].append(r["score"])
        results[off][r["source_type"]].append(r["score"])
        
    # Generate TPR vs Offset table
    tpr_offset_0_50 = {}
    
    print("\nOffset | N(All) | TPR(All) | Mean Score | Median Score | P10 Score")
    print("-" * 65)
    for off in OFFSETS_MS:
        if off not in results or not results[off]["ALL"]: continue
        
        all_s = results[off]["ALL"]
        tpr = sum(1 for s in all_s if s >= 0.50) / len(all_s)
        mean_s = np.mean(all_s)
        med_s = np.median(all_s)
        p10_s = np.percentile(all_s, 10)
        
        tpr_offset_0_50[off] = {
            "n": len(all_s),
            "tpr": tpr,
            "mean": mean_s,
            "median": med_s,
            "p10": p10_s,
            "tpr_real": sum(1 for s in results[off]["REAL"] if s >= 0.50) / len(results[off]["REAL"]) if results[off]["REAL"] else 0,
            "tpr_tts": sum(1 for s in results[off]["TTS"] if s >= 0.50) / len(results[off]["TTS"]) if results[off]["TTS"] else 0,
            "n_real": len(results[off]["REAL"]),
            "n_tts": len(results[off]["TTS"])
        }
        
        print(f"{off:>4} ms | {len(all_s):>6} | {tpr*100:>7.2f}% | {mean_s:>10.4f} | {med_s:>12.4f} | {p10_s:>9.4f}")
        
    # 4. Threshold Sweep Matrix
    matrix_rows = []
    for off in OFFSETS_MS:
        if off not in results or not results[off]["ALL"]: continue
        all_s = results[off]["ALL"]
        
        row = {"offset_ms": off}
        for thr in THRESHOLDS:
            t = sum(1 for s in all_s if s >= thr) / len(all_s)
            row[f"tpr_{thr:.2f}"] = t
        matrix_rows.append(row)
        
    with open(MATRIX_CSV, "w", newline="", encoding="utf-8") as f:
        fields = ["offset_ms"] + [f"tpr_{t:.2f}" for t in THRESHOLDS]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(matrix_rows)
        
    print(f"\nSaved threshold sweep matrix to {MATRIX_CSV}")
    
    # Save JSON summary for report generator
    import json
    summary = {
        "train_stats": train_stats,
        "test_counts": {
            "real": len([t for t in test_pos if t["is_real"]]),
            "tts": len([t for t in test_pos if not t["is_real"]]),
            "total": len(test_pos)
        },
        "offset_tpr_0_50": tpr_offset_0_50,
        "matrix": matrix_rows
    }
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f)

if __name__ == "__main__":
    main()
