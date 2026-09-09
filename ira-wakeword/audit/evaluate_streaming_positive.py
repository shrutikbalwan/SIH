# -*- coding: utf-8 -*-
"""
evaluate_streaming_positive.py
==============================
Streaming Positive Evaluation for IRA wake-word model.

Synthesizes continuous audio streams by overlaying held-out positive
test clips onto background noise and LibriSpeech streams. 
Applies the exact same streaming inference pipeline and trigger logic
used in evaluate_streaming.py.

Usage:
  python evaluate_streaming_positive.py
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).parent
MODEL_PATH     = REPO_ROOT / "cnn" / "models" / "ira_cnn_int8.tflite"
LIBRI_BASE     = REPO_ROOT / "dataset" / "negative" / "extracted" / "LibriSpeech" / "train-clean-5"
BG_DIR         = REPO_ROOT / "dataset" / "negative" / "background"
MANIFEST       = REPO_ROOT / "dataset" / "split_manifest.csv"

OUT_DIR        = REPO_ROOT / "streaming_false_negatives"
GT_CSV         = REPO_ROOT / "streaming_positive_ground_truth.csv"
SCORES_CSV     = REPO_ROOT / "positive_event_scores.csv"
FN_CSV         = REPO_ROOT / "streaming_false_negatives.csv"
SWEEP_RES_CSV  = REPO_ROOT / "streaming_sweep_results.csv"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SAMPLE_RATE        = 16000
WINDOW_SAMPLES     = 16000          # 1.0 s
STFT_FRAME_LENGTH  = 480            # 30 ms
STFT_FRAME_STEP    = 320            # 20 ms
STFT_FFT_LENGTH    = 512
NUM_FREQ_BINS      = 40

TEST_SPEAKERS = ["1088", "1737", "5789", "6848"]

# ===========================================================================
# Preprocessing & Inference (Verbatim from evaluate_streaming.py)
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
# Stream Synthesis
# ===========================================================================

def load_audio(path: str) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a

def get_test_positives():
    pos_files = []
    with open(MANIFEST, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] == 'test':
                if 'positive' in r['path'] or 'real' in r['group'] or 'piper' in r['group']:
                    path = REPO_ROOT / r['path']
                    if path.exists():
                        pos_files.append(str(path))
    return pos_files

def get_test_negatives():
    bg_files = sorted(str(p) for p in BG_DIR.glob("*.wav")) if BG_DIR.exists() else []
    lib_files = []
    for sp in TEST_SPEAKERS:
        sp_dir = LIBRI_BASE / sp
        if sp_dir.exists():
            lib_files.extend(str(p) for p in sp_dir.rglob("*.flac"))
    return bg_files, lib_files

def synthesize_stream(
    pos_files: list,
    neg_files: list,
    condition: str,
    spacing_sec: float = 4.5
):
    """
    Creates a continuous float32 array mixing negative audio and inserting 
    a positive clip every `spacing_sec` seconds.
    Returns: (audio_array, list_of_ground_truth_events)
    """
    total_pos = len(pos_files)
    est_len = int((total_pos + 1) * spacing_sec * SAMPLE_RATE)
    
    # Pre-allocate large array
    stream = np.zeros(est_len, dtype=np.float32)
    
    # 1. Fill with continuous background
    pos_idx = 0
    neg_idx = 0
    while pos_idx < est_len and len(neg_files) > 0:
        n_file = neg_files[neg_idx % len(neg_files)]
        bg = load_audio(n_file)
        # Add to stream
        end_idx = min(pos_idx + len(bg), est_len)
        chunk_len = end_idx - pos_idx
        stream[pos_idx:end_idx] += bg[:chunk_len]
        pos_idx = end_idx
        neg_idx += 1
    
    if condition == "clean":
        stream *= 0.05  # lower background significantly
    
    # 2. Insert positives
    gt_events = []
    insert_idx = int(spacing_sec * SAMPLE_RATE)
    
    for p_file in pos_files:
        if insert_idx >= est_len: break
        
        pos_audio = load_audio(p_file)
        
        # Ensure it fits
        if insert_idx + len(pos_audio) > est_len: break
        
        # Overlay logic: we just add them. This preserves the background underneath.
        # Positive clips are mostly padded 1-second segments. If they have silence padded, 
        # addition keeps the background.
        stream[insert_idx : insert_idx + len(pos_audio)] += pos_audio
        
        # Normalization clipping precaution
        # We don't hard clip here to avoid distortion, let's assume it stays roughly in [-1, 1].
        
        gt_events.append({
            "source_positive_file": p_file,
            "insertion_timestamp_sec": insert_idx / SAMPLE_RATE,
            "condition": condition
        })
        
        insert_idx += int(spacing_sec * SAMPLE_RATE)
        
    return stream[:insert_idx], gt_events

# ===========================================================================
# Inference Pass
# ===========================================================================

def score_stream(audio: np.ndarray, detector: IraDetector, stride_ms: int):
    """Returns list of (timestamp_sec, score)"""
    n = len(audio)
    stride_samp = int(SAMPLE_RATE * stride_ms / 1000)
    pos = 0
    results = []
    while pos + WINDOW_SAMPLES <= n:
        window = audio[pos : pos + WINDOW_SAMPLES].copy()
        ts = pos / SAMPLE_RATE
        score = detector.predict(window)
        results.append((ts, score))
        pos += stride_samp
    return results

def get_triggers(scored_windows: list, threshold: float, consecutive: int, cooldown_ms: int, stride_ms: int):
    """Applies debounce logic and returns a list of trigger timestamps."""
    triggers = []
    stride_samp = int(SAMPLE_RATE * stride_ms / 1000)
    cooldown_samp = int(SAMPLE_RATE * cooldown_ms / 1000)
    
    cooldown_remaining = 0
    consecutive_run = 0
    
    for ts, score in scored_windows:
        if cooldown_remaining > 0:
            cooldown_remaining = max(0, cooldown_remaining - stride_samp)
            consecutive_run = 0
        else:
            if score >= threshold:
                consecutive_run += 1
                if consecutive_run >= consecutive:
                    triggers.append(ts)
                    cooldown_remaining = cooldown_samp
                    consecutive_run = 0
            else:
                consecutive_run = 0
    return triggers


# ===========================================================================
# Alignment and Metrics
# ===========================================================================

def align_detections(gt_events: list, triggers: list, match_start: float = -0.2, match_end: float = 1.2):
    """
    Matches triggers to ground truth events.
    1-to-1 matching.
    """
    matched_gt = set()
    matched_tr = set()
    
    tp_pairs = [] # (gt, tr_ts)
    
    for i, gt in enumerate(gt_events):
        t_g = gt["insertion_timestamp_sec"]
        
        # Find first unused trigger within window
        for j, t_d in enumerate(triggers):
            if j in matched_tr: continue
            if t_g + match_start <= t_d <= t_g + match_end:
                tp_pairs.append((gt, t_d))
                matched_gt.add(i)
                matched_tr.add(j)
                break
                
    fn_list = [gt for i, gt in enumerate(gt_events) if i not in matched_gt]
    fp_list = [t for j, t in enumerate(triggers) if j not in matched_tr]
    
    return tp_pairs, fn_list, fp_list

def calculate_latencies(tp_pairs: list):
    lats = [(tr_ts - gt["insertion_timestamp_sec"])*1000.0 for gt, tr_ts in tp_pairs]
    if not lats: return {}
    return {
        "mean": np.mean(lats),
        "median": np.median(lats),
        "min": np.min(lats),
        "max": np.max(lats),
        "p90": np.percentile(lats, 90),
        "p95": np.percentile(lats, 95)
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride-ms", type=int, default=100)
    parser.add_argument("--consecutive", type=int, default=1)
    parser.add_argument("--cooldown-ms", type=int, default=1500)
    parser.add_argument("--spacing-sec", type=float, default=4.5)
    args = parser.parse_args()

    print("\nLoading model...")
    detector = IraDetector()
    
    pos_files = get_test_positives()
    bg_files, lib_files = get_test_negatives()
    
    random.seed(42)
    random.shuffle(pos_files)
    
    # We will split positive files into 3 groups for conditions
    n = len(pos_files)
    p_clean = pos_files[:n//3]
    p_ambient = pos_files[n//3 : 2*n//3]
    p_speech = pos_files[2*n//3:]
    
    print(f"Synthesizing test streams... (total positives: {n})")
    
    streams = [] # (condition, audio_array, gt_events)
    
    print(f"  -> Clean condition ({len(p_clean)} events)")
    a1, g1 = synthesize_stream(p_clean, bg_files, "clean", args.spacing_sec)
    streams.append(("clean", a1, g1))
    
    print(f"  -> Ambient condition ({len(p_ambient)} events)")
    a2, g2 = synthesize_stream(p_ambient, bg_files, "ambient", args.spacing_sec)
    streams.append(("ambient", a2, g2))
    
    print(f"  -> Speech condition ({len(p_speech)} events)")
    # Repeat libri files to ensure enough length
    lib_files_ext = lib_files * 5 if lib_files else []
    a3, g3 = synthesize_stream(p_speech, lib_files_ext, "speech", args.spacing_sec)
    streams.append(("speech", a3, g3))
    
    # Save GT
    all_gt = []
    for c, a, g in streams: all_gt.extend(g)
    
    with open(GT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream_id", "source_positive_file", "insertion_timestamp_sec", "condition"])
        w.writeheader()
        for i, gt in enumerate(all_gt):
            w.writerow({
                "stream_id": gt["condition"],
                "source_positive_file": gt["source_positive_file"],
                "insertion_timestamp_sec": f"{gt['insertion_timestamp_sec']:.3f}",
                "condition": gt["condition"]
            })
            
    # Run Inference
    print("\nRunning inference...")
    stream_scores = {}
    for c, a, g in streams:
        t0 = time.time()
        res = score_stream(a, detector, args.stride_ms)
        stream_scores[c] = {"audio": a, "gt": g, "scores": res}
        print(f"  Scored {c} stream: {len(res)} windows, {time.time()-t0:.1f}s")
        
    # Analyze Scores (Max score near event)
    print("\nAnalyzing scores...")
    event_scores = []
    for c, data in stream_scores.items():
        for gt in data["gt"]:
            t_g = gt["insertion_timestamp_sec"]
            # look between t_g - 0.2 and t_g + 1.2
            nearby = [s for ts, s in data["scores"] if t_g - 0.2 <= ts <= t_g + 1.2]
            max_s = max(nearby) if nearby else 0.0
            event_scores.append({
                "event_id": len(event_scores)+1,
                "source_file": gt["source_positive_file"],
                "condition": gt["condition"],
                "max_score": max_s,
                "detection_timestamp": t_g,
                "latency_ms": 0 # n/a here
            })
            
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["event_id", "source_file", "condition", "max_score", "detection_timestamp", "latency_ms"])
        w.writeheader()
        for e in event_scores: w.writerow(e)
        
    # Threshold Sweep
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print("\nThreshold Sweep Analysis:")
    
    results = []
    
    for thr in thresholds:
        tp_total, fn_total, fp_total = 0, 0, 0
        lats_all = []
        cond_tpr = {}
        
        all_fn = [] # for saving
        
        for c, data in stream_scores.items():
            trigs = get_triggers(data["scores"], thr, args.consecutive, args.cooldown_ms, args.stride_ms)
            tp_p, fn_l, fp_l = align_detections(data["gt"], trigs)
            
            tp = len(tp_p)
            fn = len(fn_l)
            fp = len(fp_l)
            
            tp_total += tp
            fn_total += fn
            fp_total += fp
            
            cond_tpr[c] = tp / (tp + fn) if (tp+fn)>0 else 0.0
            
            for gt, t_d in tp_p: lats_all.append((t_d - gt["insertion_timestamp_sec"])*1000)
            
            for gt in fn_l:
                all_fn.append((gt, c, data["audio"]))
                
        tpr = tp_total / (tp_total + fn_total) if (tp_total+fn_total)>0 else 0
        fnr = fn_total / (tp_total + fn_total) if (tp_total+fn_total)>0 else 0
        med_lat = np.median(lats_all) if lats_all else float("nan")
        
        results.append({
            "threshold": thr,
            "tp": tp_total,
            "fn": fn_total,
            "fp": fp_total, # Unmatched false triggers in POSITIVE streams
            "tpr": tpr,
            "fnr": fnr,
            "med_lat": med_lat,
            "cond_tpr": cond_tpr,
            "all_fn": all_fn,
            "lats_all": lats_all
        })
        
        print(f"  Thr {thr:.2f} | TPR: {tpr*100:6.2f}% | FNR: {fnr*100:6.2f}% | Latency: {med_lat:6.1f}ms")
        
    # Save False Negatives for Thr 0.50
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thr50_res = next(r for r in results if abs(r["threshold"]-0.50)<1e-6)
    
    fn_rows = []
    for idx, (gt, c, audio) in enumerate(thr50_res["all_fn"]):
        t_g = gt["insertion_timestamp_sec"]
        # Save 1s before to 2s after
        s_idx = max(0, int((t_g - 1.0) * SAMPLE_RATE))
        e_idx = min(len(audio), int((t_g + 2.0) * SAMPLE_RATE))
        clip = audio[s_idx:e_idx]
        
        clip_name = f"fn_{idx:03d}_{c}.wav"
        clip_path = OUT_DIR / clip_name
        sf.write(str(clip_path), clip, SAMPLE_RATE, subtype="PCM_16")
        
        # get max score
        es = next(e for e in event_scores if e["detection_timestamp"] == t_g)
        
        fn_rows.append({
            "event_id": idx,
            "source_positive": gt["source_positive_file"],
            "condition": c,
            "expected_timestamp": f"{t_g:.3f}",
            "maximum_score_near_event": f"{es['max_score']:.4f}",
            "threshold": 0.50,
            "audio_path": str(clip_path)
        })
        
    with open(FN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["event_id", "source_positive", "condition", "expected_timestamp", "maximum_score_near_event", "threshold", "audio_path"])
        w.writeheader()
        for r in fn_rows: w.writerow(r)
        
    print(f"\nSaved {len(fn_rows)} False Negative clips for threshold 0.50 to {OUT_DIR}")
    
    # Save summary for report generation
    import json
    with open(REPO_ROOT / "streaming_positive_results.json", "w") as f:
        json.dump(results, f, default=lambda x: str(x) if isinstance(x, np.ndarray) else x)

if __name__ == "__main__":
    main()
