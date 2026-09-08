# -*- coding: utf-8 -*-
"""
evaluate_snr_sensitivity.py
===========================
Evaluates IRA wake-word model robustness to background noise at controlled SNRs.
Also extracts feature-level diagnostics (raw spectrograms vs z-scored) to isolate
normalization failure modes.
"""

import os
import csv
import random
import time
from pathlib import Path
import json

import numpy as np
import soundfile as sf
import tensorflow as tf
import librosa
from scipy.signal import resample_poly

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).parent
MODEL_PATH  = REPO_ROOT / "cnn" / "models" / "ira_cnn_int8.tflite"
MANIFEST    = REPO_ROOT / "dataset" / "split_manifest.csv"
LIBRI_BASE  = REPO_ROOT / "dataset" / "negative" / "extracted" / "LibriSpeech" / "train-clean-5"
BG_DIR      = REPO_ROOT / "dataset" / "negative" / "background"

DIAG_CSV    = REPO_ROOT / "snr_feature_diagnostics.csv"
SCORES_CSV  = REPO_ROOT / "snr_scores.csv"
MATRIX_AMB  = REPO_ROOT / "snr_threshold_matrix_ambient.csv"
MATRIX_SP   = REPO_ROOT / "snr_threshold_matrix_speech.csv"
JSON_OUT    = REPO_ROOT / "snr_summary.json"

SAMPLE_RATE        = 16000
WINDOW_SAMPLES     = 16000
STFT_FRAME_LENGTH  = 480
STFT_FRAME_STEP    = 320
STFT_FFT_LENGTH    = 512
NUM_FREQ_BINS      = 40

SNRS_DB = [30, 20, 15, 10, 5, 0, -5]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

TEST_SPEAKERS = ["1088", "1737", "5789", "6848"]

# ===========================================================================
# Model & Preprocessing
# ===========================================================================

def make_features(window: np.ndarray):
    """Returns (raw_logspec, normalized_spec)"""
    audio_t = tf.convert_to_tensor(window, dtype=tf.float32)
    spec = tf.signal.stft(
        audio_t,
        frame_length=STFT_FRAME_LENGTH,
        frame_step=STFT_FRAME_STEP,
        fft_length=STFT_FFT_LENGTH,
    )
    spec = tf.abs(spec)
    raw_logspec = tf.math.log(spec + 1e-6)[:, :NUM_FREQ_BINS]
    
    mean = tf.reduce_mean(raw_logspec)
    std  = tf.math.reduce_std(raw_logspec) + 1e-6
    norm_spec = (raw_logspec - mean) / std
    
    return raw_logspec.numpy().astype(np.float32), norm_spec.numpy().astype(np.float32)

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

    def predict_features(self, norm_spec: np.ndarray) -> float:
        x = norm_spec[np.newaxis, ..., np.newaxis]
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

def get_test_positives():
    test_pos = []
    with open(MANIFEST, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] != 'test': continue
            path_str = r['path'].lower()
            group = r.get('group', '').lower()
            is_pos = ('positive' in path_str) or ('real' in group) or ('piper' in group)
            if 'label' in r and r['label'] == '0': is_pos = False
            
            if is_pos:
                path = REPO_ROOT / r['path']
                if not path.exists(): continue
                is_real = ('real' in group) or ('real' in path_str)
                is_tts = ('piper' in group) or ('piper' in path_str)
                if not is_real and not is_tts:
                    if 'tts' in path_str: is_tts = True
                    else: is_real = True
                test_pos.append({"path": str(path), "is_real": is_real})
    return test_pos

def get_test_negatives():
    bg_files = sorted(str(p) for p in BG_DIR.glob("*.wav")) if BG_DIR.exists() else []
    lib_files = []
    for sp in TEST_SPEAKERS:
        sp_dir = LIBRI_BASE / sp
        if sp_dir.exists():
            lib_files.extend(str(p) for p in sp_dir.rglob("*.flac"))
    return bg_files, lib_files

def vad_trim(audio: np.ndarray, top_db=25):
    yt, _ = librosa.effects.trim(audio, top_db=top_db)
    return yt

def calc_rms(audio: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(audio))) + 1e-9

def get_random_bg_segment(bg_files: list, target_len: int = WINDOW_SAMPLES) -> np.ndarray:
    path = random.choice(bg_files)
    a = load_audio(path)
    if len(a) < target_len:
        return np.pad(a, (0, target_len - len(a)))
    elif len(a) > target_len:
        start = random.randint(0, len(a) - target_len)
        return a[start:start+target_len]
    return a

# ===========================================================================
# Main Evaluation
# ===========================================================================

def main():
    random.seed(42)
    print("\nLoading files...")
    positives = get_test_positives()
    ambient_files, speech_files = get_test_negatives()
    
    print(f"  Positives: {len(positives)}")
    print(f"  Ambient backgrounds: {len(ambient_files)}")
    print(f"  Speech backgrounds: {len(speech_files)}")
    
    detector = IraDetector()
    
    diag_rows = []
    score_rows = []
    
    def record_run(event_id, path, src_type, bg_type, bg_file, snr_str, window, raw_s, norm_s, score, 
                   pos_rms, bg_rms_raw, bg_rms_scaled, mix_rms, peak, gain):
        diag_rows.append({
            "event_id": event_id,
            "source_positive": path,
            "source_type": src_type,
            "background_type": bg_type,
            "background_file": bg_file,
            "snr_db": snr_str,
            "score": score,
            "raw_logspec_mean": np.mean(raw_s),
            "raw_logspec_std": np.std(raw_s),
            "normalized_mean": np.mean(norm_s),
            "normalized_std": np.std(norm_s),
            "positive_rms": pos_rms,
            "background_rms": bg_rms_raw,
            "mixture_rms": mix_rms,
            "peak_amplitude": peak,
            "common_gain": gain
        })
        
        score_rows.append({
            "event_id": event_id,
            "source_positive": path,
            "source_type": src_type,
            "background_type": bg_type,
            "snr_db": snr_str,
            "score": score,
            "pred_0_30": 1 if score >= 0.30 else 0,
            "pred_0_50": 1 if score >= 0.50 else 0,
            "pred_0_70": 1 if score >= 0.70 else 0,
            "pred_0_90": 1 if score >= 0.90 else 0
        })

    print("\nStarting mixing and inference...")
    
    for i, item in enumerate(positives):
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(positives)}")
            
        event_id = i + 1
        path = item["path"]
        src_type = "REAL" if item["is_real"] else "TTS"
        
        orig_audio = load_audio(path)
        active_speech = vad_trim(orig_audio)
        
        if len(active_speech) > WINDOW_SAMPLES:
            active_speech = active_speech[:WINDOW_SAMPLES]
            
        pos_rms = calc_rms(active_speech)
        
        # 1. CLEAN CONTROL (Centered)
        clean_win = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        start_idx = (WINDOW_SAMPLES - len(active_speech)) // 2
        clean_win[start_idx : start_idx + len(active_speech)] = active_speech
        
        raw_c, norm_c = make_features(clean_win)
        score_c = detector.predict_features(norm_c)
        
        record_run(event_id, path, src_type, "none", "none", "clean", clean_win, raw_c, norm_c, score_c,
                   pos_rms, 0, 0, pos_rms, np.max(np.abs(clean_win)), 1.0)
                   
        # Function to process an SNR series
        def process_snr_series(bg_type, bg_list):
            bg_seg = get_random_bg_segment(bg_list)
            bg_rms = calc_rms(bg_seg)
            
            for snr in SNRS_DB:
                target_bg_rms = pos_rms / (10 ** (snr / 20.0))
                scale_factor = target_bg_rms / bg_rms
                scaled_bg = bg_seg * scale_factor
                
                mix = clean_win + scaled_bg
                peak = np.max(np.abs(mix))
                gain = 1.0
                if peak > 1.0:
                    gain = 0.99 / peak
                    mix = mix * gain
                    
                raw_m, norm_m = make_features(mix)
                score_m = detector.predict_features(norm_m)
                mix_rms = calc_rms(mix)
                
                record_run(event_id, path, src_type, bg_type, "random_segment", str(snr), mix, raw_m, norm_m, score_m,
                           pos_rms, bg_rms, target_bg_rms, mix_rms, peak, gain)
                           
        # 2. AMBIENT SNR SERIES
        process_snr_series("ambient", ambient_files)
        
        # 3. SPEECH SNR SERIES
        process_snr_series("speech", speech_files)
        
        # 4. STREAMING REPRODUCTION
        # Original padded clip added directly to unscaled ambient background
        bg_seg_stream = get_random_bg_segment(ambient_files)
        if len(orig_audio) < WINDOW_SAMPLES:
            orig_padded = np.pad(orig_audio, (0, WINDOW_SAMPLES - len(orig_audio)))
        else:
            orig_padded = orig_audio[:WINDOW_SAMPLES]
            
        stream_mix = orig_padded + bg_seg_stream
        raw_sr, norm_sr = make_features(stream_mix)
        score_sr = detector.predict_features(norm_sr)
        
        record_run(event_id, path, src_type, "streaming_repro", "random_ambient", "repro", stream_mix, raw_sr, norm_sr, score_sr,
                   pos_rms, 0, 0, calc_rms(stream_mix), np.max(np.abs(stream_mix)), 1.0)
                   
    # Save CSVs
    with open(DIAG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=diag_rows[0].keys())
        w.writeheader()
        w.writerows(diag_rows)
        
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=score_rows[0].keys())
        w.writeheader()
        w.writerows(score_rows)
        
    # Analyze TPR & Threshold Matrices
    print("\nAnalyzing TPR vs SNR at threshold 0.50...")
    
    def generate_matrix(bg_type):
        rows = []
        conditions = ["clean"] + [str(s) for s in SNRS_DB] + ["repro"]
        for cond in conditions:
            # Filter score_rows
            if cond == "clean":
                subset = [r for r in score_rows if r["snr_db"] == "clean"]
            elif cond == "repro":
                subset = [r for r in score_rows if r["snr_db"] == "repro"]
            else:
                subset = [r for r in score_rows if r["snr_db"] == cond and r["background_type"] == bg_type]
                
            if not subset: continue
            
            row = {"condition": cond}
            for t in THRESHOLDS:
                tpr = sum(1 for r in subset if r["score"] >= t) / len(subset)
                row[f"tpr_{t:.2f}"] = tpr
                
            tpr_50 = row["tpr_0.50"]
            tpr_50_real = sum(1 for r in subset if r["score"] >= 0.50 and r["source_type"]=="REAL") / len([r for r in subset if r["source_type"]=="REAL"])
            tpr_50_tts = sum(1 for r in subset if r["score"] >= 0.50 and r["source_type"]=="TTS") / len([r for r in subset if r["source_type"]=="TTS"])
            
            med_score = np.median([r["score"] for r in subset])
            p10_score = np.percentile([r["score"] for r in subset], 10)
            
            # Median score drop compared to clean
            drops = []
            if cond not in ["clean", "repro"]:
                for e_id in set(r["event_id"] for r in subset):
                    clean_score = next(r["score"] for r in score_rows if r["event_id"]==e_id and r["snr_db"]=="clean")
                    mix_score = next(r["score"] for r in subset if r["event_id"]==e_id)
                    drops.append(clean_score - mix_score)
            med_drop = np.median(drops) if drops else 0.0
            
            row["tpr_0.50_real"] = tpr_50_real
            row["tpr_0.50_tts"] = tpr_50_tts
            row["med_score"] = med_score
            row["p10_score"] = p10_score
            row["med_drop"] = med_drop
            row["n"] = len(subset)
            rows.append(row)
        return rows
        
    mat_amb = generate_matrix("ambient")
    mat_sp = generate_matrix("speech")
    
    with open(MATRIX_AMB, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mat_amb[0].keys())
        w.writeheader()
        w.writerows(mat_amb)
        
    with open(MATRIX_SP, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mat_sp[0].keys())
        w.writeheader()
        w.writerows(mat_sp)
        
    summary = {
        "counts": {
            "total": len(positives),
            "real": len([p for p in positives if p["is_real"]]),
            "tts": len([p for p in positives if not p["is_real"]])
        },
        "ambient": mat_amb,
        "speech": mat_sp
    }
    
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f)
        
    print(f"\nSaved {DIAG_CSV}")
    print(f"Saved {SCORES_CSV}")
    print(f"Saved {MATRIX_AMB}")
    print(f"Saved {MATRIX_SP}")
    
if __name__ == "__main__":
    main()
