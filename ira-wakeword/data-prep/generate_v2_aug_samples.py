# -*- coding: utf-8 -*-
import os
import csv
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).parent
MANIFEST = REPO_ROOT / "dataset" / "split_manifest.csv"
OUT_DIR = REPO_ROOT / "v2_augmentation_examples"
OUT_CSV = REPO_ROOT / "v2_augmentation_examples.csv"
SKIPPED_CSV = REPO_ROOT / "v2_augmentation_skipped.csv"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000

# Ensure reproducibility
random.seed(12345)
np.random.seed(12345)

def load_audio(path: str) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a

def vad_trim(audio: np.ndarray, top_db=25):
    yt, _ = librosa.effects.trim(audio, top_db=top_db)
    return yt

def calc_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))

def get_train_files():
    positives = []
    ambient = []
    speech_spks = set()
    speech = []
    
    with open(MANIFEST, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] != 'train': continue
            path_str = r['path'].lower()
            group = r.get('group', '').lower()
            label = int(r.get('label', -1))
            full_path = REPO_ROOT / r['path']
            
            if label == 1:
                positives.append(str(full_path))
            elif label == 0:
                if 'background' in group or 'ambient' in path_str or 'background' in path_str:
                    ambient.append(str(full_path))
                elif 'speech' in group or 'librispeech' in group or 'librispeech' in path_str:
                    speech.append(str(full_path))
    
    # Filter missing
    positives = [p for p in positives if Path(p).exists()]
    ambient = [p for p in ambient if Path(p).exists()]
    speech = [p for p in speech if Path(p).exists()]
    
    return positives, ambient, speech

def get_random_bg_segment(bg_files: list, target_len: int = WINDOW_SAMPLES):
    rejected = 0
    while True:
        path = random.choice(bg_files)
        a = load_audio(path)
        start_sample = 0
        if len(a) < target_len:
            seg = np.pad(a, (0, target_len - len(a)))
        elif len(a) > target_len:
            start_sample = random.randint(0, len(a) - target_len)
            seg = a[start_sample : start_sample + target_len]
        else:
            seg = a
            
        rms = calc_rms(seg)
        if rms >= 1e-4:
            return seg, path, start_sample, rejected
        rejected += 1

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    positives, amb_files, sp_files = get_train_files()
    print(f"Loaded {len(positives)} train positives, {len(amb_files)} train ambients, {len(sp_files)} train speech.")
    
    profiles = [
        ("clean", "none", None),
        ("clean", "none", None),
        ("clean", "none", None),
        ("ambient", "high_snr", (15, 25)),
        ("ambient", "high_snr", (15, 25)),
        ("ambient", "med_snr", (5, 15)),
        ("ambient", "med_snr", (5, 15)),
        ("ambient", "med_snr", (5, 15)),
        ("ambient", "low_snr", (-5, 5)),
        ("ambient", "low_snr", (-5, 5)),
        ("speech", "high_snr", (15, 25)),
        ("speech", "high_snr", (15, 25)),
        ("speech", "med_snr", (5, 15)),
        ("speech", "med_snr", (5, 15)),
        ("speech", "med_snr", (5, 15)),
        ("speech", "low_snr", (-5, 5)),
        ("speech", "low_snr", (-5, 5)),
        ("ambient", "random", (-5, 25)),
        ("speech", "random", (-5, 25)),
        ("ambient", "random", (-5, 25))
    ]
    random.shuffle(profiles)
    
    random.shuffle(positives)
    
    csv_rows = []
    skipped_rows = []
    
    pos_idx = 0
    generated = 0
    total_silent_rejected = 0
    
    while generated < 20 and pos_idx < len(positives):
        path = positives[pos_idx]
        pos_idx += 1
        
        orig = load_audio(path)
        active = vad_trim(orig)
        
        if len(active) > WINDOW_SAMPLES:
            skipped_rows.append({"source_positive": Path(path).name, "reason": "active_speech_longer_than_1s", "len_samples": len(active)})
            continue
            
        pos_rms = calc_rms(active)
        bg_type, label, snr_range = profiles[generated]
        
        max_shift = WINDOW_SAMPLES - len(active)
        start_idx = random.randint(0, max_shift) if max_shift > 0 else 0
            
        window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        window[start_idx:start_idx+len(active)] = active
        
        bg_src_file = "none"
        bg_start_sec = 0.0
        target_snr = "clean"
        
        if bg_type != "clean":
            bg_list = amb_files if bg_type == "ambient" else sp_files
            bg_seg, b_path, b_start_sample, rej = get_random_bg_segment(bg_list)
            total_silent_rejected += rej
            
            bg_src_file = Path(b_path).name
            bg_start_sec = b_start_sample / SAMPLE_RATE
            
            raw_bg_rms = calc_rms(bg_seg)
            t_snr = random.uniform(snr_range[0], snr_range[1])
            target_snr = f"{t_snr:.1f}"
            
            target_bg_rms = pos_rms / (10 ** (t_snr / 20.0))
            scale_factor = target_bg_rms / raw_bg_rms
            scaled_bg = bg_seg * scale_factor
            
            mix = window + scaled_bg
        else:
            scaled_bg = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            mix = window.copy()
            
        pre_gain_peak = float(np.max(np.abs(mix)))
        gain = 1.0
        if pre_gain_peak > 1.0:
            gain = 0.99 / pre_gain_peak
            mix = mix * gain
            
        final_peak = float(np.max(np.abs(mix)))
        
        # Calculate final components for verification
        final_speech_comp = window * gain
        final_bg_comp = scaled_bg * gain
        final_speech_rms = calc_rms(final_speech_comp[start_idx:start_idx+len(active)])
        final_bg_rms = calc_rms(final_bg_comp)
        
        if bg_type != "clean":
            measured_snr_db = 20 * np.log10(final_speech_rms / (final_bg_rms + 1e-9))
            m_snr = f"{measured_snr_db:.1f}"
        else:
            m_snr = "clean"
            
        out_name = f"aug_{generated+1:02d}_{bg_type}_{label}.wav"
        out_path = OUT_DIR / out_name
        sf.write(out_path, mix, SAMPLE_RATE)
        
        csv_rows.append({
            "source_positive": Path(path).name,
            "background_source_file": bg_src_file,
            "background_start_sec": f"{bg_start_sec:.3f}",
            "background_type": bg_type,
            "target_snr_db": target_snr,
            "measured_final_snr_db": m_snr,
            "speech_start_sample": start_idx,
            "speech_start_ms": f"{(start_idx/SAMPLE_RATE)*1000:.1f}",
            "speech_duration_ms": f"{(len(active)/SAMPLE_RATE)*1000:.1f}",
            "pre_gain_peak": f"{pre_gain_peak:.4f}",
            "common_gain": f"{gain:.4f}",
            "final_peak": f"{final_peak:.4f}",
            "final_speech_rms": f"{final_speech_rms:.5f}",
            "final_background_rms": f"{final_bg_rms:.5f}",
            "output_file": out_name
        })
        generated += 1
        
    # Write CSVs
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
        
    if skipped_rows:
        with open(SKIPPED_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source_positive", "reason", "len_samples"])
            w.writeheader()
            w.writerows(skipped_rows)
            
    print(f"\nGenerated 20 augmented samples in {OUT_DIR}")
    print(f"Skipped {len(skipped_rows)} samples due to >1s length.")
    print(f"Rejected {total_silent_rejected} silent background segments.")
    
    # --------------------------------------------------------
    # VALIDATION CHECK
    # --------------------------------------------------------
    all_passed = True
    for r in csv_rows:
        # Check matching SNR
        if r["target_snr_db"] != "clean":
            t = float(r["target_snr_db"])
            m = float(r["measured_final_snr_db"])
            if abs(t - m) > 0.15:
                print(f"FAIL: SNR mismatch {t} vs {m} in {r['output_file']}")
                all_passed = False
        
        # Check peak
        if float(r["final_peak"]) > 1.0:
            print(f"FAIL: Final peak > 1.0 in {r['output_file']}")
            all_passed = False
            
    if all_passed:
        print("\nAUGMENTATION VALIDATION: PASS")
    else:
        print("\nAUGMENTATION VALIDATION: FAIL")

if __name__ == "__main__":
    main()
