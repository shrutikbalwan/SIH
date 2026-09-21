# -*- coding: utf-8 -*-
# Proves the training frontend and the ESP32 device frontend are the same:
# measured 0 of 1960 INT8 mismatches (train vs device), while the TFLM
# MicroFrontend used by the retracted audit/evaluate.py differs by ~0.64 MAD.
"""
PHASE A — numerically confirm (or refute) the train/inference feature mismatch
documented in readme.txt.

Compares, on ONE wav file, three feature paths:

  A. TRAIN   : tf.signal.stft + log(|X|+1e-6) + first 40 bins + global z-score
               (train_cnn_v2_3.py::make_spectrogram)
  B. DEVICE  : esp32/main/ira_features.cpp via the prebuilt host binary
               (the code that actually runs on the ESP32-S3)
  C. MICROFE : pymicro_features MicroFrontend — the frontend used by
               audit/evaluate.py, which produced the 65.5% TPR / 864 FAPH numbers

Read-only. Nothing is trained, modified or written back.
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np
from scipy.io import wavfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

REPO = Path(__file__).resolve().parent.parent
EXE = REPO / "build" / ("ira_features_host.exe" if os.name == "nt" else "ira_features_host")

SR, N = 16000, 16000
FRAMES, BINS = 49, 40


def stats(name, a):
    print(f"  {name:<28s} shape={str(a.shape):<10s} "
          f"min={a.min():>10.4f}  max={a.max():>10.4f}  "
          f"mean={a.mean():>9.4f}  std={a.std():>8.4f}")


def pad_or_truncate(x, n=N):
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


# ---------------------------------------------------------------- A. TRAIN
def train_features(f32):
    import tensorflow as tf
    t = tf.convert_to_tensor(f32.astype(np.float32))
    s = tf.signal.stft(t, frame_length=480, frame_step=320,
                       fft_length=512, pad_end=False)
    lg = tf.math.log(tf.abs(s) + 1e-6)[:, :BINS]
    m = tf.reduce_mean(lg)
    d = tf.math.reduce_std(lg) + 1e-6
    return ((lg - m) / d).numpy().astype(np.float32)


# ---------------------------------------------------------------- B. DEVICE
def device_features(pcm16):
    r = subprocess.run([str(EXE)], input=pcm16.astype("<i2").tobytes(),
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit("host binary failed: " + r.stderr.decode(errors="replace"))
    n = FRAMES * BINS
    f = np.frombuffer(r.stdout[:n * 4], dtype="<f4").reshape(FRAMES, BINS)
    q = np.frombuffer(r.stdout[n * 4:], dtype=np.int8).reshape(FRAMES, BINS)
    return f.copy(), q.copy()


# ---------------------------------------------------------------- C. MICROFE
def microfrontend_features(pcm16):
    """Matches microwakeword.audio.audio_utils.generate_features_for_clip:
    MicroFrontend over 10 ms chunks, uint16 out, scaled by 0.0390625."""
    from pymicro_features import MicroFrontend
    fe = MicroFrontend()
    out, i, chunk = [], 0, 160  # 10 ms @ 16 kHz
    data = pcm16.astype(np.int16).tobytes()
    while i < len(data):
        res = fe.process_samples(data[i:i + chunk * 2])
        i += res.samples_read * 2
        if res.features:
            out.append(res.features)
        if res.samples_read == 0:
            break
    return np.array(out, dtype=np.float32) * 0.0390625


def main():
    wav = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO / "ira words" / "ira words" / "ira_0001.wav")
    sr, raw = wavfile.read(wav)
    assert sr == SR, f"expected 16 kHz, got {sr}"
    if raw.ndim > 1:
        raw = raw[:, 0]
    pcm16 = pad_or_truncate(raw.astype(np.int16))
    f32 = pcm16.astype(np.float32) / 32768.0

    print("=" * 78)
    print("PHASE A — feature path comparison")
    print("=" * 78)
    print(f"file    : {wav}")
    print(f"samples : {len(raw)} raw -> {len(pcm16)} after pad/truncate to 1.0 s\n")

    A = train_features(f32)
    B_f, B_q = device_features(pcm16)
    C = microfrontend_features(pcm16)

    print("VALUE RANGES")
    stats("A TRAIN  (tf.signal.stft)", A)
    stats("B DEVICE (esp32 C++)", B_f)
    stats("C MICROFE (pymicro)", C)

    print("\nPAIRWISE COMPARISON")
    print(f"  A vs B  (train vs device)   : shapes {A.shape} vs {B_f.shape}")
    if A.shape == B_f.shape:
        d = np.abs(A - B_f)
        print(f"      mean abs diff = {d.mean():.3e}     max abs diff = {d.max():.3e}")
        qa = np.clip(np.round(A / 0.06078097224235535 + 49), -128, 127).astype(np.int8)
        nmis = int((qa != B_q).sum())
        print(f"      INT8 mismatches = {nmis} / {qa.size}"
              f"   -> {'BIT-IDENTICAL' if nmis == 0 else 'DIFFERS'}")

    print(f"\n  A vs C  (train vs MicroFrontend): shapes {A.shape} vs {C.shape}")
    if A.shape != C.shape:
        print("      SHAPE MISMATCH — cannot subtract directly.")
    n = min(A.shape[0], C.shape[0])
    Ac, Cc = A[:n], C[:n, :BINS]
    print(f"      on the overlapping {n}x{BINS} region:")
    print(f"      mean abs diff = {np.abs(Ac - Cc).mean():.4f}"
          f"     max abs diff = {np.abs(Ac - Cc).max():.4f}")
    # correlation tells us whether they even encode the same structure
    cc = np.corrcoef(Ac.ravel(), Cc.ravel())[0, 1]
    print(f"      Pearson correlation = {cc:+.4f}")

    # normalised comparison: put both on a common z-scale to separate
    # "different units" from "different information"
    za = (Ac - Ac.mean()) / (Ac.std() + 1e-9)
    zc = (Cc - Cc.mean()) / (Cc.std() + 1e-9)
    print(f"      after z-scoring BOTH: mean abs diff = {np.abs(za - zc).mean():.4f}"
          f"   corr = {np.corrcoef(za.ravel(), zc.ravel())[0,1]:+.4f}")


if __name__ == "__main__":
    main()
