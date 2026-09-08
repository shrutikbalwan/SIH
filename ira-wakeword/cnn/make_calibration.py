import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import stft
import tensorflow as tf

BASE_DIR = r"E:\SIH\ira-wakeword"

TEST_DIR = os.path.join(BASE_DIR, "dataset", "splits", "test")
OUTPUT_PATH = os.path.join(
    BASE_DIR, "cnn", "models", "calibration_data.npy"
)

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
N_FFT = 512
WIN_LENGTH = 480
HOP_LENGTH = 320
NUM_BINS = 40

print("=" * 60)
print("CREATING INT8 CALIBRATION DATA")
print("=" * 60)


def make_spectrogram(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sr} Hz: {path}"
        )

    target_length = int(SAMPLE_RATE * CLIP_SECONDS)

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (target_length - len(audio), 0)
        )
    else:
        audio = audio[:target_length]

    _, _, Zxx = stft(
        audio,
        fs=SAMPLE_RATE,
        nperseg=WIN_LENGTH,
        noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False
    )

    spec = np.abs(Zxx).T
    spec = spec[:, :NUM_BINS]

    spec = np.log1p(spec)

    mean = np.mean(spec)
    std = np.std(spec) + 1e-6
    spec = (spec - mean) / std

    spec = spec.astype(np.float32)

    return spec


files = []

for category in ["positive", "negative"]:
    folder = os.path.join(TEST_DIR, category)

    wavs = glob.glob(os.path.join(folder, "*.wav"))
    files.extend(wavs)

print(f"\nFound {len(files)} test WAV files.")

# Use at most 500 samples for calibration
np.random.seed(42)

if len(files) > 500:
    files = list(
        np.random.choice(files, 500, replace=False)
    )

features = []

for i, path in enumerate(files, 1):

    try:
        spec = make_spectrogram(path)

        # CNN expects (49, 40, 1)
        if spec.shape != (49, 40):
            print(
                f"Skipping unexpected shape {spec.shape}: {path}"
            )
            continue

        features.append(spec[..., np.newaxis])

    except Exception as e:
        print(f"Skipping {path}: {e}")

    if i % 100 == 0:
        print(f"Processed {i}/{len(files)}")


X = np.stack(features).astype(np.float32)

print("\nCalibration data shape:", X.shape)

np.save(OUTPUT_PATH, X)

print("\nSaved:")
print(OUTPUT_PATH)

print(f"\nSamples: {len(X)}")
print(f"Size: {os.path.getsize(OUTPUT_PATH):,} bytes")

print("\nCalibration data creation complete.")