from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

INPUT_DIR = Path("synthetic/raw")
OUTPUT_DIR = Path("synthetic/validated")

TARGET_SR = 16000
MIN_DURATION = 0.15
MAX_DURATION = 2.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total = 0
valid = 0
failed = 0

for wav_path in INPUT_DIR.rglob("*.wav"):
    total += 1

    try:
        audio, sample_rate = sf.read(wav_path, always_2d=False)

        # Convert stereo/multi-channel to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Check duration before resampling
        duration = len(audio) / sample_rate

        if duration < MIN_DURATION or duration > MAX_DURATION:
            print(f"SKIP duration: {wav_path} ({duration:.2f}s)")
            failed += 1
            continue

        # Check for clipping
        peak = np.max(np.abs(audio))

        if peak > 1.0:
            audio = audio / peak

        # Resample to 16 kHz
        if sample_rate != TARGET_SR:
            audio = resample_poly(
                audio,
                TARGET_SR,
                sample_rate
            )

        # Normalize conservatively
        peak = np.max(np.abs(audio))

        if peak > 0.98:
            audio = audio * (0.98 / peak)

        # Preserve directory structure
        relative_path = wav_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(
            output_path,
            audio,
            TARGET_SR,
            subtype="PCM_16"
        )

        valid += 1

    except Exception as e:
        print(f"ERROR: {wav_path}")
        print(e)
        failed += 1

print()
print("=" * 50)
print("AUDIO PREPROCESSING COMPLETE")
print("=" * 50)
print(f"Total files : {total}")
print(f"Valid files : {valid}")
print(f"Failed      : {failed}")