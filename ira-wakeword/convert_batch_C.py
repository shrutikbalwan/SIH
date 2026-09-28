from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly

src = Path("./dataset/piper_generated/batch_C")
dst = Path("./dataset/piper_generated/batch_C_16k")

files = list(src.glob("*.wav"))
print(f"Found {len(files)} input files")

for i, f in enumerate(files, 1):
    audio, sr = sf.read(f, dtype="float32")

    if sr != 22050:
        print(f"Skipping {f.name}: {sr} Hz")
        continue

    audio_16k = resample_poly(audio, 16000, 22050)
    audio_16k = audio_16k.clip(-1.0, 1.0)

    sf.write(
        dst / f.name,
        audio_16k,
        16000,
        subtype="PCM_16"
    )

    if i % 100 == 0:
        print(f"Converted {i}/{len(files)}")

print("Conversion complete")
