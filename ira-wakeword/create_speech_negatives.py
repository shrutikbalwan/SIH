from pathlib import Path
import soundfile as sf
import numpy as np

src = Path("./dataset/negative/extracted/LibriSpeech/train-clean-5")
dst = Path("./dataset/negative/speech")
dst.mkdir(parents=True, exist_ok=True)

TARGET = 4000
SR = 16000
CLIP_SAMPLES = SR

files = sorted(src.rglob("*.flac"))
created = 0

for f in files:
    audio, sr = sf.read(f, dtype="float32")

    if sr != SR or audio.ndim != 1:
        continue

    # Generate evenly spaced 1-second windows.
    n_windows = max(1, len(audio) // CLIP_SAMPLES)

    if n_windows == 1:
        starts = [0]
    else:
        starts = np.linspace(
            0,
            len(audio) - CLIP_SAMPLES,
            n_windows,
            dtype=int
        )

    for j, start in enumerate(starts):
        if created >= TARGET:
            break

        clip = audio[start:start + CLIP_SAMPLES]

        if len(clip) < CLIP_SAMPLES:
            continue

        # Keep a little headroom.
        peak = np.max(np.abs(clip))
        if peak > 0.99:
            clip = clip / peak * 0.99

        out = dst / f"{f.stem}_neg_{j:03d}.wav"
        sf.write(out, clip, SR, subtype="PCM_16")

        created += 1

    if created >= TARGET:
        break

    if created % 500 == 0:
        print(f"Created {created}/{TARGET}")

print(f"Created {created} negative speech clips")
