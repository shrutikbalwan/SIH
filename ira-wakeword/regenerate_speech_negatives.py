from pathlib import Path
import soundfile as sf
import numpy as np
import shutil

SRC = Path("./dataset/negative/extracted/LibriSpeech/train-clean-5")
DST = Path("./dataset/negative/speech")

SR = 16000
CLIP = SR
TARGET = 4000

# Start clean
if DST.exists():
    for f in DST.glob("*.wav"):
        f.unlink()
else:
    DST.mkdir(parents=True)

# Find all speakers
speaker_dirs = sorted([p for p in SRC.iterdir() if p.is_dir()])

print("Speakers found:", len(speaker_dirs))

if len(speaker_dirs) != 28:
    raise RuntimeError(f"Expected 28 speakers, found {len(speaker_dirs)}")

# 4000 clips distributed as evenly as possible
base = TARGET // len(speaker_dirs)
remainder = TARGET % len(speaker_dirs)

speaker_targets = {}

for i, speaker_dir in enumerate(speaker_dirs):
    speaker_targets[speaker_dir.name] = base + (1 if i < remainder else 0)

rng = np.random.default_rng(42)
created_total = 0

for speaker_dir in speaker_dirs:
    speaker = speaker_dir.name
    target = speaker_targets[speaker]

    source_files = sorted(speaker_dir.rglob("*.flac"))

    candidates = []

    for f in source_files:
        audio, sr = sf.read(f, dtype="float32")

        if sr != SR or audio.ndim != 1:
            continue

        if len(audio) < CLIP:
            continue

        # Generate several evenly spaced windows from each recording.
        max_start = len(audio) - CLIP
        n_windows = max(1, len(audio) // CLIP)

        starts = np.linspace(
            0,
            max_start,
            n_windows,
            dtype=int
        )

        for j, start in enumerate(starts):
            candidates.append((f, int(start), j))

    if len(candidates) < target:
        raise RuntimeError(
            f"Speaker {speaker} only has {len(candidates)} candidates; "
            f"need {target}"
        )

    # Deterministic shuffle so we don't always select the earliest recordings.
    rng.shuffle(candidates)

    speaker_created = 0

    for f, start, j in candidates[:target]:
        audio, sr = sf.read(f, dtype="float32")
        clip = audio[start:start + CLIP]

        if len(clip) != CLIP:
            continue

        peak = np.max(np.abs(clip))

        if peak > 0.99:
            clip = clip / peak * 0.99

        out = DST / f"{speaker}_{f.stem}_neg_{j:03d}.wav"

        sf.write(
            out,
            clip,
            SR,
            subtype="PCM_16"
        )

        speaker_created += 1
        created_total += 1

    print(f"Speaker {speaker}: {speaker_created}/{target}")

print()
print("Created:", created_total)

if created_total != TARGET:
    raise RuntimeError(
        f"Expected {TARGET} clips, created {created_total}"
    )
