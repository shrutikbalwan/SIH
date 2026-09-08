from pathlib import Path
import random
import numpy as np
import soundfile as sf
from audiomentations import Compose, Gain

PROJECT = Path(r"E:\SIH\ira-wakeword")
SOURCE = PROJECT / "dataset" / "positive_all" / "real"
OUTPUT = PROJECT / "dataset" / "augmentation_test"

augment = Compose([
    Gain(min_gain_db=-2.0, max_gain_db=-0.5, p=1.0)
])

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    files = sorted(SOURCE.glob("*.wav"))

    if len(files) < 20:
        raise RuntimeError(f"Only {len(files)} files found.")

    random.seed(42)
    selected = random.sample(files, 20)

    print(f"Selected {len(selected)} source files")

    for index, file_path in enumerate(selected, 1):
        audio, sample_rate = sf.read(
            file_path,
            dtype="float32"
        )

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        augmented_audio = augment(
            samples=audio,
            sample_rate=sample_rate
        )

        augmented_audio = np.clip(
            augmented_audio,
            -1.0,
            1.0
        )

        output_path = OUTPUT / f"test_aug_{index:03d}.wav"

        sf.write(
            output_path,
            augmented_audio,
            sample_rate,
            subtype="PCM_16"
        )

        print(f"{index:02d}/20 -> {output_path.name}")

    print(
        "Generated:",
        len(list(OUTPUT.glob("*.wav")))
    )


if __name__ == "__main__":
    main()