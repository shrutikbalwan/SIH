from pathlib import Path
import csv
import random

INPUT_DIR = Path("synthetic/validated")
METADATA_DIR = Path("metadata")

METADATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

files = list(INPUT_DIR.rglob("*.wav"))

print(f"Found {len(files)} WAV files")

samples = []

for path in files:
    relative = path.relative_to(INPUT_DIR)

    parts = relative.parts

    # Expected:
    # voice / speed / filename.wav
    voice = parts[0] if len(parts) >= 3 else "unknown"
    speed = parts[1] if len(parts) >= 3 else "unknown"

    samples.append({
        "file": str(relative).replace("\\", "/"),
        "keyword": "Ira",
        "voice": voice,
        "speed": speed,
        "source": "synthetic"
    })

random.shuffle(samples)

total = len(samples)

train_end = int(total * 0.80)
validation_end = int(total * 0.90)

train = samples[:train_end]
validation = samples[train_end:validation_end]
test = samples[validation_end:]


def write_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "keyword",
                "voice",
                "speed",
                "source"
            ]
        )

        writer.writeheader()
        writer.writerows(data)


write_csv(METADATA_DIR / "all_samples.csv", samples)
write_csv(METADATA_DIR / "train.csv", train)
write_csv(METADATA_DIR / "validation.csv", validation)
write_csv(METADATA_DIR / "test.csv", test)

print()
print("Dataset split complete")
print("-----------------------")
print(f"Total      : {len(samples)}")
print(f"Training   : {len(train)}")
print(f"Validation : {len(validation)}")
print(f"Test       : {len(test)}")