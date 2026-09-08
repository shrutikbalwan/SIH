import csv
from collections import defaultdict

manifest_path = "dataset/split_manifest.csv"
speakers = {"train": set(), "validation": set(), "test": set()}
counts = {"train": 0, "validation": 0, "test": 0}

with open(manifest_path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["label"] == "0" and "libri" in row["group"].lower():
            spk = row.get("speaker_id", "").strip()
            split = row["split"].strip()
            if spk and split in speakers:
                speakers[split].add(spk)
                counts[split] += 1

print("TRAIN speakers:", sorted(list(speakers["train"])))
print("VALIDATION speakers:", sorted(list(speakers["validation"])))
print("TEST speakers:", sorted(list(speakers["test"])))
print(f"TRAIN count: {counts['train']} clips, {len(speakers['train'])} speakers")
print(f"VALIDATION count: {counts['validation']} clips, {len(speakers['validation'])} speakers")
print(f"TEST count: {counts['test']} clips, {len(speakers['test'])} speakers")

train_val = speakers["train"].intersection(speakers["validation"])
train_test = speakers["train"].intersection(speakers["test"])
val_test = speakers["validation"].intersection(speakers["test"])

print("Overlap train/val:", train_val)
print("Overlap train/test:", train_test)
print("Overlap val/test:", val_test)

