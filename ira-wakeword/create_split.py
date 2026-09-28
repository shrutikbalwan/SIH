from pathlib import Path
import csv
import random
from collections import defaultdict

ROOT = Path(".")
MANIFEST = ROOT / "dataset" / "manifest.csv"
OUT = ROOT / "dataset" / "split_manifest.csv"

random.seed(42)

with MANIFEST.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def split_items(items):
    items = list(items)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    for i, row in enumerate(items):
        if i < n_train:
            row["split"] = "train"
        elif i < n_train + n_val:
            row["split"] = "validation"
        else:
            row["split"] = "test"

# ---------------------------------
# LibriSpeech: speaker-safe split
# ---------------------------------

libri = [r for r in rows if r["group"] == "librispeech"]
other = [r for r in rows if r["group"] != "librispeech"]

speakers = defaultdict(list)

for row in libri:
    name = Path(row["path"]).stem

    # Regenerated filename:
    # <speaker>_<original-speaker>-<chapter>-<utterance>_neg_<window>
    #
    # Example:
    # 1088_1088-134315-0000_neg_002
    #
    # The first field is the speaker ID.
    speaker = name.split("_")[0]

    if not speaker.isdigit():
        raise RuntimeError(
            f"Could not determine LibriSpeech speaker from: {name}"
        )

    speakers[speaker].append(row)

speaker_ids = sorted(speakers)

print("LibriSpeech speakers found:", len(speaker_ids))

if len(speaker_ids) != 28:
    raise RuntimeError(
        f"Expected 28 LibriSpeech speakers, found {len(speaker_ids)}"
    )

random.shuffle(speaker_ids)

n_speakers = len(speaker_ids)
n_train = int(n_speakers * 0.80)
n_val = int(n_speakers * 0.10)

train_speakers = set(speaker_ids[:n_train])
val_speakers = set(speaker_ids[n_train:n_train + n_val])
test_speakers = set(speaker_ids[n_train + n_val:])

for speaker, items in speakers.items():

    if speaker in train_speakers:
        split = "train"
    elif speaker in val_speakers:
        split = "validation"
    else:
        split = "test"

    for row in items:
        row["split"] = split

# ---------------------------------
# Other groups
# ---------------------------------

groups = defaultdict(list)

for row in other:
    groups[row["group"]].append(row)

for group, items in groups.items():
    split_items(items)

# ---------------------------------
# Write split manifest
# ---------------------------------

fieldnames = ["path", "label", "source", "group", "split"]

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow({
            "path": row["path"],
            "label": row["label"],
            "source": row["source"],
            "group": row["group"],
            "split": row["split"],
        })

# ---------------------------------
# Report
# ---------------------------------

print()
print("Created:", OUT)
print("Total:", len(rows))

for split in ["train", "validation", "test"]:
    subset = [r for r in rows if r["split"] == split]
    pos = sum(r["label"] == "1" for r in subset)
    neg = sum(r["label"] == "0" for r in subset)

    print(f"{split}: {len(subset)}")
    print(f"  positive: {pos}")
    print(f"  negative: {neg}")

print()
print("LibriSpeech speakers:")
print("  train:", len(train_speakers))
print("  validation:", len(val_speakers))
print("  test:", len(test_speakers))
