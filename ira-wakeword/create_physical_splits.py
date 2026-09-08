from pathlib import Path
import csv
import shutil

ROOT = Path(".")
MANIFEST = ROOT / "dataset" / "split_manifest.csv"
OUT = ROOT / "dataset" / "splits"

with MANIFEST.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for split in ["train", "validation", "test"]:
    for label_name in ["positive", "negative"]:
        (OUT / split / label_name).mkdir(
            parents=True,
            exist_ok=True
        )

counts = {
    "train": {"positive": 0, "negative": 0},
    "validation": {"positive": 0, "negative": 0},
    "test": {"positive": 0, "negative": 0},
}

for i, row in enumerate(rows, 1):
    src = Path(row["path"])

    if not src.exists():
        raise FileNotFoundError(src)

    split = row["split"]
    label_name = "positive" if row["label"] == "1" else "negative"

    # Prefix with an index so filenames can never collide.
    dst = (
        OUT
        / split
        / label_name
        / f"{i:06d}_{src.name}"
    )

    shutil.copy2(src, dst)

    counts[split][label_name] += 1

    if i % 1000 == 0:
        print(f"Copied {i}/{len(rows)}")

print()
print("Copy complete")

for split in ["train", "validation", "test"]:
    print(
        f"{split}: "
        f"positive={counts[split]['positive']} "
        f"negative={counts[split]['negative']} "
        f"total={sum(counts[split].values())}"
    )
