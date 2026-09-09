import csv
from collections import defaultdict
import os

manifest_path = "dataset/split_manifest.csv"
libri_paths = {"train": [], "validation": [], "test": []}

with open(manifest_path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["label"] == "0" and ("libri" in row["group"].lower() or "speech" in row["group"].lower()):
            split = row["split"].strip()
            if split in libri_paths:
                libri_paths[split].append(row["path"])

for sp, paths in libri_paths.items():
    print(f"{sp} count: {len(paths)}")
    if paths:
        print(f"Sample paths: {paths[:3]}")

