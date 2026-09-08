import csv
import os

manifest_path = "dataset/split_manifest.csv"
spk_by_split = {"train": set(), "validation": set(), "test": set()}

with open(manifest_path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["label"] == "0" and ("libri" in row["group"].lower() or "speech" in row["group"].lower()):
            split = row["split"].strip()
            # Extract speaker id from filename: E.g., 118_118-47824-0000_neg_013.wav -> "118"
            fname = os.path.basename(row["path"])
            spk = fname.split("_")[0]
            if spk.isdigit():
                spk_by_split[split].add(spk)

for sp, spks in spk_by_split.items():
    print(f"{sp} speakers ({len(spks)}): {sorted(list(spks))}")

