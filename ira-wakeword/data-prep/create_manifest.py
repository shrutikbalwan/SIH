from pathlib import Path
import csv

ROOT = Path(".")
OUT = ROOT / "dataset" / "manifest.csv"

rows = []

def add_folder(folder, label, source, group):
    for f in sorted(folder.glob("*.wav")):
        rows.append({
            "path": str(f.resolve()),
            "label": label,
            "source": source,
            "group": group,
        })

# -------------------------
# POSITIVE DATA
# -------------------------

# Real recordings: keep each recording condition identifiable.
add_folder(
    ROOT / "real" / "positive" / "standardized" / "quiet_20cm",
    1, "real", "real_quiet_20cm"
)

add_folder(
    ROOT / "real" / "positive" / "standardized" / "quiet_50cm",
    1, "real", "real_quiet_50cm"
)

add_folder(
    ROOT / "real" / "positive" / "standardized" / "quiet_1m",
    1, "real", "real_quiet_1m"
)

# Piper-generated positives.
add_folder(
    ROOT / "dataset" / "positive_all" / "synthetic",
    1, "piper", "piper_original"
)

add_folder(
    ROOT / "dataset" / "positive_all" / "piper_batch_A",
    1, "piper", "piper_batch_A"
)

add_folder(
    ROOT / "dataset" / "positive_all" / "piper_batch_B",
    1, "piper", "piper_batch_B"
)

add_folder(
    ROOT / "dataset" / "positive_all" / "piper_batch_C",
    1, "piper", "piper_batch_C"
)

add_folder(
    ROOT / "dataset" / "positive_all" / "piper_batch_D",
    1, "piper", "piper_batch_D"
)

# -------------------------
# NEGATIVE DATA
# -------------------------

add_folder(
    ROOT / "dataset" / "negative" / "speech",
    0, "speech", "librispeech"
)

add_folder(
    ROOT / "dataset" / "negative" / "background",
    0, "background", "synthetic_noise"
)

# -------------------------
# WRITE MANIFEST
# -------------------------

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["path", "label", "source", "group"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Manifest:", OUT)
print("Total files:", len(rows))
print("Positive:", sum(r["label"] == 1 for r in rows))
print("Negative:", sum(r["label"] == 0 for r in rows))
