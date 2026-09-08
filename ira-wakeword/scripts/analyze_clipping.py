from pathlib import Path
import csv
import numpy as np
import soundfile as sf


PROJECT = Path(r"E:\SIH\ira-wakeword")

SOURCE = PROJECT / "real" / "positive"
REPORT_DIR = PROJECT / "reports"

REPORT_FILE = REPORT_DIR / "clipping_report.csv"


def analyze_file(file_path):

    audio, sample_rate = sf.read(
        file_path,
        dtype="float32",
        always_2d=False
    )

    # Convert stereo to mono if necessary
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.nan_to_num(audio)

    total_samples = len(audio)

    if total_samples == 0:
        return 0.0, 0, 0.0

    peak = float(np.max(np.abs(audio)))

    clipped_samples = int(
        np.sum(np.abs(audio) >= 0.999)
    )

    clipping_percentage = (
        clipped_samples / total_samples
    ) * 100

    return peak, clipped_samples, clipping_percentage


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        SOURCE.rglob("*.wav")
    )

    rows = []

    clipped_files = 0

    for file_path in files:

        peak, clipped_samples, clipping_percentage = (
            analyze_file(file_path)
        )

        if clipped_samples > 0:

            clipped_files += 1

            relative_path = file_path.relative_to(
                SOURCE
            )

            parts = relative_path.parts

            condition = (
                parts[0]
                if len(parts) > 1
                else "unknown"
            )

            rows.append({
                "file": str(relative_path),
                "condition": condition,
                "peak": f"{peak:.6f}",
                "clipped_samples": clipped_samples,
                "clipping_percentage": f"{clipping_percentage:.4f}"
            })

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "condition",
                "peak",
                "clipped_samples",
                "clipping_percentage"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 65)
    print(" CLIPPING ANALYSIS")
    print("=" * 65)

    print()
    print(f"Total WAV files : {len(files)}")
    print(f"Clipped files   : {clipped_files}")

    print()
    print("Report saved to:")
    print(REPORT_FILE)

    print()


if __name__ == "__main__":
    main()