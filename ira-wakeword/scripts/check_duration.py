from pathlib import Path
import soundfile as sf
import numpy as np


PROJECT = Path(r"E:\SIH\ira-wakeword")

SOURCE = PROJECT / "real" / "positive" / "standardized"


def main():

    files = sorted(
        SOURCE.rglob("*.wav")
    )

    print()
    print("=" * 70)
    print(" REAL IRA AUDIO DURATION CHECK")
    print("=" * 70)

    print()
    print(f"Total WAV files: {len(files)}")

    if not files:
        print("No WAV files found.")
        return

    durations = []

    short_files = []
    long_files = []

    for file_path in files:

        try:

            info = sf.info(file_path)

            duration = info.duration

            durations.append(duration)

            if duration < 0.5:
                short_files.append(
                    (file_path, duration)
                )

            if duration > 2.0:
                long_files.append(
                    (file_path, duration)
                )

        except Exception as e:

            print(
                f"ERROR: {file_path}"
            )

            print(e)

    durations = np.array(durations)

    print()
    print("=" * 70)
    print(" DURATION STATISTICS")
    print("=" * 70)

    print()
    print(
        f"Minimum : {durations.min():.3f} sec"
    )

    print(
        f"Maximum : {durations.max():.3f} sec"
    )

    print(
        f"Average : {durations.mean():.3f} sec"
    )

    print(
        f"Median  : {np.median(durations):.3f} sec"
    )

    print()

    # --------------------------------------------------------
    # Duration ranges
    # --------------------------------------------------------

    ranges = [
        ("< 0.5 sec", durations < 0.5),
        ("0.5 - 1.0 sec",
         (durations >= 0.5) &
         (durations < 1.0)),
        ("1.0 - 1.5 sec",
         (durations >= 1.0) &
         (durations < 1.5)),
        ("1.5 - 2.0 sec",
         (durations >= 1.5) &
         (durations < 2.0)),
        (">= 2.0 sec", durations >= 2.0)
    ]

    print("DURATION DISTRIBUTION")
    print()

    for name, mask in ranges:

        print(
            f"{name:15s}: {np.sum(mask)} files"
        )

    # --------------------------------------------------------
    # Short files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(" SHORT FILES (< 0.5 sec)")
    print("=" * 70)

    if not short_files:

        print()
        print("None")

    else:

        for path, duration in short_files:

            print(
                f"{duration:.3f} sec  {path}"
            )

    # --------------------------------------------------------
    # Long files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(" LONG FILES (> 2.0 sec)")
    print("=" * 70)

    if not long_files:

        print()
        print("None")

    else:

        for path, duration in long_files:

            print(
                f"{duration:.3f} sec  {path}"
            )

    print()
    print("=" * 70)
    print(" CHECK COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()