from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

PROJECT = Path(r"E:\SIH\ira-wakeword")

SOURCE = PROJECT / "synthetic" / "raw"
OUTPUT = PROJECT / "synthetic" / "standardized"

TARGET_SAMPLE_RATE = 16000


# ============================================================
# FUNCTIONS
# ============================================================

def convert_audio(input_file, output_file):
    """
    Convert audio to:
    - 16 kHz
    - mono
    - 16-bit PCM WAV
    """

    audio, sample_rate = sf.read(
        input_file,
        dtype="float32",
        always_2d=False
    )

    # --------------------------------------------------------
    # Convert stereo/multi-channel to mono
    # --------------------------------------------------------

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # --------------------------------------------------------
    # Remove NaN / infinity
    # --------------------------------------------------------

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Resample to 16 kHz
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        # Find greatest common divisor
        gcd = np.gcd(
            sample_rate,
            TARGET_SAMPLE_RATE
        )

        up = TARGET_SAMPLE_RATE // gcd
        down = sample_rate // gcd

        audio = resample_poly(
            audio,
            up,
            down
        ).astype(np.float32)

    # --------------------------------------------------------
    # Keep audio safely inside [-1, 1]
    # --------------------------------------------------------

    peak = np.max(np.abs(audio))

    if peak > 1.0:
        audio = audio / peak

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save as 16-bit PCM WAV
    # --------------------------------------------------------

    sf.write(
        output_file,
        audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(" IRA TINYML - AUDIO STANDARDIZATION")
    print("=" * 60)

    print()
    print("SOURCE:")
    print(SOURCE)

    print()
    print("OUTPUT:")
    print(OUTPUT)

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE.exists():

        print()
        print("ERROR: Source folder does not exist.")
        print(SOURCE)
        return

    # --------------------------------------------------------
    # Find WAV files
    # --------------------------------------------------------

    wav_files = list(
        SOURCE.rglob("*.wav")
    )

    print()
    print(f"WAV files found: {len(wav_files)}")

    if len(wav_files) == 0:

        print()
        print("No WAV files found.")
        return

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    successful = 0
    failed = 0

    print()
    print("Converting...")
    print()

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    for input_file in tqdm(
        wav_files,
        unit="file"
    ):

        try:

            relative_path = input_file.relative_to(
                SOURCE
            )

            output_file = OUTPUT / relative_path

            convert_audio(
                input_file,
                output_file
            )

            successful += 1

        except Exception as e:

            failed += 1

            print()
            print("ERROR:")
            print(input_file)
            print(e)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(" CONVERSION COMPLETE")
    print("=" * 60)

    print()
    print(f"Input WAV files : {len(wav_files)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")

    print()
    print("Standardized data:")
    print(OUTPUT)

    print()
    print("Format:")
    print("Sample rate : 16000 Hz")
    print("Channels    : Mono")
    print("Bit depth   : 16-bit PCM")
    print("Format      : WAV")

    print()


if __name__ == "__main__":
    main()