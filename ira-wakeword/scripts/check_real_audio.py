from pathlib import Path
import soundfile as sf
import numpy as np


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT = Path(r"E:\SIH\ira-wakeword")

SOURCE = PROJECT / "real" / "positive"


# ============================================================
# CHECK ONE AUDIO FILE
# ============================================================

def check_file(file_path):

    result = {
        "valid": False,
        "sample_rate": None,
        "channels": None,
        "duration": None,
        "format": None,
        "silent": False,
        "clipped": False,
        "error": None
    }

    try:

        # Get WAV information
        info = sf.info(file_path)

        result["sample_rate"] = info.samplerate
        result["channels"] = info.channels
        result["duration"] = info.duration
        result["format"] = info.subtype

        # Read audio
        audio, sr = sf.read(
            file_path,
            dtype="float32",
            always_2d=False
        )

        # Convert stereo to mono only for checking
        if audio.ndim > 1:
            audio_check = np.mean(
                audio,
                axis=1
            )
        else:
            audio_check = audio

        # Remove invalid values
        audio_check = np.nan_to_num(audio_check)

        # Peak
        peak = np.max(
            np.abs(audio_check)
        )

        # RMS
        rms = np.sqrt(
            np.mean(audio_check ** 2)
        )

        # Detect silence
        if peak < 0.005 or rms < 0.001:
            result["silent"] = True

        # Detect clipping
        if np.any(
            np.abs(audio_check) >= 0.999
        ):
            result["clipped"] = True

        result["valid"] = True

    except Exception as e:

        result["error"] = str(e)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print(" REAL IRA AUDIO DATASET CHECK")
    print("=" * 65)

    print()
    print("Source:")
    print(SOURCE)

    # --------------------------------------------------------
    # Check folder
    # --------------------------------------------------------

    if not SOURCE.exists():

        print()
        print("ERROR: Folder does not exist.")
        print(SOURCE)
        return

    # --------------------------------------------------------
    # Find all WAV files
    # --------------------------------------------------------

    files = sorted(
        SOURCE.rglob("*.wav")
    )

    print()
    print(
        f"Total WAV files found: {len(files)}"
    )

    if len(files) == 0:

        print()
        print("No WAV files found.")
        return

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    valid = 0
    invalid = 0
    silent = 0
    clipped = 0

    sample_rates = {}
    channels = {}
    formats = {}

    # --------------------------------------------------------
    # Check every file
    # --------------------------------------------------------

    print()
    print("Checking files...")
    print()

    for file_path in files:

        result = check_file(file_path)

        if not result["valid"]:

            invalid += 1

            print("[INVALID]")
            print(file_path)
            print(result["error"])
            print()

            continue

        valid += 1

        # Sample rate
        sr = result["sample_rate"]

        sample_rates[sr] = (
            sample_rates.get(sr, 0) + 1
        )

        # Channels
        ch = result["channels"]

        channels[ch] = (
            channels.get(ch, 0) + 1
        )

        # Format
        fmt = result["format"]

        formats[fmt] = (
            formats.get(fmt, 0) + 1
        )

        # Silence
        if result["silent"]:

            silent += 1

            print("[SILENT]")
            print(file_path)
            print()

        # Clipping
        if result["clipped"]:

            clipped += 1

            print("[CLIPPED]")
            print(file_path)
            print()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 65)
    print(" FINAL REPORT")
    print("=" * 65)

    print()
    print(f"Total files : {len(files)}")
    print(f"Valid       : {valid}")
    print(f"Invalid     : {invalid}")
    print(f"Silent      : {silent}")
    print(f"Clipped     : {clipped}")

    print()
    print("SAMPLE RATES")

    for sr, count in sorted(
        sample_rates.items()
    ):

        print(
            f"  {sr} Hz : {count} files"
        )

    print()
    print("CHANNELS")

    for ch, count in sorted(
        channels.items()
    ):

        print(
            f"  {ch} channel(s) : {count} files"
        )

    print()
    print("AUDIO FORMAT")

    for fmt, count in sorted(
        formats.items()
    ):

        print(
            f"  {fmt} : {count} files"
        )

    print()
    print("=" * 65)
    print("TARGET FORMAT FOR OUR TINYML PIPELINE")
    print("=" * 65)

    print()
    print("Sample rate : 16000 Hz")
    print("Channels    : Mono")
    print("Format      : 16-bit PCM WAV")

    print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()