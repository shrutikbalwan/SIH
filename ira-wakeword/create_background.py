from pathlib import Path
import numpy as np
import soundfile as sf

OUT = Path("./dataset/negative/background")
OUT.mkdir(parents=True, exist_ok=True)

SR = 16000
N = SR
TARGET = 2000

rng = np.random.default_rng(42)

def normalize(x, peak=0.25):
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m * peak
    return x.astype(np.float32)

def pink_noise(n):
    white = rng.normal(0, 1, n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1
    spectrum = np.fft.rfft(white)
    spectrum /= np.sqrt(freqs)
    return np.fft.irfft(spectrum, n=n)

def brown_noise(n):
    white = rng.normal(0, 1, n)
    return np.cumsum(white)

for i in range(TARGET):
    kind = i % 7

    if kind == 0:
        # Very quiet room
        x = rng.normal(0, 0.01, N)

    elif kind == 1:
        # White noise
        x = rng.normal(0, 1, N)

    elif kind == 2:
        # Pink noise
        x = pink_noise(N)

    elif kind == 3:
        # Brown noise
        x = brown_noise(N)

    elif kind == 4:
        # Fan-like noise: filtered low-frequency noise
        x = pink_noise(N)
        t = np.arange(N) / SR
        x += 0.15 * np.sin(2 * np.pi * 110 * t)
        x += 0.08 * np.sin(2 * np.pi * 220 * t)

    elif kind == 5:
        # Electrical/room hum
        t = np.arange(N) / SR
        x = (
            rng.normal(0, 0.08, N)
            + 0.5 * np.sin(2 * np.pi * 50 * t)
            + 0.2 * np.sin(2 * np.pi * 100 * t)
        )

    else:
        # Mixed environmental-style noise
        x = (
            0.55 * pink_noise(N)
            + 0.25 * rng.normal(0, 1, N)
        )

    x = normalize(x, rng.uniform(0.08, 0.30))

    sf.write(
        OUT / f"background_{i:05d}.wav",
        x,
        SR,
        subtype="PCM_16"
    )

print(f"Created {TARGET} background clips")
