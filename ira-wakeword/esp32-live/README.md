# esp32-live/ — Live Wake-Word Firmware

**Status: EMPTY. Firmware not yet copied into this repository.**

This folder is reserved for the **live capture firmware** that was run on the
ESP32-S3: I2S microphone capture, continuous sliding-window detection, and the
detection smoothing rule.

That firmware was developed and run on a different machine and has not yet been
copied here. Nothing in this repository currently performs audio capture.

---

## How this differs from `esp32/`

The two folders are not alternatives. They are different stages.

| | `esp32/` | `esp32-live/` (this folder) |
|---|---|---|
| Purpose | Boot self-test | Live wake-word detection |
| Microphone / I2S | **None** | I2S capture |
| Operation | One-shot validation at boot | Continuous, sliding window |
| Input | Two embedded test vectors | Live microphone audio |
| Smoothing | None | Consecutive-detection rule |
| In this repo | Present | **Not yet** |

`esp32/main/main.cpp` states this plainly in its header comment: *"NO
MICROPHONE / I2S CAPTURE."* It validates that TFLite Micro initialises, that
the feature frontend reproduces the desktop reference bit-for-bit, and that the
invalid-audio guard rejects a zero window — all against embedded vectors, not
live audio.

---

## What belongs here

When the live firmware is copied in, it should include:

- The ESP-IDF project (`CMakeLists.txt`, `main/`, `sdkconfig.defaults`)
- I2S capture and ring-buffer code
- The detection state machine: threshold and consecutive-detection smoothing
- Any serial-logging code that produced the on-device detection logs

**Do not commit `sdkconfig`, `build/`, or `managed_components/`** — all are
already covered by `.gitignore`. If the firmware holds Wi-Fi credentials, put
them in `secrets.h` (see `esp32/main/secrets.h.example`); `secrets.h` is
gitignored.

---

## Two discrepancies to resolve when the firmware lands

These are open questions, recorded here so they are not forgotten. Both concern
what the deployed configuration actually was.

1. **Detection threshold.** The committed self-test firmware uses
   `IRA_Q_THRESHOLD (-25)` in `esp32/main/ira_inference.h`, i.e. fires at
   **p >= 0.40234375**. If the live firmware used a 0.95 threshold, the two
   disagree and the deployed value must be documented.

   Note that p = 0.95 is **not exactly representable**: the output quantization
   is scale 0.00390625 (= 1/256), zero-point -128, so the achievable steps
   either side are q=115 (p = 0.94921875) and q=116 (p = 0.953125).

2. **Smoothing.** The committed firmware contains **no** smoothing or
   consecutive-detection logic — it is a single-shot self-test. Whatever
   consecutive-N rule the live firmware applied exists only in that firmware.

Measured hardware figures belong in `../results/HARDWARE_RESULTS.md`, not here.
