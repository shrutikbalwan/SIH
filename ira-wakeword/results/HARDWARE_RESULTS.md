# Hardware Results — IRA Wake-Word on ESP32-S3

> **STATUS: PARTIALLY RECORDED.**
> Timing, smoothing and one detection are measured. Memory (arena, heap, stack)
> and feature parity are **not** — the firmware would need to print them.
> Fields still reading `not yet recorded` have not been measured. Do not quote
> them.

All measured values below come from a single source: a **3.53 s excerpt** of
serial output (27 inference lines, timestamps 104823–108353 ms) captured on the
live firmware and saved as
[`screenshots/serial_log_detection.png`](screenshots/serial_log_detection.png).
That excerpt contains **one** detection. Figures derived from it are marked
*derived*; figures read directly off it are marked *measured*.

Fill a row only when you have the measurement **and** the artifact that shows
it. Replace `not yet recorded` with the observed value and replace the evidence
link. If a field was never measured, leave it as is.

---

## 1. Test setup

| Field | Value | Evidence |
|---|---|---|
| Board model | `not yet recorded` | `not yet recorded` |
| Chip revision | `not yet recorded` | `not yet recorded` |
| CPU frequency | `not yet recorded` | `not yet recorded` |
| PSRAM present | `not yet recorded` | `not yet recorded` |
| Microphone / I2S device | `not yet recorded` | `not yet recorded` |
| Firmware version / commit | `not yet recorded` | `not yet recorded` |
| ESP-IDF version | **v6.0.3** *(from the serial-monitor window title `IDF_v6.0.3_Powershell` — confirm against `idf.py --version`)* | [Screenshot](screenshots/serial_log_detection.png) |
| esp-tflite-micro version | `not yet recorded` | `not yet recorded` |
| Model file | `not yet recorded` | `not yet recorded` |
| Model SHA-256 | `not yet recorded` | `not yet recorded` |
| Date of test | `not yet recorded` | `not yet recorded` |
| Tested by | `not yet recorded` | — |

---

## 2. Memory — measured on device

| Field | Value | Evidence |
|---|---|---|
| Tensor arena used | `not yet recorded` | `not yet recorded` |
| Tensor arena configured | `not yet recorded` | `not yet recorded` |
| Free heap (internal) | `not yet recorded` | `not yet recorded` |
| Minimum free heap | `not yet recorded` | `not yet recorded` |
| Largest free block | `not yet recorded` | `not yet recorded` |
| Main task stack high-water | `not yet recorded` | `not yet recorded` |
| Model flash bytes | `not yet recorded` | `not yet recorded` |
| Frontend static RAM | `not yet recorded` | `not yet recorded` |

> For reference, `esp32/` documents a **configured** arena of 40,960 bytes
> (`IRA_TENSOR_ARENA_SIZE`), explicitly described there as "a safe placeholder,
> not a measured value". Record the **observed** `arena_used_bytes` above, then
> trim the configured size to match.

---

## 3. Timing — measured on device

| Field | Value | Evidence |
|---|---|---|
| Feature extraction time (mean) | `not separately reported` — the logged 30.92 ms appears to include it (interval − inference = 104.8 ms ≈ a 100 ms delay + slop) | [Screenshot](screenshots/serial_log_detection.png) |
| Feature extraction time (max) | `not yet recorded` | `not yet recorded` |
| **Inference time (mean)** | **30.92 ms** *(measured)* | [Screenshot](screenshots/serial_log_detection.png) |
| **Inference time (max)** | **30.94 ms** *(measured, 27 readings)* | [Screenshot](screenshots/serial_log_detection.png) |
| Total per-window time (mean) | `not yet recorded` | `not yet recorded` |
| Total per-window time (max) | `not yet recorded` | `not yet recorded` |
| **Inference interval / stride** | **135.8 ms mean** (median 135.0, range 130–150, quantised to the 10 ms FreeRTOS tick); 7.37 inferences/s *(measured, 26 intervals)* | [Screenshot](screenshots/serial_log_detection.png) |
| **Budget utilisation** | *(derived)* 30.92 / 135.8 = **22.77% of one core**, **11.39% across two cores** | [Screenshot](screenshots/serial_log_detection.png) |
| Real-time feasible | Yes — inference (30.92 ms) fits well inside the 135.8 ms interval *(derived)* | [Screenshot](screenshots/serial_log_detection.png) |
| Idle CPU | `not yet recorded` | `not yet recorded` |
| Number of timing iterations | 27 inference lines / 26 intervals in the captured excerpt | [Screenshot](screenshots/serial_log_detection.png) |

> **On the "two cores" figure.** A single inference task runs on **one** core.
> The 11.39% figure is % of total system capacity across both cores; the core
> actually doing the work sits at **22.77%**. Quote both, or state which one you
> mean.
>
> **Planned change, not yet measured.** Raising the loop delay from ~100 ms to
> 140 ms would give an interval of ~176 ms → 17.6% of one core / 8.8% of two.
> Those are **predictions** from the measured overhead, not measurements. Do not
> record them here until a new log confirms them.

---

## 4. Detection configuration — as actually flashed

| Field | Value | Evidence |
|---|---|---|
| Detection threshold (probability) | **≈ 0.98046875, NOT 0.95** *(inferred — see below; confirm in firmware source)* | [Screenshot](screenshots/serial_log_detection.png) |
| Detection threshold (INT8 `q_out`) | **q ≥ 123** *(inferred)* | [Screenshot](screenshots/serial_log_detection.png) |
| **Smoothing rule** | **3 consecutive candidates required** — log prints `IRA candidate 1/3`, `2/3`, `3/3`, then `>>> IRA DETECTED <<<` *(measured)* | [Screenshot](screenshots/serial_log_detection.png) |
| Refractory / ignore-after-accept | **≈ 410 ms** *(inferred)* — a 0.9922 reading 140 ms after detection was not counted as a candidate | [Screenshot](screenshots/serial_log_detection.png) |
| Audio guard active | `not yet recorded` — RMS is logged per line, but no rejection was observed in the excerpt | [Screenshot](screenshots/serial_log_detection.png) |

> ### ⚠️ The flashed threshold does not appear to be 0.95
>
> The log constrains it directly. Output quantization is `p = (q + 128) / 256`:
>
> | timestamp | p | q | candidate? |
> |---|---|---|---|
> | 106443 | 0.9766 | 122 | **no** |
> | 106583 | 0.9805 | 123 | **yes — 1/3** |
>
> A reading of 0.9766 did **not** start the candidate run; 0.9805 did. The
> threshold therefore sits above 0.9766 and at or below 0.9805, i.e.
> **q ≥ 123, p ≥ 251/256 = 0.98046875**. If it were 0.95 (q ≥ 116), the 0.9766
> reading would have counted.
>
> The refractory figure is inferred the same way: after the detection at
> 106863, readings of 0.9922 and 0.9766 were both skipped, and candidates
> resumed at 107273 — about 410 ms later.
>
> **Both are inferences from a single 3.53 s excerpt, not readings of the
> source.** Confirm against the live firmware before quoting. It matters: the
> project documentation elsewhere states a 0.95 threshold, and the committed
> self-test firmware uses `IRA_Q_THRESHOLD (-25)` — p ≥ 0.40234375 — with no
> smoothing at all. Three different values are in play.

> Record the value that was **flashed**, not the intended one. The committed
> self-test firmware uses `IRA_Q_THRESHOLD (-25)` — p >= 0.40234375 — and
> contains no smoothing. If the live firmware differed, that difference is the
> point of this table.
>
> p = 0.95 is not exactly representable at output scale 0.00390625: the nearest
> steps are q=115 (p = 0.94921875) and q=116 (p = 0.953125). Record the `q_out`
> value the firmware actually compared against.

---

## 5. Observed detection behaviour

| Field | Value | Evidence |
|---|---|---|
| Wake-word utterances attempted | `not yet recorded` — not determinable from the log alone | `not yet recorded` |
| **Detections** | **1** in the excerpt — `>>> IRA DETECTED <<<`, probability **0.9961** *(measured)* | [Screenshot](screenshots/serial_log_detection.png) |
| Missed | `not yet recorded` | `not yet recorded` |
| False accepts observed | **0** in 3.53 s *(measured — far too short a window to estimate a rate)* | [Screenshot](screenshots/serial_log_detection.png) |
| **Observation duration** | **3.53 s** (104823 → 108353 ms) *(measured)* | [Screenshot](screenshots/serial_log_detection.png) |
| Test environment | `not yet recorded` | `not yet recorded` |
| Distance from microphone | `not yet recorded` | `not yet recorded` |
| Speakers tested | `not yet recorded` | `not yet recorded` |

> **Note on speakers.** All real recordings used in training to date come from a
> single speaker, and that speaker's recordings appear in both the training and
> test splits. Record how many distinct speakers were tested on hardware — if
> it was one, and the same one, say so here. It changes how these numbers
> should be read.

---

## 6. Feature parity on device

| Field | Value | Evidence |
|---|---|---|
| `FEATURE_MATCH` (of 1960) | `not yet recorded` | `not yet recorded` |
| `MAX_INT8_FEATURE_DIFF` | `not yet recorded` | `not yet recorded` |
| `q_out` vs desktop reference | `not yet recorded` | `not yet recorded` |
| Detection decision match | `not yet recorded` | `not yet recorded` |

> Host parity is already established: 1960/1960 INT8 features identical, max
> diff 0, on both self-test vectors, compiled with g++. That is **host**
> evidence. Xtensa parity is a separate claim and is still unproven — a
> different compiler and FPU can reorder float operations.

---

## 7. Screenshots

Place images in `screenshots/` and link each from the tables above.

| File | Shows |
|---|---|
| [serial_log_detection.png](screenshots/serial_log_detection.png) | live firmware serial log showing detection |
