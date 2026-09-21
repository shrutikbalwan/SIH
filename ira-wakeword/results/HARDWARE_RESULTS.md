# Hardware Results — IRA Wake-Word on ESP32-S3

> **STATUS: NOT YET RECORDED.**
> Every value in this document is a placeholder. No figure below has been
> measured. Do not quote anything from this file until the value is filled in
> and its evidence linked.

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
| ESP-IDF version | `not yet recorded` | `not yet recorded` |
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
| Feature extraction time (mean) | `not yet recorded` | `not yet recorded` |
| Feature extraction time (max) | `not yet recorded` | `not yet recorded` |
| Inference time (mean) | 30.9 ms (30.91–30.94 across readings) | [Screenshot](screenshots/serial_log_detection.png) |
| Inference time (max) | `not yet recorded` | `not yet recorded` |
| Total per-window time (mean) | `not yet recorded` | `not yet recorded` |
| Total per-window time (max) | `not yet recorded` | `not yet recorded` |
| Inference interval / stride | ~136 ms, from timestamp deltas | [Screenshot](screenshots/serial_log_detection.png) |
| Budget utilisation | derived, not measured — 30.9/136 = 22.8% one core, 11.4% across two | [Screenshot](screenshots/serial_log_detection.png) |
| Real-time feasible | `not yet recorded` | `not yet recorded` |
| Idle CPU | `not yet recorded` | `not yet recorded` |
| Number of timing iterations | `not yet recorded` | — |

---

## 4. Detection configuration — as actually flashed

| Field | Value | Evidence |
|---|---|---|
| Detection threshold (probability) | inferred ~0.98, to be confirmed from live firmware source | [Screenshot](screenshots/serial_log_detection.png) |
| Detection threshold (INT8 `q_out`) | `not yet recorded` | `not yet recorded` |
| Smoothing rule | 3 consecutive candidates required ("candidate 1/3, 2/3, 3/3") | [Screenshot](screenshots/serial_log_detection.png) |
| Refractory / ignore-after-accept | a 0.9922 reading immediately after detection was not counted | [Screenshot](screenshots/serial_log_detection.png) |
| Audio guard active | `not yet recorded` | `not yet recorded` |

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
| Wake-word utterances attempted | `not yet recorded` | `not yet recorded` |
| Detections | confirmed detection: probability 0.9961 | [Screenshot](screenshots/serial_log_detection.png) |
| Missed | `not yet recorded` | `not yet recorded` |
| False accepts observed | `not yet recorded` | `not yet recorded` |
| Observation duration | `not yet recorded` | `not yet recorded` |
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
