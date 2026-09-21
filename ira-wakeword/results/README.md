# results/ — Hardware Evidence

**Status: FIRST MEASUREMENTS RECORDED.** Inference time, interval, smoothing,
and detection probability are now filled in from the serial log below.
Fields not yet measured (RAM, heap, arena) remain *not yet recorded*.

This folder holds evidence **measured on real ESP32-S3 hardware** — figures
observed on a board, not computed on a desktop, and not estimated.

![Serial log — IRA detection on ESP32-S3](screenshots/serial_log_detection.png)

---

## Contents

| File | Purpose |
|---|---|
| `HARDWARE_RESULTS.md` | The measurement table. Every field currently reads *not yet recorded*. |
| `screenshots/` | Serial-monitor screenshots backing each figure. `serial_log_detection.png` is the first. |

---

## The rule for this folder

**Every number here must be one that was observed on hardware, and must link to
the evidence that shows it.**

- No estimates.
- No desktop-computed figures presented as device figures.
- No number without a screenshot, serial log, or equivalent artifact.
- A field that has not been measured stays marked *not yet recorded*. Leaving a
  field blank is correct; filling it with a plausible value is not.

This matters because the existing reports in `esp32/` were written to the same
standard. `ESP32_ON_DEVICE_SELF_TEST_REPORT.md` states outright: *"No firmware
size, RAM figure, or timing number is reported below, because none was
measured."* Statically computable figures — model size, configured arena size,
static RAM totals — are recorded there and labelled as derived from source, not
measured. This folder continues that separation.

---

## What is NOT hardware evidence

For clarity, none of the following belongs here:

- Desktop evaluation output (`cnn/`, `audit/`, `logs/`)
- Host-compiled self-test results (`HOST_SELFTEST = PASS` is a g++ host run)
- Statically computed sizes from source constants
- Anything produced by a Python script on a laptop

Those live in `reports/` and `logs/`.
