# ESP32-S3 Feature Parity Report — IRA V2.3 Frontend

**Stage:** V2.3 preprocessing implementation + desktop parity harness.
**Microphone / I2S capture:** ❌ **NOT IMPLEMENTED (out of scope for this stage)**

**Result in one line:** on real audio the C++ frontend is **bit-identical** to the
Python reference (1960/1960 INT8 values, identical `q_out`, identical decision, on
every real WAV tested). On synthetic pure tones the two diverge — investigated
below and shown to be **float32 ill-conditioning, not an implementation defect**.

---

## 1. Exact frontend specification (as implemented)

| step | value |
|---|---|
| Input | 16000 mono PCM16 samples @ 16 kHz |
| PCM → float | `x / 32768.0f` (matches `soundfile` int16 loading) |
| Frame length | 480 |
| Frame step | 320 |
| FFT length | 512 |
| `pad_end` | `false` |
| Frames produced | **49** = (16000 − 480)/320 + 1 |
| Window | **PERIODIC Hann** (see §2) |
| Zero-pad | samples 480..511 = 0, imaginary = 0 |
| Magnitude | `sqrt(re² + im²)` |
| Compression | `logf(mag + 1e-6)` — natural log |
| Bins kept | 0..39 (first 40 of 257 rfft bins) → **49 × 40** |
| Mean | ONE global mean over all 1960 values |
| Std | ONE global **population** std (ddof = 0), matching `tf.math.reduce_std` |
| Normalise | `(v − mean) / (std + 1e-6)` |
| Quantise | `clamp(rintf(v / 0.06078097224235535) + 49, −128, 127)` |

Not MFCC. Not mel. Not MicroFrontend.

### Rounding

`rintf()` is used, **not** `roundf()`. The Python reference uses `numpy.round`,
which is **round-half-to-even**; `rintf()` with the default rounding mode is also
round-half-to-even, whereas `roundf()` is half-away-from-zero and would differ by
1 LSB on exact `.5` boundaries. Verified: 0 mismatches attributable to rounding on
any test case.

`q = round(x/scale) + zp` and `q = round(x/scale + zp)` are identical here because
`zp` is an integer, so the choice of formulation is immaterial.

---

## 2. TensorFlow Hann window — verified, not assumed

`tf.signal.stft` defaults to `tf.signal.hann_window(..., periodic=True)`.
Measured against both candidate formulas:

| formula | max abs deviation from TF |
|---|---|
| **periodic:** `0.5 − 0.5·cos(2πn/480)` | **2.62e−07** ✅ |
| symmetric: `0.5 − 0.5·cos(2πn/479)` | 5.02e−03 ❌ |

Confirmed `w[0] = 0`, `w[1] = w[479] = 4.282593727e−05`. The C++ uses the periodic
form. The symmetric form — what most naive Hann implementations produce — is wrong
by up to 5e−3 and was explicitly ruled out.

STFT output shape verified as `(49, 257)`, and a manual `rfft(frame × periodic_hann, n=512)`
reproduction matched TF to **6.39e−06**.

---

## 3. FFT backend

| item | value |
|---|---|
| Implementation | Built-in **iterative radix-2 complex FFT**, 512-point, decimation-in-time |
| Tables | Precomputed twiddles (256 complex) + bit-reversal LUT, built once in `ira_features_init()` |
| **Normalization** | **NONE** — unnormalized forward transform, matching `np.fft.rfft` / `tf.signal.stft`:  `X[k] = Σ x[n]·e^(−2πikn/N)` |
| 480 → 512 | 480 windowed real samples in slots 0..479; slots 480..511 zero; all imaginary parts zero |
| Isolation | Behind `ira_fft512()` with an `IRA_USE_ESP_DSP` compile switch |

**ESP-DSP swap:** the ESP-DSP branch is present but deliberately `#error`s rather
than silently substituting. `dsps_fft2r_fc32` uses the same unnormalized forward
convention, so no rescaling should be needed — **but the parity harness must be
re-run after wiring it**, since FFT accumulation order affects the marginal cases
described in §5.

A convention note worth recording: an FFT that divided by N would shift every
log-magnitude by a constant, and that constant would in fact be cancelled by the
per-window z-score. A `sqrt(N)` or per-bin scaling would **not** cancel. Relying
on that cancellation would be fragile, so the unnormalized convention is kept
explicit rather than assumed-harmless.

---

## 4. Parity results

Harness: `tools/test_feature_parity.py` (Python/TF reference) driving
`tools/ira_features_host.cpp`, which compiles **`esp32/main/ira_features.cpp`
unchanged** with g++ 13.1.0 `-O2 -std=c++17`. Model SHA-256 verified before use.

### Float features

| case | max \|err\| | mean \|err\| | cosine sim |
|---|---|---|---|
| all-zero | 9.130e−01 | 9.130e−01 | 1.000000000000 |
| sine-440Hz | 4.990e−02 | 1.005e−03 | 0.999993205070 |
| sine-1000Hz | 7.124e−04 | 6.204e−05 | 1.000000000000 |
| random-pcm | 1.860e−05 | 1.974e−07 | 0.999999940395 |
| impulse | 1.097e−05 | 5.698e−06 | 1.000000000000 |
| dc-offset | 6.567e−04 | 5.976e−05 | 0.999999940395 |
| near-full-scale | 6.843e−02 | 1.088e−02 | 0.999867558479 |
| wav ×6 (real) | 1.49e−06 … 1.41e−05 | ≤2.23e−07 | ≥0.999999940 |

### INT8 features (of 1960 per case)

| case | identical | mismatch | identical % | max \|dq\| |
|---|---|---|---|---|
| all-zero | 0 | 1960 | 0.0000% | 15 |
| sine-440Hz | 1902 | 58 | 97.0408% | 1 |
| sine-1000Hz | 1960 | 0 | **100.0000%** | 0 |
| random-pcm | 1960 | 0 | **100.0000%** | 0 |
| impulse | 1960 | 0 | **100.0000%** | 0 |
| dc-offset | 1960 | 0 | **100.0000%** | 0 |
| near-full-scale | 1568 | 392 | 80.0000% | 2 |
| **all 6 real WAVs** | **1960 each** | **0** | **100.0000%** | **0** |

### Model output on both feature sets

| case | q(python) | q(c++) | \|Δq\| | det(py) | det(c++) | same |
|---|---|---|---|---|---|---|
| all-zero | 104 | 102 | 2 | YES | YES | ✅ |
| sine-440Hz | −128 | −128 | 0 | no | no | ✅ |
| sine-1000Hz | −128 | −128 | 0 | no | no | ✅ |
| random-pcm | −126 | −126 | 0 | no | no | ✅ |
| impulse | −128 | −128 | 0 | no | no | ✅ |
| dc-offset | −127 | −127 | 0 | no | no | ✅ |
| near-full-scale | −128 | −128 | 0 | no | no | ✅ |
| wav ×6 | 114 / 108 / 87 / 114 / 114 / 119 | **identical** | **0** | YES | YES | ✅ |

### Aggregate

| | value |
|---|---|
| INT8 identical (all cases) | 23070 / 25480 = 90.5416% |
| **INT8 identical (real audio only)** | **11760 / 11760 = 100.0000%** |
| **Worst \|Δq_out\| (real audio)** | **0** |
| Worst \|Δq_out\| (overall) | 2 |
| **Detection decisions matching** | **13 / 13 — ALL** |

---

## 5. Diagnosis of the remaining discrepancies

Per instruction, these were diagnosed rather than papered over. **No model or
threshold change was made.**

### 5.1 all-zero — the Python reference is the one behaving oddly

The task brief predicted Python would output all zeros. **It does not.**

```
log(0 + 1e-6)            = -13.815511  (all 1960 values identical)
tf.reduce_mean           = -13.815521240234375   <-- differs from the value itself
tf.math.reduce_std       =  1.049e-05            <-- not 0
normalised               =  0.912971  (uniform, not 0)
quantised                =  64        (uniform)
```

`reduce_mean` of 1960 identical float32 values returns a value ~1e−5 away from
that value — a float32 accumulation artifact in TF's reduction. That residual is
then divided by `std + 1e-6 ≈ 1.15e−05`, an **amplification of ~87 000×**,
producing 0.913 instead of 0.

The C++ accumulates mean and variance in `double`, gets the mathematically correct
mean, a residual of exactly 0, `std = 0`, and therefore output exactly `0.0` —
**finite, no NaN, no divide-by-zero**, which is the behaviour the brief asked to verify.

Could the C++ instead reproduce TF's artifact? Tested:

| accumulation | mean |
|---|---|
| **TF `reduce_mean`** | **−13.815521240234375** |
| naive float32 sequential | −13.815823554992676 |
| numpy float32 (pairwise) | −13.815509796142578 |
| double | −13.815510749816895 |

TF's value matches **none** of them — it is Eigen's specific vectorized tree-sum
order for this shape and CPU. Reproducing it bit-exactly in portable C++ is not
achievable, and chasing it would make the firmware depend on a host BLAS
implementation detail. **Double accumulation was kept deliberately.**

### 5.2 Pure tones — float32 ill-conditioning, and the C++ is *closer* to truth

The decisive test: compare **both** implementations against a float64 "ground
truth" frontend.

| case | TF float32 vs truth | C++ float32 vs truth | closer to truth |
|---|---|---|---|
| sine-440Hz | 1.199e−01 | **7.001e−02** | **C++** |
| sine-1000Hz | 9.493e−02 | **9.422e−02** | **C++** |
| near-full-scale | 1.657e−01 | **1.176e−01** | **C++** |
| wav ×3 (real) | 1.06e−05 / 2.17e−06 / 1.15e−05 | 1.18e−05 / 3.51e−06 / 1.16e−05 | TF (marginally) |

On pure tones **both** are ~0.1 from the true value, and the C++ is consistently
*nearer*. Neither is wrong; float32 simply cannot resolve these signals stably.
On real audio both agree with truth to ~1e−05 — five orders of magnitude better.

Why pure tones specifically: bin spacing is 16000/512 = 31.25 Hz.

- **1000 Hz** = exactly bin 32.0 → no leakage → **0 mismatches**
- **440 Hz** = bin 14.08 → off-bin → leakage tails formed by cancellation of large
  terms → 58 mismatches
- **300 Hz @ 0.999 amplitude** = bin 9.6, off-bin, near full scale → largest
  dynamic range → 392 mismatches

Bin alignment predicts the outcome exactly. Real speech is broadband, has no such
cancellation structure, and shows exact parity.

### Ruled out as causes

Window definition (§2, verified numerically), FFT scaling (unnormalized both
sides; the sine-1000Hz case is exact, which a scaling error could not produce),
magnitude computation, log implementation, std definition (population/ddof=0,
verified), and float→int rounding (`rintf` = half-to-even, matches numpy).

---

## 6. ⚠️ Finding unrelated to parity: constant input triggers a wake

Worth flagging because it emerged from this harness and matters for deployment:

**For the all-zero input, both implementations produce `q_out` = 104 / 102, i.e.
`q_out >= -25` — the model FIRES on digital silence.** `dc-offset` (constant 1000)
gives −127 and does not fire, so this is specific to an exactly-constant spectrum,
where the z-score of a flat plane produces a degenerate feature map.

A real microphone will not deliver exact digital zero, so this is unlikely to
trigger in practice — but a muted/disconnected I2S channel, a codec in reset, or a
DMA underrun buffer **would** deliver exact zeros. **Recommendation for the I2S
stage: gate inference on a minimum input RMS,** so an all-zero or constant buffer
is never fed to the model. This is an input-validation guard, not a model or
threshold change.

---

## 7. Memory required for feature calculation

Static, no dynamic allocation anywhere:

| buffer | bytes |
|---|---|
| `s_samples[16000]` float | 64 000 |
| `s_logmag[49][40]` float | 7 840 |
| `s_fft_re[512]` + `s_fft_im[512]` float | 4 096 |
| `s_hann[480]` float | 1 920 |
| twiddles `s_tw_re/im[256]` float | 2 048 |
| `s_bitrev[512]` uint16 | 1 024 |
| `norm[49][40]` float (static in `ira_extract_features`) | 7 840 |
| **Total frontend static RAM** | **≈ 88.8 KB** |
| Model tensor arena (from embedding stage) | ≤ 40 KB (to be trimmed once measured) |
| Model flatbuffer (flash/rodata) | 13 312 B |

The 64 KB float sample buffer dominates. It exists only because the PCM16 entry
point converts the whole window up-front; if RAM is tight it can be removed by
converting per-frame inside the loop (480 floats at a time), at the cost of
converting overlapped samples twice. ESP32-S3 with PSRAM has ample headroom; on a
no-PSRAM part this is the first thing to optimise.

---

## 8. Is it safe to proceed to microphone / I2S integration?

**Yes, with two conditions.**

Justification: on every real recording tested the C++ frontend reproduces the
Python pipeline **bit-exactly** — all 1960 INT8 features identical, `q_out`
identical, detection identical. That is the strongest form of parity available,
and it is the case that matters, since deployment sees microphone audio and never
synthetic tones. The synthetic divergences were traced to float32 conditioning and
shown not to be implementation defects, with the C++ nearer the true value than TF.

Conditions before/during I2S work:

1. **Add an input-validation guard** (§6): skip inference when the window RMS is
   below a small floor, so constant/zero buffers cannot produce a wake.
2. **Re-run this harness after any FFT backend change** (notably the ESP-DSP swap)
   and after the first on-device build — the marginal cases are sensitive to
   accumulation order, and `-ffast-math` in particular must **not** be enabled, as
   it would licence reassociation that breaks the double-accumulated mean/std.

Also still open from the embedding stage: none of this has been compiled for
ESP32 yet (no ESP-IDF project scaffolding), and the tensor arena size remains an
unmeasured placeholder.

**Overall parity harness verdict:** `REAL-AUDIO PARITY: EXACT`,
`OVERALL PARITY: NEEDS REVIEW` — the latter driven entirely by the synthetic
cases diagnosed in §5. The pass criteria in the harness were **not** loosened to
force a green result.

---

## Files

| file | role |
|---|---|
| `esp32/main/ira_features.h` | frozen frontend contract |
| `esp32/main/ira_features.cpp` | implementation, isolated FFT backend |
| `tools/ira_features_host.cpp` | host driver (compiles the ESP32 source unchanged) |
| `tools/test_feature_parity.py` | parity harness |
| `esp32/ESP32_FEATURE_PARITY_REPORT.md` | this report |

Nothing was retrained, re-quantized, re-thresholded, or re-strided. The frozen
model and the source WAVs were opened read-only.
