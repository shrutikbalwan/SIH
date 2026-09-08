# Real-Human Development Set — Analysis Plan

**Status: PLAN ONLY. Nothing here has been executed.** No recordings exist yet.
No model is to be trained, modified, quantized, re-thresholded, or re-strided by
anything in this document.

---

## 0. Purpose

The frozen V2.3 INT8 candidate misses ~11% of real "Ira" utterances
(89.17% TPR, 100 ms stride, `q_out >= -25`). Human review confirmed 46 of the
50 hardest failures were **valid, intelligible "Ira"**, so the deficiency is
real. Failures were **strongly session-concentrated**, but the old collection
recorded no speaker, device, distance, or room metadata — every factor was
confounded, so nothing could be attributed.

This plan says exactly what will be computed once the new set exists, **declared
in advance** so the analysis cannot be steered by its own results.

### Hypotheses to discriminate

| | hypothesis | signature |
|---|---|---|
| **A** | Speaker-specific difficulty | Speaker variance large after conditioning on distance/room/device |
| **B** | Distance / SNR / reverberation | Monotone TPR decline with distance, consistent across speakers |
| **C** | Device / channel | Device effect large with speaker and room held fixed |
| **D** | Session-specific artifact | Session variance large *within* the same speaker+room combination |
| **E** | Broadly distributed failure | No factor dominates; residual variance is the bulk |
| **F** | Temporal alignment sensitivity | Fires at 10 ms but not 100 ms; narrow firing regions |

**A vs D is the question the old data could not answer.** It is answerable only
because this design repeats the same speaker+room in separate sessions.

---

## 1. Frozen evaluation configuration

| item | value |
|---|---|
| Model | `cnn/models/ira_cnn_v2_3_int8.tflite` |
| SHA-256 | `b9554c054bc9a57a00e874824e5a4ab9c3254b4505ca8a3ec67835d193103de0` (verify before every run) |
| Decision | `WAKE iff q_out >= -25` (≡ p ≥ 0.40234375) |
| Preprocessing | 16 kHz mono, 1 s window, STFT 480/320/512, first 40 bins, `log(|·|+1e-6)`, per-window z-score, `[1,49,40,1]`, int8 input via TFLite scale/zero-point |

**These are frozen.** The stride is the *only* varied quantity, and it is varied
for diagnosis, never for selection.

Any recording not natively 16 kHz is resampled **in the evaluation script only**;
the source WAV is never rewritten.

---

## 2. Stride diagnostic (script to be written when WAVs exist)

Proposed: `cnn/eval_real_human_dev_strides.py` — **do not create or run it until
recordings exist and the audit passes.**

Score every complete recording at three strides, sliding a 1 s window across the
whole file, with a final window anchored at `len − 16000` so the tail is covered.
Files under 1 s are zero-padded once.

| stride | role |
|---|---|
| **100 ms** | **PRIMARY** — matches the frozen deployment policy |
| 50 ms | secondary diagnostic |
| 10 ms | diagnostic only — establishes the model's ceiling |

A recording fires if **any** window reaches `q_out >= -25`.

### Per-recording output

`recording_id`, `filename`, and all metadata factors, plus:

- `fires_100ms`, `fires_50ms`, `fires_10ms` (bool)
- `first_fire_offset_100ms` / `_50ms` / `_10ms` (s, blank if none)
- `max_q_out_100ms` / `_50ms` / `_10ms`
- `max_score_*` (dequantized)
- `t_max_score_*`
- `n_firing_windows_*`
- `n_windows_evaluated_*`
- `firing_region_max_width_ms`, `firing_region_total_ms`, `n_firing_regions`
  — computed at 1 ms probe resolution **only for recordings that fail at 100 ms
  but fire at 10 ms** (the Group-A analogue), since that scan is expensive

Outputs: `cnn/real_human_dev_stride_results.csv`,
`cnn/real_human_dev_stride_report.json`.

**Reporting rule:** the headline TPR is the **100 ms** figure. The 10 ms figure
is never the headline — more windows means more chances to fire, at 10× the
inference cost and 10× the false-accept exposure, and this positive-only set
measures none of that cost.

---

## 3. Descriptive analysis (always reported)

Grouped TPR **with counts and exact Clopper–Pearson 95% CIs**, at 100 ms:

1. Overall
2. By `speaker_id`
3. By `distance_m`
4. By `room_id` and `room_condition`
5. By `device_id`
6. By `session_id`
7. By `noise_condition`
8. **Speaker × distance** (does distance hurt everyone, or only some speakers?)
9. **Speaker × room** (does room hurt everyone, or only some speakers?)
10. **Same speaker+room across sessions** — the A-vs-D discriminator

Also report, for the same groupings: median `max_score_100ms`, and the
100 ms → 50 ms → 10 ms recovery counts.

> **Marginal percentages are descriptive only.** With an unbalanced design a
> marginal difference can be produced entirely by another factor. No causal
> claim is made from a group percentage.

---

## 4. Inferential analysis (sample-size gated)

Outcome: `fires_100ms` (binary, per recording).

### 4.1 Primary — mixed-effects logistic regression

```
fires_100ms ~ distance_z + room_condition + device_id
              + (1 | speaker_id) + (1 | session_id)
```

- Fixed effects: distance (continuous, standardised), room condition, device.
- Random intercepts: **speaker** and **session**.
- Session nested within speaker×room, which is what makes the speaker and
  session variance components separately identifiable.

Report fixed-effect odds ratios with 95% CIs, and the **variance components**
σ²_speaker and σ²_session with the intraclass correlations. The comparison of
σ²_speaker against σ²_session is the direct test of **A vs D**.

Fit with `statsmodels` (`BinomialBayesMixedGLM` or GEE) or `pymer4`/R `lme4` if
available. If the mixed model fails to converge, fall back to §4.2 and say so
explicitly rather than reporting an unconverged fit.

**Gate:** requires **≥6 speakers** and **≥8 sessions**, with ≥1 firing and ≥1
non-firing outcome in a reasonable number of cells. Below that, random-effect
variances are not estimable — report descriptives only.

### 4.2 Fallback — fixed-effects logistic regression

```
fires_100ms ~ distance_z + C(room_condition) + C(device_id) + C(speaker_id)
```

Cluster-robust standard errors by session. Speaker as fixed effects consumes
df but is stable at small sample sizes.

### 4.3 Alignment sub-analysis (hypothesis F)

Restricted to recordings that **fail at 100 ms**:

```
fires_10ms ~ distance_z + C(room_condition) + C(device_id) + (1 | speaker_id)
```

Plus the distribution of `firing_region_max_width_ms`, compared against the
old finding (median ≈ 10 ms, all 40 narrower than the 100 ms stride). If narrow
firing regions recur here and are **not** associated with any recorded factor,
alignment sensitivity is an intrinsic model property, not an environmental one.

### 4.4 Multiplicity

The primary model is pre-specified and reported without correction. Every
grouped comparison in §3 is exploratory; where several are compared, apply
Benjamini–Hochberg and label the results exploratory. **Do not** promote an
exploratory finding to a conclusion after seeing it.

---

## 5. Decision rules — declared in advance

| finding | reading |
|---|---|
| σ²_speaker ≫ σ²_session, distance OR small | **A** — speaker-specific difficulty |
| Distance OR strong and consistent within speakers | **B** — distance/SNR/reverb |
| Device OR strong with speaker+room fixed | **C** — device/channel |
| σ²_session ≫ σ²_speaker | **D** — session artifact; investigate capture procedure before touching the model |
| No component dominates; residual is the bulk | **E** — broadly distributed model limitation |
| Many 100 ms failures fire at 10 ms with narrow regions | **F** — alignment sensitivity, orthogonal to the rest |

More than one may hold; report all that the evidence supports.

### What each verdict implies for V2.9

- **A** → positive-speaker diversity is the gap. Data problem, not augmentation.
- **B** → a *far-field* intervention (reverberation / RIR convolution), **not**
  a plain SNR change. Note that V2.8 already tested extending the hard SNR band
  to −10 dB and **regressed** (matched-recall Speech FPR 3.92% vs 3.25%), so
  plain SNR widening is not the answer.
- **C** → channel augmentation (device impulse response, codec, AGC).
- **D** → fix the capture protocol first; no model change is warranted.
- **E** → augmentation tuning is exhausted; consider architecture or a larger
  positive corpus. Five consecutive single-knob failures (V2.4–V2.8) already
  point here.
- **F** → an inference-policy or time-placement question. Address separately;
  do **not** bundle it into a V2.9 SNR/room experiment.

**Whatever the verdict, V2.9 changes exactly ONE variable and is judged on the
frozen `expanded_val_v2_3_frozen.npz` frontier — never on this development set.**

---

## 6. Data hygiene (binding)

- This set is **development/diagnostic data from creation**. It informs V2.9
  design and therefore can **never** be the untouched final positive benchmark.
- Final positive claims require **another** newly collected, never-analysed
  real-human set.
- **Do not** add these recordings to training.
- **Do not** select checkpoints, thresholds, or strides against them.
- Still consumed and unusable for final selection: the old 831 real-human
  recordings, and the one-shot `unseen_test` speech negatives.
- The positive-only nature of this set means it measures **recall only**. False
  accepts and FAPH require a separate continuous-negative recording, which does
  not exist yet.

---

## 7. Execution order

1. Collect per `dataset/real_human_dev/README.md`.
2. `python cnn/audit_real_human_dev_dataset.py` — must reach PASS or
   PASS_WITH_WARNINGS.
3. Write and run the stride diagnostic (§2).
4. Descriptive analysis (§3).
5. Inferential analysis (§4) **if** the sample-size gate is met.
6. Assign a verdict from §5 and specify **one** V2.9 intervention.
7. **Stop.** Do not train V2.9 in the same step as diagnosing it.
