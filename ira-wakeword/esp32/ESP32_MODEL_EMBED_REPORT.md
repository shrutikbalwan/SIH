# ESP32-S3 Model Embedding Report — IRA V2.3 INT8

**Stage:** model embedding + inference API only.
**Status:** ✅ **COMPLETE**
**Audio capture / STFT preprocessing:** ❌ **NOT STARTED (deliberately out of scope)**

---

## Task 1 — Model verification

| check | expected | actual | result |
|---|---|---|---|
| File exists | `cnn/models/ira_cnn_v2_3_int8.tflite` | present | ✅ |
| Size | 13312 bytes | **13312** | ✅ |
| SHA-256 | `b9554c05…03de0` | `b9554c054bc9a57a00e874824e5a4ab9c3254b4505ca8a3ec67835d193103de0` | ✅ |
| Flatbuffer identifier | `TFL3` | `TFL3` | ✅ |
| Input dtype / shape | INT8 `[1,49,40,1]` | INT8 `[1,49,40,1]` | ✅ |
| Input quantization | scale 0.06078097224235535, zp 49 | scale 0.06078097224235535, zp 49 | ✅ |
| Output dtype / shape | INT8 `[1,1]` | INT8 `[1,1]` | ✅ |
| Output quantization | scale 0.00390625, zp −128 | scale 0.00390625, zp −128 | ✅ |
| Full INT8 | no float fallback | **0 float32 tensors** | ✅ |

The `.tflite` file was opened read-only and **not modified**.

### Operators in the model

`CONV_2D` ×3, `MAX_POOL_2D` ×2, `MEAN` (GlobalAveragePooling2D), `FULLY_CONNECTED` ×2, `LOGISTIC` — **5 unique operators, 9 nodes**.

(The desktop interpreter also lists a `DELEGATE` node; that is the host XNNPACK delegate and does not exist on ESP32.)

---

## Task 2 — C array conversion

Generator: [`tools/convert_tflite_to_c.py`](../tools/convert_tflite_to_c.py) — reproducible, refuses to emit anything if the SHA-256 or size does not match the frozen model, and round-trip-verifies the emitted array back to bytes.

| output | size |
|---|---|
| `esp32/main/ira_model_data.h` | 739 bytes |
| `esp32/main/ira_model_data.cc` | 86 370 bytes |

```c
extern const unsigned char ira_model_data[];
extern const unsigned int  ira_model_data_len;   // = 13312
extern const char          ira_model_data_sha256[];
```

- **Generated array length: 13312** — identical to the source model.
- **Round-trip verified:** the emitted bytes re-hash to `b9554c05…03de0`, byte-identical to the `.tflite`.
- **Alignment: 16 bytes** (`__attribute__((aligned(16)))`, with an `alignas` fallback). TFLite Micro requires ≥8-byte alignment or `GetModel()` faults.

Regenerate at any time with:

```bash
python tools/convert_tflite_to_c.py          # write
python tools/convert_tflite_to_c.py --check  # verify without rewriting
```

---

## Task 3 — Inference wrapper

Files: `esp32/main/ira_inference.h`, `esp32/main/ira_inference.cpp` (TensorFlow Lite Micro).

### `ira_model_init()` performs, in order

1. Checks the embedded array is non-empty and ≥8-byte aligned.
2. `tflite::GetModel()` and **schema version check** against `TFLITE_SCHEMA_VERSION`.
3. Builds a `MicroMutableOpResolver<5>` registering **exactly** the five operators the model uses — nothing more, to avoid bloating the image.
4. Constructs the `MicroInterpreter` and calls `AllocateTensors()`.
5. Verifies **input**: type INT8, shape `[1,49,40,1]`, scale, zero-point, byte size.
6. Verifies **output**: type INT8, shape `[1,1]`, scale, zero-point.
7. **Logs the ACTUAL scale and zero-point read from the model** (not the constants), plus arena bytes used.
8. Returns `false` on any mismatch, leaving the module not-ready so inference is refused.

No dynamic allocation: the resolver and interpreter are placement-new'd into static buffers, and the tensor arena is a static 16-byte-aligned array.

### Public API

```c
bool    ira_model_init(void);
void    ira_model_deinit(void);
bool    ira_model_is_ready(void);

int8_t  ira_run_inference(const int8_t features[49][40]);
int8_t  ira_run_inference_flat(const int8_t *features, size_t len);

static inline bool ira_detected(int8_t q_out) { return q_out >= IRA_Q_THRESHOLD; }

float   ira_dequantize(int8_t q_out);                 // logging/debug ONLY
const ira_model_info_t *ira_model_get_info(void);
```

### Frozen decision

```c
#define IRA_Q_THRESHOLD (-25)
static inline bool ira_detected(int8_t q_out) { return q_out >= IRA_Q_THRESHOLD; }
```

**Pure integer comparison — the decision never converts to float.**

Derivation, from output scale and zero-point only:

```
q_min = ceil(0.40 / 0.00390625 + (-128)) = ceil(-25.6) = -25
q = -25  ->  p = 0.4023437500   fires
q = -26  ->  p = 0.3984375000   does not fire
```

`ira_dequantize()` exists solely for logging and is explicitly documented as never valid for the wake decision.

### Failure safety

`IRA_Q_INVALID` is `-128`, the most-negative INT8. A failed or uninitialised inference therefore returns a value far **below** the threshold and can never be misread as a detection.

### Tensor arena

`IRA_TENSOR_ARENA_SIZE` defaults to **40 KB**. The largest intermediate is the first conv output (49 × 40 × 8 = 15 680 bytes), so this has comfortable headroom. `ira_model_init()` logs `arena_used_bytes`, so the figure should be **measured on hardware and trimmed** — the 40 KB is a safe placeholder, not a measured value.

---

## Frozen contract summary

| item | value |
|---|---|
| Model | `ira_cnn_v2_3_int8.tflite` |
| SHA-256 | `b9554c054bc9a57a00e874824e5a4ab9c3254b4505ca8a3ec67835d193103de0` |
| Model bytes | 13312 |
| Embedded array length | 13312 |
| Input | INT8 `[1,49,40,1]`, scale 0.06078097224235535, zp 49 |
| Output | INT8 `[1,1]`, scale 0.00390625, zp −128 |
| **Threshold** | **`q_out >= -25`** (p ≥ 0.40234375) |
| Primary stride | 100 ms (caller's responsibility, not this module's) |

Nothing was retrained, re-quantized, re-thresholded, or re-strided.

---

## Stage status

| step | status |
|---|---|
| `.tflite` verification | ✅ complete |
| `.tflite` → C array | ✅ complete, round-trip verified |
| TFLite Micro model loading | ✅ implemented |
| Schema + tensor verification | ✅ implemented |
| Raw INT8 inference API | ✅ implemented |
| Microphone capture | ⛔ not started |
| STFT preprocessing | ⛔ not started |

**MODEL EMBEDDING STEP: COMPLETE. STOPPED HERE as instructed.**

### Not yet done — required before this runs on hardware

1. **Not compiled.** These sources have not been built against ESP-IDF or `esp-tflite-micro`; there is no ESP-IDF project scaffolding (`CMakeLists.txt`, `sdkconfig`) yet. Compilation may surface API differences depending on the `esp-tflite-micro` version pinned.
2. **Arena size unmeasured** — trim once `arena_used_bytes` is observed on device.
3. **No feature extraction.** The caller must eventually produce the exact V2.3 pipeline: STFT `frame_length=480, frame_step=320, fft_length=512, pad_end=false` → `|X|` → `log(x + 1e-6)` → first 40 bins → per-window z-score over all 49×40 values → quantize with the input scale/zero-point. **MFCC, mel spectrogram, and MicroFrontend must not be used** — they would not reproduce the training features.
