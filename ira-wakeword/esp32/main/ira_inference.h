// IRA wake-word — frozen V2.3 INT8 model inference wrapper (TensorFlow Lite Micro).
//
// FROZEN CONTRACT — do not change any of these without re-freezing the model:
//   model   : cnn/models/ira_cnn_v2_3_int8.tflite
//   sha256  : b9554c054bc9a57a00e874824e5a4ab9c3254b4505ca8a3ec67835d193103de0
//   input   : [1,49,40,1] INT8, scale 0.06078097224235535, zero_point 49
//   output  : [1,1]       INT8, scale 0.00390625,          zero_point -128
//   decide  : WAKE iff q_out >= -25
//   stride  : 100 ms (owned by the caller, not by this module)
//
// This module performs ONLY model loading, tensor verification, and raw INT8
// inference. Microphone capture and STFT feature extraction are deliberately
// NOT implemented yet.

#ifndef IRA_INFERENCE_H_
#define IRA_INFERENCE_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Frozen model contract
// ---------------------------------------------------------------------------
#define IRA_INPUT_ROWS 49  // STFT frames
#define IRA_INPUT_COLS 40  // frequency bins
#define IRA_INPUT_ELEMENTS (IRA_INPUT_ROWS * IRA_INPUT_COLS)

// Frozen firmware decision boundary. Derived as
//   q_min = ceil(0.40 / output_scale + output_zero_point)
//         = ceil(0.40 / 0.00390625 + (-128)) = ceil(-25.6) = -25
// q_out == -25 fires (p = 0.40234375); q_out == -26 does not (p = 0.3984375).
#define IRA_Q_THRESHOLD (-25)

// Expected quantization parameters (verified at init against the model).
#define IRA_EXPECTED_INPUT_SCALE 0.06078097224235535f
#define IRA_EXPECTED_INPUT_ZERO_POINT 49
#define IRA_EXPECTED_OUTPUT_SCALE 0.00390625f
#define IRA_EXPECTED_OUTPUT_ZERO_POINT (-128)

// Returned by ira_run_inference() when the model is not initialised or the
// call fails. -128 is the most-negative INT8 and is far below the threshold,
// so a failed inference can never be mistaken for a detection.
#define IRA_Q_INVALID ((int8_t)-128)

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Loads the embedded model, checks the schema version, builds the interpreter,
// allocates tensors, and verifies input/output dtype, shape and quantization
// against the frozen contract above. Logs the ACTUAL scale/zero-point read
// from the model. Returns false on any mismatch — the caller must not run
// inference in that case.
bool ira_model_init(void);

// Frees interpreter state. Safe to call when uninitialised.
void ira_model_deinit(void);

// True once ira_model_init() has succeeded.
bool ira_model_is_ready(void);

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

// Runs one inference on an already-quantized 49x40 INT8 feature window.
//
// The caller is responsible for producing features with the EXACT V2.3
// pipeline (STFT 480/320/512, first 40 bins, log(|X| + 1e-6), per-window
// z-score) and quantizing with the model's input scale/zero-point:
//     q = clamp(round(f / input_scale + input_zero_point), -128, 127)
//
// Returns the raw INT8 output, or IRA_Q_INVALID on failure.
int8_t ira_run_inference(const int8_t features[IRA_INPUT_ROWS][IRA_INPUT_COLS]);

// Flat-buffer variant; `len` must be IRA_INPUT_ELEMENTS (row-major, 49x40).
int8_t ira_run_inference_flat(const int8_t *features, size_t len);

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

// The frozen decision. Pure integer comparison — no float conversion.
static inline bool ira_detected(int8_t q_out) { return q_out >= IRA_Q_THRESHOLD; }

// Dequantized probability. FOR LOGGING/DEBUG ONLY — never use this for the
// wake decision; ira_detected() is the contract.
float ira_dequantize(int8_t q_out);

// ---------------------------------------------------------------------------
// Introspection (populated by ira_model_init)
// ---------------------------------------------------------------------------
typedef struct {
  float input_scale;
  int32_t input_zero_point;
  float output_scale;
  int32_t output_zero_point;
  int32_t input_dims[4];
  int32_t output_dims[2];
  size_t arena_used_bytes;
  const char *model_sha256;
  unsigned int model_len;
} ira_model_info_t;

// Returns model metadata after a successful init, or NULL if not ready.
const ira_model_info_t *ira_model_get_info(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IRA_INFERENCE_H_
