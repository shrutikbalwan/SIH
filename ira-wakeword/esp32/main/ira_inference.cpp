// IRA wake-word — frozen V2.3 INT8 model inference wrapper (TensorFlow Lite Micro).
//
// Model loading, tensor verification, and raw INT8 inference ONLY.
// Microphone capture and STFT preprocessing are intentionally NOT implemented
// at this stage.

#include "ira_inference.h"

#include <math.h>
#include <string.h>

#include "esp_log.h"
#include "ira_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "ira_inference";

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------
// The model is tiny (13312 bytes, 21 tensors, all int8/int32). The largest
// intermediate is the first conv output: 49 x 40 x 8 = 15680 bytes. 40 KB
// gives comfortable headroom; ira_model_init() logs the ACTUAL bytes used so
// this can be trimmed once measured on hardware.
#ifndef IRA_TENSOR_ARENA_SIZE
#define IRA_TENSOR_ARENA_SIZE (40 * 1024)
#endif

// 16-byte aligned, as TFLite Micro requires for the arena.
alignas(16) static uint8_t s_tensor_arena[IRA_TENSOR_ARENA_SIZE];

// ---------------------------------------------------------------------------
// Interpreter state
// ---------------------------------------------------------------------------
namespace {

// This model uses exactly five operators:
//   CONV_2D x3, MAX_POOL_2D x2, MEAN (GlobalAveragePooling2D),
//   FULLY_CONNECTED x2, LOGISTIC (sigmoid)
// A MicroMutableOpResolver with only these keeps the binary small; adding an
// operator the model does not use would silently bloat the image.
constexpr int kNumOps = 5;
using IraOpResolver = tflite::MicroMutableOpResolver<kNumOps>;

const tflite::Model *s_model = nullptr;
tflite::MicroInterpreter *s_interpreter = nullptr;
IraOpResolver *s_resolver = nullptr;
TfLiteTensor *s_input = nullptr;
TfLiteTensor *s_output = nullptr;
bool s_ready = false;
ira_model_info_t s_info;

// Storage for placement-new so we avoid any dynamic allocation.
alignas(IraOpResolver) uint8_t s_resolver_buf[sizeof(IraOpResolver)];
alignas(tflite::MicroInterpreter) uint8_t s_interpreter_buf[sizeof(tflite::MicroInterpreter)];

bool nearly_equal(float a, float b, float tol) { return fabsf(a - b) <= tol; }

}  // namespace

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
extern "C" bool ira_model_init(void) {
  if (s_ready) {
    ESP_LOGW(TAG, "already initialised");
    return true;
  }
  memset(&s_info, 0, sizeof(s_info));

  ESP_LOGI(TAG, "=== IRA V2.3 INT8 model init ===");
  ESP_LOGI(TAG, "embedded model : %u bytes", ira_model_data_len);
  ESP_LOGI(TAG, "model sha256   : %s", ira_model_data_sha256);

  if (ira_model_data_len == 0) {
    ESP_LOGE(TAG, "embedded model is empty");
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(ira_model_data) & 0x7) != 0) {
    ESP_LOGE(TAG, "model data is not 8-byte aligned");
    return false;
  }

  // --- schema version -------------------------------------------------------
  s_model = tflite::GetModel(ira_model_data);
  if (s_model == nullptr) {
    ESP_LOGE(TAG, "GetModel returned null");
    return false;
  }
  if (s_model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "schema version %lu != supported %d",
             static_cast<unsigned long>(s_model->version()), TFLITE_SCHEMA_VERSION);
    return false;
  }
  ESP_LOGI(TAG, "schema version : %lu (OK)", static_cast<unsigned long>(s_model->version()));

  // --- op resolver ----------------------------------------------------------
  s_resolver = new (s_resolver_buf) IraOpResolver();
  if (s_resolver->AddConv2D() != kTfLiteOk ||
      s_resolver->AddMaxPool2D() != kTfLiteOk ||
      s_resolver->AddMean() != kTfLiteOk ||
      s_resolver->AddFullyConnected() != kTfLiteOk ||
      s_resolver->AddLogistic() != kTfLiteOk) {
    ESP_LOGE(TAG, "failed to register operators");
    return false;
  }

  // --- interpreter ----------------------------------------------------------
  s_interpreter = new (s_interpreter_buf)
      tflite::MicroInterpreter(s_model, *s_resolver, s_tensor_arena, IRA_TENSOR_ARENA_SIZE);

  if (s_interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors failed (arena %d bytes too small?)",
             IRA_TENSOR_ARENA_SIZE);
    return false;
  }
  const size_t used = s_interpreter->arena_used_bytes();
  ESP_LOGI(TAG, "arena          : %u / %d bytes used", (unsigned)used, IRA_TENSOR_ARENA_SIZE);

  s_input = s_interpreter->input(0);
  s_output = s_interpreter->output(0);
  if (s_input == nullptr || s_output == nullptr) {
    ESP_LOGE(TAG, "null input/output tensor");
    return false;
  }

  // --- verify INPUT ---------------------------------------------------------
  ESP_LOGI(TAG, "--- input tensor ---");
  ESP_LOGI(TAG, "type=%s dims=%d [%d,%d,%d,%d]", TfLiteTypeGetName(s_input->type),
           s_input->dims->size,
           s_input->dims->size > 0 ? s_input->dims->data[0] : -1,
           s_input->dims->size > 1 ? s_input->dims->data[1] : -1,
           s_input->dims->size > 2 ? s_input->dims->data[2] : -1,
           s_input->dims->size > 3 ? s_input->dims->data[3] : -1);
  ESP_LOGI(TAG, "scale=%.17g zero_point=%ld",
           (double)s_input->params.scale, (long)s_input->params.zero_point);

  bool ok = true;
  if (s_input->type != kTfLiteInt8) {
    ESP_LOGE(TAG, "input type is %s, expected INT8", TfLiteTypeGetName(s_input->type));
    ok = false;
  }
  if (s_input->dims->size != 4 || s_input->dims->data[0] != 1 ||
      s_input->dims->data[1] != IRA_INPUT_ROWS ||
      s_input->dims->data[2] != IRA_INPUT_COLS || s_input->dims->data[3] != 1) {
    ESP_LOGE(TAG, "input shape mismatch, expected [1,%d,%d,1]", IRA_INPUT_ROWS,
             IRA_INPUT_COLS);
    ok = false;
  }
  if (!nearly_equal(s_input->params.scale, IRA_EXPECTED_INPUT_SCALE, 1e-9f)) {
    ESP_LOGE(TAG, "input scale mismatch, expected %.17g", (double)IRA_EXPECTED_INPUT_SCALE);
    ok = false;
  }
  if (s_input->params.zero_point != IRA_EXPECTED_INPUT_ZERO_POINT) {
    ESP_LOGE(TAG, "input zero_point mismatch, expected %d", IRA_EXPECTED_INPUT_ZERO_POINT);
    ok = false;
  }
  if (s_input->bytes != (size_t)IRA_INPUT_ELEMENTS) {
    ESP_LOGE(TAG, "input bytes %u, expected %d", (unsigned)s_input->bytes,
             IRA_INPUT_ELEMENTS);
    ok = false;
  }

  // --- verify OUTPUT --------------------------------------------------------
  ESP_LOGI(TAG, "--- output tensor ---");
  ESP_LOGI(TAG, "type=%s dims=%d [%d,%d]", TfLiteTypeGetName(s_output->type),
           s_output->dims->size,
           s_output->dims->size > 0 ? s_output->dims->data[0] : -1,
           s_output->dims->size > 1 ? s_output->dims->data[1] : -1);
  ESP_LOGI(TAG, "scale=%.17g zero_point=%ld",
           (double)s_output->params.scale, (long)s_output->params.zero_point);

  if (s_output->type != kTfLiteInt8) {
    ESP_LOGE(TAG, "output type is %s, expected INT8", TfLiteTypeGetName(s_output->type));
    ok = false;
  }
  if (s_output->dims->size != 2 || s_output->dims->data[0] != 1 ||
      s_output->dims->data[1] != 1) {
    ESP_LOGE(TAG, "output shape mismatch, expected [1,1]");
    ok = false;
  }
  if (!nearly_equal(s_output->params.scale, IRA_EXPECTED_OUTPUT_SCALE, 1e-12f)) {
    ESP_LOGE(TAG, "output scale mismatch, expected %.17g", (double)IRA_EXPECTED_OUTPUT_SCALE);
    ok = false;
  }
  if (s_output->params.zero_point != IRA_EXPECTED_OUTPUT_ZERO_POINT) {
    ESP_LOGE(TAG, "output zero_point mismatch, expected %d", IRA_EXPECTED_OUTPUT_ZERO_POINT);
    ok = false;
  }

  if (!ok) {
    ESP_LOGE(TAG, "MODEL VERIFICATION FAILED — inference disabled");
    s_ready = false;
    return false;
  }

  s_info.input_scale = s_input->params.scale;
  s_info.input_zero_point = s_input->params.zero_point;
  s_info.output_scale = s_output->params.scale;
  s_info.output_zero_point = s_output->params.zero_point;
  for (int i = 0; i < 4; ++i) s_info.input_dims[i] = s_input->dims->data[i];
  for (int i = 0; i < 2; ++i) s_info.output_dims[i] = s_output->dims->data[i];
  s_info.arena_used_bytes = used;
  s_info.model_sha256 = ira_model_data_sha256;
  s_info.model_len = ira_model_data_len;

  ESP_LOGI(TAG, "--- decision rule (frozen) ---");
  ESP_LOGI(TAG, "WAKE iff q_out >= %d  (p >= %.10f)", IRA_Q_THRESHOLD,
           (double)((IRA_Q_THRESHOLD - s_output->params.zero_point) * s_output->params.scale));
  ESP_LOGI(TAG, "MODEL VERIFICATION PASSED — ready");

  s_ready = true;
  return true;
}

extern "C" void ira_model_deinit(void) {
  if (s_interpreter != nullptr) {
    s_interpreter->~MicroInterpreter();
    s_interpreter = nullptr;
  }
  if (s_resolver != nullptr) {
    s_resolver->~IraOpResolver();
    s_resolver = nullptr;
  }
  s_model = nullptr;
  s_input = nullptr;
  s_output = nullptr;
  s_ready = false;
  memset(&s_info, 0, sizeof(s_info));
}

extern "C" bool ira_model_is_ready(void) { return s_ready; }

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
extern "C" int8_t ira_run_inference_flat(const int8_t *features, size_t len) {
  if (!s_ready) {
    ESP_LOGE(TAG, "ira_run_inference called before successful init");
    return IRA_Q_INVALID;
  }
  if (features == nullptr || len != (size_t)IRA_INPUT_ELEMENTS) {
    ESP_LOGE(TAG, "bad feature buffer (len=%u, expected %d)", (unsigned)len,
             IRA_INPUT_ELEMENTS);
    return IRA_Q_INVALID;
  }

  memcpy(s_input->data.int8, features, (size_t)IRA_INPUT_ELEMENTS);

  if (s_interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke failed");
    return IRA_Q_INVALID;
  }
  return s_output->data.int8[0];
}

extern "C" int8_t ira_run_inference(
    const int8_t features[IRA_INPUT_ROWS][IRA_INPUT_COLS]) {
  return ira_run_inference_flat(reinterpret_cast<const int8_t *>(features),
                                (size_t)IRA_INPUT_ELEMENTS);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
extern "C" float ira_dequantize(int8_t q_out) {
  const float scale = s_ready ? s_output->params.scale : IRA_EXPECTED_OUTPUT_SCALE;
  const int32_t zp = s_ready ? s_output->params.zero_point : IRA_EXPECTED_OUTPUT_ZERO_POINT;
  return ((int32_t)q_out - zp) * scale;
}

extern "C" const ira_model_info_t *ira_model_get_info(void) {
  return s_ready ? &s_info : nullptr;
}
