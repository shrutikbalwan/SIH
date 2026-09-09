// IRA wake-word — ESP32-S3 boot self-test.
//
// NO MICROPHONE / I2S CAPTURE. This stage validates, on real hardware:
//   1. TFLite Micro initialises and AllocateTensors succeeds
//   2. the on-device tensor metadata matches the frozen model
//   3. the C++ feature frontend reproduces the desktop reference bit-for-bit
//   4. q_out and the detection decision match the desktop reference
//   5. the invalid-audio guard rejects an all-zero window BEFORE inference
//   6. memory and timing are measured (never invented)
//
// Frozen: model bytes, preprocessing math, quantization constants, q_out >= -25.

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "ira_audio_guard.h"
#include "ira_features.h"
#include "ira_inference.h"
#include "ira_model_data.h"
#include "ira_selftest_data.h"

static const char *TAG = "ira_selftest";

#define TIMING_ITERATIONS 20

// Static so we never risk the main task stack with 32 KB + 2 KB buffers.
static int16_t s_pcm[IRA_SELFTEST_SAMPLES];
static int8_t s_feat[IRA_NUM_FRAMES][IRA_NUM_BINS];

// ---------------------------------------------------------------------------
struct FeatureCompare {
  int match;
  int mismatch;
  int max_abs_diff;
};

static FeatureCompare compare_features(const int8_t got[IRA_NUM_FRAMES][IRA_NUM_BINS],
                                       const int8_t *expected) {
  FeatureCompare c = {0, 0, 0};
  for (int f = 0; f < IRA_NUM_FRAMES; ++f) {
    for (int k = 0; k < IRA_NUM_BINS; ++k) {
      const int a = got[f][k];
      const int b = expected[f * IRA_NUM_BINS + k];
      const int d = a > b ? a - b : b - a;
      if (d == 0) {
        c.match++;
      } else {
        c.mismatch++;
        if (d > c.max_abs_diff) c.max_abs_diff = d;
      }
    }
  }
  return c;
}

// Returns true if this vector passed every check.
static bool run_vector(const char *name, const int8_t *expected_feat,
                       int expected_q, int expected_det) {
  printf("\n--- VECTOR: %s ---\n", name);

  const ira_audio_status_t st = ira_audio_window_check(s_pcm);
  printf("GUARD_STATUS = %s\n", ira_audio_status_str(st));
  if (st != IRA_AUDIO_OK) {
    printf("RESULT = REJECTED_BEFORE_INFERENCE\n");
    return false;
  }

  if (!ira_extract_features(s_pcm, s_feat)) {
    printf("ERROR: feature extraction failed\n");
    return false;
  }

  const FeatureCompare c = compare_features(s_feat, expected_feat);
  printf("FEATURE_MATCH = %d/%d\n", c.match, IRA_SELFTEST_FEATURES);
  printf("FEATURE_MISMATCH = %d\n", c.mismatch);
  printf("MAX_INT8_FEATURE_DIFF = %d\n", c.max_abs_diff);

  const int8_t q = ira_run_inference(s_feat);
  const bool det = ira_detected(q);

  printf("EXPECTED_Q_OUT = %d\n", expected_q);
  printf("ESP32_Q_OUT = %d\n", (int)q);
  printf("Q_OUT_MATCH = %s\n", ((int)q == expected_q) ? "YES" : "NO");
  printf("EXPECTED_DETECTION = %s\n", expected_det ? "YES" : "NO");
  printf("ESP32_DETECTION = %s\n", det ? "YES" : "NO");
  printf("DETECTION_MATCH = %s\n", (det == (expected_det != 0)) ? "YES" : "NO");
  printf("SCORE_DEBUG_ONLY = %.6f\n", (double)ira_dequantize(q));

  // Parity is asserted on the FEATURES, not merely on the final decision.
  const bool ok = (c.mismatch == 0) && ((int)q == expected_q) &&
                  (det == (expected_det != 0));
  printf("VECTOR_RESULT = %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------------
static void run_timing(void) {
  printf("\n=== TIMING (%d iterations, ESP32-S3) ===\n", TIMING_ITERATIONS);

  int64_t feat_sum = 0, feat_max = 0;
  int64_t inf_sum = 0, inf_max = 0;
  int64_t tot_sum = 0, tot_max = 0;

  for (int i = 0; i < TIMING_ITERATIONS; ++i) {
    const int64_t t0 = esp_timer_get_time();
    ira_extract_features(s_pcm, s_feat);
    const int64_t t1 = esp_timer_get_time();
    (void)ira_run_inference(s_feat);
    const int64_t t2 = esp_timer_get_time();

    const int64_t f = t1 - t0, n = t2 - t1, t = t2 - t0;
    feat_sum += f; inf_sum += n; tot_sum += t;
    if (f > feat_max) feat_max = f;
    if (n > inf_max) inf_max = n;
    if (t > tot_max) tot_max = t;
  }

  const double fm = feat_sum / (double)TIMING_ITERATIONS / 1000.0;
  const double im = inf_sum / (double)TIMING_ITERATIONS / 1000.0;
  const double tm = tot_sum / (double)TIMING_ITERATIONS / 1000.0;

  printf("FEATURE_TIME_MEAN_MS = %.3f\n", fm);
  printf("FEATURE_TIME_MAX_MS = %.3f\n", feat_max / 1000.0);
  printf("INFERENCE_TIME_MEAN_MS = %.3f\n", im);
  printf("INFERENCE_TIME_MAX_MS = %.3f\n", inf_max / 1000.0);
  printf("TOTAL_TIME_MEAN_MS = %.3f\n", tm);
  printf("TOTAL_TIME_MAX_MS = %.3f\n", tot_max / 1000.0);
  printf("STRIDE_BUDGET_MS = 100.000\n");
  printf("BUDGET_UTILISATION_MEAN_PCT = %.2f\n", tm / 100.0 * 100.0);
  printf("BUDGET_UTILISATION_MAX_PCT = %.2f\n", (tot_max / 1000.0) / 100.0 * 100.0);
  printf("REALTIME_FEASIBLE = %s\n", (tot_max / 1000.0) < 100.0 ? "YES" : "NO");
}

// ---------------------------------------------------------------------------
static void report_memory(size_t heap_before, size_t heap_after) {
  printf("\n=== MEMORY ===\n");
  printf("MODEL_FLASH_BYTES = %u\n", ira_model_data_len);

  const ira_model_info_t *info = ira_model_get_info();
  if (info != nullptr) {
    printf("ARENA_USED_BYTES = %u\n", (unsigned)info->arena_used_bytes);
  }

  printf("FRONTEND_STATIC_RAM_BYTES = %u\n",
         (unsigned)(sizeof(float) * IRA_WINDOW_SAMPLES +          // s_samples
                    sizeof(float) * IRA_NUM_FRAMES * IRA_NUM_BINS + // s_logmag
                    sizeof(float) * IRA_FFT_LENGTH * 2 +           // fft re/im
                    sizeof(float) * IRA_FRAME_LENGTH +             // hann
                    sizeof(float) * IRA_FFT_LENGTH +               // twiddles
                    sizeof(uint16_t) * IRA_FFT_LENGTH +            // bitrev
                    sizeof(float) * IRA_NUM_FRAMES * IRA_NUM_BINS));// norm
  printf("SELFTEST_STATIC_RAM_BYTES = %u\n", (unsigned)(sizeof(s_pcm) + sizeof(s_feat)));

  printf("HEAP_INTERNAL_FREE_BEFORE_INIT = %u\n", (unsigned)heap_before);
  printf("HEAP_INTERNAL_FREE_AFTER_INIT = %u\n", (unsigned)heap_after);
  printf("HEAP_CONSUMED_BY_INIT = %d\n", (int)((int64_t)heap_before - (int64_t)heap_after));
  printf("HEAP_INTERNAL_LARGEST_FREE_BLOCK = %u\n",
         (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

  const size_t psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
  printf("PSRAM_PRESENT = %s\n", psram > 0 ? "YES" : "NO");
  printf("PSRAM_TOTAL_BYTES = %u\n", (unsigned)psram);
  printf("PSRAM_RELIED_UPON = NO\n");

  printf("MAIN_TASK_STACK_HIGH_WATER_BYTES = %u\n",
         (unsigned)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t)));
}

// ---------------------------------------------------------------------------
extern "C" void app_main(void) {
  printf("\n\n");
  printf("================================================================\n");
  printf("IRA WAKE-WORD — ESP32-S3 BOOT SELF-TEST (no microphone)\n");
  printf("================================================================\n");

  esp_chip_info_t chip;
  esp_chip_info(&chip);
  printf("CHIP_CORES = %d\n", chip.cores);
  printf("CHIP_REVISION = %d\n", chip.revision);
  printf("IDF_VERSION = %s\n", esp_get_idf_version());

  const size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

  // ---- Task 3: model init + metadata ------------------------------------
  printf("\n=== MODEL INIT ===\n");
  printf("MODEL_SHA256 = %s\n", ira_model_data_sha256);
  printf("MODEL_LENGTH_BYTES = %u\n", ira_model_data_len);

  if (!ira_model_init()) {
    printf("\nMODEL_INIT = FAIL\n");
    printf("VERDICT = HARDWARE SELF-TEST FAIL\n");
    return;
  }
  printf("MODEL_INIT = OK\n");

  const size_t heap_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

  const ira_model_info_t *info = ira_model_get_info();
  if (info != nullptr) {
    printf("INPUT_SHAPE = [%d,%d,%d,%d]\n", (int)info->input_dims[0],
           (int)info->input_dims[1], (int)info->input_dims[2], (int)info->input_dims[3]);
    printf("INPUT_SCALE = %.17g\n", (double)info->input_scale);
    printf("INPUT_ZERO_POINT = %d\n", (int)info->input_zero_point);
    printf("OUTPUT_SHAPE = [%d,%d]\n", (int)info->output_dims[0], (int)info->output_dims[1]);
    printf("OUTPUT_SCALE = %.17g\n", (double)info->output_scale);
    printf("OUTPUT_ZERO_POINT = %d\n", (int)info->output_zero_point);
    printf("ARENA_USED_BYTES = %u\n", (unsigned)info->arena_used_bytes);
  }
  printf("Q_THRESHOLD = %d\n", IRA_Q_THRESHOLD);

  ira_features_init();

  // ---- Task 5: invalid-audio guard (checked BEFORE any inference) --------
  printf("\n=== INVALID-AUDIO GUARD ===\n");
  memset(s_pcm, 0, sizeof(s_pcm));
  const ira_audio_status_t zero_st = ira_audio_window_check(s_pcm);
  const bool zero_valid = ira_audio_window_valid(s_pcm);
  printf("ALL_ZERO_STATUS = %s\n", ira_audio_status_str(zero_st));
  printf("ALL_ZERO_VALID = %s\n", zero_valid ? "true" : "false");
  printf("ALL_ZERO_REJECTED = %s\n", !zero_valid ? "YES" : "NO");
  const bool guard_zero_ok = !zero_valid;

  for (int i = 0; i < IRA_SELFTEST_SAMPLES; ++i) s_pcm[i] = 1234;
  const ira_audio_status_t const_st = ira_audio_window_check(s_pcm);
  printf("CONSTANT_STATUS = %s\n", ira_audio_status_str(const_st));
  const bool guard_const_ok = (const_st == IRA_AUDIO_INVALID_CONSTANT);
  printf("GUARD_RESULT = %s\n", (guard_zero_ok && guard_const_ok) ? "PASS" : "FAIL");
  printf("NOTE: no minimum-RMS speech threshold is applied; it must be derived\n");
  printf("      from real microphone data, not from development recordings.\n");

  // ---- Task 4: feature + inference parity --------------------------------
  printf("\n=== FEATURE + INFERENCE PARITY ===\n");

  ira_selftest_fill_lcg(s_pcm);
  printf("LCG_PCM_FIRST5 = %d %d %d %d %d\n", (int)s_pcm[0], (int)s_pcm[1],
         (int)s_pcm[2], (int)s_pcm[3], (int)s_pcm[4]);
  const bool ok_lcg = run_vector("lcg (deterministic, expect NO detection)",
                                 ira_selftest_lcg_features,
                                 IRA_SELFTEST_LCG_Q_OUT, IRA_SELFTEST_LCG_DETECTED);

  memcpy(s_pcm, ira_selftest_speech_pcm, sizeof(s_pcm));
  const bool ok_speech = run_vector("speech (real recording, expect DETECTION)",
                                    ira_selftest_speech_features,
                                    IRA_SELFTEST_SPEECH_Q_OUT,
                                    IRA_SELFTEST_SPEECH_DETECTED);

  // ---- Task 6: timing + memory -------------------------------------------
  run_timing();
  report_memory(heap_before, heap_after);

  // ---- Verdict ------------------------------------------------------------
  const bool pass = ok_lcg && ok_speech && guard_zero_ok && guard_const_ok;
  printf("\n================================================================\n");
  printf("VERDICT = %s\n", pass ? "HARDWARE SELF-TEST PASS" : "HARDWARE SELF-TEST FAIL");
  printf("================================================================\n");

  while (true) vTaskDelay(pdMS_TO_TICKS(10000));
}
