// IRA wake-word — V2.3 feature frontend implementation.
//
// See ira_features.h for the frozen contract. No dynamic allocation.
//
// FFT BACKEND
// -----------
// Default: a self-contained iterative radix-2 complex FFT (see kFftBackendName).
// It is deliberately isolated behind ira_fft512() so ESP-DSP can be swapped in
// without touching any of the framing/log/normalise/quantise logic.
//
// To switch to ESP-DSP on ESP32-S3, build with -DIRA_USE_ESP_DSP=1 and link
// esp-dsp. dsps_fft2r_fc32 uses the SAME unnormalized forward-transform
// convention as numpy/TensorFlow rfft, so no scaling change is needed --
// but the parity harness MUST be re-run after the swap to confirm it.
//
// NORMALIZATION CONVENTION (critical)
// -----------------------------------
// tf.signal.stft / np.fft.rfft apply NO scaling to the forward transform:
//     X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)
// Any FFT that divides by N (or sqrt(N)) would shift every log-magnitude by a
// constant. Note that a *constant* offset would actually be removed by the
// per-window z-score -- but a per-bin or sqrt(N) scaling would not, and relying
// on that cancellation would be fragile. We therefore keep the unnormalized
// convention explicitly.
//
// 480 -> 512 HANDLING
// -------------------
// Each frame takes 480 samples, multiplies by the 480-point periodic Hann
// window, writes them into the low 480 slots of a 512-point complex buffer, and
// ZERO-PADS slots 480..511. Imaginary parts are all zero. This is exactly what
// tf.signal.stft does when fft_length > frame_length.

#include "ira_features.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(ESP_PLATFORM)
#include "esp_log.h"
#define IRA_LOGE(tag, fmt, ...) ESP_LOGE(tag, fmt, ##__VA_ARGS__)
#else
#include <stdio.h>
#define IRA_LOGE(tag, fmt, ...) fprintf(stderr, "[%s] " fmt "\n", tag, ##__VA_ARGS__)
#endif

static const char *TAG = "ira_features";

// ---------------------------------------------------------------------------
// Static state (no heap)
// ---------------------------------------------------------------------------
namespace {

constexpr int kN = IRA_FFT_LENGTH;      // 512
constexpr int kLogN = 9;                // log2(512)

#if defined(IRA_USE_ESP_DSP) && IRA_USE_ESP_DSP
const char *const kFftBackendName = "ESP-DSP dsps_fft2r_fc32";
#else
const char *const kFftBackendName = "builtin iterative radix-2 complex FFT";
#endif

bool s_inited = false;

float s_hann[IRA_FRAME_LENGTH];          // 480 floats  = 1920 B
float s_tw_re[kN / 2];                   // 256 floats  = 1024 B
float s_tw_im[kN / 2];                   // 256 floats  = 1024 B
uint16_t s_bitrev[kN];                   // 512 u16     = 1024 B

// FFT working buffers
float s_fft_re[kN];                      // 512 floats  = 2048 B
float s_fft_im[kN];                      // 512 floats  = 2048 B

// log-magnitude plane, reused by both public entry points
float s_logmag[IRA_NUM_FRAMES][IRA_NUM_BINS];   // 49*40 floats = 7840 B

// Float sample scratch for the PCM16 entry point.
float s_samples[IRA_WINDOW_SAMPLES];     // 16000 floats = 64000 B

// ---------------------------------------------------------------------------
// Builtin FFT
// ---------------------------------------------------------------------------
void build_tables() {
  // PERIODIC Hann, matching tf.signal.hann_window(480, periodic=True):
  //     w[n] = 0.5 - 0.5 * cos(2*pi*n / 480)
  // Note w[0] == 0 exactly. A SYMMETRIC Hann would divide by (480-1) and is a
  // different window -- verified to differ by up to 5.0e-3, which is not
  // acceptable here.
  for (int n = 0; n < IRA_FRAME_LENGTH; ++n) {
    s_hann[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / (float)IRA_FRAME_LENGTH);
  }
  for (int k = 0; k < kN / 2; ++k) {
    const double ang = -2.0 * M_PI * (double)k / (double)kN;
    s_tw_re[k] = (float)cos(ang);
    s_tw_im[k] = (float)sin(ang);
  }
  for (int i = 0; i < kN; ++i) {
    unsigned r = 0, v = (unsigned)i;
    for (int b = 0; b < kLogN; ++b) {
      r = (r << 1) | (v & 1u);
      v >>= 1;
    }
    s_bitrev[i] = (uint16_t)r;
  }
}

// In-place complex FFT over s_fft_re / s_fft_im. Unnormalized forward
// transform, matching numpy/TensorFlow.
void ira_fft512() {
#if defined(IRA_USE_ESP_DSP) && IRA_USE_ESP_DSP
  // ESP-DSP path: interleaved [re,im] buffer. Left as an explicit TODO so the
  // swap is a deliberate, re-verified step rather than a silent substitution.
#error "ESP-DSP FFT backend not wired up yet; re-run tools/test_feature_parity.py after wiring."
#else
  // bit-reversal permutation
  for (int i = 0; i < kN; ++i) {
    const int j = s_bitrev[i];
    if (j > i) {
      float t = s_fft_re[i]; s_fft_re[i] = s_fft_re[j]; s_fft_re[j] = t;
      t = s_fft_im[i];       s_fft_im[i] = s_fft_im[j]; s_fft_im[j] = t;
    }
  }
  for (int len = 2; len <= kN; len <<= 1) {
    const int half = len >> 1;
    const int step = kN / len;
    for (int i = 0; i < kN; i += len) {
      int tw = 0;
      for (int j = 0; j < half; ++j, tw += step) {
        const float wr = s_tw_re[tw], wi = s_tw_im[tw];
        const int a = i + j, b = i + j + half;
        const float xr = s_fft_re[b], xi = s_fft_im[b];
        const float tr = xr * wr - xi * wi;
        const float ti = xr * wi + xi * wr;
        s_fft_re[b] = s_fft_re[a] - tr;
        s_fft_im[b] = s_fft_im[a] - ti;
        s_fft_re[a] += tr;
        s_fft_im[a] += ti;
      }
    }
  }
#endif
}

// Core: float samples in [-1,1) -> normalised 49x40 plane.
bool compute(const float *samples, float out[IRA_NUM_FRAMES][IRA_NUM_BINS]) {
  for (int f = 0; f < IRA_NUM_FRAMES; ++f) {
    const int off = f * IRA_FRAME_STEP;

    // window the 480 real samples, zero-pad 480..511, imag = 0
    for (int n = 0; n < IRA_FRAME_LENGTH; ++n) {
      s_fft_re[n] = samples[off + n] * s_hann[n];
      s_fft_im[n] = 0.0f;
    }
    for (int n = IRA_FRAME_LENGTH; n < kN; ++n) {
      s_fft_re[n] = 0.0f;
      s_fft_im[n] = 0.0f;
    }

    ira_fft512();

    // keep bins 0..39; magnitude then log(x + 1e-6)
    for (int k = 0; k < IRA_NUM_BINS; ++k) {
      const float re = s_fft_re[k], im = s_fft_im[k];
      const float mag = sqrtf(re * re + im * im);
      s_logmag[f][k] = logf(mag + IRA_LOG_EPSILON);
    }
  }

  // ONE global mean and POPULATION std over all 49*40 values.
  // Accumulate in double: the naive float sum of 1960 values near -13.8
  // (log(1e-6)) loses precision, which showed up as visible parity error.
  const int total = IRA_NUM_FRAMES * IRA_NUM_BINS;
  double sum = 0.0;
  for (int f = 0; f < IRA_NUM_FRAMES; ++f)
    for (int k = 0; k < IRA_NUM_BINS; ++k) sum += (double)s_logmag[f][k];
  const double mean = sum / (double)total;

  double var = 0.0;
  for (int f = 0; f < IRA_NUM_FRAMES; ++f)
    for (int k = 0; k < IRA_NUM_BINS; ++k) {
      const double d = (double)s_logmag[f][k] - mean;
      var += d * d;
    }
  var /= (double)total;                       // ddof = 0, matches reduce_std
  const float stddev = (float)sqrt(var);

  // (v - mean) / (std + 1e-6). For an all-constant input std == 0, so the
  // denominator is exactly 1e-6 and every output is 0 -- no NaN, no div-by-0.
  const float denom = stddev + IRA_STD_EPSILON;
  const float fmean = (float)mean;
  for (int f = 0; f < IRA_NUM_FRAMES; ++f)
    for (int k = 0; k < IRA_NUM_BINS; ++k)
      out[f][k] = (s_logmag[f][k] - fmean) / denom;

  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
extern "C" bool ira_features_init(void) {
  if (!s_inited) {
    build_tables();
    s_inited = true;
  }
  return true;
}

extern "C" int8_t ira_quantize_value(float v) {
  // Python reference: np.clip(np.round(x / scale + zp), -128, 127)
  // numpy.round is round-half-to-EVEN. rintf() with the default rounding mode
  // is also round-half-to-even; roundf() would be half-away-from-zero and can
  // differ by 1 LSB on exact .5 boundaries.
  const float q = rintf(v / IRA_FEATURE_INPUT_SCALE) + (float)IRA_FEATURE_INPUT_ZERO_POINT;
  if (q <= -128.0f) return (int8_t)-128;
  if (q >= 127.0f) return (int8_t)127;
  return (int8_t)q;
}

extern "C" void ira_quantize_features(const float in[IRA_NUM_FRAMES][IRA_NUM_BINS],
                                      int8_t out[IRA_NUM_FRAMES][IRA_NUM_BINS]) {
  for (int f = 0; f < IRA_NUM_FRAMES; ++f)
    for (int k = 0; k < IRA_NUM_BINS; ++k) out[f][k] = ira_quantize_value(in[f][k]);
}

extern "C" bool ira_extract_features_float_in(const float samples[IRA_WINDOW_SAMPLES],
                                              float output[IRA_NUM_FRAMES][IRA_NUM_BINS]) {
  if (samples == nullptr || output == nullptr) {
    IRA_LOGE(TAG, "null argument");
    return false;
  }
  if (!ira_features_init()) return false;
  return compute(samples, output);
}

extern "C" bool ira_extract_features_float(const int16_t pcm[IRA_WINDOW_SAMPLES],
                                           float output[IRA_NUM_FRAMES][IRA_NUM_BINS]) {
  if (pcm == nullptr || output == nullptr) {
    IRA_LOGE(TAG, "null argument");
    return false;
  }
  if (!ira_features_init()) return false;
  for (int i = 0; i < IRA_WINDOW_SAMPLES; ++i)
    s_samples[i] = (float)pcm[i] / IRA_PCM_SCALE;
  return compute(s_samples, output);
}

extern "C" bool ira_extract_features(const int16_t pcm[IRA_WINDOW_SAMPLES],
                                     int8_t output[IRA_NUM_FRAMES][IRA_NUM_BINS]) {
  if (pcm == nullptr || output == nullptr) {
    IRA_LOGE(TAG, "null argument");
    return false;
  }
  static float norm[IRA_NUM_FRAMES][IRA_NUM_BINS];   // 7840 B, static not stack
  if (!ira_extract_features_float(pcm, norm)) return false;
  ira_quantize_features(norm, output);
  return true;
}

// ---------------------------------------------------------------------------
// Introspection for the parity harness / boot logs
// ---------------------------------------------------------------------------
extern "C" const char *ira_features_backend_name(void) { return kFftBackendName; }
