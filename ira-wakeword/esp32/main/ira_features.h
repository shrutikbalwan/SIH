// IRA wake-word — V2.3 feature frontend (STFT log-magnitude + per-window z-score).
//
// FROZEN CONTRACT — must byte-match the Python training/eval pipeline:
//   window       : 16000 samples @ 16 kHz mono
//   STFT         : frame_length=480, frame_step=320, fft_length=512, pad_end=false
//   window fn    : PERIODIC Hann, w[n] = 0.5 - 0.5*cos(2*pi*n/480), n = 0..479
//   frames       : exactly 49
//   bins kept    : 0..39 (first 40 of the 257 rfft bins)
//   magnitude    : |X| = sqrt(re^2 + im^2)
//   compression  : log(|X| + 1e-6)   (natural log)
//   normalise    : (v - mean) / (std + 1e-6), ONE global mean/std over all 49*40
//                  std is the POPULATION std (ddof = 0), matching tf.math.reduce_std
//   quantise     : q = clamp(rint(v / 0.06078097224235535) + 49, -128, 127)
//
// NOT MFCC. NOT mel. NOT MicroFrontend. Those would not reproduce the training
// features and the model would be operating out of distribution.
//
// No dynamic allocation anywhere in this module.

#ifndef IRA_FEATURES_H_
#define IRA_FEATURES_H_

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IRA_SAMPLE_RATE 16000
#define IRA_WINDOW_SAMPLES 16000
#define IRA_FRAME_LENGTH 480
#define IRA_FRAME_STEP 320
#define IRA_FFT_LENGTH 512
#define IRA_NUM_FRAMES 49   // (16000 - 480) / 320 + 1
#define IRA_NUM_BINS 40     // first 40 of 257 rfft bins
#define IRA_LOG_EPSILON 1e-6f
#define IRA_STD_EPSILON 1e-6f

// Must match the model's input tensor quantization exactly.
#define IRA_FEATURE_INPUT_SCALE 0.06078097224235535f
#define IRA_FEATURE_INPUT_ZERO_POINT 49

// PCM16 -> float uses /32768.0f, matching Python audio loading
// (soundfile/librosa read int16 WAV as x / 32768).
#define IRA_PCM_SCALE 32768.0f

// ---------------------------------------------------------------------------
// One-time initialisation: builds the Hann window and FFT twiddle tables.
// Safe to call repeatedly. Called implicitly by the extract functions, but
// call it explicitly at boot to keep the first inference cheap.
// ---------------------------------------------------------------------------
bool ira_features_init(void);

// ---------------------------------------------------------------------------
// Full frontend: PCM16 window -> quantized INT8 features ready for the model.
// `pcm` must hold exactly IRA_WINDOW_SAMPLES samples.
// Returns false only on a null argument or init failure.
// ---------------------------------------------------------------------------
bool ira_extract_features(const int16_t pcm[IRA_WINDOW_SAMPLES],
                          int8_t output[IRA_NUM_FRAMES][IRA_NUM_BINS]);

// Debug/parity variant: stops after normalisation, before quantisation.
bool ira_extract_features_float(const int16_t pcm[IRA_WINDOW_SAMPLES],
                                float output[IRA_NUM_FRAMES][IRA_NUM_BINS]);

// Float32 input variant (samples already in [-1, 1)), for host parity testing
// against Python without a PCM16 round-trip.
bool ira_extract_features_float_in(const float samples[IRA_WINDOW_SAMPLES],
                                   float output[IRA_NUM_FRAMES][IRA_NUM_BINS]);

// Quantise an already-normalised feature plane. Exposed so the rounding rule
// can be tested in isolation.
void ira_quantize_features(const float in[IRA_NUM_FRAMES][IRA_NUM_BINS],
                           int8_t out[IRA_NUM_FRAMES][IRA_NUM_BINS]);

// Single-value quantisation, identical rule. Uses rintf() = round-half-to-EVEN,
// matching numpy.round used by the Python reference. (roundf() would be
// round-half-away-from-zero and would differ on exact .5 cases.)
int8_t ira_quantize_value(float v);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IRA_FEATURES_H_
