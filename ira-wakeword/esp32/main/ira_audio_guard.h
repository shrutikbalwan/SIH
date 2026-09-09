// IRA wake-word — input-validity guard.
//
// PURPOSE
// -------
// The desktop parity harness showed that an EXACTLY-ZERO 16000-sample window
// makes the trained model fire (q_out = 104 >= -25). The cause is degenerate:
// a constant spectrum has std == 0, so the per-window z-score produces a flat
// feature plane that the model happens to score high. A real microphone will
// not produce digital zero, but a muted I2S channel, a codec in reset, or a
// DMA underrun buffer will.
//
// This guard rejects windows that are UNQUESTIONABLY invalid audio. It is NOT
// a speech/silence detector and NOT a minimum-loudness threshold.
//
// DELIBERATELY NOT IMPLEMENTED YET
// --------------------------------
// No general minimum-RMS threshold. Choosing one from the 66 development
// recordings would tune a deployment parameter against development data, and
// would risk rejecting legitimately quiet speech — the prior diagnostic found
// genuine missed detections were the QUIETEST recordings (RMS 0.082 vs 0.137),
// so a naive RMS floor would discard exactly the cases that already fail.
// Any broader guard must be derived from real microphone data later.

#ifndef IRA_AUDIO_GUARD_H_
#define IRA_AUDIO_GUARD_H_

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IRA_GUARD_SAMPLES 16000

typedef enum {
  IRA_AUDIO_OK = 0,
  IRA_AUDIO_INVALID_ALL_ZERO,   // every sample exactly 0
  IRA_AUDIO_INVALID_CONSTANT,   // every sample identical (stuck line / DC latch)
  IRA_AUDIO_INVALID_NULL,       // null buffer
} ira_audio_status_t;

// Returns IRA_AUDIO_OK only if the window is plausibly real audio.
ira_audio_status_t ira_audio_window_check(const int16_t pcm[IRA_GUARD_SAMPLES]);

// Convenience wrapper. Exactly-zero input MUST return false.
bool ira_audio_window_valid(const int16_t pcm[IRA_GUARD_SAMPLES]);

// Human-readable status, for logging.
const char *ira_audio_status_str(ira_audio_status_t s);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IRA_AUDIO_GUARD_H_
