// IRA wake-word — input-validity guard implementation. See header for rationale.

#include "ira_audio_guard.h"

extern "C" ira_audio_status_t ira_audio_window_check(const int16_t pcm[IRA_GUARD_SAMPLES]) {
  if (pcm == nullptr) return IRA_AUDIO_INVALID_NULL;

  // A single pass answers both questions: is everything zero, and is
  // everything identical? Both are early-exit, so the common (valid) case
  // costs only a few samples.
  const int16_t first = pcm[0];
  bool all_zero = (first == 0);
  bool all_same = true;

  for (int i = 1; i < IRA_GUARD_SAMPLES; ++i) {
    const int16_t v = pcm[i];
    if (v != 0) all_zero = false;
    if (v != first) all_same = false;
    if (!all_zero && !all_same) return IRA_AUDIO_OK;  // definitely varying
  }

  if (all_zero) return IRA_AUDIO_INVALID_ALL_ZERO;
  if (all_same) return IRA_AUDIO_INVALID_CONSTANT;
  return IRA_AUDIO_OK;
}

extern "C" bool ira_audio_window_valid(const int16_t pcm[IRA_GUARD_SAMPLES]) {
  return ira_audio_window_check(pcm) == IRA_AUDIO_OK;
}

extern "C" const char *ira_audio_status_str(ira_audio_status_t s) {
  switch (s) {
    case IRA_AUDIO_OK: return "OK";
    case IRA_AUDIO_INVALID_ALL_ZERO: return "INVALID_ALL_ZERO";
    case IRA_AUDIO_INVALID_CONSTANT: return "INVALID_CONSTANT";
    case IRA_AUDIO_INVALID_NULL: return "INVALID_NULL";
    default: return "UNKNOWN";
  }
}
