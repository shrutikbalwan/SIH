// Host-side driver for the ESP32 feature frontend, used by
// tools/test_feature_parity.py. Compiles esp32/main/ira_features.cpp unchanged.
//
// Protocol (all little-endian, stdin -> stdout, binary):
//   stdin : 16000 x int16   (one 1-second PCM window)
//   stdout: 1960 x float32  (normalised 49x40 features, row-major)
//           1960 x int8     (quantised 49x40 features, row-major)
//
// Build:
//   g++ -O2 -std=c++17 -I esp32/main tools/ira_features_host.cpp \
//       esp32/main/ira_features.cpp -o build/ira_features_host

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#include "ira_features.h"

extern "C" const char *ira_features_backend_name(void);

static int16_t pcm[IRA_WINDOW_SAMPLES];
static float feats[IRA_NUM_FRAMES][IRA_NUM_BINS];
static int8_t q[IRA_NUM_FRAMES][IRA_NUM_BINS];

int main(int argc, char **argv) {
#if defined(_WIN32)
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  if (argc > 1 && std::strcmp(argv[1], "--backend") == 0) {
    std::printf("%s\n", ira_features_backend_name());
    return 0;
  }

  const size_t want = sizeof(pcm);
  size_t got = 0;
  while (got < want) {
    const size_t n = std::fread(reinterpret_cast<uint8_t *>(pcm) + got, 1, want - got, stdin);
    if (n == 0) break;
    got += n;
  }
  if (got != want) {
    std::fprintf(stderr, "expected %zu bytes of PCM, got %zu\n", want, got);
    return 2;
  }

  if (!ira_features_init()) {
    std::fprintf(stderr, "ira_features_init failed\n");
    return 3;
  }
  if (!ira_extract_features_float(pcm, feats)) {
    std::fprintf(stderr, "ira_extract_features_float failed\n");
    return 4;
  }
  ira_quantize_features(feats, q);

  if (std::fwrite(feats, sizeof(float), IRA_NUM_FRAMES * IRA_NUM_BINS, stdout)
          != (size_t)(IRA_NUM_FRAMES * IRA_NUM_BINS)) return 5;
  if (std::fwrite(q, sizeof(int8_t), IRA_NUM_FRAMES * IRA_NUM_BINS, stdout)
          != (size_t)(IRA_NUM_FRAMES * IRA_NUM_BINS)) return 6;
  std::fflush(stdout);
  return 0;
}
