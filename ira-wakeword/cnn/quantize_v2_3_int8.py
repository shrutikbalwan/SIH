# -*- coding: utf-8 -*-
"""
Freeze V2.3 best_loss at threshold 0.43 and convert it to FULL INT8.

Representative data is drawn from the TRAIN split ONLY, using the exact V2.3
preprocessing and positive-augmentation pipeline. unseen_test is never loaded;
the frozen validation artifact is NOT used for calibration.

No retraining. Conversion + freeze manifest only.
"""
import os, csv, json, random, hashlib, collections
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO = Path(__file__).parent.parent
MANIFEST = REPO / "dataset" / "split_manifest_v3.csv"
MODELS = REPO / "cnn" / "models"
SRC_KERAS = MODELS / "ira_cnn_v2_3_best_loss.keras"
OUT_TFLITE = MODELS / "ira_cnn_v2_3_int8.tflite"
OUT_REP = MODELS / "v2_3_int8_representative_train_only.npy"
FREEZE_JSON = REPO / "cnn" / "ira_cnn_v2_3_freeze_manifest.json"

OPERATING_THRESHOLD = 0.43

SAMPLE_RATE = WINDOW_SAMPLES = 16000
STFT_FRAME_LEN, STFT_FRAME_STEP, STFT_FFT_LEN, N_FREQ_BINS = 480, 320, 512, 40

# V2.3 augmentation constants (unchanged)
SNR_RANGES = {"easy": (15, 30), "medium": (5, 15), "hard": (-5, 5)}
SNR_WEIGHTS = {"easy": 1, "medium": 1, "hard": 1}
CLEAN_PROB = 0.15
RMS_THRESH = 1e-4

# Representative set: mirrors the V2.3 effective batch composition
# (32 positives : 27 speech negatives : 5 ambient negatives per 64).
N_REPRESENTATIVE = 1024
REP_SEED = 20260906


# ---------------------------------------------------------------- audio utils
def load_audio(path):
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE:
        a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    return a.astype(np.float32)


def pad_to_window(a):
    if len(a) < WINDOW_SAMPLES:
        return np.pad(a, (0, WINDOW_SAMPLES - len(a)))
    return a[:WINDOW_SAMPLES]


def make_spectrogram(a):
    t = tf.convert_to_tensor(a, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=STFT_FRAME_LEN,
                       frame_step=STFT_FRAME_STEP, fft_length=STFT_FFT_LEN)
    s = tf.math.log(tf.abs(s) + 1e-6)[:, :N_FREQ_BINS]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)


def calc_rms(a):
    r = float(np.sqrt(np.mean(a ** 2)))
    return r if r > 1e-9 else 1e-9


def vad_trim(a, top_db=25):
    thr = calc_rms(a) / (10 ** (top_db / 20))
    above = np.where(np.abs(a) > thr)[0]
    return a if len(above) == 0 else a[above[0]:above[-1] + 1]


def mix_snr(speech, noise, snr_db):
    scale = (calc_rms(speech) / (10 ** (snr_db / 20))) / calc_rms(noise)
    mixed = speech + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def bg_segment(bg_paths):
    while True:
        p = random.choice(bg_paths)
        a = load_audio(p)
        if len(a) < WINDOW_SAMPLES:
            seg = np.pad(a, (0, WINDOW_SAMPLES - len(a)))
        elif len(a) > WINDOW_SAMPLES:
            st = random.randint(0, len(a) - WINDOW_SAMPLES)
            seg = a[st:st + WINDOW_SAMPLES]
        else:
            seg = a
        if calc_rms(seg) >= RMS_THRESH:
            return seg


def augment_positive(path, amb, sp):
    """Exact V2.3 positive augmentation."""
    active = vad_trim(load_audio(path))
    max_shift = max(0, WINDOW_SAMPLES - len(active))
    off = random.randint(0, max_shift)
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    end = min(off + len(active), WINDOW_SAMPLES)
    padded[off:end] = active[:end - off]

    if random.random() < CLEAN_PROB or (not amb and not sp):
        return padded, "clean"
    if sp and (random.random() < 0.5 or not amb):
        seg, kind = bg_segment(sp), "speech"
    else:
        seg, kind = bg_segment(amb), "ambient"
    tier = random.choices(list(SNR_RANGES), weights=list(SNR_WEIGHTS.values()))[0]
    lo, hi = SNR_RANGES[tier]
    return mix_snr(padded, seg, random.uniform(lo, hi)), kind


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_only():
    """TRAIN split only. unseen_test is never read."""
    pos, speech, ambient = [], [], []
    skipped_splits = collections.Counter()
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = r["split"].strip()
            if sp != "train":
                skipped_splits[sp] += 1
                continue
            p = str(REPO / r["path"])
            if not Path(p).exists():
                continue
            g = r.get("group", "").lower()
            if r["label"] == "1":
                pos.append(p)
            elif "libri" in g or "speech" in g:
                speech.append(p)
            elif "ambient" in g or "background" in g or "noise" in g:
                ambient.append(p)
    return pos, speech, ambient, skipped_splits


def main():
    print("=" * 72)
    print("STEP 1 -- FREEZE V2.3 best_loss @ threshold 0.43")
    print("=" * 72)
    if not SRC_KERAS.exists():
        raise SystemExit("STOP: %s not found" % SRC_KERAS)
    src_sha = sha256(SRC_KERAS)
    print("  checkpoint : %s" % SRC_KERAS)
    print("  sha256     : %s" % src_sha)
    print("  size       : %.1f KB" % (SRC_KERAS.stat().st_size / 1024))
    print("  operating threshold (frozen): %.2f" % OPERATING_THRESHOLD)

    model = tf.keras.models.load_model(str(SRC_KERAS))
    print("  input shape: %s   params: %d" % (model.input_shape, model.count_params()))

    print("\n" + "=" * 72)
    print("STEP 2 -- BUILD REPRESENTATIVE DATASET (TRAIN ONLY)")
    print("=" * 72)
    pos, speech, ambient, skipped = load_train_only()
    print("  TRAIN positives=%d  speech_neg=%d  ambient_neg=%d" % (len(pos), len(speech), len(ambient)))
    print("  splits NOT loaded: %s" % dict(skipped))
    assert "unseen_test" in skipped, "sanity: unseen_test rows should exist and be skipped"

    random.seed(REP_SEED)
    np.random.seed(REP_SEED)

    # Effective V2.3 batch composition: 32 pos : 27 speech : 5 ambient per 64
    n_pos = int(round(N_REPRESENTATIVE * 32 / 64))
    n_sp = int(round(N_REPRESENTATIVE * 27 / 64))
    n_amb = N_REPRESENTATIVE - n_pos - n_sp
    print("  composition: %d positives (augmented), %d speech neg, %d ambient neg"
          % (n_pos, n_sp, n_amb))

    X, kinds = [], collections.Counter()
    for p in random.choices(pos, k=n_pos):
        a, kind = augment_positive(p, ambient, speech)
        X.append(np.expand_dims(make_spectrogram(a), -1))
        kinds["pos_" + kind] += 1
    for p in random.sample(speech, n_sp):
        X.append(np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1))
        kinds["speech_neg"] += 1
    for p in random.choices(ambient, k=n_amb):
        X.append(np.expand_dims(make_spectrogram(pad_to_window(load_audio(p))), -1))
        kinds["ambient_neg"] += 1

    X = np.array(X, dtype=np.float32)
    print("  representative tensor: %s" % (X.shape,))
    print("  breakdown: %s" % dict(kinds))
    print("  stats: min=%.4f max=%.4f mean=%.4f std=%.4f"
          % (X.min(), X.max(), X.mean(), X.std()))
    np.save(OUT_REP, X)
    print("  saved %s" % OUT_REP)

    print("\n" + "=" * 72)
    print("STEP 3 -- CONVERT TO FULL INT8")
    print("=" * 72)

    def representative_dataset():
        for i in range(len(X)):
            yield [X[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    # FULL integer: no float fallback kernels allowed
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    tflite = conv.convert()
    OUT_TFLITE.write_bytes(tflite)
    print("  wrote %s" % OUT_TFLITE)
    print("  size: %.2f KB (float keras: %.2f KB)"
          % (len(tflite) / 1024, SRC_KERAS.stat().st_size / 1024))

    interp = tf.lite.Interpreter(model_path=str(OUT_TFLITE))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print("\n  INPUT  dtype=%s shape=%s quant=%s" % (inp["dtype"].__name__, inp["shape"], inp["quantization"]))
    print("  OUTPUT dtype=%s shape=%s quant=%s" % (out["dtype"].__name__, out["shape"], out["quantization"]))

    # verify no float kernels remain
    float_ops = []
    for d in interp.get_tensor_details():
        if d["dtype"] == np.float32 and d["name"]:
            float_ops.append(d["name"])
    print("\n  float32 tensors remaining: %d %s"
          % (len(float_ops), ("(" + ", ".join(float_ops[:4]) + ")") if float_ops else ""))
    assert inp["dtype"] == np.int8 and out["dtype"] == np.int8, "interface is not int8"
    print("  FULL INT8 INTERFACE: CONFIRMED")

    manifest = {
        "frozen_model": str(SRC_KERAS),
        "frozen_model_sha256": src_sha,
        "operating_threshold": OPERATING_THRESHOLD,
        "int8_model": str(OUT_TFLITE),
        "int8_model_sha256": sha256(OUT_TFLITE),
        "int8_size_bytes": len(tflite),
        "representative_data": {
            "source_split": "train",
            "n": int(len(X)),
            "composition": dict(kinds),
            "seed": REP_SEED,
            "file": str(OUT_REP),
        },
        "input_quantization": {"scale": float(inp["quantization"][0]),
                               "zero_point": int(inp["quantization"][1])},
        "output_quantization": {"scale": float(out["quantization"][0]),
                                "zero_point": int(out["quantization"][1])},
        "unseen_test_loaded": False,
        "retrained": False,
    }
    json.dump(manifest, open(FREEZE_JSON, "w"), indent=2)
    print("\n  freeze manifest -> %s" % FREEZE_JSON)


if __name__ == "__main__":
    main()
