# -*- coding: utf-8 -*-
"""
Feature parity harness: Python (TensorFlow) V2.3 frontend vs the C++
implementation intended for ESP32.

Compares float features, INT8 features, and the frozen V2.3 INT8 model's
q_out on both feature sets, for deterministic synthetic inputs plus real WAVs.

Real WAVs are opened READ-ONLY. Nothing is trained, modified, or written back.
"""
import argparse
import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent
HOST_SRC = REPO / "tools" / "ira_features_host.cpp"
FEAT_SRC = REPO / "esp32" / "main" / "ira_features.cpp"
INC = REPO / "esp32" / "main"
BUILD = REPO / "build"
EXE = BUILD / ("ira_features_host.exe" if os.name == "nt" else "ira_features_host")
TFLITE = REPO / "cnn" / "models" / "ira_cnn_v2_3_int8.tflite"
EXPECT_SHA = "b9554c054bc9a57a00e874824e5a4ab9c3254b4505ca8a3ec67835d193103de0"

SR = 16000
N = 16000
FRAMES, BINS = 49, 40
IN_SCALE, IN_ZP = 0.06078097224235535, 49
Q_MIN = -25


# ---------------------------------------------------------------- reference
def py_features_float(samples):
    """EXACT V2.3 frontend: float32 samples in [-1,1) -> normalised 49x40."""
    t = tf.convert_to_tensor(samples.astype(np.float32))
    s = tf.signal.stft(t, frame_length=480, frame_step=320, fft_length=512,
                       pad_end=False)
    lg = tf.math.log(tf.abs(s) + 1e-6)[:, :BINS]
    m = tf.reduce_mean(lg)
    d = tf.math.reduce_std(lg) + 1e-6
    return ((lg - m) / d).numpy().astype(np.float32)


def py_quantize(f):
    return np.clip(np.round(f / IN_SCALE + IN_ZP), -128, 127).astype(np.int8)


# ---------------------------------------------------------------- C++ bridge
def build_host(verbose=True):
    BUILD.mkdir(exist_ok=True)
    cmd = ["g++", "-O2", "-std=c++17", "-I", str(INC),
           str(HOST_SRC), str(FEAT_SRC), "-o", str(EXE)]
    if verbose:
        print("  building: " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("STOP: host build failed")
    if verbose:
        b = subprocess.run([str(EXE), "--backend"], capture_output=True, text=True)
        print("  FFT backend: %s" % b.stdout.strip())
    return True


def cpp_features(pcm16):
    raw = pcm16.astype("<i2").tobytes()
    r = subprocess.run([str(EXE)], input=raw, capture_output=True)
    if r.returncode != 0:
        raise SystemExit("STOP: host binary failed: %s" % r.stderr.decode(errors="replace"))
    n = FRAMES * BINS
    exp = n * 4 + n
    if len(r.stdout) != exp:
        raise SystemExit("STOP: host returned %d bytes, expected %d" % (len(r.stdout), exp))
    f = np.frombuffer(r.stdout[:n * 4], dtype="<f4").reshape(FRAMES, BINS)
    q = np.frombuffer(r.stdout[n * 4:], dtype=np.int8).reshape(FRAMES, BINS)
    return f.copy(), q.copy()


# ---------------------------------------------------------------- test inputs
def make_cases(n_wav):
    cases = []
    cases.append(("all-zero", np.zeros(N, dtype=np.int16)))

    t = np.arange(N) / SR
    for hz in (440.0, 1000.0):
        w = 0.5 * np.sin(2 * np.pi * hz * t)
        cases.append(("sine-%dHz" % int(hz), np.round(w * 32767).astype(np.int16)))

    rng = np.random.default_rng(12345)
    cases.append(("random-pcm", rng.integers(-8000, 8000, N, dtype=np.int16)))

    imp = np.zeros(N, dtype=np.int16)
    imp[8000] = 20000
    cases.append(("impulse", imp))

    # extra edge cases worth covering
    cases.append(("dc-offset", np.full(N, 1000, dtype=np.int16)))
    full = np.round(0.999 * 32767 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    cases.append(("near-full-scale", full))

    # real recordings, read-only
    src = REPO / "dataset" / "new data set"
    if src.exists() and n_wav > 0:
        import soundfile as sf
        wavs = sorted(p for p in src.rglob("*.wav"))[:n_wav]
        for p in wavs:
            a, sr = sf.read(str(p), dtype="int16", always_2d=False)
            if sr != SR:
                continue
            if a.ndim > 1:
                a = a[:, 0]
            a = a[:N] if len(a) >= N else np.pad(a, (0, N - len(a)))
            cases.append(("wav:" + p.name[:34], a.astype(np.int16)))
    return cases


# ---------------------------------------------------------------- model
def load_model():
    blob = TFLITE.read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    if got != EXPECT_SHA:
        raise SystemExit("STOP: model SHA256 mismatch\n  expected %s\n  got      %s"
                         % (EXPECT_SHA, got))
    it = tf.lite.Interpreter(model_path=str(TFLITE))
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0], got


def run_model(it, inp, out, q):
    it.set_tensor(inp["index"], q.reshape(1, FRAMES, BINS, 1))
    it.invoke()
    return int(it.get_tensor(out["index"])[0, 0])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", type=int, default=6)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    print("=" * 92)
    print("V2.3 FEATURE PARITY — Python (TensorFlow) vs C++ (ESP32 frontend)")
    print("=" * 92)

    if not args.no_build:
        build_host()
    elif not EXE.exists():
        raise SystemExit("STOP: %s not built" % EXE)

    it, inp, out, sha = load_model()
    print("  model sha256: %s (verified)" % sha)
    print("  frozen rule : q_out >= %d" % Q_MIN)

    cases = make_cases(args.wavs)
    print("  test cases  : %d\n" % len(cases))

    rows = []
    tot_ident = tot_vals = 0
    worst_q = 0
    all_decisions_match = True

    for name, pcm in cases:
        samples = pcm.astype(np.float32) / 32768.0
        pf = py_features_float(samples)
        pq = py_quantize(pf)
        cf, cq = cpp_features(pcm)

        aerr = np.abs(pf - cf)
        max_err, mean_err = float(aerr.max()), float(aerr.mean())
        a, b = pf.ravel(), cf.ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = 1.0 if denom == 0 else float(a @ b / denom)

        ident = int((pq == cq).sum())
        mism = pq.size - ident
        maxd = int(np.abs(pq.astype(int) - cq.astype(int)).max())
        tot_ident += ident
        tot_vals += pq.size
        worst_q = max(worst_q, maxd)

        qp = run_model(it, inp, out, pq)
        qc = run_model(it, inp, out, cq)
        dp, dc = qp >= Q_MIN, qc >= Q_MIN
        same = dp == dc
        all_decisions_match &= same

        rows.append(dict(name=name, max_err=max_err, mean_err=mean_err, cos=cos,
                         ident=ident, mism=mism, maxd=maxd,
                         qp=qp, qc=qc, dq=abs(qp - qc), dp=dp, dc=dc, same=same))

        if name == "all-zero":
            finite = np.isfinite(cf).all()
            allzero = np.allclose(cf, 0.0, atol=0)
            print("  [all-zero special case] C++ finite=%s  all-exactly-zero=%s  "
                  "py all-zero=%s" % (finite, allzero, np.allclose(pf, 0.0, atol=0)))

    print("\n" + "-" * 92)
    print("FLOAT FEATURES")
    print("-" * 92)
    print("  %-38s %12s %12s %14s" % ("case", "max |err|", "mean |err|", "cosine sim"))
    for r in rows:
        print("  %-38s %12.3e %12.3e %14.12f" % (r["name"], r["max_err"], r["mean_err"], r["cos"]))

    print("\n" + "-" * 92)
    print("INT8 FEATURES (out of %d per case)" % (FRAMES * BINS))
    print("-" * 92)
    print("  %-38s %10s %10s %10s %10s" % ("case", "identical", "mismatch", "identical%", "max |dq|"))
    for r in rows:
        print("  %-38s %10d %10d %9.4f%% %10d"
              % (r["name"], r["ident"], r["mism"], r["ident"] / (FRAMES * BINS) * 100, r["maxd"]))

    print("\n" + "-" * 92)
    print("MODEL OUTPUT ON EACH FEATURE SET (frozen V2.3 INT8)")
    print("-" * 92)
    print("  %-38s %10s %10s %8s %8s %8s %7s"
          % ("case", "q(python)", "q(c++)", "|dq|", "det(py)", "det(c++)", "same"))
    for r in rows:
        print("  %-38s %10d %10d %8d %8s %8s %7s"
              % (r["name"], r["qp"], r["qc"], r["dq"],
                 "YES" if r["dp"] else "no", "YES" if r["dc"] else "no",
                 "OK" if r["same"] else "**NO**"))

    print("\n" + "=" * 92)
    print("AGGREGATE")
    print("=" * 92)
    pct = tot_ident / tot_vals * 100
    max_dq = max(r["dq"] for r in rows)
    print("  INT8 values identical      : %d / %d = %.4f%%" % (tot_ident, tot_vals, pct))
    print("  worst INT8 feature diff    : %d LSB" % worst_q)
    print("  worst |q_out| difference   : %d" % max_dq)
    print("  detection decisions match  : %s (%d/%d cases)"
          % ("ALL" if all_decisions_match else "NO",
             sum(1 for r in rows if r["same"]), len(rows)))
    print("  worst float max |err|      : %.3e" % max(r["max_err"] for r in rows))
    print("  worst cosine similarity    : %.12f" % min(r["cos"] for r in rows))

    # Split real audio from synthetic: deployment only ever sees the former.
    real = [r for r in rows if r["name"].startswith("wav:")]
    synth = [r for r in rows if not r["name"].startswith("wav:")]
    if real:
        ri = sum(r["ident"] for r in real); rv = len(real) * FRAMES * BINS
        print("")
        print("  --- REAL AUDIO ONLY (%d cases) ---" % len(real))
        print("  INT8 identical             : %d / %d = %.4f%%" % (ri, rv, ri / rv * 100))
        print("  worst INT8 feature diff    : %d LSB" % max(r["maxd"] for r in real))
        print("  worst |q_out| difference   : %d" % max(r["dq"] for r in real))
        print("  decisions match            : %d / %d" % (sum(1 for r in real if r["same"]), len(real)))
    if synth:
        si = sum(r["ident"] for r in synth); sv = len(synth) * FRAMES * BINS
        print("")
        print("  --- SYNTHETIC ONLY (%d cases) ---" % len(synth))
        print("  INT8 identical             : %d / %d = %.4f%%" % (si, sv, si / sv * 100))
        print("  worst |q_out| difference   : %d" % max(r["dq"] for r in synth))
    real_exact = bool(real) and all(r["mism"] == 0 and r["dq"] == 0 for r in real)

    ok = all_decisions_match and pct >= 99.0 and max_dq <= 1
    print("")
    print("  REAL-AUDIO PARITY : %s" % ("EXACT (bit-identical INT8 and q_out)"
                                        if real_exact else "NOT EXACT"))
    print("  OVERALL PARITY    : %s" % ("PASS" if ok else "NEEDS REVIEW"))
    if not ok:
        print("  Diagnose before proceeding: window definition, FFT scaling, magnitude,")
        print("  log implementation, std definition (ddof), float->int rounding mode.")
        print("  Do NOT compensate by changing the model or the threshold.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
