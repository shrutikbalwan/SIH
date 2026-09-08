# -*- coding: utf-8 -*-
"""
Float vs FULL-INT8 equivalence report for V2.3 best_loss on the permanently
frozen expanded_val_v2_3_frozen.npz.

READ-ONLY on the validation artifact. unseen_test is never loaded.
No retraining.
"""
import os, json
import numpy as np
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

REPO = Path(__file__).parent.parent
NPZ = REPO / "dataset" / "evaluation" / "expanded_val_v2_3_frozen.npz"
KERAS = REPO / "cnn" / "models" / "ira_cnn_v2_3_best_loss.keras"
TFLITE = REPO / "cnn" / "models" / "ira_cnn_v2_3_int8.tflite"
OUT = REPO / "cnn" / "v2_3_float_vs_int8_report.json"

THR = 0.43
REF = {"TP": 949, "FN": 49, "speech_FP": 58, "speech_TN": 1728, "ambient_FP": 0}


def metrics(s_pos, s_sp, s_amb, thr):
    tp = int((s_pos >= thr).sum())
    fn = len(s_pos) - tp
    sfp = int((s_sp >= thr).sum())
    stn = len(s_sp) - sfp
    afp = int((s_amb >= thr).sum())
    neg = np.concatenate([s_sp, s_amb])
    return {
        "threshold": round(float(thr), 2),
        "TP": tp, "FN": fn,
        "TPR": tp / len(s_pos) * 100,
        "speech_FP": sfp, "speech_TN": stn,
        "speech_FPR": sfp / len(s_sp) * 100,
        "ambient_FP": afp,
        "ambient_FPR": afp / len(s_amb) * 100,
        "overall_FPR": int((neg >= thr).sum()) / len(neg) * 100,
    }


def main():
    d = np.load(str(NPZ), allow_pickle=True)
    X, lab, cat = d["X"], d["labels"], d["negative_categories"]
    pos_i = (lab == 1.0).flatten()
    sp_i = (cat == "speech")
    am_i = (cat == "ambient")
    print("=" * 72)
    print("FLOAT vs FULL-INT8 EQUIVALENCE  (frozen EXPANDED_VAL, thr=%.2f)" % THR)
    print("=" * 72)
    print("  artifact: pos=%d speech=%d ambient=%d" % (pos_i.sum(), sp_i.sum(), am_i.sum()))

    # ---------------- float ----------------
    m = tf.keras.models.load_model(str(KERAS))
    f_all = m.predict(X, batch_size=64, verbose=0).flatten()

    # ---------------- int8 ----------------
    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_s, in_z = inp["quantization"]
    out_s, out_z = out["quantization"]
    print("  int8 input  scale=%.8f zero_point=%d" % (in_s, in_z))
    print("  int8 output scale=%.8f zero_point=%d  (=> %d probability levels)"
          % (out_s, out_z, int(round(1 / out_s))))

    q_all = np.empty(len(X), dtype=np.float32)
    raw_q = np.empty(len(X), dtype=np.int16)
    for i in range(len(X)):
        q = np.round(X[i] / in_s + in_z)
        q = np.clip(q, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None, ...])
        interp.invoke()
        o = interp.get_tensor(out["index"])[0, 0]
        raw_q[i] = int(o)
        q_all[i] = (float(o) - out_z) * out_s

    f_pos, f_sp, f_amb = f_all[pos_i], f_all[sp_i], f_all[am_i]
    q_pos, q_sp, q_amb = q_all[pos_i], q_all[sp_i], q_all[am_i]

    # ---------------- reference gate ----------------
    fm = metrics(f_pos, f_sp, f_amb, THR)
    ok = all(fm[k] == v for k, v in REF.items())
    print("\n  float reference check @0.43: TP=%d FN=%d spFP=%d spTN=%d ambFP=%d  %s"
          % (fm["TP"], fm["FN"], fm["speech_FP"], fm["speech_TN"], fm["ambient_FP"],
             "EXACT" if ok else "MISMATCH"))
    if not ok:
        raise SystemExit("STOP: float model no longer reproduces the frozen reference.")

    # ---------------- score agreement ----------------
    diff = q_all - f_all
    print("\n" + "-" * 72)
    print("SCORE AGREEMENT (all %d frozen examples)" % len(X))
    print("-" * 72)
    print("  Pearson r        : %.6f" % np.corrcoef(f_all, q_all)[0, 1])
    print("  Spearman-ish rho : %.6f" % np.corrcoef(
        np.argsort(np.argsort(f_all)), np.argsort(np.argsort(q_all)))[0, 1])
    print("  mean abs diff    : %.6f" % np.abs(diff).mean())
    print("  median abs diff  : %.6f" % np.median(np.abs(diff)))
    print("  P99 abs diff     : %.6f" % np.percentile(np.abs(diff), 99))
    print("  max abs diff     : %.6f" % np.abs(diff).max())
    print("  mean signed diff : %+.6f  (int8 - float)" % diff.mean())
    print("  output LSB       : %.6f  (1 int8 step)" % out_s)
    print("  |diff| <= 1 LSB  : %.2f%%" % ((np.abs(diff) <= out_s).mean() * 100))
    print("  |diff| <= 2 LSB  : %.2f%%" % ((np.abs(diff) <= 2 * out_s).mean() * 100))

    # ---------------- decision agreement at 0.43 ----------------
    fd = f_all >= THR
    qd = q_all >= THR
    agree = (fd == qd)
    flips = np.where(~agree)[0]
    print("\n" + "-" * 72)
    print("DECISION AGREEMENT @ %.2f" % THR)
    print("-" * 72)
    print("  identical decisions : %d/%d = %.4f%%" % (agree.sum(), len(X), agree.mean() * 100))
    print("  flips               : %d" % len(flips))
    for i in flips:
        grp = "positive" if pos_i[i] else ("speech" if sp_i[i] else "ambient")
        print("    idx %-5d %-8s float=%.4f int8=%.4f  %s"
              % (i, grp, f_all[i], q_all[i],
                 "float NEG -> int8 POS" if qd[i] else "float POS -> int8 NEG"))

    # ---------------- operating point ----------------
    qm = metrics(q_pos, q_sp, q_amb, THR)
    print("\n" + "-" * 72)
    print("OPERATING POINT @ %.2f" % THR)
    print("-" * 72)
    print("  %-14s %12s %12s %10s" % ("metric", "float", "int8", "delta"))
    for k, unit in [("TP", ""), ("FN", ""), ("TPR", "%"), ("speech_FP", ""),
                    ("speech_TN", ""), ("speech_FPR", "%"), ("ambient_FP", ""),
                    ("ambient_FPR", "%"), ("overall_FPR", "%")]:
        fv, qv = fm[k], qm[k]
        if unit:
            print("  %-14s %11.2f%% %11.2f%% %+9.2f pp" % (k, fv, qv, qv - fv))
        else:
            print("  %-14s %12d %12d %+10d" % (k, fv, qv, qv - fv))

    tgt = lambda mm: (mm["TPR"] >= 95.0 and mm["speech_FPR"] <= 3.0 and mm["ambient_FPR"] <= 1.0)
    print("\n  primary target (TPR>=95 & SpFPR<=3 & AmbFPR<=1):  float=%s  int8=%s"
          % ("PASS" if tgt(fm) else "FAIL", "PASS" if tgt(qm) else "FAIL"))

    # ---------------- threshold sweep ----------------
    print("\n" + "-" * 72)
    print("THRESHOLD SWEEP 0.30-0.95 (float vs int8)")
    print("-" * 72)
    print("  %5s | %7s %7s %7s | %7s %7s %7s" %
          ("thr", "f_TPR", "f_SpFPR", "f_AFPR", "q_TPR", "q_SpFPR", "q_AFPR"))
    sweep = []
    for thr in np.arange(0.30, 0.951, 0.01):
        a, b = metrics(f_pos, f_sp, f_amb, thr), metrics(q_pos, q_sp, q_amb, thr)
        sweep.append({"thr": round(float(thr), 2), "float": a, "int8": b})
        print("  %5.2f | %6.2f%% %6.2f%% %6.2f%% | %6.2f%% %6.2f%% %6.2f%%"
              % (thr, a["TPR"], a["speech_FPR"], a["ambient_FPR"],
                 b["TPR"], b["speech_FPR"], b["ambient_FPR"]))

    def best(pref):
        cand = [s for s in sweep if s[pref]["TPR"] >= 95.0]
        return min(cand, key=lambda s: s[pref]["speech_FPR"]) if cand else None

    for tag in ("float", "int8"):
        b = best(tag)
        if b:
            mm = b[tag]
            print("\n  best %s threshold with TPR>=95%%: thr=%.2f  TPR=%.2f%%  SpFPR=%.2f%%  AmbFPR=%.2f%%"
                  % (tag, b["thr"], mm["TPR"], mm["speech_FPR"], mm["ambient_FPR"]))
        else:
            print("\n  best %s threshold with TPR>=95%%: none" % tag)

    json.dump({
        "threshold": THR,
        "float": fm, "int8": qm,
        "agreement": {
            "pearson_r": float(np.corrcoef(f_all, q_all)[0, 1]),
            "mean_abs_diff": float(np.abs(diff).mean()),
            "max_abs_diff": float(np.abs(diff).max()),
            "output_lsb": float(out_s),
            "decision_agreement_pct": float(agree.mean() * 100),
            "flips": int(len(flips)),
        },
        "sweep": sweep,
    }, open(OUT, "w"), indent=2)
    print("\n  report -> %s" % OUT)


if __name__ == "__main__":
    main()
