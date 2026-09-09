# -*- coding: utf-8 -*-
"""
mine_v2_train_speech_hard_negatives.py
======================================
Score TRAIN-SPLIT speech negatives with ira_cnn_v2_best_loss.keras to
identify which clips the V2 model already finds confusable.

Uses ONLY train-split speakers. Does NOT touch validation or test splits.

Saves: v2_train_speech_negative_scores.csv
"""
import os, csv
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT   = Path(__file__).parent
MODEL_PATH  = REPO_ROOT / "cnn" / "models" / "ira_cnn_v2_best_loss.keras"
MANIFEST    = REPO_ROOT / "dataset" / "split_manifest.csv"
OUT_CSV     = REPO_ROOT / "v2_train_speech_negative_scores.csv"

# Score threshold for hard-speech pool selection
HARD_SCORE_THRESHOLD = 0.30

SAMPLE_RATE    = 16000
WINDOW_SAMPLES = 16000


def load_audio(path):
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1: a = np.mean(a, axis=1)
    if sr != SAMPLE_RATE: a = resample_poly(a, SAMPLE_RATE, sr).astype(np.float32)
    if len(a) < WINDOW_SAMPLES: a = np.pad(a, (0, WINDOW_SAMPLES - len(a)))
    else: a = a[:WINDOW_SAMPLES]
    return a.astype(np.float32)


def make_spectrogram(audio):
    t = tf.convert_to_tensor(audio, dtype=tf.float32)
    s = tf.signal.stft(t, frame_length=480, frame_step=320, fft_length=512)
    s = tf.abs(s)
    s = tf.math.log(s + 1e-6)
    s = s[:, :40]
    m = tf.reduce_mean(s)
    d = tf.math.reduce_std(s) + 1e-6
    return ((s - m) / d).numpy().astype(np.float32)


def extract_speaker(path_str):
    """Extract LibriSpeech speaker ID from path like .../1737_1737-146161-0005_neg_010.wav"""
    stem = Path(path_str).stem
    parts = stem.split("_")
    return parts[0] if parts else "unknown"


def main():
    print("="*60)
    print("MINING V2 TRAIN SPEECH HARD NEGATIVES")
    print("="*60)

    # Load model
    print(f"\nLoading: {MODEL_PATH.name}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    # Gather TRAIN speech negatives ONLY
    train_speech = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "train": continue
            if int(r.get("label", -1)) != 0: continue
            g = r.get("group", "").lower()
            if "libri" not in g and "speech" not in g: continue
            p = REPO_ROOT / r["path"]
            if not p.exists():
                print(f"  WARNING missing: {p}"); continue
            spk = extract_speaker(r["path"])
            train_speech.append({"path": str(p), "speaker_id": spk})

    print(f"\nTrain speech negatives found: {len(train_speech)}")
    speakers = sorted({d["speaker_id"] for d in train_speech})
    print(f"Unique speakers: {len(speakers)}")
    print(f"Speakers: {speakers}")

    # Score all in batch
    print(f"\nScoring all {len(train_speech)} clips...")
    paths = [d["path"] for d in train_speech]
    specs = [np.expand_dims(make_spectrogram(load_audio(p)), -1) for p in paths]
    X = np.stack(specs, 0)
    scores = model.predict(X, batch_size=64, verbose=1).flatten().astype(np.float32)

    # Save CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "speaker_id", "V2_score"])
        for d, s in zip(train_speech, scores):
            w.writerow([d["path"], d["speaker_id"], f"{s:.6f}"])
    print(f"\nSaved: {OUT_CSV}")

    # Distribution stats
    print("\n" + "="*60)
    print("SCORE DISTRIBUTION (all train speech negatives)")
    print("="*60)
    for stat, val in [
        ("count",  len(scores)),
        ("mean",   np.mean(scores)),
        ("median", np.median(scores)),
        ("P90",    np.percentile(scores, 90)),
        ("P95",    np.percentile(scores, 95)),
        ("P99",    np.percentile(scores, 99)),
        ("max",    np.max(scores)),
    ]:
        print(f"  {stat:8s}: {val:.4f}" if stat != "count" else f"  {stat:8s}: {int(val)}")

    # Hard pool selection
    hard_mask = scores >= HARD_SCORE_THRESHOLD
    hard_paths = [paths[i] for i in range(len(paths)) if hard_mask[i]]
    hard_scores = scores[hard_mask]

    print(f"\n{'='*60}")
    print(f"HARD SPEECH POOL (V2 score >= {HARD_SCORE_THRESHOLD})")
    print("="*60)
    print(f"  Count: {len(hard_paths)} / {len(paths)}")
    print(f"  ({len(hard_paths)/len(paths)*100:.1f}% of train speech negatives)")

    if len(hard_paths) > 0:
        print(f"  Score range: [{hard_scores.min():.4f}, {hard_scores.max():.4f}]")
        print(f"  Mean score:  {hard_scores.mean():.4f}")
        # Per-speaker breakdown
        spk_hard = {}
        for i, p in enumerate(paths):
            if hard_mask[i]:
                spk = train_speech[i]["speaker_id"]
                spk_hard[spk] = spk_hard.get(spk, 0) + 1
        print("\n  Per-speaker hard-speech counts:")
        for spk in sorted(spk_hard):
            print(f"    Speaker {spk}: {spk_hard[spk]}")
    else:
        print("  WARNING: No clips meet threshold. Will use top-20% by score.")
        # Fall back
        top_n = max(50, int(len(paths) * 0.20))
        top_idxs = np.argsort(scores)[::-1][:top_n]
        hard_paths = [paths[i] for i in top_idxs]
        print(f"  Fallback pool size: {len(hard_paths)}")

    print("\nDone. Use v2_train_speech_negative_scores.csv in train_cnn_v2_1.py")


if __name__ == "__main__":
    main()
