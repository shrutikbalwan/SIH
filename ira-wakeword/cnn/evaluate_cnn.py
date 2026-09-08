import os
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# Configuration
# ============================================================

BASE_DIR = r"E:\SIH\ira-wakeword"

MODEL_PATH = os.path.join(
    BASE_DIR, "cnn", "models", "ira_cnn_best.keras"
)

TEST_DIR = os.path.join(
    BASE_DIR, "dataset", "splits", "test"
)

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
NUM_SAMPLES = 16000

# We will test several wake-word thresholds
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# ============================================================
# Audio preprocessing
# ============================================================

def load_audio(path):
    audio, sr = sf.read(path)

    # Convert stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Resample if necessary
    if sr != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sr
        ).astype(np.float32)

    # Fixed 1-second input
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )
    else:
        audio = audio[:NUM_SAMPLES]

    return audio


def audio_to_spectrogram(audio):
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    # Same STFT configuration used during training
    spec = tf.signal.stft(
        audio,
        frame_length=480,
        frame_step=320,
        fft_length=512
    )

    spec = tf.abs(spec)

    # Keep first 40 frequency bins
    spec = spec[:, :40]

    # Log compression
    spec = tf.math.log(spec + 1e-6)

    # Per-sample normalization
    mean = tf.reduce_mean(spec)
    std = tf.math.reduce_std(spec)

    spec = (spec - mean) / (std + 1e-6)

    # Add CNN channel dimension
    spec = spec[..., tf.newaxis]

    return spec.numpy().astype(np.float32)


# ============================================================
# Load test dataset
# ============================================================

def load_test_data():

    X = []
    y = []

    positive_dir = os.path.join(TEST_DIR, "positive")
    negative_dir = os.path.join(TEST_DIR, "negative")

    positive_files = [
        os.path.join(positive_dir, f)
        for f in os.listdir(positive_dir)
        if f.lower().endswith(".wav")
    ]

    negative_files = [
        os.path.join(negative_dir, f)
        for f in os.listdir(negative_dir)
        if f.lower().endswith(".wav")
    ]

    print(f"Positive test files: {len(positive_files)}")
    print(f"Negative test files: {len(negative_files)}")

    # Positive = 1
    for i, path in enumerate(positive_files):

        audio = load_audio(path)
        spec = audio_to_spectrogram(audio)

        X.append(spec)
        y.append(1)

        if (i + 1) % 200 == 0:
            print(f"  positive processed: {i + 1}/{len(positive_files)}")

    # Negative = 0
    for i, path in enumerate(negative_files):

        audio = load_audio(path)
        spec = audio_to_spectrogram(audio)

        X.append(spec)
        y.append(0)

        if (i + 1) % 200 == 0:
            print(f"  negative processed: {i + 1}/{len(negative_files)}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    return X, y


# ============================================================
# Main evaluation
# ============================================================

print("=" * 60)
print("IRA CNN EVALUATION")
print("=" * 60)

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")

X_test, y_test = load_test_data()

print("\nTest shape:", X_test.shape)
print("Positive samples:", np.sum(y_test == 1))
print("Negative samples:", np.sum(y_test == 0))


# ============================================================
# Predictions
# ============================================================

print("\nRunning predictions...")

probabilities = model.predict(
    X_test,
    batch_size=64,
    verbose=1
).reshape(-1)


# Save probabilities for later analysis
output_path = os.path.join(
    BASE_DIR,
    "cnn",
    "models",
    "test_probabilities.npy"
)

np.save(output_path, probabilities)

print("\nSaved prediction probabilities:")
print(output_path)


# ============================================================
# Threshold analysis
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

print(
    f"\n{'Threshold':>10} "
    f"{'Accuracy':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'FAR':>10} "
    f"{'FRR':>10}"
)

best_threshold = None
best_score = -1

for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = (tp + tn) / len(y_test)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    # False Acceptance Rate:
    # negative samples incorrectly detected as "Ira"
    far = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    # False Rejection Rate:
    # positive samples incorrectly rejected
    frr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    print(
        f"{threshold:10.2f} "
        f"{accuracy:10.4f} "
        f"{precision:10.4f} "
        f"{recall:10.4f} "
        f"{far:10.4f} "
        f"{frr:10.4f}"
    )

    # Prefer high recall while minimizing FAR
    score = recall - far

    if score > best_score:
        best_score = score
        best_threshold = threshold


# ============================================================
# Final report using selected threshold
# ============================================================

predictions = (
    probabilities >= best_threshold
).astype(np.int32)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

print("\n" + "=" * 60)
print("SELECTED THRESHOLD")
print("=" * 60)

print(f"\nThreshold: {best_threshold:.2f}")

print("\nConfusion Matrix:")
print(
    f"True Negative : {tn}"
)
print(
    f"False Positive: {fp}"
)
print(
    f"False Negative: {fn}"
)
print(
    f"True Positive : {tp}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "background",
            "Ira"
        ],
        digits=4
    )
)

far = fp / (fp + tn)
frr = fn / (fn + tp)

print(f"False Acceptance Rate : {far:.6f}")
print(f"False Rejection Rate  : {frr:.6f}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)