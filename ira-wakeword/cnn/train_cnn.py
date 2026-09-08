import os
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.signal import resample_poly

# ============================================================
# Ira Wake Word - Lightweight CNN
# ============================================================

PROJECT = r"E:\SIH\ira-wakeword"
DATASET = os.path.join(PROJECT, "dataset", "splits")
MODEL_DIR = os.path.join(PROJECT, "cnn", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

BATCH_SIZE = 64
EPOCHS = 30
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------
# Audio loading
# ------------------------------------------------------------

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)

    # Fix length to exactly 1 second
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    else:
        audio = audio[:NUM_SAMPLES]

    return audio


# ------------------------------------------------------------
# Spectrogram
# ------------------------------------------------------------

def make_spectrogram(audio):
    spec = tf.signal.stft(
        audio,
        frame_length=480,       # 30 ms
        frame_step=320,         # 20 ms
        fft_length=512
    )

    spec = tf.abs(spec)

    # Log compression
    spec = tf.math.log(spec + 1e-6)

    # Remove DC / highest unused bins
    spec = spec[:, :40]

    # Normalize per sample
    mean = tf.reduce_mean(spec)
    std = tf.math.reduce_std(spec) + 1e-6
    spec = (spec - mean) / std

    return spec.numpy().astype(np.float32)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

def load_dataset(split):
    positive_dir = os.path.join(DATASET, split, "positive")
    negative_dir = os.path.join(DATASET, split, "negative")

    paths = []
    labels = []

    for filename in os.listdir(positive_dir):
        if filename.lower().endswith(".wav"):
            paths.append(os.path.join(positive_dir, filename))
            labels.append(1)

    for filename in os.listdir(negative_dir):
        if filename.lower().endswith(".wav"):
            paths.append(os.path.join(negative_dir, filename))
            labels.append(0)

    # Shuffle
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(paths))

    paths = [paths[i] for i in indices]
    labels = np.array([labels[i] for i in indices], dtype=np.float32)

    X = []
    y = []

    print(f"\nLoading {split}: {len(paths)} files")

    for i, path in enumerate(paths):
        audio = load_audio(path)
        spec = make_spectrogram(audio)

        X.append(spec)
        y.append(labels[i])

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(paths)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # CNN channel dimension
    X = X[..., np.newaxis]

    print(f"{split} shape: {X.shape}")
    print(f"{split} positives: {int(np.sum(y == 1))}")
    print(f"{split} negatives: {int(np.sum(y == 0))}")

    return X, y


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

X_train, y_train = load_dataset("train")
X_val, y_val = load_dataset("validation")
X_test, y_test = load_dataset("test")


# ------------------------------------------------------------
# CNN model
# ------------------------------------------------------------

input_shape = X_train.shape[1:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),

    tf.keras.layers.Conv2D(
        8,
        (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)


model.summary()


# ------------------------------------------------------------
# Class weighting
# ------------------------------------------------------------

positive_count = np.sum(y_train == 1)
negative_count = np.sum(y_train == 0)

total = positive_count + negative_count

class_weight = {
    0: total / (2.0 * negative_count),
    1: total / (2.0 * positive_count),
}

print("\nClass weights:")
print(class_weight)


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "ira_cnn_best.keras"),
        monitor="val_recall",
        mode="max",
        save_best_only=True
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("\n========================================")
print("Training Ira CNN")
print("========================================\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

print("\n========================================")
print("TEST RESULTS")
print("========================================\n")

results = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.6f}")


# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------

final_path = os.path.join(
    MODEL_DIR,
    "ira_cnn_final.keras"
)

model.save(final_path)

print("\nSaved:")
print(final_path)
