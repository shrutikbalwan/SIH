import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import stft
import tensorflow as tf

BASE_DIR = r"E:\SIH\ira-wakeword"

KERAS_MODEL = os.path.join(
    BASE_DIR, "cnn", "models", "ira_cnn_best.keras"
)

INT8_MODEL = os.path.join(
    BASE_DIR, "cnn", "models", "ira_cnn_int8.tflite"
)

TEST_DIR = os.path.join(
    BASE_DIR, "dataset", "splits", "test"
)

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 480
HOP_LENGTH = 320
NUM_BINS = 40
NUM_SAMPLES = 16000


def load_audio(path):

    audio, sr = sf.read(
        path,
        dtype="float32"
    )

    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected 16000 Hz, got {sr} Hz: {path}"
        )

    # IMPORTANT:
    # Same padding/truncation as train_cnn.py
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )
    else:
        audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


def make_spectrogram(audio):

    # IMPORTANT:
    # Same STFT implementation as train_cnn.py
    spec = tf.signal.stft(
        audio,
        frame_length=480,
        frame_step=320,
        fft_length=512
    )

    spec = tf.abs(spec)

    # IMPORTANT:
    # Same log operation as train_cnn.py
    spec = tf.math.log(
        spec + 1e-6
    )

    spec = spec[:, :40]

    # Same per-sample normalization
    mean = tf.reduce_mean(spec)
    std = tf.math.reduce_std(spec) + 1e-6

    spec = (
        spec - mean
    ) / std

    return spec.numpy().astype(
        np.float32
    )


print("=" * 60)
print("FLOAT vs INT8 MODEL COMPARISON")
print("=" * 60)


# ---------------------------------------------------------
# Load FLOAT model
# ---------------------------------------------------------

print("\nLoading FLOAT model:")
print(KERAS_MODEL)

keras_model = tf.keras.models.load_model(
    KERAS_MODEL,
    compile=False
)


# ---------------------------------------------------------
# Load INT8 model
# ---------------------------------------------------------

print("\nLoading INT8 model:")
print(INT8_MODEL)

interpreter = tf.lite.Interpreter(
    model_path=INT8_MODEL
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]
output_index = output_details["index"]

input_scale, input_zero = (
    input_details["quantization"]
)

output_scale, output_zero = (
    output_details["quantization"]
)

print("\nINT8 input:")
print("dtype      =", input_details["dtype"])
print("scale      =", input_scale)
print("zero_point =", input_zero)

print("\nINT8 output:")
print("dtype      =", output_details["dtype"])
print("scale      =", output_scale)
print("zero_point =", output_zero)


# ---------------------------------------------------------
# Test files
# ---------------------------------------------------------

files = []
labels = []

for category, label in [
    ("positive", 1),
    ("negative", 0)
]:

    folder = os.path.join(
        TEST_DIR,
        category
    )

    wavs = sorted(
        glob.glob(
            os.path.join(
                folder,
                "*.wav"
            )
        )
    )

    for path in wavs:
        files.append(path)
        labels.append(label)

labels = np.array(
    labels
)

print("\nTotal test files:", len(files))
print("Positive:", np.sum(labels))
print(
    "Negative:",
    len(labels) - np.sum(labels)
)


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

float_predictions = []
int8_predictions = []

print("\nRunning comparison...")

for i, path in enumerate(
    files,
    1
):

    audio = load_audio(path)

    spec = make_spectrogram(
        audio
    )

    x = spec[
        np.newaxis,
        ...,
        np.newaxis
    ]


    # -------------------------
    # FLOAT
    # -------------------------

    float_output = (
        keras_model.predict(
            x,
            verbose=0
        )
    )

    float_probability = float(
        float_output[0, 0]
    )


    # -------------------------
    # INT8
    # -------------------------

    x_quantized = np.round(
        x / input_scale
        + input_zero
    )

    x_quantized = np.clip(
        x_quantized,
        -128,
        127
    ).astype(
        np.int8
    )

    interpreter.set_tensor(
        input_index,
        x_quantized
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_index
    )

    int8_probability = float(
        (
            output.astype(
                np.float32
            )[0, 0]
            - output_zero
        )
        * output_scale
    )

    float_predictions.append(
        float_probability
    )

    int8_predictions.append(
        int8_probability
    )

    if i % 200 == 0:
        print(
            f"Processed {i}/{len(files)}"
        )


float_predictions = np.array(
    float_predictions
)

int8_predictions = np.array(
    int8_predictions
)


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

difference = np.abs(
    float_predictions
    - int8_predictions
)

print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)


print("\nFLOAT prediction:")

print(
    "min  =",
    np.min(float_predictions)
)

print(
    "max  =",
    np.max(float_predictions)
)

print(
    "mean =",
    np.mean(float_predictions)
)


print("\nINT8 prediction:")

print(
    "min  =",
    np.min(int8_predictions)
)

print(
    "max  =",
    np.max(int8_predictions)
)

print(
    "mean =",
    np.mean(int8_predictions)
)


print("\nAbsolute difference:")

print(
    "mean =",
    np.mean(difference)
)

print(
    "max  =",
    np.max(difference)
)

print(
    "median =",
    np.median(difference)
)


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

float_pred = (
    float_predictions >= 0.5
).astype(int)

int8_pred = (
    int8_predictions >= 0.5
).astype(int)


float_accuracy = np.mean(
    float_pred == labels
)

int8_accuracy = np.mean(
    int8_pred == labels
)


print(
    "\nAccuracy @ threshold 0.50:"
)

print(
    "FLOAT:",
    float_accuracy
)

print(
    "INT8 :",
    int8_accuracy
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

np.save(
    os.path.join(
        BASE_DIR,
        "cnn",
        "models",
        "float_predictions_compare.npy"
    ),
    float_predictions
)

np.save(
    os.path.join(
        BASE_DIR,
        "cnn",
        "models",
        "int8_predictions_compare.npy"
    ),
    int8_predictions
)


print(
    "\nComparison complete."
)