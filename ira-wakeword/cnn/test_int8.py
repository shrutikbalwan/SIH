import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import stft
import tensorflow as tf

BASE_DIR = r"E:\SIH\ira-wakeword"

MODEL_PATH = os.path.join(
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


def make_spectrogram(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    target_length = SAMPLE_RATE

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (target_length - len(audio), 0)
        )
    else:
        audio = audio[:target_length]

    _, _, Zxx = stft(
        audio,
        fs=SAMPLE_RATE,
        nperseg=WIN_LENGTH,
        noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False
    )

    spec = np.abs(Zxx).T
    spec = spec[:, :NUM_BINS]

    spec = np.log1p(spec)

    mean = np.mean(spec)
    std = np.std(spec) + 1e-6
    spec = (spec - mean) / std

    return spec.astype(np.float32)


print("=" * 60)
print("IRA INT8 TFLITE EVALUATION")
print("=" * 60)

print("\nLoading INT8 model:")
print(MODEL_PATH)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput:")
print(input_details[0])

print("\nOutput:")
print(output_details[0])

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_scale, input_zero = input_details[0]["quantization"]
output_scale, output_zero = output_details[0]["quantization"]

print("\nInput quantization:")
print("scale =", input_scale)
print("zero_point =", input_zero)

print("\nOutput quantization:")
print("scale =", output_scale)
print("zero_point =", output_zero)


files = []
labels = []

for category, label in [
    ("positive", 1),
    ("negative", 0)
]:
    folder = os.path.join(TEST_DIR, category)

    wavs = glob.glob(os.path.join(folder, "*.wav"))

    for path in wavs:
        files.append(path)
        labels.append(label)

print("\nTotal test files:", len(files))
print("Positive:", sum(labels))
print("Negative:", len(labels) - sum(labels))


predictions = []

print("\nRunning INT8 inference...")

for i, path in enumerate(files, 1):

    spec = make_spectrogram(path)

    x = spec[np.newaxis, ..., np.newaxis]

    # Convert float input to INT8 using model quantization.
    x_quantized = np.round(
        x / input_scale + input_zero
    )

    x_quantized = np.clip(
        x_quantized,
        -128,
        127
    ).astype(np.int8)

    interpreter.set_tensor(
        input_index,
        x_quantized
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_index
    )

    # Convert INT8 output back to float.
    probability = (
        (output.astype(np.float32) - output_zero)
        * output_scale
    )

    predictions.append(float(probability[0][0]))

    if i % 200 == 0:
        print(f"Processed {i}/{len(files)}")


predictions = np.array(predictions)
labels = np.array(labels)

np.save(
    os.path.join(
        BASE_DIR,
        "cnn",
        "models",
        "int8_test_probabilities.npy"
    ),
    predictions
)

print("\n" + "=" * 60)
print("INT8 RESULTS")
print("=" * 60)

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

print(
    f"\n{'Threshold':>10} "
    f"{'Accuracy':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'FAR':>10} "
    f"{'FRR':>10}"
)

for threshold in thresholds:

    predicted = (predictions >= threshold).astype(int)

    tp = np.sum((labels == 1) & (predicted == 1))
    tn = np.sum((labels == 0) & (predicted == 0))
    fp = np.sum((labels == 0) & (predicted == 1))
    fn = np.sum((labels == 1) & (predicted == 0))

    accuracy = (tp + tn) / len(labels)

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

    far = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

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


threshold = 0.90

predicted = (predictions >= threshold).astype(int)

tp = np.sum((labels == 1) & (predicted == 1))
tn = np.sum((labels == 0) & (predicted == 0))
fp = np.sum((labels == 0) & (predicted == 1))
fn = np.sum((labels == 1) & (predicted == 0))

print("\nUsing threshold:", threshold)

print("\nConfusion Matrix:")
print("True Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)

print("\nFalse Acceptance Rate :",
      fp / (fp + tn))

print("False Rejection Rate :",
      fn / (fn + tp))

print("\nINT8 evaluation complete.")