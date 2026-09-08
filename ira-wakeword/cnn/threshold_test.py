import time
import numpy as np
import sounddevice as sd
import tensorflow as tf

MODEL_PATH = r"E:\SIH\ira-wakeword\cnn\models\ira_cnn_int8.tflite"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
BLOCK_SIZE = 8000
DEVICE = 1

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero = input_details["quantization"]
output_scale, output_zero = output_details["quantization"]

buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)


def make_features(audio):
    audio = audio[:WINDOW_SAMPLES]

    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(
            audio,
            (0, WINDOW_SAMPLES - len(audio))
        )

    spec = tf.abs(
        tf.signal.stft(
            audio,
            frame_length=480,
            frame_step=320,
            fft_length=512
        )
    )

    spec = tf.math.log(spec + 1e-6)
    spec = spec[:, :40]

    spec = (
        spec - tf.reduce_mean(spec)
    ) / tf.math.reduce_std(spec)

    return spec.numpy().astype(np.float32)


def predict(audio):
    features = make_features(audio)
    features = features[np.newaxis, :, :, np.newaxis]

    q = np.round(
        features / input_scale + input_zero
    )

    q = np.clip(q, -128, 127).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        q
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details["index"]
    )

    probability = (
        output.astype(np.float32) - output_zero
    ) * output_scale

    return float(probability[0][0])


def callback(indata, frames, time_info, status):
    global buffer

    samples = indata[:, 0]

    if frames >= WINDOW_SAMPLES:
        buffer[:] = samples[-WINDOW_SAMPLES:]
    else:
        buffer[:-frames] = buffer[frames:]
        buffer[-frames:] = samples


print("=" * 60)
print("IRA THRESHOLD TEST")
print("=" * 60)
print()
print("Device: 1 - Headset (EarPods)")
print()
print("TEST PROCEDURE")
print("----------------")
print("1. Stay SILENT for 5 seconds.")
print("2. Say IRA clearly 5 times.")
print("3. Say OTHER WORDS for 5 seconds.")
print("4. Say IRA clearly 5 more times.")
print()
print("The program prints one score every 0.5 seconds.")
print("Press Ctrl+C when finished.")
print("=" * 60)

scores = []

start_time = time.time()

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    dtype="float32",
    device=DEVICE,
    callback=callback,
):
    while True:
        time.sleep(0.5)

        probability = predict(buffer)
        elapsed = time.time() - start_time

        scores.append(probability)

        print(
            f"{elapsed:7.1f}s    "
            f"Ira probability = {probability:.3f}",
            flush=True
        )