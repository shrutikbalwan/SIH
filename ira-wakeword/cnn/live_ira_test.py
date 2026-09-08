import numpy as np
import sounddevice as sd
import tensorflow as tf

MODEL_PATH = r"E:\SIH\ira-wakeword\cnn\models\ira_cnn_int8.tflite"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
DEVICE = 1

# Load INT8 TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero = input_details["quantization"]
output_scale, output_zero = output_details["quantization"]

print("=" * 60)
print("LIVE IRA WAKE-WORD TEST")
print("=" * 60)
print("Microphone device:", DEVICE)
print("Model:", MODEL_PATH)
print("Input:", input_details["shape"], input_details["dtype"])
print("Output:", output_details["shape"], output_details["dtype"])
print()
print("Speak 'Ira' clearly.")
print("Press Ctrl+C to stop.")
print("=" * 60)


def make_features(audio):
    audio = np.asarray(audio, dtype=np.float32)

    # Match training preprocessing exactly
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(
            audio,
            (0, WINDOW_SAMPLES - len(audio))
        )
    else:
        audio = audio[:WINDOW_SAMPLES]

    spec = tf.abs(
        tf.signal.stft(
            audio,
            frame_length=480,
            frame_step=320,
            fft_length=512
        )
    )

    spec = tf.math.log(spec + 1e-6)

    # First 40 frequency bins
    spec = spec[:, :40]

    # Per-sample normalization
    spec = (
        spec - tf.reduce_mean(spec)
    ) / tf.math.reduce_std(spec)

    return spec.numpy().astype(np.float32)


def predict(audio):
    features = make_features(audio)

    # Add channel dimension
    features = features[np.newaxis, :, :, np.newaxis]

    # Quantize input
    quantized = np.round(
        features / input_scale + input_zero
    )

    quantized = np.clip(
        quantized,
        -128,
        127
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        quantized
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details["index"]
    )

    probability = (
        output.astype(np.float32) - output_zero
    ) * output_scale

    return float(probability[0][0])


# Rolling buffer
audio_buffer = np.zeros(
    WINDOW_SAMPLES,
    dtype=np.float32
)


def audio_callback(indata, frames, time, status):
    global audio_buffer

    if status:
        print("Audio status:", status)

    samples = indata[:, 0]

    # Shift old audio out
    audio_buffer[:-frames] = audio_buffer[frames:]

    # Add new audio
    audio_buffer[-frames:] = samples


# Start microphone
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=1600,
    channels=1,
    dtype="float32",
    device=DEVICE,
    callback=audio_callback,
):
    print("Listening...\n")

    while True:
        sd.sleep(500)

        probability = predict(audio_buffer)

        print(
            f"Ira probability: {probability:.3f}",
            flush=True
        )