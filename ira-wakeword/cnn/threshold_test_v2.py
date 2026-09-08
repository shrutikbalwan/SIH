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
    audio = np.asarray(audio, dtype=np.float32)

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
print("IRA CONTROLLED THRESHOLD TEST")
print("=" * 60)
print()
print("Device: 1 - Headset (EarPods)")
print()
print("PHASE 1: SILENCE")
print("Stay completely silent for 10 seconds.")
print()
print("PHASE 2: IRA")
print("Say 'Ira' clearly 5-10 times during 10 seconds.")
print()
print("PHASE 3: OTHER SPEECH")
print("Say unrelated words for 10 seconds.")
print()
print("The program will automatically move between phases.")
print("=" * 60)

all_scores = []


with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    dtype="float32",
    device=DEVICE,
    callback=callback,
):

    # Let the audio buffer fill
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    phases = [
        ("SILENCE", 10),
        ("IRA", 10),
        ("OTHER_SPEECH", 10),
    ]

    for phase_name, duration in phases:

        print()
        print("=" * 60)
        print(f"NOW: {phase_name}")
        print("=" * 60)

        phase_scores = []
        start = time.time()

        while time.time() - start < duration:

            probability = predict(buffer)
            phase_scores.append(probability)

            elapsed = time.time() - start

            print(
                f"{phase_name:12s} "
                f"{elapsed:5.1f}s  "
                f"probability = {probability:.3f}",
                flush=True
            )

            time.sleep(0.5)

        phase_scores = np.array(phase_scores)

        print()
        print(f"{phase_name} SUMMARY")
        print(f"  Samples : {len(phase_scores)}")
        print(f"  Minimum : {phase_scores.min():.3f}")
        print(f"  Maximum : {phase_scores.max():.3f}")
        print(f"  Mean    : {phase_scores.mean():.3f}")
        print(f"  Median  : {np.median(phase_scores):.3f}")

        all_scores.append(
            (phase_name, phase_scores)
        )


print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for phase_name, scores in all_scores:
    print(
        f"{phase_name:12s} "
        f"min={scores.min():.3f}  "
        f"max={scores.max():.3f}  "
        f"mean={scores.mean():.3f}  "
        f"median={np.median(scores):.3f}"
    )

print("=" * 60)