import os
import numpy as np
import tensorflow as tf

BASE_DIR = r"E:\SIH\ira-wakeword"

MODEL_PATH = os.path.join(
    BASE_DIR, "cnn", "models", "ira_cnn_best.keras"
)

CALIBRATION_PATH = os.path.join(
    BASE_DIR, "cnn", "models", "calibration_data.npy"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "cnn", "models", "ira_cnn_int8_v2.tflite"
)

print("=" * 60)
print("IRA CNN -> CORRECTED INT8 TFLITE")
print("=" * 60)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)

# ------------------------------------------------------------
# Load representative calibration data
# ------------------------------------------------------------

print("\nLoading calibration data:")
print(CALIBRATION_PATH)

X = np.load(CALIBRATION_PATH).astype(np.float32)

print("Calibration shape:", X.shape)

# Make sure shape is exactly what the model expects.
X = X.reshape((-1, 49, 40, 1))

print("Final calibration shape:", X.shape)

# Use all 500 calibration samples.
num_samples = len(X)

print("Calibration samples:", num_samples)


def representative_dataset():
    for i in range(num_samples):
        sample = X[i:i + 1]
        yield [sample]


# ------------------------------------------------------------
# Convert
# ------------------------------------------------------------

print("\nStarting full INT8 conversion...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

# Require every operation to use INT8.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# Force INT8 input and output.
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_bytes = os.path.getsize(OUTPUT_PATH)
size_kb = size_bytes / 1024

print("\n" + "=" * 60)
print("CORRECTED INT8 CONVERSION COMPLETE")
print("=" * 60)

print("\nOutput:")
print(OUTPUT_PATH)

print(f"\nModel size: {size_bytes:,} bytes")
print(f"Model size: {size_kb:.2f} KB")

# ------------------------------------------------------------
# Inspect tensors
# ------------------------------------------------------------

print("\nChecking model tensors...")

interpreter = tf.lite.Interpreter(
    model_path=OUTPUT_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput dtype:")
print(input_details[0]["dtype"])

print("Input quantization:")
print(input_details[0]["quantization"])

print("\nOutput dtype:")
print(output_details[0]["dtype"])

print("Output quantization:")
print(output_details[0]["quantization"])

print("\nDone.")