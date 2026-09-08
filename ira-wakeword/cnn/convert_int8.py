import os
import numpy as np
import tensorflow as tf

BASE_DIR = r"E:\SIH\ira-wakeword"

MODEL_PATH = os.path.join(
    BASE_DIR, "cnn", "models", "ira_cnn_best.keras"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "cnn", "models", "ira_cnn_int8.tflite"
)

CALIBRATION_PATH = os.path.join(
    BASE_DIR, "cnn", "models", "calibration_data.npy"
)

print("=" * 60)
print("IRA CNN -> INT8 TFLITE CONVERSION")
print("=" * 60)

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)

# ------------------------------------------------------------
# Calibration data
# ------------------------------------------------------------

print("\nLoading calibration data:")

X = np.load(CALIBRATION_PATH)

print("Calibration shape:", X.shape)

# Use a subset for representative calibration
num_samples = min(500, len(X))
X_calibration = X[:num_samples].astype(np.float32)

print("Calibration samples:", num_samples)

def representative_dataset():
    for i in range(num_samples):
        yield [X_calibration[i:i+1]]

# ------------------------------------------------------------
# Convert
# ------------------------------------------------------------

print("\nConverting to INT8...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

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
print("CONVERSION COMPLETE")
print("=" * 60)

print("Output:")
print(OUTPUT_PATH)

print(f"\nINT8 model size: {size_bytes:,} bytes")
print(f"INT8 model size: {size_kb:.2f} KB")

print("\nDone.")