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
    BASE_DIR, "cnn", "models", "ira_cnn_quantized_v3.tflite"
)

print("=" * 60)
print("IRA CNN - QUANTIZATION DIAGNOSTIC")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded.")
print("Input shape:", model.input_shape)

X = np.load(CALIBRATION_PATH).astype(np.float32)

print("\nCalibration data:")
print("Shape:", X.shape)
print("Min:", X.min())
print("Max:", X.max())
print("Mean:", X.mean())
print("Std:", X.std())


def representative_dataset():
    for i in range(len(X)):
        yield [X[i:i + 1]]


print("\nCreating quantized model...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

# INT8 weights/operations, but FLOAT32 interface.
# This lets us diagnose the quantization without
# forcing the input tensor to INT8 yet.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]

converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("\nSaved:")
print(OUTPUT_PATH)

print(
    f"Size: {os.path.getsize(OUTPUT_PATH) / 1024:.2f} KB"
)

print("\nInspecting tensors...")

interpreter = tf.lite.Interpreter(
    model_path=OUTPUT_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nINPUT")
print("dtype:", input_details[0]["dtype"])
print("shape:", input_details[0]["shape"])
print("quantization:", input_details[0]["quantization"])

print("\nOUTPUT")
print("dtype:", output_details[0]["dtype"])
print("shape:", output_details[0]["shape"])
print("quantization:", output_details[0]["quantization"])

print("\nQuantization diagnostic complete.")