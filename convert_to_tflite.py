import tensorflow as tf
import os

print("=" * 50)
print("  Model Conversion: .h5 -> .tflite")
print("=" * 50)

MODEL_PATH = "pretrained_global_model.h5"
OUTPUT_PATH = "emotion_model.tflite"

# Step 1: Load the trained model
print(f"\n[*] Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("[OK] Model loaded successfully!")
print(f"   Input shape  : {model.input_shape}")
print(f"   Output shape : {model.output_shape}")
print(f"   Total params : {model.count_params():,}")

# Step 2: Convert to TFLite
print("\n[*] Converting to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: optimize for size/speed (recommended for mobile)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
print("[OK] Conversion successful!")

# Step 3: Save the .tflite file
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

# Step 4: Show file size comparison
h5_size   = os.path.getsize(MODEL_PATH)   / (1024 * 1024)
tflite_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print(f"\n[INFO] File Size Comparison:")
print(f"   Original  .h5     : {h5_size:.2f} MB")
print(f"   Converted .tflite : {tflite_size:.2f} MB")
print(f"   Size reduction    : {((h5_size - tflite_size) / h5_size * 100):.1f}% smaller")

print(f"\n[DONE] Saved as: {OUTPUT_PATH}")
print("   This file is ready to use in your Flutter app.")
print("=" * 50)
