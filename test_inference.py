import numpy as np
from PIL import Image
import os

# Use tflite-runtime if available, otherwise fallback to tensorflow
try:
    import tflite_runtime.interpreter as tflite
    TFLiteInterpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter

# Model Configuration
TFLITE_MODEL_PATH = "emotion_model.tflite"
EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
IMAGE_SIZE = (224, 224)

def test_model(image_path):
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"❌ Error: Model not found at '{TFLITE_MODEL_PATH}'.")
        print("Please run convert_to_tflite.py first.")
        return

    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at '{image_path}'.")
        return

    print(f"✅ Loading TFLite model from '{TFLITE_MODEL_PATH}'...")
    interpreter = TFLiteInterpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"📷 Processing image '{image_path}'...")
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    print("🧠 Running inference...")
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"])[0]

    # Display results
    top_idx = int(np.argmax(scores))
    confidence = float(scores[top_idx])
    predicted_emotion = EMOTION_LABELS[top_idx] if top_idx < len(EMOTION_LABELS) else f"Unknown ({top_idx})"

    print("\n" + "="*30)
    print(f"🎉 Predicted Emotion: {predicted_emotion.upper()}")
    print(f"📊 Confidence Score:  {confidence*100:.2f}%")
    print("="*30)
    
    print("\nDetailed Probabilities:")
    for i, score in enumerate(scores):
        label = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f"Class {i}"
        print(f" - {label.ljust(10)} : {score*100:.2f}%")

if __name__ == "__main__":
    # Feel free to change this path to any image you want to test!
    TEST_IMAGE = "test_photo.jpg" 
    test_model(TEST_IMAGE)
