from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import json
import os
import glob
import io
import numpy as np
from PIL import Image
from config import METRICS_FILE, MODEL_DIR

# Use lightweight tflite-runtime on cloud; fall back to full TF locally
try:
    import tflite_runtime.interpreter as tflite
    TFLiteInterpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter

# ── Model config ─────────────────────────────────────────────────────────────
TFLITE_MODEL_PATH = "emotion_model.tflite"
EMOTION_LABELS    = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
IMAGE_SIZE        = (224, 224)

# ── Load TFLite interpreter once at startup ──────────────────────────────────
if os.path.exists(TFLITE_MODEL_PATH):
    _interpreter = TFLiteInterpreter(model_path=TFLITE_MODEL_PATH)
    _interpreter.allocate_tensors()
    _input_details  = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()
    print(f"[OK] TFLite model loaded from {TFLITE_MODEL_PATH}")
else:
    _interpreter = None
    print(f"[WARN] TFLite model not found at {TFLITE_MODEL_PATH} — /predict will be unavailable")

app = FastAPI(title="FL Emotion System Tracker")

def read_metrics():
    if not os.path.exists(METRICS_FILE):
        return []
    with open(METRICS_FILE, "r") as f:
        return [json.loads(line) for line in f.readlines()]

@app.get("/info")
def get_info():
    metrics = read_metrics()
    if not metrics:
        return {"status": "Training not started", "current_round": 0}
    latest = metrics[-1]
    return {
        "current_round": latest["round"],
        "latest_accuracy": latest["accuracy"],
        "total_clients": latest["total_clients_participated"]
    }

@app.get("/metrics")
def get_metrics_history():
    return {"history": read_metrics()}

@app.get("/model")
def download_model():
    # Find the latest .h5 file in the model directory
    list_of_files = glob.glob(f"{MODEL_DIR}/*.h5")
    if not list_of_files:
        return {"error": "No model found"}
    latest_file = max(list_of_files, key=os.path.getctime)
    return FileResponse(path=latest_file, filename=os.path.basename(latest_file))


@app.get("/model/version")
def get_model_version():
    """
    Returns the current model version (= latest FL round number).
    The Flutter app calls this on startup to check if a newer model
    is available than the one it already has saved locally.
    """
    metrics = read_metrics()
    if not metrics:
        return {"version": 0, "note": "Using base pre-trained model"}
    latest = metrics[-1]
    return {
        "version": latest["round"],
        "accuracy": latest["accuracy"],
    }


@app.get("/model/tflite")
def download_tflite():
    """
    Serves the latest emotion_model.tflite file.
    The Flutter app downloads this when /model/version returns
    a higher version than what the app currently has.
    """
    if not os.path.exists(TFLITE_MODEL_PATH):
        raise HTTPException(
            status_code=404,
            detail="TFLite model not found. Run convert_to_tflite.py first."
        )
    return FileResponse(
        path=TFLITE_MODEL_PATH,
        filename="emotion_model.tflite",
        media_type="application/octet-stream"
    )


@app.get("/health")
def health_check():
    """Quick sanity check — confirms the server and model are ready."""
    return {
        "status": "ok",
        "model_loaded": _interpreter is not None,
        "tflite_path": TFLITE_MODEL_PATH,
        "labels": EMOTION_LABELS,
    }


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Upload an image (jpg/png) and get back the predicted emotion.

    Returns:
        emotion      – top predicted emotion label
        confidence   – confidence score (0–1)
        all_scores   – probability for every emotion class
    """
    # ── Guard: model must be loaded ──────────────────────────────────────────
    if _interpreter is None:
        raise HTTPException(
            status_code=503,
            detail=f"TFLite model not found at '{TFLITE_MODEL_PATH}'. "
                   "Run convert_to_tflite.py first."
        )

    # ── 1. Read & validate the uploaded file ─────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Send a JPEG or PNG."
        )

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    # ── 2. Preprocess: resize → numpy → normalize ────────────────────────────
    img = img.resize(IMAGE_SIZE)                          # (224, 224)
    img_array = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)         # add batch dim → (1, 224, 224, 3)

    # ── 3. Run TFLite inference ───────────────────────────────────────────────
    _interpreter.set_tensor(_input_details[0]["index"], img_array)
    _interpreter.invoke()
    scores = _interpreter.get_tensor(_output_details[0]["index"])[0]  # shape: (num_classes,)

    # ── 4. Build response ─────────────────────────────────────────────────────
    top_idx    = int(np.argmax(scores))
    confidence = float(scores[top_idx])
    emotion    = EMOTION_LABELS[top_idx] if top_idx < len(EMOTION_LABELS) else f"class_{top_idx}"

    all_scores = {
        (EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f"class_{i}"): round(float(s), 4)
        for i, s in enumerate(scores)
    }

    return {
        "emotion":    emotion,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    }