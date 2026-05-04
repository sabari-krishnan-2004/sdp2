from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import glob
import io
import time
import numpy as np
from PIL import Image
import flwr as fl
from pydantic import BaseModel
from typing import List
from config import METRICS_FILE, MODEL_DIR, TFLITE_MODEL_PATH, EMOTION_LABELS, IMAGE_SIZE, SERVER_ADDRESS, logger

# Use lightweight tflite-runtime on cloud; fall back to full TF locally
try:
    import tflite_runtime.interpreter as tflite
    TFLiteInterpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter

# ── Global Interpreter Instance ──────────────────────────────────────────────
_interpreter = None
_input_details = None
_output_details = None
_last_model_mtime = 0

def load_interpreter():
    global _interpreter, _input_details, _output_details, _last_model_mtime
    if os.path.exists(TFLITE_MODEL_PATH):
        try:
            mtime = os.path.getmtime(TFLITE_MODEL_PATH)
            if mtime > _last_model_mtime:
                _interpreter = TFLiteInterpreter(model_path=TFLITE_MODEL_PATH)
                _interpreter.allocate_tensors()
                _input_details  = _interpreter.get_input_details()
                _output_details = _interpreter.get_output_details()
                _last_model_mtime = mtime
                logger.info(f"TFLite model loaded/updated from {TFLITE_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
    else:
        _interpreter = None
        logger.warning(f"TFLite model not found at {TFLITE_MODEL_PATH}")

# Initial load
load_interpreter()

app = FastAPI(
    title="FL Emotion System API",
    description="Production-ready API for Federated Learning model serving and inference.",
    version="2.0.0"
)

# ── Middleware ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ── Helper Functions ────────────────────────────────────────────────────────
def read_metrics():
    if not os.path.exists(METRICS_FILE):
        return []
    try:
        with open(METRICS_FILE, "r") as f:
            return [json.loads(line) for line in f.readlines()]
    except Exception as e:
        logger.error(f"Error reading metrics file: {e}")
        return []

# ── Flutter Bridge Schema ─────────────────────────────────────────────────────
class WeightSyncRequest(BaseModel):
    client_id: str
    weights: List[List[float]]  # List of flattened layer weights
    sample_count: int
    accuracy: float

class ProxyFlowerClient(fl.client.NumPyClient):
    """
    A temporary client that acts as a bridge for a mobile app.
    It takes pre-computed weights from the app and feeds them into the Flower Server.
    """
    def __init__(self, weights, sample_count, accuracy):
        self.weights = [np.array(w, dtype=np.float32) for w in weights]
        self.sample_count = sample_count
        self.accuracy = accuracy

    def get_parameters(self, config):
        return self.weights

    def fit(self, parameters, config):
        # We don't train here; we just return the weights we got from the Flutter app
        logger.info("ProxyClient: Delivering weights from mobile app to FL server.")
        return self.weights, self.sample_count, {}

    def evaluate(self, parameters, config):
        # Return the accuracy the app reported locally
        return 0.0, self.sample_count, {"accuracy": self.accuracy}

# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/model/sync")
async def sync_weights(request: WeightSyncRequest):
    """
    Endpoint for the Flutter app to submit its locally trained weights.
    The API server will then act as a Flower Client and join the FL round.
    """
    logger.info(f"Received weight sync request from {request.client_id} ({request.sample_count} samples)")
    
    try:
        # 1. Start a temporary Flower Client to bridge this to the Flower Server
        client = ProxyFlowerClient(request.weights, request.sample_count, request.accuracy)
        
        # We run the client sync in-process. 
        # Note: This blocks until the FL server finishes the round or rejects the client.
        fl.client.start_numpy_client(
            server_address=SERVER_ADDRESS,
            client=client
        )
        
        return {"status": "success", "message": "Weights aggregated into global model."}
    except Exception as e:
        logger.error(f"Failed to bridge weights to Flower server: {e}")
        raise HTTPException(status_code=500, detail=f"Flower Bridge Error: {e}")

@app.get("/info")
async def get_info():
    metrics = read_metrics()
    if not metrics:
        return {"status": "Training not started", "current_round": 0}
    latest = metrics[-1]
    return {
        "current_round": latest.get("round", 0),
        "latest_accuracy": latest.get("accuracy", 0),
        "total_clients": latest.get("total_clients_participated", 0)
    }

@app.get("/metrics")
async def get_metrics_history():
    return {"history": read_metrics()}

@app.get("/model/latest")
async def download_latest_h5():
    """Finds and serves the latest trained .h5 model."""
    list_of_files = glob.glob(os.path.join(MODEL_DIR, "*.h5"))
    if not list_of_files:
        raise HTTPException(status_code=404, detail="No aggregated models found yet.")
    
    latest_file = max(list_of_files, key=os.path.getctime)
    return FileResponse(
        path=latest_file, 
        filename=os.path.basename(latest_file),
        media_type="application/octet-stream"
    )

@app.get("/model/version")
async def get_model_version():
    """Returns metadata about the latest global model version."""
    metrics = read_metrics()
    if not metrics:
        return {"version": 0, "note": "Using base pre-trained model", "accuracy": 0}
    latest = metrics[-1]
    return {
        "version": latest.get("round", 0),
        "accuracy": latest.get("accuracy", 0),
        "timestamp": time.time()
    }

@app.get("/model/tflite")
async def download_tflite():
    """Serves the latest TFLite model for mobile deployment."""
    if not os.path.exists(TFLITE_MODEL_PATH):
        raise HTTPException(status_code=404, detail="TFLite model not yet generated.")
    
    return FileResponse(
        path=TFLITE_MODEL_PATH,
        filename="emotion_model.tflite",
        media_type="application/octet-stream"
    )

@app.get("/health")
async def health_check():
    # Check if model needs reloading
    load_interpreter()
    return {
        "status": "healthy",
        "model_loaded": _interpreter is not None,
        "mtime": _last_model_mtime
    }

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """Upload an image and get an emotion prediction using the latest TFLite model."""
    # Ensure model is up to date
    load_interpreter()

    if _interpreter is None:
        raise HTTPException(status_code=503, detail="Model server is initializing or model is missing.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Inference
        _interpreter.set_tensor(_input_details[0]["index"], img_array)
        _interpreter.invoke()
        scores = _interpreter.get_tensor(_output_details[0]["index"])[0]

        top_idx = int(np.argmax(scores))
        confidence = float(scores[top_idx])
        emotion = EMOTION_LABELS[top_idx] if top_idx < len(EMOTION_LABELS) else f"class_{top_idx}"

        all_scores = {
            (EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f"class_{i}"): round(float(s), 4)
            for i, s in enumerate(scores)
        }

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "version": _last_model_mtime
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Error during image processing or inference.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))