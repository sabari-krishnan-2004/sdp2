import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", 2))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 5))

MODEL_DIR = os.getenv("MODEL_DIR", "./global_model")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")

# Paths shared between API and Flower Server
TFLITE_MODEL_PATH = os.getenv("TFLITE_MODEL_PATH", "emotion_model.tflite")
EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
IMAGE_SIZE = (224, 224)

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("system.log")
    ]
)
logger = logging.getLogger("FL-System")