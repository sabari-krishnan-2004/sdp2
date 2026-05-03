import os

NUM_ROUNDS = 5
BATCH_SIZE = 32
LOCAL_EPOCHS = 2
NUM_CLASSES = 5
MODEL_DIR = "./global_model"
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
SERVER_ADDRESS = "127.0.0.1:8080"

os.makedirs(MODEL_DIR, exist_ok=True)