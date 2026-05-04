import flwr as fl
import tensorflow as tf
import argparse
import numpy as np
import os
from PIL import Image
from config import BATCH_SIZE, NUM_CLASSES, LOCAL_EPOCHS, SERVER_ADDRESS, IMAGE_SIZE, logger

def load_data_manual(data_dir):
    """Manually loads images from subdirectories to avoid Keras split issues on small datasets."""
    logger.info(f"Loading data manually from: {data_dir}")
    
    X = []
    y = []
    
    # Map folder names to class indices (0-4 based on config)
    # We'll use the subdirectories as labels
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {name: i for i, name in enumerate(subdirs)}
    
    for label_name in subdirs:
        label_path = os.path.join(data_dir, label_name)
        for img_name in os.listdir(label_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    img_path = os.path.join(label_path, img_name)
                    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
                    X.append(np.array(img, dtype=np.float32) / 255.0)
                    y.append(label_map[label_name])
                except Exception as e:
                    logger.warning(f"Skipping image {img_name}: {e}")
    
    if not X:
        raise ValueError(f"No valid images found in {data_dir}")
        
    X = np.array(X)
    y = np.array(y)
    
    # Create TF dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(BATCH_SIZE)
    return dataset, dataset, len(X), len(X) # Use same for val in small dataset simulation

def build_model():
    """Builds the exact same MobileNetV2 architecture as the server."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

class EmotionClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, val_dataset, n_train, n_val):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_train = n_train
        self.n_val = n_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        logger.info("Received global weights. Starting local training...")
        self.model.set_weights(parameters)
        
        self.model.fit(
            self.train_dataset,
            epochs=LOCAL_EPOCHS,
            verbose=1
        )
        
        logger.info("Local training complete. Sending updated weights.")
        return self.model.get_weights(), self.n_train, {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.val_dataset, verbose=0)
        logger.info(f"Local Evaluation - Accuracy: {accuracy:.4f}")
        return loss, self.n_val, {"accuracy": accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    try:
        train_ds, val_ds, n_train, n_val = load_data_manual(args.data_dir)
        model = build_model()
        logger.info(f"Connecting to FL Server at {SERVER_ADDRESS}...")
        fl.client.start_numpy_client(
            server_address=SERVER_ADDRESS,
            client=EmotionClient(model, train_ds, val_ds, n_train, n_val)
        )
    except Exception as e:
        logger.critical(f"Client failed: {e}")
        exit(1)