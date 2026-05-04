import flwr as fl
import tensorflow as tf
import flwr.common as flwr_common
import numpy as np
import os
import json
import logging
from config import METRICS_FILE, MODEL_DIR, TFLITE_MODEL_PATH, NUM_CLASSES, SERVER_ADDRESS, logger

# Create a folder to save the federated models if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Custom Strategy that saves the Global Model after each round and converts to TFLite."""
    
    def aggregate_fit(self, server_round, results, failures):
        # 1. Call the parent class to do the actual FedAvg math
        aggregated_weights, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        # Track how many clients participated this round
        self.last_num_clients = len(results)

        # 2. If the aggregation was successful, save the model
        if aggregated_weights is not None:
            logger.info(f"Round {server_round} completed! Aggregating weights...")
            
            try:
                # Convert Flower parameters back to lists of numpy arrays
                weights = flwr_common.parameters_to_ndarrays(aggregated_weights)
                
                # Build an empty shell of the model to hold the weights
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
                
                # Inject the newly averaged weights into the shell
                model.set_weights(weights)
                
                # Save the file to the hard drive (.h5)
                h5_path = os.path.join(MODEL_DIR, f"federated_model_round_{server_round}.h5")
                model.save(h5_path)
                logger.info(f"Global model saved to: {h5_path}")

                # 3. AUTOMATIC TFLITE CONVERSION
                logger.info(f"Generating TFLite model for round {server_round}...")
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                tflite_model = converter.convert()
                with open(TFLITE_MODEL_PATH, "wb") as f:
                    f.write(tflite_model)
                logger.info(f"Successfully updated {TFLITE_MODEL_PATH}")

            except Exception as e:
                logger.error(f"Error during model aggregation or conversion: {e}")
            
        return aggregated_weights, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregates evaluation results and writes accuracy metrics to metrics.json."""
        aggregated_loss, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        if results:
            try:
                total_examples = sum(res.num_examples for _, res in results)
                weighted_accuracy = sum(
                    res.metrics["accuracy"] * res.num_examples
                    for _, res in results
                    if "accuracy" in res.metrics
                ) / total_examples if total_examples > 0 else 0.0

                record = {
                    "round": server_round,
                    "accuracy": round(weighted_accuracy, 4),
                    "total_clients_participated": getattr(self, "last_num_clients", len(results)),
                    "timestamp": os.path.getmtime(TFLITE_MODEL_PATH) if os.path.exists(TFLITE_MODEL_PATH) else 0
                }

                with open(METRICS_FILE, "a") as f:
                    f.write(json.dumps(record) + "\n")

                logger.info(f"METRICS | Round {server_round} | Accuracy: {weighted_accuracy:.4f} | Clients: {record['total_clients_participated']}")
            except Exception as e:
                logger.error(f"Error during metrics aggregation: {e}")

        return aggregated_loss, metrics_aggregated

def main():
    logger.info("Starting Federated Emotion Recognition Server...")
    
    # 1. Warm Start Model Check
    WARM_START_MODEL = "pretrained_global_model.h5"
    if not os.path.exists(WARM_START_MODEL):
        logger.error(f"Missing warm start model '{WARM_START_MODEL}'. Please run trained_model.py first.")
        return

    try:
        logger.info(f"Loading warm start model: {WARM_START_MODEL}")
        pretrained_model = tf.keras.models.load_model(WARM_START_MODEL)
        weights = pretrained_model.get_weights()
        initial_parameters = flwr_common.ndarrays_to_parameters(weights)
    except Exception as e:
        logger.error(f"Failed to load warm start model: {e}")
        return

    # 2. Strategy Setup
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=int(os.getenv("MIN_FIT_CLIENTS", 2)),
        min_evaluate_clients=int(os.getenv("MIN_EVAL_CLIENTS", 2)),
        min_available_clients=int(os.getenv("MIN_AVAIL_CLIENTS", 2)),
        initial_parameters=initial_parameters
    )

    # 3. Start Flower Server
    try:
        logger.info(f"FL Server listening on {SERVER_ADDRESS}")
        fl.server.start_server(
            server_address=SERVER_ADDRESS,
            config=fl.server.ServerConfig(num_rounds=int(os.getenv("NUM_ROUNDS", 5))),
            strategy=strategy,
        )
    except Exception as e:
        logger.critical(f"FL Server crashed: {e}")

if __name__ == "__main__":
    main()