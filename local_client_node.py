import os
import shutil
import subprocess
import sys
import argparse
from config import SERVER_ADDRESS

class LocalClientNode:
    """
    Simulates a mobile device app that:
    1. Collects/Classifies images locally.
    2. Stores them in a private local directory.
    3. Joins the Federated Learning round to train and send weights.
    """
    
    def __init__(self, client_id):
        self.client_id = client_id
        self.storage_dir = f"local_storage/{client_id}"
        os.makedirs(self.storage_dir, exist_ok=True)
        print(f"Client {client_id} initialized. Storage: {self.storage_dir}")

    def capture_image(self, source_image_path, label):
        """
        Simulates the user taking a photo and providing feedback (the label).
        The image is saved locally in the client's private storage.
        """
        label_dir = os.path.join(self.storage_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        filename = os.path.basename(source_image_path)
        # Add a timestamp or unique ID to prevent overwriting
        dest_path = os.path.join(label_dir, f"captured_{len(os.listdir(label_dir))}_{filename}")
        
        shutil.copy(source_image_path, dest_path)
        print(f"Image captured and saved locally as '{label}': {dest_path}")

    def run_federated_sync(self):
        """
        Connects to the Flower server and participates in the training round.
        Only the weights are sent; the images stay on this 'device'.
        """
        print(f"\nStarting Federated Learning sync for {self.client_id}...")
        # We call the existing flower_client1.py logic
        cmd = [
            sys.executable, "flower_client1.py", 
            "--data_dir", self.storage_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Sync complete for {self.client_id}!")
        except subprocess.CalledProcessError as e:
            print(f"Error during sync: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a Local App Client Node")
    parser.add_argument("--id", type=str, default="user_1", help="Unique ID for this client")
    parser.add_argument("--capture", type=str, help="Path to an image to capture for training")
    parser.add_argument("--label", type=str, help="Label for the captured image")
    parser.add_argument("--sync", action="store_true", help="Start the Federated Learning sync")
    
    args = parser.parse_args()
    
    node = LocalClientNode(args.id)
    
    if args.capture and args.label:
        node.capture_image(args.capture, args.label)
    
    if args.sync:
        node.run_federated_sync()
