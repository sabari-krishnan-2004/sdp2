import subprocess
import time
import sys
import os

# Create local storage if it doesn't exist
os.makedirs("local_storage/client_1", exist_ok=True)
os.makedirs("local_storage/client_2", exist_ok=True)

# We use the new local_client_node.py to simulate app users
clients = [
    {"id": "user_1", "actions": [("test_photo.jpg", "happy"), ("test_photo.jpg", "sad")]},
    {"id": "user_2", "actions": [("test_photo.jpg", "happy"), ("test_photo.jpg", "sad")]},
]

print("Starting Simulation...")

# Step 1: Simulate users capturing images
for client in clients:
    for photo, label in client["actions"]:
        print(f"Capture: {client['id']} capturing {label}...")
        subprocess.run([
            sys.executable, "local_client_node.py",
            "--id", client["id"],
            "--capture", photo,
            "--label", label
        ])

print("\nConnecting clients to FL Server for training...")
processes = []
for client in clients:
    # Step 2: Trigger Federated Sync for each user
    p = subprocess.Popen([
        sys.executable, "local_client_node.py",
        "--id", client["id"],
        "--sync"
    ])
    processes.append(p)
    time.sleep(2) # Stagger starts

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    for p in processes:
        p.terminate()

print("\nSimulation Complete.")