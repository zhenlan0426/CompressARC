#!/bin/bash

# This script starts NVIDIA MPS, runs train.py, and stops MPS.
# It assumes sudo access without password or proper configuration.

# Start MPS daemon in background
echo "Starting MPS..."
sudo nvidia-cuda-mps-control -d

# Check if MPS started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start MPS. Exiting."
    exit 1
fi

# Run the Python training script
echo "Running train.py..."
python train.py

# Stop MPS
echo "Stopping MPS..."
echo quit | sudo nvidia-cuda-mps-control

# Check if MPS stopped successfully
if [ $? -ne 0 ]; then
    echo "Failed to stop MPS. Please stop it manually."
else
    echo "MPS stopped successfully."
fi

echo "Script completed."