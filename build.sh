#!/bin/bash

echo "Building FAISS-based RAG Pipeline with CUDA support..."

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q "nvidia"; then
    echo "Warning: NVIDIA Docker runtime not detected. GPU support may not work."
    echo "Please install nvidia-docker2: https://github.com/NVIDIA/nvidia-docker"
fi

# Build the Docker image
echo "Building Docker image..."
docker-compose build

echo "Build completed!"
echo ""
echo "To run the container:"
echo "  docker-compose up"
echo ""
echo "To run in background:"
echo "  docker-compose up -d"
echo ""
echo "To stop the container:"
echo "  docker-compose down" 