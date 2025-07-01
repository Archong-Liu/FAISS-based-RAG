#!/bin/bash

echo "Running FAISS-based RAG Pipeline in Docker..."

# Check if container is already running
if docker ps | grep -q "faiss-rag-pipeline"; then
    echo "Container is already running. Stopping it first..."
    docker-compose down
fi

# Run the container
echo "Starting container..."
docker-compose up

echo "Container stopped." 