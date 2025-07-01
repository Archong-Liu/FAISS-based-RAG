# FAISS-based RAG Pipeline - Docker Setup

This guide explains how to run the FAISS-based RAG pipeline using Docker with CUDA support.

## Prerequisites

### 1. Install Docker
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows/Mac
- [Docker Engine](https://docs.docker.com/engine/install/) for Linux

### 2. Install NVIDIA Docker Runtime
For GPU support, you need nvidia-docker2:

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Windows:**
- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 3. Verify GPU Support
```bash
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Build the Docker Image
```bash
# Make scripts executable (Linux/Mac)
chmod +x build.sh run.sh

# Build the image
./build.sh
# or
docker-compose build
```

### 2. Run the Pipeline
```bash
# Run with logs
./run.sh
# or
docker-compose up

# Run in background
docker-compose up -d
```

### 3. Stop the Container
```bash
docker-compose down
```

## File Structure

```
FAISS-based-RAG/
├── data/                    # Mounted to /app/data
│   └── Sample.pdf          # Your PDF files
├── models/                  # Mounted to /app/models
│   └── llama-2-7b-chat.Q3_K_S.gguf  # Your LLM models
├── vector_store/           # Mounted to /app/vector_store
│   └── ...                 # Auto-generated vector stores
├── backend/
│   └── rag_pipeline.py     # Main RAG pipeline
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
└── build.sh, run.sh        # Helper scripts
```

## Configuration

### Environment Variables
You can modify these in `docker-compose.yml`:

- `CUDA_VISIBLE_DEVICES`: Which GPU to use (default: 0)
- `NVIDIA_VISIBLE_DEVICES`: Which GPUs are visible (default: all)

### GPU Memory
If you encounter GPU memory issues, you can:
1. Use a smaller model (Q2_K instead of Q3_K_S)
2. Reduce `n_ctx` in the LLM configuration
3. Use fewer GPU layers with `n_gpu_layers` parameter

## Troubleshooting

### 1. GPU Not Detected
```bash
# Check if nvidia-docker is working
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

### 2. CUDA Version Mismatch
If you get CUDA version errors, update the Dockerfile:
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04  # Use newer CUDA version
```

### 3. Memory Issues
Add memory limits to `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
    reservations:
      memory: 4G
```

### 4. Build Failures
If the build fails, try:
```bash
# Clean build
docker-compose build --no-cache

# Or rebuild specific stage
docker-compose build --no-cache rag-pipeline
```

## Development

### Interactive Shell
```bash
# Run container with interactive shell
docker-compose run --rm rag-pipeline bash

# Inside container, you can run:
python3 backend/rag_pipeline.py
```

### Debug Mode
```bash
# Run with debug output
docker-compose run --rm rag-pipeline python3 -u backend/rag_pipeline.py
```

## Performance Tips

1. **Use SSD storage** for better I/O performance
2. **Mount volumes** instead of copying large files
3. **Use appropriate model size** for your GPU memory
4. **Enable GPU layers** in llama-cpp-python for faster inference

## Next Steps

- Add a web interface (FastAPI/Streamlit)
- Implement batch processing
- Add model caching
- Set up monitoring and logging 