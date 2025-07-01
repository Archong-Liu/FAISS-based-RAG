# 使用官方 CUDA 基底映像檔
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 安裝必要工具與 Python 環境
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential cmake ninja-build git curl \
    && apt-get clean

# 建立工作目錄
WORKDIR /app

# clone 最新原始碼（或指定版本 tag）
RUN git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python /tmp/llama-cpp
WORKDIR /tmp/llama-cpp

ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip3 install . --no-cache-dir

# 回到工作目錄
WORKDIR /app


# Copy requirements first for better caching
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

# Install sentence-transformers separately (can be slow)
RUN pip3 install sentence-transformers

# Copy application code
COPY . .

# Expose port (if you want to add a web interface later)
EXPOSE 8000

# Set the default command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
