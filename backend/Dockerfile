FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Required for bitsandbytes to find CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV BNB_CUDA_VERSION=121

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy new script only
COPY qwen_model_server_script.py /app/qwen_model_server_script.py

# Expose API port
EXPOSE 9000

# Start FastAPI server with new script
CMD ["uvicorn", "qwen_model_server_script:app", "--host", "0.0.0.0", "--port", "9000"]
