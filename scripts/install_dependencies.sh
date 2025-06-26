#!/bin/bash
# scripts/install_dependencies_simple.sh
set -e

echo "🚀 Installing dependencies (without flash-attn)..."

# Update system
apt update && apt upgrade -y
apt install -y git wget curl htop nvtop

# Install Python packages
pip install --upgrade pip

# Install PyTorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements (no flash-attn)
pip install transformers==4.36.0
pip install accelerate bitsandbytes optimum
pip install qwen-vl-utils pillow requests
pip install fastapi uvicorn python-multipart pydantic

# Try xformers for better performance
pip install xformers==0.0.23 || echo "⚠️  xformers failed, using standard attention"

mkdir -p logs
echo "✅ Dependencies installed successfully (standard attention)!"