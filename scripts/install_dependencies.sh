#!/bin/bash
set -e

echo "🚀 Installing dependencies for Qwen2.5-VL-32B..."

# Update system
apt update && apt upgrade -y

# Install system dependencies
apt install -y git wget curl htop nvtop supervisor nginx

# Install Python packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt

# Install flash-attention (can be tricky)
pip install flash-attn --no-build-isolation || echo "⚠️  Flash attention failed, continuing without it"

# Create log directories
mkdir -p logs
mkdir -p /var/log/qwen-server

echo "✅ Dependencies installed successfully!"
echo "💡 Run './scripts/start_server.sh' to start the server"