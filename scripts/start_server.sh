#!/bin/bash
cd "$(dirname "$0")/.."

echo "🚀 Starting Qwen2.5-VL-32B server..."

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface

# Create cache directory
mkdir -p /workspace/.cache/huggingface

# Start server
python src/model_server.py 2>&1 | tee logs/server.log