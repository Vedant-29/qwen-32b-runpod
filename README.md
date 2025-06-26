# Qwen2.5-VL-32B RunPod Server

Self-hosted Qwen2.5-VL-32B model for CAD analysis on RunPod.

## Quick Setup

```bash
git clone <your-repo-url>
cd qwen-32b-runpod
chmod +x scripts/*.sh
./scripts/install_dependencies.sh
./scripts/start_server.sh
```

## API Endpoints

- `POST /analyze-cad-variations` - Analyze CAD models
- `GET /health` - Health check
- `GET /` - API info

## Hardware Requirements

- A100 80GB (recommended)
- 60GB+ storage