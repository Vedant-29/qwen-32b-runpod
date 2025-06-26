import torch
import logging
import json
from typing import Dict, Any

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/server.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_info[f"gpu_{i}"] = {
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "allocated_memory_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
            "cached_memory_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2)
        }
    
    return gpu_info

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def validate_json_response(text: str) -> Dict[str, Any]:
    """Validate and clean JSON response from model"""
    try:
        # Try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        
        # Return default structure
        return {
            "canCreateVariations": False,
            "reason": "Invalid response format from model",
            "variations": []
        }