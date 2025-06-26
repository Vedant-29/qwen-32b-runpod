import os
from typing import Dict, Any

class Config:
    # Model settings
    MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"
    DEVICE_MAP = "auto"
    TORCH_DTYPE = "bfloat16"
    LOAD_IN_8BIT = True
    USE_FLASH_ATTENTION = True
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    LOG_LEVEL = "info"
    
    # Generation settings
    MAX_NEW_TOKENS = 1000
    TEMPERATURE = 0.8
    DO_SAMPLE = True
    
    # Cache settings
    CACHE_DIR = "/workspace/.cache/huggingface"
    
    @classmethod
    def get_model_kwargs(cls) -> Dict[str, Any]:
        kwargs = {
            "torch_dtype": getattr(__import__('torch'), cls.TORCH_DTYPE),
            "device_map": cls.DEVICE_MAP,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if cls.LOAD_IN_8BIT:
            kwargs["load_in_8bit"] = True
            
        if cls.USE_FLASH_ATTENTION:
            kwargs["attn_implementation"] = "flash_attention_2"
            
        return kwargs