import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import gc
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

from config import Config
from utils import setup_logging, get_gpu_info, clear_gpu_cache, validate_json_response

# Setup logging
logger = setup_logging()

app = FastAPI(
    title="Qwen2.5-VL-32B CAD Analyzer",
    description="Self-hosted Qwen2.5-VL-32B for CAD model analysis",
    version="1.0.0"
)

# Global variables
model = None
processor = None

class AnalysisRequest(BaseModel):
    stepFileContent: str
    originalPrompt: str

def load_model():
    global model, processor
    
    logger.info(f"Loading {Config.MODEL_NAME}...")
    logger.info(f"GPU Info: {get_gpu_info()}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        Config.MODEL_NAME, 
        trust_remote_code=True,
        cache_dir=Config.CACHE_DIR
    )
    
    # Load model with config
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=Config.CACHE_DIR,
        **Config.get_model_kwargs()
    )
    
    logger.info("Model loaded successfully!")
    logger.info(f"Final GPU Info: {get_gpu_info()}")
    
    # Clear cache
    gc.collect()
    clear_gpu_cache()

def analyze_cad_model(step_content: str, original_prompt: str) -> dict:
    system_prompt = """You are a CAD model analyzer. Analyze the STEP file and original prompt to determine if meaningful variations can be created.

Respond in JSON format:
{
  "canCreateVariations": boolean,
  "reason": "Brief explanation",
  "variations": [
    {
      "prompt": "Complete but concise CAD prompt for variation",
      "description": "What makes this different"
    }
  ]
}

PROMPT RULES:
1. Each variation prompt must be complete and standalone (not a modification instruction)
2. Keep prompts concise - restate core concept + add one key change
3. Match the brevity and style of the original prompt

Generate 2-4 variations for most models. Only generate 5-6 variations if the model is highly complex AND the original prompt is very detailed. Focus on: different dimensions, quantities, shapes, or arrangements."""
    
    user_content = f"""Original prompt: "{original_prompt}"

STEP file content:
{step_content}

Analyze this CAD model and determine if meaningful variations can be created."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process inputs
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE,
            do_sample=Config.DO_SAMPLE,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/analyze-cad-variations")
async def analyze_variations(request: AnalysisRequest, background_tasks: BackgroundTasks):
    try:
        logger.info("Analyzing CAD variations...")
        result = analyze_cad_model(request.stepFileContent, request.originalPrompt)
        
        # Clean up in background
        background_tasks.add_task(clear_gpu_cache)
        
        # Validate and return JSON
        return validate_json_response(result)
            
    except Exception as e:
        logger.error(f"Error analyzing CAD model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing CAD model: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": Config.MODEL_NAME,
        "gpu_info": get_gpu_info()
    }

@app.get("/")
async def root():
    return {
        "message": "Qwen2.5-VL-32B CAD Analyzer API",
        "model": Config.MODEL_NAME,
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level=Config.LOG_LEVEL)