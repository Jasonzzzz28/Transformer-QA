from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time
import os
from prometheus_fastapi_instrumentator import Instrumentator

# TODO: Test the fastapi server

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define API models
class QARequest(BaseModel):
    # context: str
    question: str
    max_length: Optional[int] = 512
    
class QAResponse(BaseModel):
    answer: str
    elapsed_time: float

# Initialize FastAPI app
app = FastAPI(
    title="Question Answering API",
    description="API for answering questions based on context using a transformer model",
    version="1.0.0"
)

# Global variables for model and tokenizer
# base_model_name = "meta-llama/Llama-3.2-1B-Instruct" # TODO: Change this to the final model, or move to environment variable
base_model_name = "facebook/opt-125m"
# uncomment this to load model from local files
# base_model_name = "Llama-3.1-8B-Instruct"
model = None
tokenizer = None
device = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model and tokenizer when the application starts"""
    global model, tokenizer, device
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model_path = "opt-125m.pth"
    # model_path = os.getenv("MODEL_PATH", None)
    # if model_path is None:
    #     logger.warning("MODEL_PATH environment variable not set.")
    #     return
    
    logger.info(f"Loading model: {model_path}")
    
    try:
        # Load tokenizer and model
        model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # TODO: Use vllm behind fastapi for inference
        # state_dict = torch.load(model_path, map_location=torch.device(device))
        # model.load_state_dict(state_dict)

        model.eval()
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.post("/answer", response_model=QAResponse)
async def answer_question(request: QARequest):
    """Answer a question based on the provided context"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    start_time = time.time()
    
    try:
        answer = get_model_response(request.question)
        elapsed_time = time.time() - start_time
        
        # Log the request and response
        logger.info(f"Question: {request.question}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Processing time: {elapsed_time:.4f} seconds")
        
        return QAResponse(
            answer=answer,
            elapsed_time=elapsed_time
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
def get_model_response(question_text):
    # Tokenize input
    inputs = tokenizer(question_text, return_tensors="pt")

    # TODO: Add prompt to make this extractive QA
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the generated output
    if full_output.startswith(question_text):
        return full_output[len(question_text):].strip()
    else:
        return full_output.strip()

Instrumentator().instrument(app).expose(app)