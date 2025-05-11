import os
import logging
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer, device

def generate_response(model, tokenizer, prompt, device, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    latency = time.time() - start_time
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text, latency

def worker(model, tokenizer, device, request_queue, result_queue):
    while True:
        try:
            prompt = request_queue.get_nowait()
            response, latency = generate_response(model, tokenizer, prompt, device)
            result_queue.put((response, latency))
        except queue.Empty:
            break

def run_load_test(model, tokenizer, device, num_requests=100, num_threads=4):
    latencies = []
    request_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Prepare test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "What is the meaning of life?",
        "How does photosynthesis work?",
        "What is machine learning?"
    ] * (num_requests // len(test_prompts) + 1)
    
    for prompt in test_prompts[:num_requests]:
        request_queue.put(prompt)
    
    # Run load test
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(worker, model, tokenizer, device, request_queue, result_queue)
            for _ in range(num_threads)
        ]
    
    # Collect results
    while not result_queue.empty():
        _, latency = result_queue.get()
        latencies.append(latency)
    
    # Calculate metrics
    metrics = {
        "total_requests": num_requests,
        "avg_latency": np.mean(latencies),
        "p50_latency": np.percentile(latencies, 50),
        "p90_latency": np.percentile(latencies, 90),
        "p95_latency": np.percentile(latencies, 95),
        "p99_latency": np.percentile(latencies, 99),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "throughput": num_requests / sum(latencies)
    }
    
    return metrics

def main():
    model_path = os.environ.get("MODEL_PATH")
    output_file = os.environ.get("LOAD_TEST_OUTPUT", "load_test_results.json")
    
    if not model_path:
        logging.error("MODEL_PATH environment variable not set")
        return
    
    try:
        model, tokenizer, device = load_model_and_tokenizer(model_path)
        results = run_load_test(model, tokenizer, device)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Load test results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Load test failed: {e}", exc_info=True)

if __name__ == "__main__":
    main() 