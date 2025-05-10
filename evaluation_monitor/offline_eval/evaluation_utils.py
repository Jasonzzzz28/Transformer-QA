import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(model_path):
    """Loads the Causal LM model and tokenizer."""
    try:
        logging.info(f"Loading model and tokenizer from: {model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        # Ensure the model path exists or is a valid Hugging Face identifier
        if not os.path.exists(model_path) and "/" not in model_path:
             raise FileNotFoundError(f"Model path {model_path} not found and is not a valid Hugging Face identifier.")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Add padding token if it doesn't exist (common for Llama models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # Use bfloat16 for faster GPU inference if available
            device_map="auto" # Automatically distribute model across available GPUs/CPU
        )
        model.eval() # Set model to evaluation mode
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer, device
    except Exception as e:
        logging.error(f"Error loading model/tokenizer from {model_path}: {e}", exc_info=True)
        raise

def load_evaluation_dataset(data_path):
    """Loads the evaluation dataset from the specified path (assuming JSON list format)."""
    if not data_path or not os.path.exists(data_path):
        logging.error(f"Data path not provided or does not exist: {data_path}")
        return None
    try:
        logging.info(f"Loading dataset from: {data_path}")
        # Load JSON, assuming it's a list of objects [{question: ..., answer: ..., context: ...}, ...]
        # The 'train' split is a default for datasets loaded this way when not specified otherwise.
        dataset = load_dataset('json', data_files=data_path, split='train')
        # Add basic validation for expected keys
        if not all(key in dataset.column_names for key in ['question', 'context', 'answer']):
             logging.error(f"Dataset at {data_path} is missing required keys ('question', 'context', 'answer'). Found keys: {dataset.column_names}")
             return None
        logging.info(f"Dataset loaded successfully with {len(dataset)} examples.")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset from {data_path}: {e}", exc_info=True)
        return None

def generate_predictions(model, tokenizer, dataset, device, max_new_tokens=50):
    """Generates predictions for the dataset using the model."""
    predictions = []
    references = []

    logging.info("Generating predictions...")
    for i, example in enumerate(tqdm(dataset)):
        question = example['question']
        context = example['context']
        reference_answer = example['answer'] # Get the single answer string

        # Simple prompt for extractive QA with Llama Instruct
        # You might need to experiment with different prompt formats for optimal results
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert extractive question answering system. Given the context and question, extract the exact answer span from the context. If the answer is not present in the context, respond with "Answer not found".<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question:
{question}

Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - max_new_tokens).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False # Use greedy decoding for extractive tasks
            )
            
        # Decode the generated tokens, excluding the prompt
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        # Format prediction for SQuAD metric
        # Use index 'i' as a fallback ID if no 'id' field exists in the data
        prediction_formatted = {'prediction_text': generated_text, 'id': example.get('id', str(i))}
        predictions.append(prediction_formatted)

        # Format reference for SQuAD metric using the new 'answer' field
        # SQuAD metric expects 'answers' field as a dict with 'text' (list) and 'answer_start' (list)
        reference_formatted = {
            'answers': {'text': [reference_answer], 'answer_start': [-1]}, # Use -1 for answer_start as it's not provided
            'id': example.get('id', str(i))
        }
        references.append(reference_formatted)

    logging.info("Prediction generation complete.")
    return predictions, references

def calculate_metrics(predictions, references):
    """Calculates EM and F1 scores using the SQuAD metric."""
    try:
        logging.info("Calculating EM and F1 metrics...")
        squad_metric = evaluate.load("squad")
        results = squad_metric.compute(predictions=predictions, references=references)
        logging.info(f"Metrics calculated: {results}")
        return results
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}", exc_info=True)
        return None

def run_evaluation(model, tokenizer, device, data_path, eval_type_name):
    """Runs the full evaluation pipeline for a given dataset."""
    logging.info(f"--- Starting {eval_type_name} Evaluation ---")
    dataset = load_evaluation_dataset(data_path)
    if dataset is None:
        logging.warning(f"Skipping {eval_type_name} evaluation due to dataset loading error.")
        return None
        
    predictions, references = generate_predictions(model, tokenizer, dataset, device)
    metrics = calculate_metrics(predictions, references)
    logging.info(f"--- Finished {eval_type_name} Evaluation ---")
    return metrics