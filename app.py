from flask import Flask, request, jsonify, render_template
import os
import time
import numpy as np
import json
# import tritonclient.http as httpclient
# from tritonclient.utils import triton_to_np_dtype
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
import requests
import boto3 # Added for MinIO
import uuid # Added for unique IDs
from datetime import datetime # Added for timestamping
from concurrent.futures import ThreadPoolExecutor # Added for asynchronous tasks

app = Flask(__name__)

# Initialize ThreadPoolExecutor for asynchronous tasks
executor = ThreadPoolExecutor(max_workers=2)

# MinIO Configuration
MINIO_URL = os.environ.get('MINIO_URL')
MINIO_USER = os.environ.get('MINIO_USER')
MINIO_PASSWORD = os.environ.get('MINIO_PASSWORD')
BUCKET_NAME = "production" # Bucket name as per minio-init

s3 = None
if MINIO_URL and MINIO_USER and MINIO_PASSWORD:
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_USER,
            aws_secret_access_key=MINIO_PASSWORD,
            region_name='us-east-1' # Standard practice, MinIO doesn't strictly use it
        )
        print("Successfully connected to MinIO.")
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
else:
    print("Warning: MinIO environment variables (MINIO_URL, MINIO_USER, MINIO_PASSWORD) not fully set. Q&A logging to MinIO will be disabled.")


# local model configuration
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
# state_dict = torch.load("opt-125m.pth", map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# FastAPI server configuration
FASTAPI_SERVER_URL = os.environ['FASTAPI_SERVER_URL']

# Triton server configuration
# TRITON_SERVER_URL = os.environ['TRITON_SERVER_URL']
# MODEL_NAME = os.environ['FOOD11_MODEL_NAME']

# def get_model_response_local_automodel(question_text):
#     inputs = tokenizer(question_text, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Take the embedding for [CLS] or first token
#     embedding = outputs.last_hidden_state[0][0]
    
#     # Convert to a simple list for display or further processing
#     return embedding.tolist()

def get_model_response_local(question_text):
    # Tokenize input
    inputs = tokenizer(question_text, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=100,
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
    
def get_model_response_fastapi(question_text):
    response = requests.post(f"{FASTAPI_SERVER_URL}/answer", json={"question": question_text})
    response.raise_for_status()
    return response.json().get("answer", "")

# Function to save Q&A data to MinIO
def save_qa_to_minio(question, answer, qa_id):
    if not s3:
        print(f"MinIO client not initialized. Skipping save for Q&A ID: {qa_id}")
        return

    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    # Store Q&A logs in a subfolder for organization, filename includes timestamp and ID
    s3_key = f"qa_logs/{timestamp}_{qa_id}.json" 
    qa_data = {
        "id": qa_id,
        "question": question,
        "answer": answer,
        "timestamp": timestamp
    }
    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(qa_data),
            ContentType='application/json'
        )
        print(f"Q&A {qa_id} saved to MinIO: {BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"Error saving Q&A {qa_id} to MinIO: {e}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    # For debugging
    print(f"Received question: {question}")
    
    # Get response from model
    start_time = time.time()
    response_text = get_model_response_fastapi(question)
    # response_text = get_model_response_local(question)
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f} seconds")

    # Save Q&A to MinIO asynchronously if s3 client is available
    if s3 and question and response_text: 
        qa_id = str(uuid.uuid4())
        try:
            executor.submit(save_qa_to_minio, question, response_text, qa_id)
        except Exception as e:
            print(f"Error submitting Q&A save task to executor: {e}")
    
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)