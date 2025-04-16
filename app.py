from flask import Flask, request, jsonify, render_template
import os
import time
import numpy as np
import json
# import tritonclient.http as httpclient
# from tritonclient.utils import triton_to_np_dtype
import torch
from transformers import AutoTokenizer

app = Flask(__name__)

# local model configuration
model = torch.load("opt-125m.pth", map_location=torch.device('cpu'))
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Triton server configuration
# TRITON_SERVER_URL = os.environ['TRITON_SERVER_URL']
# MODEL_NAME = os.environ['FOOD11_MODEL_NAME']

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

    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_model_response_triton(question_text):
    try:
        # Create a Triton client
        triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
        
        # Check if the server is ready
        if not triton_client.is_server_ready():
            return "Triton server is not ready."
        
        # Check if the model is ready
        if not triton_client.is_model_ready(MODEL_NAME):
            return f"Model {MODEL_NAME} is not ready."
        
        prompt = f"<|system|>\nYou are a helpful AI assistant.\n<|user|>\n{question_text}\n<|assistant|>"
        
        # Create the input data as a numpy array of strings
        input_data = np.array([prompt], dtype=np.object_)
        
        # Input parameters for the model
        inputs = []
        inputs.append(httpclient.InferInput("text_input", input_data.shape, "BYTES"))
        inputs[0].set_data_from_numpy(input_data)
        
        # Parameters for text generation
        parameters = {
            "temperature": 0.7,
            "max_tokens": 256,
            "top_p": 0.95,
            "top_k": 40
        }
        
        # Convert parameters to JSON string
        parameters_json = json.dumps(parameters)
        parameters_data = np.array([parameters_json], dtype=np.object_)
        
        # Add parameters as another input
        inputs.append(httpclient.InferInput("parameters", parameters_data.shape, "BYTES"))
        inputs[1].set_data_from_numpy(parameters_data)
        
        # Define the expected output
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("text_output"))
        
        # Execute the inference request
        results = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs
        )
        
        # Process the results
        output_data = results.as_numpy("text_output")
        response_text = output_data[0].decode('utf-8')
        
        return response_text
    
    except Exception as e:
        print(f"Error when calling Triton server: {e}")
        return f"Error processing request: {str(e)}"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    # For debugging
    print(f"Received question: {question}")
    
    # Get response from Triton server
    start_time = time.time()
    response = get_model_response_local(question)
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f} seconds")
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)