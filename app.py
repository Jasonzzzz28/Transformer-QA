from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
import time
import random

app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# Mock responses
MOCK_RESPONSES = [
    "Transformer models are a type of neural network architecture that relies on self-attention mechanisms to process sequential data. The Hugging Face Transformers library provides thousands of pretrained models for NLP tasks like text classification, question answering, and text generation.",
    "To fine-tune a BERT model, you'll need to: 1) Prepare your dataset, 2) Load the pretrained BERT model using `AutoModelForSequenceClassification`, 3) Set up a Trainer with your training arguments, and 4) Call trainer.train(). The Hugging Face documentation provides detailed examples.",
    "GPT (Generative Pre-trained Transformer) is a decoder-only model designed for text generation, while BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only model designed for understanding tasks like question answering. GPT processes text left-to-right, while BERT considers full context bidirectionally."
]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    # For debugging
    print(f"Received question: {question}")
    
    # Simulate processing delay
    time.sleep(1.5)
    
    # For now, return a mock response
    response = random.choice(MOCK_RESPONSES)
    # response = get_model_response(question)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)