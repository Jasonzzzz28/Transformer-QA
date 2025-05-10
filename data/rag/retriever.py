import torch
from transformers import AutoModel, AutoTokenizer
from typing import List

class Retriever:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        self.model = AutoModel.from_pretrained(model_name, device_map="auto").half()
        self.config = self.model.config
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)

    def encode(self, text: List[str]|str):
        with torch.no_grad():
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]
            return embeddings