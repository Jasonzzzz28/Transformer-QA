# memory class
import torch
import faiss
import pickle
import numpy as np
from typing import List
from dataclasses import dataclass
import os
from retriever import Retriever
from tqdm import tqdm

@dataclass
class MemoryChunk:
    """Data structure for storing a single memory chunk"""
    text: str  # Original text
    context: str  # Context

@dataclass
class Memory:
    """Data structure for storing memories feed into the model"""
    key_states: torch.Tensor  # attention key states (batch_size, num_heads, seq_len, head_dim)
    value_states: torch.Tensor  # attention value states (batch_size, num_heads, seq_len, head_dim)

class Base_Memory_3():
    def __init__(
        self,
        retrieval_model: Retriever,  # Model for text embedding
    ):
        self.retrieval_model = retrieval_model
        
        # Initialize vector database
        self.vector_db = self.init_vector_db()
        self.memory_chunks: List[MemoryChunk] = []
        self.load_path = "/root/autodl-tmp/memory"
        if not os.path.exists(self.load_path):
            os.makedirs(self.load_path, exist_ok=True)

    def init_vector_db(self):
        d = self.retrieval_model.config.hidden_size
        n_list = 150
        m = 16
        nbits = 8
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, n_list, m, nbits, faiss.METRIC_INNER_PRODUCT)
        return index
    
    # Chunck the knowledgebase and store them to the disk without model involving
    def process_knowledge_base(self, knowledge_base: List[str], save_path: str):
        """Process knowledge base and store as memory chunks"""
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        chunk_texts = []
        chunk_embeddings = []
        
        idx = 0
        for text, context in tqdm(knowledge_base):
            chunk_texts.append(text)
            with open(os.path.join(save_path, f"memory_chunks_{idx}.pkl"), "wb") as f:
                pickle.dump(MemoryChunk(text=text, context=context), f)
                idx += 1
            with torch.no_grad():
                embedding = self.retrieval_model.encode(text)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)[0]
                chunk_embeddings.append(embedding.cpu().numpy())

        # Build and store vector database
        chunk_embeddings = np.vstack(chunk_embeddings)
        self.vector_db.train(chunk_embeddings)
        self.vector_db.add(chunk_embeddings)
        self.vector_db.nprobe = 32
        # Save to disk
        self._save_to_disk(save_path)
    
    def retrieve_memory(self, query: str, top_k: int):
        """Retrieve memory from disk"""
        with torch.no_grad():
            query_embedding = self.retrieval_model.encode(query)
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)[0]
            query_embedding = query_embedding.cpu().numpy()
            distances, indices = self.vector_db.search(query_embedding.reshape(1, -1), top_k)
        return distances, indices
    
    def _save_to_disk(self, save_path: str):
        """Save database to disk"""
        faiss.write_index(self.vector_db, os.path.join(save_path, "vector_db.index"))

    
    def load_from_disk(self, load_path: str):
        self.vector_db = faiss.read_index(os.path.join(load_path, "vector_db.index"))
        self.load_path = load_path
    
    def _load_memory_chunk_from_disk(self, load_path: str, indices: List[int]):
        self.memory_chunks = []
        for idx in indices:
            with open(os.path.join(load_path, f"memory_chunks_{idx}.pkl"), "rb") as f:
                chunk = pickle.load(f)
            self.memory_chunks.append(chunk)

    def rag_preprocess(self, contexts: List[str]):
        for i in range(len(contexts)):
            _, indices = self.retrieve_memory(contexts[i], 5)
            self._load_memory_chunk_from_disk(self.load_path, indices[0], update_disk=False, use_cache=False, encode=False)
            reference_text = "\n".join([self.memory_chunks[j].context for j in range(len(self.memory_chunks))])
        return reference_text
            
