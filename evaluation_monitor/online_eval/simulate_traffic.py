import os
import requests
import json
import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from statistics import mean
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Metrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: List[float] = None
    em_scores: List[float] = None
    f1_scores: List[float] = None
    start_time: float = None
    end_time: float = None

    def __post_init__(self):
        self.latencies = []
        self.em_scores = []
        self.f1_scores = []
        self.start_time = time.time()

    def calculate_qps(self) -> float:
        if not self.end_time:
            self.end_time = time.time()
        duration = self.end_time - self.start_time
        return self.successful_requests / duration if duration > 0 else 0

    def calculate_error_rate(self) -> float:
        return (self.failed_requests / self.total_requests * 100) if self.total_requests > 0 else 0

    def calculate_avg_latency(self) -> float:
        return mean(self.latencies) if self.latencies else 0

    def calculate_avg_em(self) -> float:
        return mean(self.em_scores) if self.em_scores else 0

    def calculate_avg_f1(self) -> float:
        return mean(self.f1_scores) if self.f1_scores else 0

def normalize_text(text: str) -> str:
    """Normalize text for comparison by converting to lowercase and removing punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())

def calculate_em(prediction: str, ground_truth: str) -> float:
    """Calculate Exact Match score."""
    pred_norm = normalize_text(prediction)
    truth_norm = normalize_text(ground_truth)
    return 1.0 if pred_norm == truth_norm else 0.0

def calculate_f1(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score."""
    pred_tokens = set(normalize_text(prediction).split())
    truth_tokens = set(normalize_text(ground_truth).split())
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = pred_tokens.intersection(truth_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def load_online_eval_data(data_path: str) -> List[Dict[str, str]]:
    """Loads the online evaluation data from a JSON file."""
    if not data_path or not os.path.exists(data_path):
        logging.error(f"Online evaluation data path not provided or does not exist: {data_path}")
        return []
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logging.error(f"Data in {data_path} is not a list.")
            return []
        # Basic validation for expected keys in the first item (if list is not empty)
        if data and not all(key in data[0] for key in ['question', 'context', 'answer']):
             logging.warning(f"Data in {data_path} might be missing expected keys ('question', 'context', 'answer'). Proceeding anyway.")
        logging.info(f"Successfully loaded {len(data)} records from {data_path}")
        return data
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {data_path}")
        return []
    except Exception as e:
        logging.error(f"Error loading data from {data_path}: {e}", exc_info=True)
        return []

def send_request(api_url: str, payload: Dict[str, str], index: int, metrics: Metrics) -> None:
    """Sends a single POST request to the FastAPI endpoint and tracks metrics."""
    metrics.total_requests += 1
    start_time = time.time()
    
    try:
        logging.info(f"Sending request {index+1}: question='{payload.get('question', '')[:50]}...'")
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        # Calculate metrics
        latency = time.time() - start_time
        metrics.latencies.append(latency)
        metrics.successful_requests += 1
        
        response_data = response.json()
        if 'answer' in response_data and 'ground_truth' in payload:
            em_score = calculate_em(response_data['answer'], payload['ground_truth'])
            f1_score = calculate_f1(response_data['answer'], payload['ground_truth'])
            metrics.em_scores.append(em_score)
            metrics.f1_scores.append(f1_score)
        
        logging.info(f"Request {index+1} successful. Status: {response.status_code}, Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        metrics.failed_requests += 1
        logging.error(f"Request {index+1} failed: {e}")
    except Exception as e:
        metrics.failed_requests += 1
        logging.error(f"An unexpected error occurred during request {index+1}: {e}", exc_info=True)

def simulate_traffic(api_url: str, data: List[Dict[str, str]], delay_seconds: float = 0.1) -> None:
    """Iterates through data and sends requests to the API while tracking metrics."""
    if not api_url:
        logging.error("FASTAPI_URL is not set. Cannot send requests.")
        return
    if not data:
        logging.warning("No data loaded. No requests will be sent.")
        return

    metrics = Metrics()
    logging.info(f"Starting traffic simulation to {api_url} with {len(data)} requests...")
    
    for i, item in enumerate(data):
        payload = {
            "question": item.get("question"),
            "context": item.get("context"),
            "ground_truth": item.get("answer")  # Include ground truth for metrics
        }
        
        if payload["question"] is None or payload["context"] is None:
            logging.warning(f"Skipping request {i+1} due to missing 'question' or 'context'.")
            continue

        send_request(api_url, payload, i, metrics)
        time.sleep(delay_seconds)

    # Calculate and log final metrics
    metrics.end_time = time.time()
    logging.info("\n=== Simulation Metrics ===")
    logging.info(f"Total Requests: {metrics.total_requests}")
    logging.info(f"Successful Requests: {metrics.successful_requests}")
    logging.info(f"Failed Requests: {metrics.failed_requests}")
    logging.info(f"Error Rate: {metrics.calculate_error_rate():.2f}%")
    logging.info(f"Average Latency: {metrics.calculate_avg_latency()*1000:.2f}ms")
    logging.info(f"QPS: {metrics.calculate_qps():.2f}")
    logging.info(f"Average EM Score: {metrics.calculate_avg_em():.4f}")
    logging.info(f"Average F1 Score: {metrics.calculate_avg_f1():.4f}")
    logging.info("========================\n")

def main():
    """Main function to load config and run simulation."""
    api_url = os.environ.get("FASTAPI_URL")
    data_path = os.environ.get("ONLINE_EVAL_DATA_PATH")

    if not api_url:
        logging.error("Critical: FASTAPI_URL environment variable not set. Exiting.")
        return
    if not data_path:
        logging.error("Critical: ONLINE_EVAL_DATA_PATH environment variable not set. Exiting.")
        return

    eval_data = load_online_eval_data(data_path)
    simulate_traffic(api_url, eval_data)

if __name__ == "__main__":
    main()