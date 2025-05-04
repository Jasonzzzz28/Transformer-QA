import os
import requests
import json
import logging
import time
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def send_request(api_url: str, payload: Dict[str, str], index: int) -> None:
    """Sends a single POST request to the FastAPI endpoint."""
    try:
        logging.info(f"Sending request {index+1}: question='{payload.get('question', '')[:50]}...'") # Log snippet
        response = requests.post(api_url, json=payload, timeout=30) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        logging.info(f"Request {index+1} successful. Status: {response.status_code}, Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request {index+1} failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during request {index+1}: {e}", exc_info=True)

def simulate_traffic(api_url: str, data: List[Dict[str, str]], delay_seconds: float = 0.1) -> None:
    """Iterates through data and sends requests to the API."""
    if not api_url:
        logging.error("FASTAPI_URL is not set. Cannot send requests.")
        return
    if not data:
        logging.warning("No data loaded. No requests will be sent.")
        return

    logging.info(f"Starting traffic simulation to {api_url} with {len(data)} requests...")
    for i, item in enumerate(data):
        payload = {
            "question": item.get("question"),
            "context": item.get("context")
        }
        # Ensure question and context are present before sending
        if payload["question"] is None or payload["context"] is None:
            logging.warning(f"Skipping request {i+1} due to missing 'question' or 'context'.")
            continue

        send_request(api_url, payload, i)
        time.sleep(delay_seconds) # Add a small delay between requests

    logging.info("Traffic simulation finished.")

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