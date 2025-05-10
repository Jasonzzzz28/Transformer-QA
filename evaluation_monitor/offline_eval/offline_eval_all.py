import os
import logging
import json
from evaluation_utils import load_model_and_tokenizer, run_evaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_standard_eval(model, tokenizer, device):
    """Runs evaluation on the standard dataset."""
    data_path = os.environ.get("STANDARD_EVAL_DATA_PATH")
    if not data_path:
        logging.error("STANDARD_EVAL_DATA_PATH environment variable not set.")
        return None
    return run_evaluation(model, tokenizer, device, data_path, "Standard")

def run_slice_eval(model, tokenizer, device):
    """Runs evaluation on the slice of interest dataset."""
    data_path = os.environ.get("SLICE_EVAL_DATA_PATH")
    if not data_path:
        logging.error("SLICE_EVAL_DATA_PATH environment variable not set.")
        return None
    return run_evaluation(model, tokenizer, device, data_path, "Slice of Interest")

def run_failure_eval(model, tokenizer, device):
    """Runs evaluation on the known failure modes dataset."""
    data_path = os.environ.get("FAILURE_EVAL_DATA_PATH")
    if not data_path:
        logging.error("FAILURE_EVAL_DATA_PATH environment variable not set.")
        return None
    return run_evaluation(model, tokenizer, device, data_path, "Known Failure Modes")

def run_template_eval(model, tokenizer, device):
    """Runs evaluation on the template-based dataset."""
    data_path = os.environ.get("TEMPLATE_EVAL_DATA_PATH")
    if not data_path:
        logging.error("TEMPLATE_EVAL_DATA_PATH environment variable not set.")
        return None
    return run_evaluation(model, tokenizer, device, data_path, "Template-Based")

# --- Main Execution Logic ---

def main():
    # --- Configuration ---
    model_path = os.environ.get("MODEL_PATH")
    output_file = os.environ.get("EVAL_OUTPUT_FILE", "evaluation_results.json") # Optional: specify output file path

    if not model_path:
        logging.error("Critical: MODEL_PATH environment variable not set. Exiting.")
        return

    all_results = {}

    # --- Load Model (once) ---
    try:
        model, tokenizer, device = load_model_and_tokenizer(model_path)
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}. Cannot proceed with evaluations. Error: {e}", exc_info=True)
        return # Exit if model loading fails

    # --- Run Evaluations ---
    logging.info("Starting all offline evaluations...")

    # 1. Standard Evaluation
    standard_results = run_standard_eval(model, tokenizer, device) # Now calls local function
    if standard_results:
        all_results["standard"] = standard_results
        logging.info(f"Standard Eval Results: {standard_results}")
    else:
        logging.warning("Standard evaluation did not produce results.")

    # 2. Slice of Interest Evaluation
    slice_results = run_slice_eval(model, tokenizer, device) # Now calls local function
    if slice_results:
        all_results["slice_of_interest"] = slice_results
        logging.info(f"Slice Eval Results: {slice_results}")
    else:
        logging.warning("Slice evaluation did not produce results.")

    # 3. Known Failure Modes Evaluation
    failure_results = run_failure_eval(model, tokenizer, device) # Now calls local function
    if failure_results:
        all_results["known_failure_modes"] = failure_results
        logging.info(f"Failure Modes Eval Results: {failure_results}")
    else:
        logging.warning("Failure modes evaluation did not produce results.")

    # 4. Template-Based Evaluation
    template_results = run_template_eval(model, tokenizer, device) # Now calls local function
    if template_results:
        all_results["template_based"] = template_results
        logging.info(f"Template Eval Results: {template_results}")
    else:
        logging.warning("Template evaluation did not produce results.")

    # --- Save Results ---
    if all_results:
        try:
            logging.info(f"Saving evaluation results to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=4)
            logging.info("Results saved successfully.")
        except IOError as e:
            logging.error(f"Error saving results to {output_file}: {e}")
    else:
        logging.warning("No evaluation results were generated to save.")

    logging.info("All offline evaluations finished.")


if __name__ == "__main__":
    main()