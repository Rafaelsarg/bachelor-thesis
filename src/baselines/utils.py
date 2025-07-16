import os 
import json


def save_results(results, results_dir, filename):
    """
    Save classification results and best parameters to a JSON file.
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved: {filepath}")