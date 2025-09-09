"""
Utility script to extract sample entries from the test dataset for GEPA optimization.
"""

import os
import json
import random
import pickle
from typing import List, Dict, Any, Optional

import numpy as np

from src.config import GlobalPathConfig


def extract_sample_entries(
    dataset_path: str,
    num_samples: int = 3,
    seed: int = 42,
    save_path: Optional[str] = None
) -> List[int]:
    """
    Extract random sample entries from the test dataset.
    
    Args:
        dataset_path: Path to the dataset JSON file
        num_samples: Number of samples to extract
        seed: Random seed for reproducibility
        save_path: Optional path to save the sample task IDs
        
    Returns:
        List of task IDs for the selected samples
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Load the dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Get all task IDs
    task_ids = list(dataset.keys())
    
    # Ensure we don't try to sample more than available
    num_samples = min(num_samples, len(task_ids))
    
    # Randomly select task IDs
    selected_task_ids = random.sample(task_ids, num_samples)
    
    # Convert task IDs from strings to integers
    selected_task_ids = [int(task_id) for task_id in selected_task_ids]
    
    print(f"Selected {num_samples} task IDs: {selected_task_ids}")
    
    # Save the selected task IDs if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({"task_ids": selected_task_ids}, f, indent=4)
        print(f"Saved selected task IDs to {save_path}")
    
    return selected_task_ids


if __name__ == "__main__":
    # Define paths
    dataset_path = os.path.join(GlobalPathConfig.dataset_path, "test_task_samples.json")
    save_path = os.path.join(GlobalPathConfig.result_path, "gepa_samples", "sample_task_ids.json")
    
    # Extract samples
    extract_sample_entries(dataset_path, num_samples=3, save_path=save_path)