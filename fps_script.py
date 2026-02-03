#!/usr/bin/env python3
"""
Run training for multiple transforms files with different methods and numbers.
Iterates over numbers first, then methods.
"""

import subprocess
import sys
from pathlib import Path
import os

def main():
    # sleep for 1 hour
    # Data directories to process
    data_dirs = [
        "../tat_dataproecssing/dataset_3/truck/",
        "../tat_dataproecssing/dataset_3/horse/",
    ]
    
    # Numbers first (30, 50)
    numbers = [10, 20, 30]
    
    # Methods second (fisher, fps, mani, ours)
    methods = ["ours", ]
    # methods = ["ours", "fps", "mani", "fisher"]
    
    # Base script
    script = "simple_trainer_custom_train.py"
    config = "default"
    
    # Common arguments
    dataset_type = "nerf"
    data_factor = 1
    
    # Iterate: numbers first, then data_dirs, then methods
    for number in numbers:
        for data_dir in data_dirs:
            data_dir_path = Path(data_dir)
            dataset_name = data_dir_path.name  # e.g., "truck" or "horse"
            colmap_dir = data_dir_path / "sparse" / "0"
            
            for method in methods:
                # Construct custom_train_json filename
                custom_train_json = f"transforms_{method}_{number}.json"
                if method == "ours":
                    custom_train_json = f"transforms_{method}_{number}_v3.json"
                
                
                # Construct result_dir
                result_dir = f"./results/simple_trainer_original/{dataset_name}_type_nerf_factor_1_{method}{number}_g"
                if method == "ours":
                    result_dir = f"./results/simple_trainer_original/{dataset_name}_type_nerf_factor_1_{method}{number}_v3_g"

                if os.path.exists(os.path.join(result_dir, "stats", "val_step29999.json")):
                    print(f"Skipping {result_dir} because it already exists")
                    continue
                # Build command
                cmd = [
                    "python",
                    script,
                    config,
                    "--data_dir", str(data_dir),
                    "--dataset_type", dataset_type,
                    "--data_factor", str(data_factor),
                    "--custom_train_json", custom_train_json,
                    "--result_dir", result_dir,
                    "--colmap_dir", str(colmap_dir),
                ]
                
                # Print command
                print(f"\n{'='*80}")
                print(f"Running: {' '.join(cmd)}")
                print(f"{'='*80}\n")
                
                # Execute command
                try:
                    result = subprocess.run(cmd, check=True)
                    print(f"\n✓ Successfully completed: {custom_train_json} for {dataset_name}\n")
                except subprocess.CalledProcessError as e:
                    print(f"\n✗ Failed: {custom_train_json} for {dataset_name}")
                    print(f"Error code: {e.returncode}\n")
                    # Continue with next iteration instead of stopping
                    continue


if __name__ == "__main__":
    main()

