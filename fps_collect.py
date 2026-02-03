#!/usr/bin/env python3
"""
Collect results from completed training runs.
Reads val_step29999.json and val_step6999.json files and collects metrics into CSV.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional


def main():
    # Data directories to process (same as fps_script.py)
    data_dirs = [
        "../tat_dataproecssing/dataset_3/truck/",
        "../tat_dataproecssing/dataset_3/horse/",
    ]
    
    # Numbers first (same order as fps_script.py)
    numbers = [10, 20, 30, 40, 50]
    
    # Methods second (same order as fps_script.py)
    methods = ["ours", "fps", "mani", "fisher"]
    
    # Result directory base
    result_dir_base = Path("./results/simple_trainer_original")
    
    # Steps to collect
    steps = [29999]
    
    # Collect all results
    all_results: List[Dict[str, any]] = []
    
    for data_dir in data_dirs:
        data_dir_path = Path(data_dir)
        dataset_name = data_dir_path.name  # e.g., "truck" or "horse"
        
        # Iterate: numbers first, then methods (same order as fps_script.py)
        for number in numbers:
            for method in methods:
                # Skip if this combination was skipped in training
                # if number == 50 and method == "ours":
                #     continue
                
                # Construct result directory path
                result_dir_name = f"{dataset_name}_type_nerf_factor_1_{method}{number}_g"
                result_dir = result_dir_base / result_dir_name
                
                # Collect for each step
                for step in steps:
                    # Path to stats file
                    stats_file = result_dir / "stats" / f"val_step{step}.json"
                    
                    # Try to read the stats file
                    if stats_file.exists():
                        try:
                            with open(stats_file, 'r') as f:
                                stats = json.load(f)
                            
                            # Extract metrics
                            result = {
                                "dataset": dataset_name,
                                "method": method,
                                "number": number,
                                "step": step,
                                "psnr": stats.get("psnr", None),
                                "ssim": stats.get("ssim", None),
                                "lpips": stats.get("lpips", None),
                                "num_GS": stats.get("num_GS", None),
                            }
                            all_results.append(result)
                            print(f"✓ Collected: {dataset_name} {method} {number} step={step}")
                        except Exception as e:
                            print(f"✗ Error reading {stats_file}: {e}")
                    else:
                        print(f"✗ Not found: {stats_file}")
    
                if method == "ours":
                    for v in range(2, 4):
                        result_dir_name = f"{dataset_name}_type_nerf_factor_1_{method}{number}_v{v}_g"
                        result_dir = result_dir_base / result_dir_name
                        
                        # Collect for each step
                        for step in steps:
                            # Path to stats file
                            stats_file = result_dir / "stats" / f"val_step{step}.json"
                            
                            # Try to read the stats file
                            if stats_file.exists():
                                try:
                                    with open(stats_file, 'r') as f:
                                        stats = json.load(f)
                                    
                                    # Extract metrics
                                    result = {
                                        "dataset": dataset_name,
                                        "method": method+f"_v{v}",
                                        "number": number,
                                        "step": step,
                                        "psnr": stats.get("psnr", None),
                                        "ssim": stats.get("ssim", None),
                                        "lpips": stats.get("lpips", None),
                                        "num_GS": stats.get("num_GS", None),
                                    }
                                    all_results.append(result)
                                    print(f"✓ Collected: {dataset_name} {method+f"_v{v}"} {number} step={step}")
                                except Exception as e:
                                    print(f"✗ Error reading {stats_file}: {e}")
                            else:
                                print(f"✗ Not found: {stats_file}")

    # Write to CSV
    if all_results:
        # Sort by step - dataset - number - method (left to right)
        all_results.sort(key=lambda x: (x["step"], x["dataset"], x["number"], x["method"]))
        
        csv_file = "results_collected.csv"
        fieldnames = ["step", "dataset", "number", "method", "psnr", "ssim", "lpips", "num_GS"]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            prev_key = None
            for result in all_results:
                # Current key: (step, dataset, number)
                current_key = (result["step"], result["dataset"], result["number"])
                
                # If number changed (different step-dataset-number set), add empty line
                if prev_key is not None and prev_key != current_key:
                    writer.writerow({})  # Write empty row
                
                writer.writerow(result)
                prev_key = current_key
        
        print(f"\n✓ Results collected to: {csv_file}")
        print(f"  Total entries: {len(all_results)}")
    else:
        print("\n✗ No results collected!")


if __name__ == "__main__":
    main()

