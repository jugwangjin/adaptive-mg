#!/usr/bin/env python3
"""
Move ply and ckpts subdirectories from result directories to ckpts_dir.
Only moves if stats file exists (completed training).
"""

import shutil
from pathlib import Path


ckpts_dir = './results/camerasplat_ckpts'


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
    
    # Step to check (final step)
    step = 29999
    
    # Create ckpts_dir if it doesn't exist
    ckpts_dir_path = Path(ckpts_dir)
    ckpts_dir_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    skipped_count = 0
    
    for data_dir in data_dirs:
        data_dir_path = Path(data_dir)
        dataset_name = data_dir_path.name  # e.g., "truck" or "horse"
        
        # Iterate: numbers first, then methods (same order as fps_script.py)
        for number in numbers:
            for method in methods:
                # Construct result directory path
                result_dir_name = f"{dataset_name}_type_nerf_factor_1_{method}{number}_g"
                result_dir = result_dir_base / result_dir_name
                
                # Path to stats file
                stats_file = result_dir / "stats" / f"val_step{step}.json"
                
                # Only move if stats file exists (completed training)
                if stats_file.exists():
                    # Target directory in ckpts_dir
                    target_dir = ckpts_dir_path / result_dir_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move ply directory if it exists
                    ply_source = result_dir / "ply"
                    ply_target = target_dir / "ply"
                    if ply_source.exists() and ply_source.is_dir():
                        if ply_target.exists():
                            print(f"⚠ Target already exists, skipping: {ply_target}")
                        else:
                            shutil.move(str(ply_source), str(ply_target))
                            print(f"✓ Moved: {ply_source} -> {ply_target}")
                            moved_count += 1
                    elif ply_source.exists():
                        print(f"⚠ Not a directory, skipping: {ply_source}")
                    
                    # Move ckpts directory if it exists
                    ckpts_source = result_dir / "ckpts"
                    ckpts_target = target_dir / "ckpts"
                    if ckpts_source.exists() and ckpts_source.is_dir():
                        if ckpts_target.exists():
                            print(f"⚠ Target already exists, skipping: {ckpts_target}")
                        else:
                            shutil.move(str(ckpts_source), str(ckpts_target))
                            print(f"✓ Moved: {ckpts_source} -> {ckpts_target}")
                            moved_count += 1
                    elif ckpts_source.exists():
                        print(f"⚠ Not a directory, skipping: {ckpts_source}")
                    
                    # Move visualizations directory if it exists
                    viz_source = result_dir / "visualizations"
                    viz_target = target_dir / "visualizations"
                    if viz_source.exists() and viz_source.is_dir():
                        if viz_target.exists():
                            print(f"⚠ Target already exists, skipping: {viz_target}")
                        else:
                            shutil.move(str(viz_source), str(viz_target))
                            print(f"✓ Moved: {viz_source} -> {viz_target}")
                            moved_count += 1
                    elif viz_source.exists():
                        print(f"⚠ Not a directory, skipping: {viz_source}")

                    # Move visualizations directory if it exists
                    viz_source = result_dir / "videos"
                    viz_target = target_dir / "videos"
                    if viz_source.exists() and viz_source.is_dir():
                        if viz_target.exists():
                            print(f"⚠ Target already exists, skipping: {viz_target}")
                        else:
                            shutil.move(str(viz_source), str(viz_target))
                            print(f"✓ Moved: {viz_source} -> {viz_target}")
                            moved_count += 1
                    elif viz_source.exists():
                        print(f"⚠ Not a directory, skipping: {viz_source}")

                else:
                    skipped_count += 1
                    print(f"✗ Skipped (no stats file): {result_dir_name}")
        
        # Handle ours with version variants
        for number in numbers:
            method = "ours"
            for v in range(2, 4):
                result_dir_name = f"{dataset_name}_type_nerf_factor_1_{method}{number}_v{v}_g"
                result_dir = result_dir_base / result_dir_name
                
                # Path to stats file
                stats_file = result_dir / "stats" / f"val_step{step}.json"
                
                # Only move if stats file exists (completed training)
                if stats_file.exists():
                    # Target directory in ckpts_dir
                    target_dir = ckpts_dir_path / result_dir_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move ply directory if it exists
                    ply_source = result_dir / "ply"
                    ply_target = target_dir / "ply"
                    if ply_source.exists() and ply_source.is_dir():
                        if ply_target.exists():
                            print(f"⚠ Target already exists, skipping: {ply_target}")
                        else:
                            shutil.move(str(ply_source), str(ply_target))
                            print(f"✓ Moved: {ply_source} -> {ply_target}")
                            moved_count += 1
                    elif ply_source.exists():
                        print(f"⚠ Not a directory, skipping: {ply_source}")
                    
                    # Move ckpts directory if it exists
                    ckpts_source = result_dir / "ckpts"
                    ckpts_target = target_dir / "ckpts"
                    if ckpts_source.exists() and ckpts_source.is_dir():
                        if ckpts_target.exists():
                            print(f"⚠ Target already exists, skipping: {ckpts_target}")
                        else:
                            shutil.move(str(ckpts_source), str(ckpts_target))
                            print(f"✓ Moved: {ckpts_source} -> {ckpts_target}")
                            moved_count += 1
                    elif ckpts_source.exists():
                        print(f"⚠ Not a directory, skipping: {ckpts_source}")
                    
                    # Move visualizations directory if it exists
                    viz_source = result_dir / "visualizations"
                    viz_target = target_dir / "visualizations"
                    if viz_source.exists() and viz_source.is_dir():
                        if viz_target.exists():
                            print(f"⚠ Target already exists, skipping: {viz_target}")
                        else:
                            shutil.move(str(viz_source), str(viz_target))
                            print(f"✓ Moved: {viz_source} -> {viz_target}")
                            moved_count += 1
                    elif viz_source.exists():
                        print(f"⚠ Not a directory, skipping: {viz_source}")
                else:
                    skipped_count += 1
                    print(f"✗ Skipped (no stats file): {result_dir_name}")
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Moved directories: {moved_count}")
    print(f"  Skipped (no stats file): {skipped_count}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
