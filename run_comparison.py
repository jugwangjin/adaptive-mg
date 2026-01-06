#!/usr/bin/env python3
"""
Script to run both vcycle_trainer and simple_trainer for comparison.

This script:
1. Runs vcycle_trainer with evaluation every 200 steps
2. Runs simple_trainer with evaluation every 200 steps
3. Both use max_steps=30000

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_comparison.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Common configuration - shared across all trainers
DATASET_TYPE = "colmap"
DATA_DIR = "/Bean/data/gwangjin/2025/kdgs/360_v2/garden"
DATA_FACTOR = 8
MAX_STEPS = 30000
BATCH_SIZE = 1
WHITE_BACKGROUND = False
NORMALIZE_WORLD_SPACE = True  # Default value in trainers
PATCH_SIZE = None  # Default value in trainers


def run_vcycle_trainer(max_level: Optional[int] = None):
    """Run vcycle trainer with custom config.
    
    Args:
        max_level: Maximum level for hierarchical structure. If None, uses default (8).
                  If 1, should behave like regular 3DGS.
    """
    # Configuration (use shared constants)
    dataset_type = DATASET_TYPE
    data_dir = DATA_DIR
    data_factor = DATA_FACTOR
    max_steps = MAX_STEPS
    batch_size = BATCH_SIZE
    white_background = WHITE_BACKGROUND
    
    # Metric measurement interval (every 200 steps)
    metric_interval = 500
    
    print("=" * 60)
    print("Running V-cycle Trainer")
    if max_level is not None:
        print(f"  (max_level={max_level})")
    print("=" * 60)
    print(f"Dataset: {dataset_type}")
    print(f"Data directory: {data_dir}")
    print(f"Max steps: {max_steps}")
    print(f"Metric interval: {metric_interval} (every {metric_interval} steps)")
    if max_level is not None:
        print(f"Max level: {max_level}")
    print("=" * 60)
    
    # Build command
    cmd_parts = [
        "python", "vcycle_trainer.py",
        "--dataset_type", dataset_type,
        "--data_dir", data_dir,
        "--data_factor", str(data_factor),
        "--max_steps", str(max_steps),
        "--batch_size", str(batch_size),
        "--metric_interval", str(metric_interval),
    ]
    
    if max_level is not None:
        cmd_parts.extend(["--max_level", str(max_level)])
    
    if white_background:
        cmd_parts.append("--white_background")
    
    cmd = " ".join(cmd_parts)
    print(f"Command: {cmd}")
    print("=" * 60)
    return
    # Run command using os.system
    result = os.system(cmd)
    
    if result != 0:
        raise RuntimeError(f"V-cycle trainer failed with return code {result}")
    
    # Extract result directory from the config pattern (matching vcycle_trainer.py's __post_init__)
    dataset_name = Path(data_dir).name
    
    # Build settings string (exactly matching vcycle_trainer.py's __post_init__ logic)
    settings_parts = [
        f"type_{dataset_type}",
        f"factor_{data_factor}",
        "vcycle",
    ]
    if dataset_type == "nerf" and white_background:
        settings_parts.append("whitebg")
    if NORMALIZE_WORLD_SPACE:
        settings_parts.append("norm")
    if PATCH_SIZE is not None:
        settings_parts.append(f"patch_{PATCH_SIZE}")
    
    settings_str = "_".join(settings_parts)
    result_dir = f"/Bean/log/gwangjin/2025/gsplat/vcycle/{dataset_name}_{settings_str}"
    
    return result_dir


def run_simple_trainer(strategy: str = "default"):
    """Run simple trainer with custom config.
    
    Args:
        strategy: Strategy to use ("default" or "mcmc")
    """
    # Configuration (use shared constants)
    dataset_type = DATASET_TYPE
    data_dir = DATA_DIR
    data_factor = DATA_FACTOR
    max_steps = MAX_STEPS
    batch_size = BATCH_SIZE
    white_background = WHITE_BACKGROUND
    
    # Metric measurement interval (every 200 steps)
    metric_interval = 500
    
    print("\n" + "=" * 60)
    print(f"Running Simple Trainer ({strategy.upper()} strategy)")
    print("=" * 60)
    print(f"Dataset: {dataset_type}")
    print(f"Data directory: {data_dir}")
    print(f"Max steps: {max_steps}")
    print(f"Metric interval: {metric_interval} (every {metric_interval} steps)")
    print(f"Strategy: {strategy}")
    print("=" * 60)
    
    # Build command (use subcommand for strategy selection)
    cmd_parts = [
        "python", "simple_trainer.py",
        strategy,  # "default" or "mcmc"
        "--dataset_type", dataset_type,
        "--data_dir", data_dir,
        "--data_factor", str(data_factor),
        "--max_steps", str(max_steps),
        "--batch_size", str(batch_size),
        "--metric_interval", str(metric_interval),
    ]
    
    if white_background:
        cmd_parts.append("--white_background")
    
    cmd = " ".join(cmd_parts)
    print(f"Command: {cmd}")
    print("=" * 60)
    return
    # Run command using os.system
    result = os.system(cmd)
    
    if result != 0:
        raise RuntimeError(f"Simple trainer ({strategy}) failed with return code {result}")
    
    # Extract result directory from the config pattern (matching simple_trainer.py's __post_init__)
    dataset_name = Path(data_dir).name
    
    # Build settings string (exactly matching simple_trainer.py's __post_init__ logic)
    settings_parts = [
        f"type_{dataset_type}",
        f"factor_{data_factor}",
    ]
    if dataset_type == "nerf" and white_background:
        settings_parts.append("whitebg")
    if NORMALIZE_WORLD_SPACE:
        settings_parts.append("norm")
    if PATCH_SIZE is not None:
        settings_parts.append(f"patch_{PATCH_SIZE}")
    
    settings_str = "_".join(settings_parts)
    result_dir = f"/Bean/log/gwangjin/2025/gsplat/baseline/{dataset_name}_{settings_str}"
    
    return result_dir


def run_comparison_script(vcycle_result_dir: str, simple_result_dir: str, output_path: str = None):
    """Run compare_results.py to generate comparison graph."""
    print("\n" + "=" * 60)
    print("Running Comparison Script")
    print("=" * 60)
    
    # Build command
    cmd_parts = [
        "python", "compare_results.py",
        "--vcycle_dir", vcycle_result_dir,
        "--simple_dir", simple_result_dir,
    ]
    
    if output_path:
        cmd_parts.extend(["--output", output_path])
    
    cmd = " ".join(cmd_parts)
    print(f"Command: {cmd}")
    print("=" * 60)
    return
    # Run command using os.system
    result = os.system(cmd)
    
    if result != 0:
        raise RuntimeError(f"Comparison script failed with return code {result}")
    
    print("Comparison graph generated successfully!")


def main():
    """Main function to run both trainers and generate comparison graph."""
    print("=" * 60)
    print("Running Comparison: V-cycle vs Simple Trainer (Default & MCMC)")
    print("=" * 60)
    print("This will run both trainers sequentially, then generate comparison graph.")
    print("Make sure you have enough GPU memory and time.")
    print("=" * 60)
    
    # Run V-cycle trainer with default max_level (for comparison)
    try:
        vcycle_default_result_dir = run_vcycle_trainer(max_level=None)
        print(f"\nV-cycle trainer (default max_level) completed! Result directory: {vcycle_default_result_dir}")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        # Don't exit, just continue without this result
        vcycle_default_result_dir = None


    # Run Simple trainer with Default strategy (baseline 3DGS)
    try:
        simple_default_result_dir = run_simple_trainer(strategy="default")
        print(f"\nSimple trainer (default) completed! Result directory: {simple_default_result_dir}")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    # Run V-cycle trainer with max_level=1 (should be equivalent to baseline 3DGS)
    try:
        vcycle_maxlevel1_result_dir = run_vcycle_trainer(max_level=1)
        print(f"\nV-cycle trainer (max_level=1) completed! Result directory: {vcycle_maxlevel1_result_dir}")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    # Run Simple trainer with MCMC strategy
    try:
        simple_mcmc_result_dir = run_simple_trainer(strategy="mcmc")
        print(f"\nSimple trainer (MCMC) completed! Result directory: {simple_mcmc_result_dir}")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Simple trainer (default) result directory: {simple_default_result_dir}")
    print(f"V-cycle trainer (max_level=1) result directory: {vcycle_maxlevel1_result_dir}")
    print(f"Simple trainer (MCMC) result directory: {simple_mcmc_result_dir}")
    if vcycle_default_result_dir:
        print(f"V-cycle trainer (default max_level) result directory: {vcycle_default_result_dir}")
    
    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
