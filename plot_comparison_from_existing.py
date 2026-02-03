#!/usr/bin/env python3
"""
Script to plot comparison graphs from existing training results.

This script:
1. Automatically finds result directories based on run_comparison.py configuration
2. Reads stats files from those directories
3. Creates comparison graphs for PSNR, SSIM, LPIPS, and num_GS
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
# import simple_trainer_c2f

# Configuration matching trainer configurations
DATASET_TYPE = "colmap"
DATA_DIR = "./dataset/360_v2/garden"  # Matching train_them.sh
DATA_FACTOR = 8  # Matching train_them.sh
WHITE_BACKGROUND = False
NORMALIZE_WORLD_SPACE = False  # Default from hierarchy_trainer_simple.py
PATCH_SIZE = None


def get_result_directories() -> Dict[str, Optional[str]]:
    """Get result directories for default configurations:
    1. hierarchy_trainer_simple.py (default)
    2. hierarchy_trainer_simple.py with --use_coarse_to_fine
    3. hierarchy_trainer_vcycle.py
    4. hierarchy_trainer_vcycle_v2.py
    
    Returns:
        Dictionary with keys: 'simple', 'simple_c2f', 'vcycle', 'vcycle_v2'
        Values are directory paths or None if not found
    """
    dataset_name = Path(DATA_DIR).name
    
    # Build settings string (matching hierarchy_trainer_simple.py and hierarchy_trainer_vcycle.py)
    settings_parts = [
        f"type_{DATASET_TYPE}",
        f"factor_{DATA_FACTOR}",
    ]
    if DATASET_TYPE == "nerf" and WHITE_BACKGROUND:
        settings_parts.append("whitebg")
    if NORMALIZE_WORLD_SPACE:
        settings_parts.append("norm")
    if PATCH_SIZE is not None:
        settings_parts.append(f"patch_{PATCH_SIZE}")
    settings_str = "_".join(settings_parts)
    
    result = {}
    
    result["simple"] = f"./results/hierarchy_trainer_simple/{dataset_name}_{settings_str}_hierarchy_hierarchy"
    result["simple_c2f"] = f"./results/hierarchy_trainer_simple/{dataset_name}_{settings_str}_hierarchy_hierarchy_c2f"
    result["vcycle"] = f"./results/hierarchy_trainer_vcycle/{dataset_name}_{settings_str}_hierarchy_hierarchy"
    result["vcycle_v2"] = f"./results/hierarchy_trainer_vcycle_v2/{dataset_name}_{settings_str}_hierarchy_hierarchy"
    result["vcycle_v3"] = f"./results/hierarchy_trainer_vcycle_v3/{dataset_name}_{settings_str}_hierarchy_hierarchy"
    
    return result


def parse_stats_files(result_dir: Path) -> Dict[str, Dict[int, float]]:
    """
    Parse all stats files from a result directory.
    
    Args:
        result_dir: Path to the trainer's result directory (contains stats/ subdirectory)
    
    Returns:
        Dictionary mapping metric_name -> {step: value}
    """
    stats_dir = result_dir / "stats"
    if not stats_dir.exists():
        print(f"Warning: Stats directory not found: {stats_dir}")
        return {}
    
    metrics = {
        'psnr': {},
        'ssim': {},
        'lpips': {},
        'num_GS': {},
        'num_GS_finest': {},
        'ellipse_time': {},
    }
    
    # Find all val_step JSON files
    for json_file in sorted(stats_dir.glob("val_step*.json")):
        # Extract step number from filename (e.g., "val_step0200.json" -> 200)
        step_str = json_file.stem.replace("val_step", "")
        try:
            step = int(step_str)
        except ValueError:
            continue
        
        # Read metrics from JSON
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'psnr' in data:
                    metrics['psnr'][step] = float(data['psnr'])
                if 'ssim' in data:
                    metrics['ssim'][step] = float(data['ssim'])
                if 'lpips' in data:
                    metrics['lpips'][step] = float(data['lpips'])
                if 'num_GS' in data:
                    metrics['num_GS'][step] = float(data['num_GS'])
                if 'num_GS_finest' in data:
                    metrics['num_GS_finest'][step] = float(data['num_GS_finest'])
                if 'ellipse_time' in data:
                    metrics['ellipse_time'][step] = float(data['ellipse_time'])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")
            continue
    
    return metrics


def plot_metric_on_axis(
    ax,
    all_data: Dict[str, Dict[str, Dict[int, float]]],
    metric_name: str,
    ylabel: Optional[str] = None,
):
    """
    Plot a metric on a given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        all_data: Dictionary mapping method_name -> {metric_name: {step: value}}
        metric_name: Name of the metric (e.g., 'psnr', 'ssim', 'lpips', 'num_GS')
        ylabel: Y-axis label (if None, uses metric_name.upper())
    """
    if ylabel is None:
        ylabel = metric_name.upper()
    
    # Special handling for num_GS_finest: fallback to num_GS if not available
    if metric_name == 'num_GS_finest':
        methods_with_data = {}
        for name, data in all_data.items():
            # Try num_GS_finest first
            if metric_name in data and len(data[metric_name]) > 0:
                methods_with_data[name] = data[metric_name]
            # Fallback to num_GS if num_GS_finest is not available
            elif 'num_GS' in data and len(data['num_GS']) > 0:
                methods_with_data[name] = data['num_GS']
    else:
        # Filter to only methods that have data for this metric
        methods_with_data = {
            name: data.get(metric_name, {})
            for name, data in all_data.items()
            if metric_name in data and len(data[metric_name]) > 0
        }   
        

    
    if not methods_with_data:
        ax.text(0.5, 0.5, f'No data for {metric_name}', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Debug: Print step information
    print(f"\n  Debug for {metric_name}:")
    
    # Color and marker styles for different methods
    styles = {
        'hierarchy_simple': ('s-', 'red', 'Hierarchy Simple'),
        'hierarchy_simple_c2f': ('d-', 'orange', 'Hierarchy Simple (C2F)'),
        'hierarchy_vcycle': ('o-', 'blue', 'Hierarchy V-cycle'),
        'vcycle': ('o-', 'blue', 'V-cycle'),
        'vcycle_v2': ('^-', 'purple', 'V-cycle v2'),
        'vcycle_default': ('o-', 'blue', 'V-cycle (default)'),
        'vcycle_maxlevel1': ('^-', 'cyan', 'V-cycle (max_level=1)'),
        'simple': ('s-', 'red', 'Simple'),
        'simple_default': ('s-', 'red', 'Simple (default)'),
        'simple_mcmc': ('d-', 'orange', 'Simple (MCMC)'),
        'inv_fcycle': ('^-', 'green', 'Inv-F cycle'),
        'vcycle_v3': ('^-', 'cyan', 'V-cycle v3'),
    }
    
    # Plot each method
    for method_name, method_data in methods_with_data.items():
        style, _, label = styles.get(method_name, ('o-', 'gray', method_name))
        
        # Debug: Print method-specific data
        method_steps = sorted(method_data.keys())
        print(f"    {method_name}: {len(method_steps)} data points")
        if len(method_steps) > 0:
            print(f"      Steps: {method_steps[:5]}..." if len(method_steps) > 5 else f"      Steps: {method_steps}")
        
        # Plot using only steps that have data for this method (no NaN filling)
        # This prevents graph breaks when different methods have different evaluation steps
        steps_with_data = sorted(method_data.keys())
        values = [method_data[step] for step in steps_with_data]
        
        if len(steps_with_data) > 0:
            # Check for gaps in steps
            if len(steps_with_data) > 1:
                gaps = [steps_with_data[i+1] - steps_with_data[i] for i in range(len(steps_with_data)-1)]
                if max(gaps) > min(gaps) * 2:
                    print(f"      WARNING: Large gaps detected in {method_name} steps (max gap: {max(gaps)})")
            ax.plot(steps_with_data, values, label=label, 
                   linewidth=2, markersize=4, alpha=0.8, marker='o')
    
    # Formatting
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def print_summary(all_data: Dict[str, Dict[str, Dict[int, float]]]):
    """Print summary statistics for all methods and metrics."""
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    for method_name, method_metrics in all_data.items():
        if not method_metrics:
            continue
        
        print(f"\n{method_name.upper()}:")
        for metric_name, metric_data in method_metrics.items():
            if not metric_data:
                continue
            
            values = list(metric_data.values())
            steps = list(metric_data.keys())
            if values:
                print(f"  {metric_name.upper()}:")
                print(f"    Steps: {len(values)}")
                print(f"    Final: {values[-1]:.3f} (at step {steps[-1]})")
                print(f"    Max: {max(values):.3f} (at step {steps[np.argmax(values)]})")
                print(f"    Mean: {np.mean(values):.3f}")
                print(f"    Std: {np.std(values):.3f}")
    
    print("="*80 + "\n")


def main():
    """Main function to find results and create comparison graphs."""
    print("="*80)
    print("Plotting Comparison Graphs from Existing Results")
    print("="*80)
    
    # Get result directories
    result_dirs = get_result_directories()
    
    print("\nSearching for result directories...")
    found_dirs = []
    for name, dir_path in result_dirs.items():
        # print(dir_path)
        # print(os.path.list)
        if dir_path and os.path.exists(dir_path):
            print(f"  Found {name}: {dir_path}")
            found_dirs.append((name, dir_path))
        else:
            print(f"  Not found {name}: {dir_path}")
    
    if not found_dirs:
        print("\nERROR: No result directories found!")
        print("Please check that training has been completed.")
        return
    
    # Parse stats files
    print("\nParsing stats files...")
    all_data = {}
    for name, dir_path in found_dirs:
        print(f"  Parsing {name}...")
        metrics = parse_stats_files(Path(dir_path))
        if metrics:
            all_data[name] = metrics
            # print(metrics)
            print(f"    Found metrics: {list(metrics.keys())}")
            for metric_name, metric_data in metrics.items():
                print(f"      {metric_name}: {len(metric_data)} data points")
        else:
            print(f"    No stats found")
    
    if not all_data:
        print("\nERROR: No stats data found in any directory!")
        return
    
    # Print summary
    print_summary(all_data)
    
    # Create output directory
    output_dir = Path("comparison_graphs")
    output_dir.mkdir(exist_ok=True)
    
    # Create combined graph with metrics (excluding num_GS, including time-psnr)
    print("Creating combined comparison graph...")
    
    # Color and marker styles for different methods
    styles = {
        'hierarchy_simple': ('s-', 'red', 'Hierarchy Simple'),
        'hierarchy_simple_c2f': ('d-', 'orange', 'Hierarchy Simple (C2F)'),
        'hierarchy_vcycle': ('o-', 'blue', 'Hierarchy V-cycle'),
        'vcycle': ('o-', 'blue', 'V-cycle'),
        'vcycle_v2': ('^-', 'purple', 'V-cycle v2'),
        'vcycle_default': ('o-', 'blue', 'V-cycle (default)'),
        'vcycle_maxlevel1': ('^-', 'cyan', 'V-cycle (max_level=1)'),
        'simple': ('s-', 'red', 'Simple'),
        'simple_default': ('s-', 'red', 'Simple (default)'),
        'simple_mcmc': ('d-', 'orange', 'Simple (MCMC)'),
        'inv_fcycle': ('^-', 'green', 'Inv-F cycle'),
        'vcycle_v3': ('^-', 'cyan', 'V-cycle v3'),
    }
    
    # Metrics to plot: psnr, ssim, lpips, ellipse_time, and time-psnr (5 total)
    # Exclude num_GS and num_GS_finest
    metrics_to_plot = ['psnr', 'ssim', 'lpips', 'ellipse_time']
    ylabels = {
        'psnr': 'PSNR (dB)',
        'ssim': 'SSIM',
        'lpips': 'LPIPS',
        'ellipse_time': 'Training Time (s)',
    }
    
    # Create figure with 2x3 subplots (5 metrics + 1 time-psnr = 6 total)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot standard metrics on subplots
    for idx, metric_name in enumerate(metrics_to_plot):
        if idx < len(axes):
            ax = axes[idx]
            plot_metric_on_axis(
                ax,
                all_data,
                metric_name,
                ylabel=ylabels.get(metric_name, metric_name.upper())
            )
    
    # Plot time-psnr on the 5th subplot (index 4)
    ax_time_psnr = axes[4]
    for method_name, method_data in all_data.items():
        if 'psnr' not in method_data or 'ellipse_time' not in method_data:
            continue
        
        psnr_data = method_data['psnr']
        time_data = method_data['ellipse_time']
        
        # Extract time-psnr pairs for each step in this method
        # Each method is processed independently with its own step data
        time_psnr_pairs = []
        for step in sorted(set(psnr_data.keys()) & set(time_data.keys())):
            # For each step, extract the corresponding time and psnr values
            time_psnr_pairs.append((time_data[step], psnr_data[step]))
        
        if len(time_psnr_pairs) == 0:
            continue
        
        # Sort by time (x-axis) for plotting
        time_psnr_pairs.sort(key=lambda x: x[0])
        time_values = [pair[0] for pair in time_psnr_pairs]
        psnr_values = [pair[1] for pair in time_psnr_pairs]
        
        style, color, label = styles.get(method_name, ('o-', 'gray', method_name))
        ax_time_psnr.plot(time_values, psnr_values, style, label=label, 
               linewidth=2, markersize=6, alpha=0.8, color=color)
    
    ax_time_psnr.set_xlabel('Training Time per Image (s)', fontsize=11)
    ax_time_psnr.set_ylabel('PSNR (dB)', fontsize=11)
    ax_time_psnr.set_title('PSNR vs Training Time', fontsize=12, fontweight='bold')
    ax_time_psnr.legend(fontsize=9)
    ax_time_psnr.grid(True, alpha=0.3)
    
    # Hide unused subplot (6th one, index 5)
    axes[5].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = output_dir / "comparison_all.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined graph saved to: {output_path}")
    plt.close()
    
    print(f"\nGraphs saved to: {output_dir}")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

