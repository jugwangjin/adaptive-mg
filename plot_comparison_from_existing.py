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

# Configuration matching run_comparison.py
DATASET_TYPE = "colmap"
DATA_DIR = "/Bean/data/gwangjin/2025/kdgs/360_v2/garden"
DATA_FACTOR = 8
WHITE_BACKGROUND = False
NORMALIZE_WORLD_SPACE = True
PATCH_SIZE = None


def get_result_directories() -> Dict[str, Optional[str]]:
    """Get result directories based on run_comparison.py configuration.
    
    Returns:
        Dictionary with keys: 'vcycle_default', 'vcycle_maxlevel1', 'simple_default', 'simple_mcmc'
        Values are directory paths or None if not found
    """
    dataset_name = Path(DATA_DIR).name
    
    # Build settings string for vcycle
    vcycle_settings_parts = [
        f"type_{DATASET_TYPE}",
        f"factor_{DATA_FACTOR}",
        "vcycle",
    ]
    if DATASET_TYPE == "nerf" and WHITE_BACKGROUND:
        vcycle_settings_parts.append("whitebg")
    if NORMALIZE_WORLD_SPACE:
        vcycle_settings_parts.append("norm")
    if PATCH_SIZE is not None:
        vcycle_settings_parts.append(f"patch_{PATCH_SIZE}")
    vcycle_settings_str = "_".join(vcycle_settings_parts)
    vcycle_base_dir = f"/Bean/log/gwangjin/2025/gsplat/vcycle/{dataset_name}_{vcycle_settings_str}"
    
    # Build settings string for simple
    simple_settings_parts = [
        f"type_{DATASET_TYPE}",
        f"factor_{DATA_FACTOR}",
    ]
    if DATASET_TYPE == "nerf" and WHITE_BACKGROUND:
        simple_settings_parts.append("whitebg")
    if NORMALIZE_WORLD_SPACE:
        simple_settings_parts.append("norm")
    if PATCH_SIZE is not None:
        simple_settings_parts.append(f"patch_{PATCH_SIZE}")
    simple_settings_str = "_".join(simple_settings_parts)
    simple_base_dir = f"/Bean/log/gwangjin/2025/gsplat/baseline/{dataset_name}_{simple_settings_str}"
    
    # Check for vcycle directories
    vcycle_dir = vcycle_base_dir if os.path.exists(vcycle_base_dir) else None
    
    # Check for simple directories (strategy doesn't change path, so just check base)
    simple_dir = simple_base_dir if os.path.exists(simple_base_dir) else None
    
    # Also search for any directories matching the pattern (in case there are variations)
    result_dirs = {}
    
    # Search vcycle directory
    vcycle_parent = Path("/Bean/log/gwangjin/2025/gsplat/vcycle")
    if vcycle_parent.exists():
        for dir_path in vcycle_parent.iterdir():
            if dir_path.is_dir() and dataset_name in dir_path.name and "vcycle" in dir_path.name:
                # Check if it has stats directory
                if (dir_path / "stats").exists():
                    if vcycle_dir is None:
                        vcycle_dir = str(dir_path)
                        result_dirs['vcycle'] = str(dir_path)
                    break
    
    # Search simple/baseline directories
    simple_parent = Path("/Bean/log/gwangjin/2025/gsplat/baseline")
    if simple_parent.exists():
        for dir_path in simple_parent.iterdir():
            if dir_path.is_dir() and dataset_name in dir_path.name:
                # Check if it has stats directory
                if (dir_path / "stats").exists():
                    if simple_dir is None:
                        simple_dir = str(dir_path)
                        result_dirs['simple'] = str(dir_path)
                    break
    
    # Return found directories
    result = {}
    
    result['simple'] = "/Bean/log/gwangjin/2025/gsplat/baseline/garden_type_colmap_factor_8"
    # result['vcycle'] = "/Bean/log/gwangjin/2025/gsplat/vcycle/garden_type_colmap_factor_8_vcycle_norm"
    # result['vcycle2'] = "/Bean/log/gwangjin/2025/gsplat/vcycle/garden_type_colmap_factor_8_vcycle_norm_2"
    
    # result['vc_v2'] = "/Bean/log/gwangjin/2025/gsplat/vcycle/garden_type_colmap_factor_8_vcycle_v2"
    # result['vc_v3'] = "/Bean/log/gwangjin/2025/gsplat/vcycle/garden_type_colmap_factor_8_vcycle_v3"
    # result['vcycle'] = "/Bean/log/gwangjin/2025/gsplat/multigrid_vcycle/garden_type_colmap_factor_8_v1"
    result['inv_fcycle_old'] = "/Bean/log/gwangjin/2025/gsplat/multigrid_inv_fcycle/garden_type_colmap_factor_8_v7_randbkgd"
    # result['inv_fcycle'] = "/Bean/log/gwangjin/2025/gsplat/multigrid_inv_fcycle/garden_type_colmap_factor_8_v13_randbkgd"
    result['vcycle'] = "/Bean/log/gwangjin/2025/gsplat/multigrid_vcycle/garden_type_colmap_factor_8_v14_randbkgd"

    # if vcycle_dir:
    #     result['vcycle'] = vcycle_dir
    # if simple_dir:
    #     result['simple'] = simple_dir
    
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
        'vcycle': ('o-', 'blue', 'V-cycle'),
        'vcycle_default': ('o-', 'blue', 'V-cycle (default)'),
        'vcycle_maxlevel1': ('^-', 'cyan', 'V-cycle (max_level=1)'),
        'simple': ('s-', 'red', 'Simple'),
        'simple_default': ('s-', 'red', 'Simple (default)'),
        'simple_mcmc': ('d-', 'orange', 'Simple (MCMC)'),
        'inv_fcycle': ('^-', 'green', 'Inv-F cycle'),
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
            print(metrics)
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
    
    # Create combined graph with all metrics
    print("Creating combined comparison graph...")
    metrics_to_plot = ['psnr', 'ssim', 'lpips', 'num_GS', 'ellipse_time']
    ylabels = {
        'psnr': 'PSNR (dB)',
        'ssim': 'SSIM',
        'lpips': 'LPIPS',
        'num_GS': 'Number of Gaussians',
        'ellipse_time': 'Training Time (s)',
    }
    
    # Create figure with 2x3 subplots (5 metrics, so 2 rows x 3 cols)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot each metric on a subplot
    for idx, metric_name in enumerate(metrics_to_plot):
        if idx < len(axes):
            ax = axes[idx]
            plot_metric_on_axis(
                ax,
                all_data,
                metric_name,
                ylabel=ylabels.get(metric_name, metric_name.upper())
            )
    
    # Hide unused subplots
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = output_dir / "comparison_all.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined graph saved to: {output_path}")
    plt.close()
    
    print(f"\nGraph saved to: {output_path}")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

