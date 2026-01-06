#!/usr/bin/env python3
"""
Script to compare PSNR results between vcycle_trainer and simple_trainer.

This script reads evaluation results from both trainers and creates a comparison graph.
For multigrid gaussian (vcycle), it uses the finest level rendering for evaluation.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_eval_results(result_dir: Path) -> Dict[int, float]:
    """
    Parse evaluation results from a trainer's result directory.
    
    Args:
        result_dir: Path to the trainer's result directory (contains stats/ subdirectory)
    
    Returns:
        Dictionary mapping step -> PSNR value
    """
    stats_dir = result_dir / "stats"
    if not stats_dir.exists():
        raise ValueError(f"Stats directory not found: {stats_dir}")
    
    psnr_data = {}
    
    # Find all eval JSON files
    for json_file in sorted(stats_dir.glob("val_step*.json")):
        # Extract step number from filename (e.g., "val_step0200.json" -> 200)
        step_str = json_file.stem.replace("val_step", "")
        try:
            step = int(step_str)
        except ValueError:
            continue
        
        # Read PSNR from JSON
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'psnr' in data:
                    psnr_data[step] = float(data['psnr'])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")
            continue
    
    return psnr_data


def create_comparison_graph(
    vcycle_data: Dict[int, float],
    simple_data: Dict[int, float],
    output_path: Path,
    vcycle_name: str = "V-cycle Trainer",
    simple_name: str = "Simple Trainer",
):
    """
    Create a comparison graph showing PSNR over training steps.
    
    Args:
        vcycle_data: Dictionary mapping step -> PSNR for vcycle trainer
        simple_data: Dictionary mapping step -> PSNR for simple trainer
        output_path: Path to save the graph
        vcycle_name: Label for vcycle trainer in the graph
        simple_name: Label for simple trainer in the graph
    """
    # Get all unique steps
    all_steps = sorted(set(list(vcycle_data.keys()) + list(simple_data.keys())))
    
    # Extract PSNR values for each method
    vcycle_psnr = [vcycle_data.get(step, np.nan) for step in all_steps]
    simple_psnr = [simple_data.get(step, np.nan) for step in all_steps]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot both methods
    plt.plot(all_steps, vcycle_psnr, 'o-', label=vcycle_name, linewidth=2, markersize=4, alpha=0.8)
    plt.plot(all_steps, simple_psnr, 's-', label=simple_name, linewidth=2, markersize=4, alpha=0.8)
    
    # Formatting
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('PSNR Comparison: V-cycle vs Simple Trainer', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison graph saved to: {output_path}")
    
    # Also save as PNG with lower DPI for quick viewing
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Comparison graph (PNG) saved to: {png_path}")
    
    plt.close()


def print_statistics(vcycle_data: Dict[int, float], simple_data: Dict[int, float]):
    """Print summary statistics for both methods."""
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    # V-cycle statistics
    if vcycle_data:
        vcycle_values = list(vcycle_data.values())
        print(f"\nV-cycle Trainer:")
        print(f"  Number of evaluations: {len(vcycle_values)}")
        print(f"  Final PSNR: {vcycle_values[-1]:.3f} dB")
        print(f"  Max PSNR: {max(vcycle_values):.3f} dB (at step {max(vcycle_data, key=vcycle_data.get)})")
        print(f"  Mean PSNR: {np.mean(vcycle_values):.3f} dB")
        print(f"  Std PSNR: {np.std(vcycle_values):.3f} dB")
    else:
        print("\nV-cycle Trainer: No data found")
    
    # Simple trainer statistics
    if simple_data:
        simple_values = list(simple_data.values())
        print(f"\nSimple Trainer:")
        print(f"  Number of evaluations: {len(simple_values)}")
        print(f"  Final PSNR: {simple_values[-1]:.3f} dB")
        print(f"  Max PSNR: {max(simple_values):.3f} dB (at step {max(simple_data, key=simple_data.get)})")
        print(f"  Mean PSNR: {np.mean(simple_values):.3f} dB")
        print(f"  Std PSNR: {np.std(simple_values):.3f} dB")
    else:
        print("\nSimple Trainer: No data found")
    
    # Comparison
    if vcycle_data and simple_data:
        common_steps = sorted(set(vcycle_data.keys()) & set(simple_data.keys()))
        if common_steps:
            print(f"\nComparison (at common steps):")
            vcycle_common = [vcycle_data[s] for s in common_steps]
            simple_common = [simple_data[s] for s in common_steps]
            diff = np.array(vcycle_common) - np.array(simple_common)
            print(f"  Average PSNR difference: {np.mean(diff):.3f} dB (V-cycle - Simple)")
            print(f"  Max improvement: {np.max(diff):.3f} dB (at step {common_steps[np.argmax(diff)]})")
            print(f"  Max degradation: {np.min(diff):.3f} dB (at step {common_steps[np.argmin(diff)]})")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PSNR results between vcycle_trainer and simple_trainer"
    )
    parser.add_argument(
        "--vcycle_dir",
        type=str,
        required=True,
        help="Path to vcycle_trainer result directory (contains stats/ subdirectory)",
    )
    parser.add_argument(
        "--simple_dir",
        type=str,
        required=True,
        help="Path to simple_trainer result directory (contains stats/ subdirectory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_graph.pdf",
        help="Output path for the comparison graph (default: comparison_graph.pdf)",
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    vcycle_dir = Path(args.vcycle_dir)
    simple_dir = Path(args.simple_dir)
    output_path = Path(args.output)
    
    # Validate directories
    if not vcycle_dir.exists():
        raise ValueError(f"V-cycle result directory not found: {vcycle_dir}")
    if not simple_dir.exists():
        raise ValueError(f"Simple trainer result directory not found: {simple_dir}")
    
    print("="*60)
    print("PSNR Comparison: V-cycle vs Simple Trainer")
    print("="*60)
    print(f"V-cycle results: {vcycle_dir}")
    print(f"Simple trainer results: {simple_dir}")
    print(f"Output graph: {output_path}")
    print("="*60)
    
    # Parse evaluation results
    print("\nParsing evaluation results...")
    vcycle_data = parse_eval_results(vcycle_dir)
    simple_data = parse_eval_results(simple_dir)
    
    if not vcycle_data:
        raise ValueError(f"No evaluation results found in {vcycle_dir}/stats/")
    if not simple_data:
        raise ValueError(f"No evaluation results found in {simple_dir}/stats/")
    
    print(f"Found {len(vcycle_data)} evaluations for V-cycle trainer")
    print(f"Found {len(simple_data)} evaluations for Simple trainer")
    
    # Print statistics
    print_statistics(vcycle_data, simple_data)
    
    # Create comparison graph
    print("Creating comparison graph...")
    create_comparison_graph(vcycle_data, simple_data, output_path)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

