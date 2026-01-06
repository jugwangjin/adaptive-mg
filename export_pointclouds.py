"""
Export level-wise point clouds from trained multigrid checkpoint.

Usage:
    python export_pointclouds.py --ckpt_dir /path/to/ckpt/dir --output_dir /path/to/output
    python export_pointclouds.py --ckpt_file /path/to/ckpt/file.pt --output_dir /path/to/output
"""

import argparse
import os
import glob
from pathlib import Path

import torch

from multigrid_gaussians import MultigridGaussians
from export_multigrid import save_level_pointclouds


def load_multigrid_from_checkpoint(
    ckpt_path: str,
    device: str = "cuda",
):
    """
    Load MultigridGaussians from checkpoint file.
    
    Args:
        ckpt_path: Path to checkpoint file (.pt)
        device: Device to load on
    
    Returns:
        multigrid_gaussians: MultigridGaussians instance
        step: Training step number
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract checkpoint information
    step = ckpt.get("step", 0)
    splats_dict = ckpt["splats"]
    levels = ckpt["levels"].to(device)
    parent_indices = ckpt["parent_indices"].to(device)
    level_indices = ckpt.get("level_indices", {})
    position_scale_reduction = ckpt.get("position_scale_reduction", 0.75)
    max_level = ckpt.get("max_level", None)
    sh_degree = ckpt.get("sh_degree", 3)
    
    # Get number of gaussians
    N = len(levels)
    print(f"Loaded {N} gaussians from step {step}")
    
    # Create a minimal MultigridGaussians instance
    # We need to create a dummy parser for initialization
    # But we'll replace the splats with loaded ones
    class DummyParser:
        def __init__(self):
            self.points = torch.zeros((N, 3)).numpy()
            self.points_rgb = torch.zeros((N, 3)).numpy()
            self.scene_scale = 1.0
    
    dummy_parser = DummyParser()
    
    # Initialize MultigridGaussians with dummy data
    multigrid_gaussians = MultigridGaussians(
        parser=dummy_parser,
        init_type="random",
        init_num_pts=N,
        init_extent=1.0,
        init_opacity=0.1,
        init_scale=1.0,
        scene_scale=1.0,
        sh_degree=sh_degree,
        device=device,
        position_scale_reduction=position_scale_reduction,
        max_level=max_level,
    )
    
    # Replace splats with loaded ones
    for key, value in splats_dict.items():
        if key in multigrid_gaussians.splats:
            multigrid_gaussians.splats[key].data = value.to(device)
    
    # Set hierarchical structure
    multigrid_gaussians.levels = levels
    multigrid_gaussians.parent_indices = parent_indices
    multigrid_gaussians.level_indices = level_indices
    multigrid_gaussians.max_level = max_level
    
    print(f"Loaded hierarchical structure: {len(level_indices)} levels")
    for level, indices in level_indices.items():
        print(f"  Level {level}: {len(indices)} gaussians")
    
    return multigrid_gaussians, step


def main():
    parser = argparse.ArgumentParser(description="Export level-wise point clouds from multigrid checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory containing checkpoint files")
    parser.add_argument("--ckpt_file", type=str, default=None, help="Path to specific checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for point clouds")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--step", type=int, default=None, help="Specific step to export (if multiple checkpoints)")
    
    args = parser.parse_args()
    
    # Determine checkpoint file(s)
    if args.ckpt_file:
        ckpt_files = [args.ckpt_file]
    elif args.ckpt_dir:
        # Find all checkpoint files in directory
        ckpt_pattern = os.path.join(args.ckpt_dir, "*.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        if not ckpt_files:
            # Try subdirectories
            ckpt_pattern = os.path.join(args.ckpt_dir, "**", "*.pt")
            ckpt_files = glob.glob(ckpt_pattern, recursive=True)
        
        if not ckpt_files:
            raise ValueError(f"No checkpoint files found in {args.ckpt_dir}")
        
        # Filter by step if specified
        if args.step is not None:
            ckpt_files = [f for f in ckpt_files if f"step_{args.step}" in f]
        
        # Sort by step number
        def extract_step(f):
            try:
                # Extract step number from filename
                parts = Path(f).stem.split("_")
                for i, part in enumerate(parts):
                    if part == "step" and i + 1 < len(parts):
                        return int(parts[i + 1])
            except:
                return 0
            return 0
        
        ckpt_files.sort(key=extract_step, reverse=True)  # Latest first
        print(f"Found {len(ckpt_files)} checkpoint file(s)")
    else:
        raise ValueError("Either --ckpt_dir or --ckpt_file must be specified")
    
    # Process each checkpoint
    for ckpt_file in ckpt_files:
        print(f"\n{'='*60}")
        print(f"Processing: {ckpt_file}")
        print(f"{'='*60}")
        
        try:
            # Load checkpoint
            multigrid_gaussians, step = load_multigrid_from_checkpoint(
                ckpt_file,
                device=args.device,
            )
            
            # Export point clouds
            save_level_pointclouds(
                multigrid_gaussians=multigrid_gaussians,
                save_dir=args.output_dir,
                step=step,
            )
            
            print(f"\nSuccessfully exported point clouds to: {args.output_dir}")
            
        except Exception as e:
            print(f"Error processing {ckpt_file}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

