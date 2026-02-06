"""
Build hierarchy and visualize levels from simple_trainer_original.py results.

This script:
1. Loads config from result_dir/cfg.yml
2. Finds PLY file from result_dir/ply directory
3. Runs build_hierarchy.py to build hierarchy
4. Runs visualize_hierarchy_levels.py to render each level
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import yaml
import glob


def load_config_from_yaml(cfg_path: str) -> dict:
    """Load config from yaml file."""
    with open(cfg_path, "r") as f:
        try:
            cfg = yaml.unsafe_load(f)
        except AttributeError:
            # Fallback for older PyYAML versions
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def find_latest_ply(ply_dir: str) -> Optional[str]:
    """Find the latest PLY file in the directory."""
    ply_files = glob.glob(os.path.join(ply_dir, "point_cloud_*.ply"))
    if not ply_files:
        return None
    
    # Sort by modification time, get latest
    ply_files.sort(key=os.path.getmtime, reverse=True)
    return ply_files[0]


def find_ply_by_step(ply_dir: str, step: Optional[int] = None) -> Optional[str]:
    """Find PLY file by step number, or latest if step is None."""
    if step is not None:
        ply_path = os.path.join(ply_dir, f"point_cloud_{step}.ply")
        if os.path.exists(ply_path):
            return ply_path
        return None
    else:
        return find_latest_ply(ply_dir)


def run_build_hierarchy(
    ply_path: str,
    output_dir: Optional[str] = None,
    num_levels: int = 4,
    reduction_factor: float = 0.5,
    clustering_method: str = "kmeans",
) -> str:
    """Run build_hierarchy.py and return the hierarchy output directory."""
    if output_dir is None:
        ply_dir = Path(ply_path).parent
        ply_name = Path(ply_path).stem
        output_dir = str(ply_dir / f"{ply_name}_hierarchy")
    
    cmd = [
        sys.executable,
        "build_hierarchy.py",
        ply_path,
        "--num_levels", str(num_levels),
        "--reduction_factor", str(reduction_factor),
        "--clustering_method", clustering_method,
        "--output_dir", output_dir,
    ]
    
    print(f"\n{'='*60}")
    print(f"Running build_hierarchy.py...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, check=True)
    
    hierarchy_path = os.path.join(output_dir, "hierarchy.pt")
    if not os.path.exists(hierarchy_path):
        raise FileNotFoundError(f"Hierarchy file not found: {hierarchy_path}")
    
    print(f"\n✓ Hierarchy built successfully: {hierarchy_path}")
    return output_dir


def run_visualize_hierarchy(
    hierarchy_path: str,
    data_dir: str,
    dataset_type: str = "colmap",
    output_dir: Optional[str] = None,
    num_cameras: int = 3,
    sh_degree: int = 3,
    white_background: bool = False,
    level_resolution_factor: float = 0.5,
    device: str = "cuda",
    data_factor: int = 1,
    normalize_world_space: bool = False,
    test_every: int = 8,
):
    """Run visualize_hierarchy_levels.py."""
    cmd = [
        sys.executable,
        "visualize_hierarchy_levels.py",
        hierarchy_path,
        data_dir,
        "--dataset_type", dataset_type,
        "--num_cameras", str(num_cameras),
        "--sh_degree", str(sh_degree),
        "--level_resolution_factor", str(level_resolution_factor),
        "--device", device,
        "--data_factor", str(data_factor),
        "--test_every", str(test_every),
    ]
    
    if normalize_world_space:
        cmd.append("--normalize_world_space")
    
    if output_dir is not None:
        cmd.extend(["--output_dir", output_dir])
    
    if white_background:
        cmd.append("--white_background")
    
    print(f"\n{'='*60}")
    print(f"Running visualize_hierarchy_levels.py...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, check=True)
    
    print(f"\n✓ Visualization completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Build hierarchy and visualize levels from simple_trainer_original.py results"
    )
    parser.add_argument(
        "result_dir",
        type=str,
        help="Result directory from simple_trainer_original.py (contains cfg.yml, ply/, etc.)"
    )
    parser.add_argument(
        "--ply_step",
        type=int,
        default=None,
        help="Step number for PLY file (default: use latest)"
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        default=4,
        help="Number of hierarchy levels (default: 4)"
    )
    parser.add_argument(
        "--reduction_factor",
        type=float,
        default=0.33,
        help="Factor to reduce number of Gaussians per level (default: 0.5)"
    )
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="fps_faiss",
        help="Clustering method (ignored; uses FPS + closest points only)"
    )
    parser.add_argument(
        "--hierarchy_output_dir",
        type=str,
        default=None,
        help="Output directory for hierarchy (default: ply_dir/point_cloud_XXX_hierarchy)"
    )
    parser.add_argument(
        "--visualization_output_dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: hierarchy_dir/level_visualizations)"
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=3,
        help="Number of cameras to render (default: 3)"
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=None,
        help="Spherical harmonics degree (default: from cfg.yml)"
    )
    parser.add_argument(
        "--level_resolution_factor",
        type=float,
        default=0.5,
        help="Resolution factor for each level (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip building hierarchy (use existing hierarchy.pt)"
    )
    parser.add_argument(
        "--skip_visualize",
        action="store_true",
        help="Skip visualization (only build hierarchy)"
    )
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")
    
    # Load config
    cfg_path = result_dir / "cfg.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    print(f"Loading config from: {cfg_path}")
    cfg = load_config_from_yaml(str(cfg_path))
    
    # Extract config values
    data_dir = cfg.get("data_dir")
    dataset_type = cfg.get("dataset_type", "colmap")
    white_background = cfg.get("white_background", False)
    data_factor = cfg.get("data_factor", 1)
    normalize_world_space = cfg.get("normalize_world_space", False)
    test_every = cfg.get("test_every", 8)
    sh_degree = args.sh_degree if args.sh_degree is not None else cfg.get("sh_degree", 3)
    
    if data_dir is None:
        raise ValueError("data_dir not found in config. Please specify --data_dir.")
    
    print(f"\nConfig loaded:")
    print(f"  data_dir: {data_dir}")
    print(f"  dataset_type: {dataset_type}")
    print(f"  data_factor: {data_factor}")
    print(f"  normalize_world_space: {normalize_world_space}")
    print(f"  test_every: {test_every}")
    print(f"  white_background: {white_background}")
    print(f"  sh_degree: {sh_degree}")
    
    # Find PLY file
    ply_dir = result_dir / "ply"
    if not ply_dir.exists():
        raise FileNotFoundError(f"PLY directory not found: {ply_dir}")
    
    ply_path = find_ply_by_step(str(ply_dir), args.ply_step)
    if ply_path is None:
        if args.ply_step is not None:
            raise FileNotFoundError(f"PLY file for step {args.ply_step} not found in {ply_dir}")
        else:
            raise FileNotFoundError(f"No PLY files found in {ply_dir}")
    
    print(f"\nUsing PLY file: {ply_path}")
    
    # Build hierarchy
    hierarchy_output_dir = args.hierarchy_output_dir
    if not args.skip_build:
        hierarchy_output_dir = run_build_hierarchy(
            ply_path=ply_path,
            output_dir=hierarchy_output_dir,
            num_levels=args.num_levels,
            reduction_factor=args.reduction_factor,
            clustering_method=args.clustering_method,
        )
        hierarchy_path = os.path.join(hierarchy_output_dir, "hierarchy.pt")
    else:
        # Find existing hierarchy
        if hierarchy_output_dir is None:
            # Try to find hierarchy in ply_dir
            hierarchy_dirs = glob.glob(str(ply_dir / "*_hierarchy"))
            if hierarchy_dirs:
                hierarchy_output_dir = hierarchy_dirs[0]
            else:
                raise FileNotFoundError("No hierarchy found. Run without --skip_build first.")
        
        hierarchy_path = os.path.join(hierarchy_output_dir, "hierarchy.pt")
        if not os.path.exists(hierarchy_path):
            raise FileNotFoundError(f"Hierarchy file not found: {hierarchy_path}")
        
        print(f"\nUsing existing hierarchy: {hierarchy_path}")
    
    # Visualize hierarchy
    if not args.skip_visualize:
        run_visualize_hierarchy(
            hierarchy_path=hierarchy_path,
            data_dir=data_dir,
            dataset_type=dataset_type,
            output_dir=args.visualization_output_dir,
            num_cameras=args.num_cameras,
            sh_degree=sh_degree,
            white_background=white_background,
            level_resolution_factor=args.level_resolution_factor,
            device=args.device,
            data_factor=data_factor,
            normalize_world_space=normalize_world_space,
            test_every=test_every,
        )
    else:
        print(f"\nSkipping visualization (--skip_visualize)")
    
    print(f"\n{'='*60}")
    print(f"✓ All tasks completed!")
    print(f"{'='*60}")
    if not args.skip_build:
        print(f"Hierarchy output: {hierarchy_output_dir}")
    if not args.skip_visualize:
        viz_dir = args.visualization_output_dir
        if viz_dir is None:
            viz_dir = os.path.join(hierarchy_output_dir, "level_visualizations")
        print(f"Visualization output: {viz_dir}")


if __name__ == "__main__":
    main()

