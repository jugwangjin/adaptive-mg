"""
Export functions for multigrid Gaussian Splatting.

This module extends export_splats to support saving multigrid hierarchical structure.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Literal

import numpy as np
import torch
from torch import Tensor

from gsplat import export_splats
from multigrid_gaussians_v8 import MultigridGaussians


def export_multigrid_splats(
    multigrid_gaussians: MultigridGaussians,
    format: Literal["ply", "splat", "ply_compressed"] = "ply",
    save_to: Optional[str] = None,
    sh_degree: int = 3,
) -> bytes:
    """
    Export multigrid Gaussian Splatting model with hierarchical structure.
    
    This function exports the full hierarchical structure including:
    - All gaussian parameters (means, scales, quats, opacities, sh0, shN)
    - Hierarchical structure (levels, parent_indices, level_indices)
    
    Args:
        multigrid_gaussians: MultigridGaussians instance to export
        format: Export format. Options: "ply", "splat", "ply_compressed". Default: "ply"
        save_to: Output file path. If provided, the bytes will be written to file.
        sh_degree: Spherical harmonics degree
    
    Returns:
        bytes: Binary file representing the model (if format supports it)
    """
    # Get all splats with hierarchical structure applied
    all_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False)
    
    # Prepare parameters for export_splats
    means = all_splats["means"]  # [N, 3]
    scales = all_splats["scales"]  # [N, 3] (log space)
    quats = all_splats["quats"]  # [N, 4]
    opacities = all_splats["opacities"]  # [N,]
    
    # Handle colors (SH coefficients or features)
    if "sh0" in all_splats and "shN" in all_splats:
        sh0 = all_splats["sh0"]  # [N, 1, 3]
        shN = all_splats["shN"]  # [N, K-1, 3]
    elif "features" in all_splats and "colors" in all_splats:
        # For appearance optimization mode, convert colors to SH
        # This is a simplified conversion - may need adjustment
        colors = torch.sigmoid(all_splats["colors"])  # [N, 3]
        from utils import rgb_to_sh
        sh0 = rgb_to_sh(colors.unsqueeze(1))  # [N, 1, 3]
        shN = torch.zeros((len(colors), sh_degree * (sh_degree + 1), 3), 
                          device=colors.device, dtype=colors.dtype)  # [N, K-1, 3]
    else:
        raise ValueError("Invalid splats structure: missing sh0/shN or features/colors")
    
    # Export using standard export_splats
    data = export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format=format,
        save_to=save_to,
    )
    
    return data


def save_multigrid_checkpoint(
    multigrid_gaussians: MultigridGaussians,
    step: int,
    save_dir: str,
    sh_degree: int = 3,
):
    """
    Save multigrid gaussians checkpoint including hierarchical structure.
    
    Args:
        multigrid_gaussians: MultigridGaussians instance to save
        step: Training step number
        save_dir: Directory to save checkpoint
        sh_degree: Spherical harmonics degree
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save checkpoint with hierarchical structure
    checkpoint_path = os.path.join(save_dir, f"multigrid_checkpoint_step_{step}.pt")
    
    checkpoint = {
        "step": step,
        "splats": multigrid_gaussians.splats.state_dict(),
        "levels": multigrid_gaussians.levels.cpu(),
        "parent_indices": multigrid_gaussians.parent_indices.cpu(),
        "level_indices": {
            k: v for k, v in multigrid_gaussians.level_indices.items()
        },
        "position_scale_reduction": multigrid_gaussians.position_scale_reduction,
        "max_level": multigrid_gaussians.max_level,
        "sh_degree": sh_degree,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved multigrid checkpoint to: {checkpoint_path}")
    
    return checkpoint_path


def save_level_pointclouds(
    multigrid_gaussians: MultigridGaussians,
    save_dir: str,
    step: Optional[int] = None,
):
    """
    Save level-wise point clouds (mean3d only) using Open3D.
    
    Each level's visible gaussians (using visible mask) are saved as a separate point cloud file.
    Only 3D positions (means) are saved for visualization.
    
    Args:
        multigrid_gaussians: MultigridGaussians instance
        save_dir: Directory to save point clouds
        step: Optional step number for filename
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Warning: open3d is not installed. Skipping point cloud export.")
        print("Install it with: pip install open3d")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all splats with hierarchical structure applied
    all_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False)
    means = all_splats["means"].detach().cpu().numpy()  # [N, 3]
    levels = multigrid_gaussians.levels.cpu().numpy()  # [N,]
    
    # Get unique levels
    unique_levels = np.unique(levels)
    max_level = int(unique_levels.max())
    
    # Save point cloud for each level using visible mask
    for level in unique_levels:
        level_int = int(level)
        
        # Get visible mask for this level
        visible_mask = multigrid_gaussians.set_visible_mask(level_int)
        visible_mask_np = visible_mask.cpu().numpy()
        
        # Filter means by visible mask
        level_means = means[visible_mask_np]  # [M, 3]
        
        if len(level_means) == 0:
            print(f"Level {level_int}: No visible gaussians, skipping...")
            continue
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(level_means)
        
        # Set color based on level (for visualization)
        # Use a color map: level 1 = red, level 2 = green, level 3 = blue, etc.
        num_points = len(level_means)
        colors = np.zeros((num_points, 3))
        if level_int == 1:
            colors[:, 0] = 1.0  # Red
        elif level_int == 2:
            colors[:, 1] = 1.0  # Green
        elif level_int == 3:
            colors[:, 2] = 1.0  # Blue
        else:
            # For higher levels, use a gradient
            hue = (level_int - 1) / max_level if max_level > 0 else 0.0
            colors[:, 0] = hue
            colors[:, 1] = 1.0 - hue
            colors[:, 2] = 0.5
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save point cloud
        if step is not None:
            filename = f"level_{level_int}_step_{step}.ply"
        else:
            filename = f"level_{level_int}.ply"
        
        filepath = os.path.join(save_dir, filename)
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"Saved level {level_int} point cloud ({len(level_means)} visible points) to: {filepath}")
    
    # Also save combined point cloud with all visible gaussians at max level
    max_level_visible_mask = multigrid_gaussians.set_visible_mask(max_level)
    max_level_visible_mask_np = max_level_visible_mask.cpu().numpy()
    all_visible_means = means[max_level_visible_mask_np]
    
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(all_visible_means)
    
    # Color by level for visible gaussians
    all_colors = np.zeros((len(all_visible_means), 3))
    visible_levels = levels[max_level_visible_mask_np]
    for level in unique_levels:
        level_mask = (visible_levels == level)
        level_int = int(level)
        if level_int == 1:
            all_colors[level_mask, 0] = 1.0  # Red
        elif level_int == 2:
            all_colors[level_mask, 1] = 1.0  # Green
        elif level_int == 3:
            all_colors[level_mask, 2] = 1.0  # Blue
        else:
            hue = (level_int - 1) / max_level if max_level > 0 else 0.0
            all_colors[level_mask, 0] = hue
            all_colors[level_mask, 1] = 1.0 - hue
            all_colors[level_mask, 2] = 0.5
    
    pcd_all.colors = o3d.utility.Vector3dVector(all_colors)
    
    if step is not None:
        filename_all = f"all_levels_step_{step}.ply"
    else:
        filename_all = "all_levels.ply"
    
    filepath_all = os.path.join(save_dir, filename_all)
    o3d.io.write_point_cloud(filepath_all, pcd_all)
    print(f"Saved combined point cloud ({len(all_visible_means)} visible points) to: {filepath_all}")

