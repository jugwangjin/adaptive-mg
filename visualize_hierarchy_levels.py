"""
Visualize hierarchy levels by rendering each level separately.

This script:
1. Loads hierarchy from hierarchy.pt (built by build_hierarchy.py)
2. Loads camera data from dataset
3. Renders each level separately and saves images
"""

import argparse
import os
import math
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import open3d as o3d

# Lazy import to avoid pycolmap issues when not using colmap dataset
# from datasets.colmap import Dataset as ColmapDataset, Parser as ColmapParser
# from datasets.nerf import Dataset as NerfDataset, Parser as NerfParser
from multigrid_gaussians_v8 import MultigridGaussians
from utils import knn

# load_hierarchy_multigrid is now a static method of MultigridGaussians
# Use MultigridGaussians.load_hierarchy_multigrid() instead


def save_hierarchy_visualization(
    multigrid_gaussians: MultigridGaussians,
    output_dir: str,
):
    """
    Save hierarchy visualization: level-wise point clouds and parent-child linesets.
    
    Saves:
    - level_{level}.ply: Point cloud for each level
    - lineset_{level}_{level+1}.ply: LineSet connecting level to level+1
    
    Args:
        multigrid_gaussians: MultigridGaussians instance with hierarchy
        output_dir: Output directory for visualization files
    """
    if len(multigrid_gaussians.levels) == 0:
        return
    
    # Create hierarchy visualization subdirectory
    hierarchy_dir = os.path.join(output_dir, "hierarchy_visualization")
    os.makedirs(hierarchy_dir, exist_ok=True)
    
    # Get means and levels
    # Children are now individual (not residual), so we can use splats["means"] directly
    means = multigrid_gaussians.splats["means"].detach().cpu().numpy()  # [N, 3]
    levels = multigrid_gaussians.levels.cpu().numpy()  # [N,]
    parent_indices = multigrid_gaussians.parent_indices.cpu().numpy()  # [N,]
    
    # Get unique levels
    unique_levels = sorted(np.unique(levels))
    
    print(f"\nSaving hierarchy visualization to {hierarchy_dir}...")
    
    # Save point cloud for each level
    for level in unique_levels:
        level_mask = (levels == level)
        if not level_mask.any():
            continue
        
        level_means = means[level_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(level_means)
        
        # Color based on level (use a colormap)
        # Simple color scheme: level 0 = red, level 1 = green, level 2 = blue, etc.
        colors = np.zeros((len(level_means), 3))
        level_idx = unique_levels.index(level)
        if level_idx % 3 == 0:
            colors[:, 0] = 1.0  # Red
        elif level_idx % 3 == 1:
            colors[:, 1] = 1.0  # Green
        else:
            colors[:, 2] = 1.0  # Blue
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        filepath = os.path.join(hierarchy_dir, f"level_{level}.ply")
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"  Saved level {level} point cloud: {filepath} ({len(level_means)} points)")
    
    # Save linesets for level -> level+1 connections
    for i in range(len(unique_levels) - 1):
        level = unique_levels[i]
        next_level = unique_levels[i + 1]
        
        # Get indices for current level and next level
        level_mask = (levels == level)
        next_level_mask = (levels == next_level)
        
        if not level_mask.any() or not next_level_mask.any():
            continue
        
        level_indices = np.where(level_mask)[0]  # Indices in full array
        next_level_indices = np.where(next_level_mask)[0]  # Indices in full array
        
        # Create mapping: level index -> position in level_indices
        level_index_map = {idx: i for i, idx in enumerate(level_indices)}
        
        # Build lines: each next_level point connects to its parent level point
        lines = []
        for next_level_idx in next_level_indices:
            parent_idx = parent_indices[next_level_idx]
            if parent_idx >= 0 and parent_idx in level_index_map:
                # Both indices are in the full array, so we can use them directly
                lines.append([int(parent_idx), int(next_level_idx)])
        
        if len(lines) > 0:
            # Create LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(means)  # All points
            line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
            
            # Color lines (cyan for visibility)
            line_colors = np.ones((len(lines), 3)) * np.array([0.0, 1.0, 1.0])  # Cyan
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            
            filepath = os.path.join(hierarchy_dir, f"lineset_{level}_{next_level}.ply")
            o3d.io.write_line_set(filepath, line_set)
            print(f"  Saved lineset {level}->{next_level}: {filepath} ({len(lines)} lines)")
    
    print(f"  Hierarchy visualization saved to {hierarchy_dir}")


def render_levels(
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
    """
    Render each hierarchy level separately and save images.
    
    Args:
        hierarchy_path: Path to hierarchy.pt file
        data_dir: Path to dataset directory
        dataset_type: "colmap" or "nerf"
        output_dir: Output directory for rendered images
        num_cameras: Number of cameras to render
        sh_degree: Spherical harmonics degree
        white_background: Use white background
        level_resolution_factor: Resolution factor for each level
        device: Device to use
    """
    # Set default output directory
    if output_dir is None:
        hierarchy_dir = Path(hierarchy_path).parent
        output_dir = str(hierarchy_dir / "level_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset parser (lazy import to avoid pycolmap issues when not using colmap)
    print(f"Loading dataset from {data_dir}...")
    print(f"  data_factor: {data_factor}")
    print(f"  normalize: {normalize_world_space}")
    print(f"  test_every: {test_every}")
    if dataset_type == "colmap":
        from datasets.colmap import Dataset as ColmapDataset, Parser as ColmapParser
        parser = ColmapParser(
            data_dir=data_dir,
            factor=data_factor,
            normalize=normalize_world_space,
            test_every=test_every,
        )
        valset = ColmapDataset(parser, split="val")
    elif dataset_type == "nerf":
        from datasets.nerf import Dataset as NerfDataset, Parser as NerfParser
        parser = NerfParser(
            data_dir=data_dir,
            factor=data_factor,
            normalize=normalize_world_space,
            test_every=test_every,
            white_background=white_background,
            extension=".png",
        )
        valset = NerfDataset(parser, split="val")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    scene_scale = parser.scene_scale * 1.1
    
    # Load hierarchy
    multigrid_gaussians, _ = MultigridGaussians.load_hierarchy_multigrid(
        hierarchy_path=hierarchy_path,
        parser=parser,
        scene_scale=scene_scale,
        sh_degree=sh_degree,
        device=device,
    )
    
    # Get max level and verify level structure
    if len(multigrid_gaussians.levels) > 0:
        actual_max_level = int(multigrid_gaussians.levels.max().item())
        actual_min_level = int(multigrid_gaussians.levels.min().item())
        unique_levels = sorted(multigrid_gaussians.levels.unique().cpu().tolist())
        print(f"Hierarchy level structure:")
        print(f"  Actual levels in data: {unique_levels}")
        print(f"  Min level: {actual_min_level}, Max level: {actual_max_level}")
        for level in unique_levels:
            level_count = (multigrid_gaussians.levels == level).sum().item()
            print(f"  Level {level}: {level_count} gaussians")
    else:
        actual_max_level = 0
        actual_min_level = 0
        unique_levels = [0]
    
    max_level = int(multigrid_gaussians.max_level) if multigrid_gaussians.max_level is not None else actual_max_level
    # Use actual unique levels from the hierarchy (multigrid levels: 1=coarsest, N=finest)
    if len(multigrid_gaussians.levels) > 0:
        levels_to_render = sorted(unique_levels)  # Multigrid levels: 1 (coarsest) to N (finest)
    else:
        levels_to_render = [1]
    
    print(f"Rendering {len(levels_to_render)} levels: {levels_to_render}")
    print(f"  Note: Level {levels_to_render[0]} = coarsest (lowest resolution), Level {levels_to_render[-1]} = finest (highest resolution)")
    
    # Select cameras
    val_indices = list(range(len(valset)))
    if len(val_indices) >= num_cameras:
        import random
        random.seed(42)
        camera_indices = random.sample(val_indices, num_cameras)
    else:
        camera_indices = val_indices[:num_cameras] if len(val_indices) > 0 else []
    
    print(f"Rendering {len(camera_indices)} cameras: {camera_indices}")
    
    # Prepare backgrounds
    backgrounds = None
    if white_background:
        backgrounds = torch.ones(1, 3, device=device)
    
    # Render each camera and each level
    for cam_idx in camera_indices:
        # Get camera data
        data = valset[cam_idx]
        camtoworlds = data["camtoworld"].unsqueeze(0).to(device)  # [1, 4, 4]
        Ks = data["K"].unsqueeze(0).to(device)  # [1, 3, 3]
        image_data = data["image"].to(device) / 255.0  # [H, W, C]
        masks = data["mask"].to(device).unsqueeze(0) if "mask" in data else None  # [1, H, W]
        
        # Handle RGBA images
        if image_data.shape[-1] == 4:
            pixels_gt = image_data[..., :3]  # [H, W, 3]
        else:
            pixels_gt = image_data  # [H, W, 3]
        
        height, width = pixels_gt.shape[:2]
        
        # Save GT image
        gt_path = os.path.join(output_dir, f"cam_{cam_idx:04d}_GT.png")
        gt_image = (pixels_gt.detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(gt_path, gt_image)
        print(f"  Saved GT to {gt_path}")
        
        # Render each level (multigrid levels: 1=coarsest, N=finest)
        for render_level in levels_to_render:
            # Calculate downsample factor: Level 1 (coarsest) = highest downsample, Level N (finest) = 1
            # Formula: downsample_factor = (1/level_resolution_factor)^(max_level - render_level)
            # Level 1 (coarsest): downsample_factor = (1/level_resolution_factor)^(max_level - 1)
            # Level N (finest): downsample_factor = 1 (no downsampling)
            level_diff = max_level - render_level
            downsample_factor = (1.0 / level_resolution_factor) ** level_diff
            downsample_factor = max(1, int(downsample_factor))
            
            print(f"  Level {render_level}: downsample_factor={downsample_factor}, "
                  f"render_size={max(1, int(height // downsample_factor))}x{max(1, int(width // downsample_factor))}")
            
            # Downsample for rendering
            if downsample_factor > 1:
                # Calculate new dimensions
                render_height = max(1, int(height // downsample_factor))
                render_width = max(1, int(width // downsample_factor))
                
                # Downsample Ks (camera intrinsics)
                Ks_downsampled = Ks.clone()
                Ks_downsampled[:, 0, 0] = Ks[:, 0, 0] / downsample_factor  # fx
                Ks_downsampled[:, 1, 1] = Ks[:, 1, 1] / downsample_factor  # fy
                Ks_downsampled[:, 0, 2] = Ks[:, 0, 2] / downsample_factor  # cx
                Ks_downsampled[:, 1, 2] = Ks[:, 1, 2] / downsample_factor  # cy
                
                # Downsample masks if provided
                masks_downsampled = None
                if masks is not None:
                    masks_bchw = masks.unsqueeze(1).float()  # [1, 1, H, W]
                    masks_downsampled = F.interpolate(
                        masks_bchw,
                        size=(render_height, render_width),
                        mode='nearest',
                    )
                    masks_downsampled = masks_downsampled.squeeze(1).bool()  # [1, H, W]
            else:
                # No downsample needed
                render_height = height
                render_width = width
                Ks_downsampled = Ks
                masks_downsampled = masks
            
            # Check how many gaussians are visible at this level before rendering
            multigrid_gaussians.set_visible_mask(render_level)
            visible_count = multigrid_gaussians.visible_mask.sum().item()
            total_count = len(multigrid_gaussians.levels)
            level_mask = (multigrid_gaussians.levels == render_level)
            level_count = level_mask.sum().item()
            print(f"    Level {render_level}: {visible_count}/{total_count} gaussians visible "
                  f"(level {render_level} has {level_count} gaussians)")
            
            # Render at downsampled resolution
            colors, alphas, info = multigrid_gaussians.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks_downsampled,
                width=render_width,
                height=render_height,
                level=render_level,
                sh_degree=sh_degree,
                near_plane=0.01,
                far_plane=1e10,  # Use same far_plane as training
                masks=masks_downsampled,
                packed=False,
                sparse_grad=False,
                distributed=False,
                camera_model="pinhole",
                backgrounds=backgrounds,
            )  # colors: [1, render_H, render_W, 3]
            
            colors = colors[0]  # [render_H, render_W, 3]
            
            # Debug: Check value ranges BEFORE clamping
            color_min = colors.min().item()
            color_max = colors.max().item()
            color_sum = colors.sum().item()
            color_mean = colors.mean().item()
            print(f"    Level {render_level} value range: min={color_min:.6f}, max={color_max:.6f}, sum={color_sum:.6f}, mean={color_mean:.6f}")
            
            colors = torch.clamp(colors, 0.0, 1.0)
            
            # Debug: Check if rendered image is all black or has very low values
            if color_sum < 1e-6 or color_mean < 1e-6:
                print(f"    WARNING: Level {render_level} rendered image is all black or very dark!")
                print(f"      color_sum={color_sum:.6f}, color_mean={color_mean:.6f}")
                print(f"      visible_count={visible_count}, level_count={level_count}")
                if "visible_mask" in info:
                    rendered_count = info.get("visible_mask", torch.tensor(0)).sum().item() if isinstance(info.get("visible_mask"), torch.Tensor) else 0
                    print(f"      rendered_count={rendered_count}")
            
            # Upsample to original resolution if needed
            if downsample_factor > 1:
                # Convert to [1, C, H, W] format for F.interpolate
                colors_bchw = colors.permute(2, 0, 1).unsqueeze(0)  # [1, 3, render_H, render_W]
                colors_upsampled = F.interpolate(
                    colors_bchw,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False,
                )
                # Convert back to [H, W, 3] format
                colors = colors_upsampled.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
            
            # Save rendered image
            render_path = os.path.join(output_dir, f"cam_{cam_idx:04d}_level_{render_level}.png")
            render_image = (colors.detach().cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(render_path, render_image)
            print(f"  Saved level {render_level} to {render_path}")
    
    # Save hierarchy visualization (point clouds and linesets)
    save_hierarchy_visualization(
        multigrid_gaussians=multigrid_gaussians,
        output_dir=output_dir,
    )
    
    print(f"\nAll visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hierarchy levels by rendering each level separately"
    )
    parser.add_argument(
        "hierarchy_path",
        type=str,
        help="Path to hierarchy.pt file (built by build_hierarchy.py)"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="colmap",
        choices=["colmap", "nerf"],
        help="Dataset type (default: colmap)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for rendered images (default: hierarchy_dir/level_visualizations)"
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
        default=3,
        help="Spherical harmonics degree (default: 3)"
    )
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="Use white background (for NeRF datasets)"
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
        "--data_factor",
        type=int,
        default=1,
        help="Downsample factor for the dataset (default: 1)"
    )
    parser.add_argument(
        "--normalize_world_space",
        action="store_true",
        help="Normalize world space"
    )
    parser.add_argument(
        "--no_normalize_world_space",
        action="store_true",
        help="Do not normalize world space (default)"
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=8,
        help="Every N images there is a test image (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Handle normalize_world_space flag
    normalize_world_space = args.normalize_world_space and not args.no_normalize_world_space
    
    render_levels(
        hierarchy_path=args.hierarchy_path,
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir,
        num_cameras=args.num_cameras,
        sh_degree=args.sh_degree,
        white_background=args.white_background,
        level_resolution_factor=args.level_resolution_factor,
        device=args.device,
        data_factor=args.data_factor,
        normalize_world_space=normalize_world_space,
        test_every=args.test_every,
    )


if __name__ == "__main__":
    main()

