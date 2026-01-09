"""Debug utility functions for multigrid Gaussian Splatting.

This module contains debug code that can be temporarily enabled/disabled.
"""

import os
import sys
from pathlib import Path


def save_initialization_point_clouds(runner, world_rank: int = 0):
    """Save level1, level2 point clouds and parent connections for debugging.
    
    This function saves:
    - level1_init.ply: Level 1 points (red)
    - level2_init.ply: Level 2 points (green)
    - combined_init.ply: All points colored by level
    - parent_connections.ply: LineSet showing level2 -> level1 parent connections
    
    Args:
        runner: The Runner instance with multigrid_gaussians
        world_rank: Process rank (only rank 0 saves)
    """
    if world_rank != 0:
        return
    
    try:
        import open3d as o3d
        import numpy as np
        
        print("\n" + "="*60)
        print("DEBUG MODE: Saving level1 and level2 point clouds...")
        print("="*60)
        
        # Create debug directory in project root
        project_root = Path(__file__).parent.absolute()
        debug_dir = project_root / "tmp"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Get means and levels
        means = runner.multigrid_gaussians.splats["means"].detach().cpu().numpy()  # [N, 3]
        levels = runner.multigrid_gaussians.levels.cpu().numpy()  # [N,]
        
        # Save level 1
        level1_mask = (levels == 1)
        if level1_mask.any():
            level1_means = means[level1_mask]
            pcd_level1 = o3d.geometry.PointCloud()
            pcd_level1.points = o3d.utility.Vector3dVector(level1_means)
            # Color: red
            colors_level1 = np.zeros((len(level1_means), 3))
            colors_level1[:, 0] = 1.0  # Red
            pcd_level1.colors = o3d.utility.Vector3dVector(colors_level1)
            
            filepath_level1 = os.path.join(debug_dir, "level1_init.ply")
            o3d.io.write_point_cloud(filepath_level1, pcd_level1)
            print(f"Saved level 1 point cloud ({len(level1_means)} points) to: {filepath_level1}")
        
        # Save level 2
        level2_mask = (levels == 2)
        if level2_mask.any():
            level2_means = means[level2_mask]
            pcd_level2 = o3d.geometry.PointCloud()
            pcd_level2.points = o3d.utility.Vector3dVector(level2_means)
            # Color: green
            colors_level2 = np.zeros((len(level2_means), 3))
            colors_level2[:, 1] = 1.0  # Green
            pcd_level2.colors = o3d.utility.Vector3dVector(colors_level2)
            
            filepath_level2 = os.path.join(debug_dir, "level2_init.ply")
            o3d.io.write_point_cloud(filepath_level2, pcd_level2)
            print(f"Saved level 2 point cloud ({len(level2_means)} points) to: {filepath_level2}")
        
        # Save combined point cloud
        pcd_combined = o3d.geometry.PointCloud()
        pcd_combined.points = o3d.utility.Vector3dVector(means)
        # Color based on level
        colors_combined = np.zeros((len(means), 3))
        colors_combined[level1_mask, 0] = 1.0  # Red for level 1
        colors_combined[level2_mask, 1] = 1.0  # Green for level 2
        pcd_combined.colors = o3d.utility.Vector3dVector(colors_combined)
        
        filepath_combined = os.path.join(debug_dir, "combined_init.ply")
        o3d.io.write_point_cloud(filepath_combined, pcd_combined)
        print(f"Saved combined point cloud ({len(means)} points) to: {filepath_combined}")
        
        # Save LineSet: level2 -> level1 parent connections
        if level2_mask.any() and level1_mask.any():
            parent_indices = runner.multigrid_gaussians.parent_indices.cpu().numpy()  # [N,]
            
            # Get level2 indices and their parent indices
            level2_indices = np.where(level2_mask)[0]  # Indices in the full array
            level1_indices = np.where(level1_mask)[0]  # Indices in the full array
            
            # Create mapping: level1 index in full array -> index in level1 subset
            level1_index_map = {idx: i for i, idx in enumerate(level1_indices)}
            
            # Build lines: each level2 point connects to its parent level1 point
            lines = []
            valid_level2_points = []
            valid_level2_indices = []
            
            for level2_idx in level2_indices:
                parent_idx = parent_indices[level2_idx]
                if parent_idx >= 0 and parent_idx in level1_index_map:
                    # level2_idx is the index in full array
                    # We need to map it to the index in the combined point cloud
                    # Since combined point cloud has same order as means, level2_idx is correct
                    # parent_idx is also in full array, so it's correct too
                    lines.append([parent_idx, level2_idx])
                    valid_level2_points.append(level2_idx)
                    valid_level2_indices.append(level2_idx)
            
            if len(lines) > 0:
                # Create LineSet
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(means)  # All points
                line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
                
                # Color lines (cyan for visibility)
                line_colors = np.ones((len(lines), 3)) * np.array([0.0, 1.0, 1.0])  # Cyan
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                
                filepath_lineset = os.path.join(debug_dir, "parent_connections.ply")
                o3d.io.write_line_set(filepath_lineset, line_set)
                print(f"Saved parent connections LineSet ({len(lines)} lines) to: {filepath_lineset}")
            else:
                print("Warning: No valid parent connections found for LineSet")
        
        print("="*60)
        print("Point clouds and LineSet saved. Exiting...")
        print("="*60)
        
    except ImportError:
        print("Warning: open3d is not installed. Skipping point cloud export.")
        print("Install it with: pip install open3d")
    except Exception as e:
        print(f"Error saving point clouds: {e}")
    
    # Exit after saving
    sys.exit(0)

