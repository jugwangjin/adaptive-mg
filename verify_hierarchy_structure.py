"""
Verify hierarchy structure consistency between build_hierarchy.py and load_hierarchy_multigrid.

This script:
1. Loads hierarchy from hierarchy.pt
2. Verifies the structure matches what multigrid_gaussians_v8.py expects
3. Checks parent-child relationships
4. Validates level numbering
"""

import argparse
import torch
import numpy as np
from pathlib import Path


def verify_hierarchy_structure(hierarchy_path: str):
    """
    Verify hierarchy structure consistency.
    
    Checks:
    1. Level numbering: Level 0 = finest, Level N = coarsest
    2. Parent indices: Each level's parent_indices point to previous (finer) level
    3. Structure matches multigrid_gaussians_v8.py expectations
    """
    print(f"Loading hierarchy from {hierarchy_path}...")
    checkpoint = torch.load(hierarchy_path, map_location="cpu", weights_only=False)
    hierarchy = checkpoint["hierarchy"]
    
    levels_data = hierarchy["levels"]  # List of dicts, one per level
    parent_indices_list = hierarchy["parent_indices"]  # List of tensors, one per level
    num_levels = len(levels_data)
    
    print(f"\n{'='*60}")
    print(f"Hierarchy Structure Verification")
    print(f"{'='*60}")
    print(f"Total levels: {num_levels}")
    print(f"Level 0 = finest (highest resolution)")
    print(f"Level {num_levels-1} = coarsest (lowest resolution)")
    print(f"{'='*60}\n")
    
    # Check each level
    level_start_indices = [0]
    all_means = []
    all_levels = []
    all_parent_indices = []
    
    for level_idx, level_data in enumerate(levels_data):
        num_gaussians = level_data["means"].shape[0]
        all_means.append(level_data["means"])
        all_levels.append(torch.full((num_gaussians,), level_idx, dtype=torch.long))
        
        print(f"Level {level_idx}:")
        print(f"  Number of Gaussians: {num_gaussians}")
        print(f"  Means shape: {level_data['means'].shape}")
        print(f"  Scales shape: {level_data['scales'].shape}")
        
        # Check parent indices
        if level_idx == 0:
            # Level 0 has no parents
            parent_indices = torch.full((num_gaussians,), -1, dtype=torch.long)
            print(f"  Parent indices: None (root level)")
        else:
            # Get parent indices from hierarchy
            parent_indices_in_level = parent_indices_list[level_idx]
            if parent_indices_in_level is None:
                print(f"  WARNING: Level {level_idx} has no parent_indices!")
                parent_indices = torch.full((num_gaussians,), -1, dtype=torch.long)
            else:
                parent_indices_in_level = parent_indices_in_level.cpu()
                print(f"  Parent indices shape: {parent_indices_in_level.shape}")
                print(f"  Parent indices range: [{parent_indices_in_level.min().item()}, {parent_indices_in_level.max().item()}]")
                
                # Convert to global indices (as done in load_hierarchy_multigrid)
                parent_start_idx = level_start_indices[level_idx - 1]
                parent_indices = parent_indices_in_level + parent_start_idx
                
                # Verify parent indices point to previous level
                prev_level_size = level_start_indices[level_idx] - level_start_indices[level_idx - 1]
                valid_parent_mask = (parent_indices_in_level >= 0) & (parent_indices_in_level < prev_level_size)
                num_valid = valid_parent_mask.sum().item()
                print(f"  Valid parent indices: {num_valid}/{num_gaussians}")
                print(f"  Parent indices point to Level {level_idx - 1} (range: [0, {prev_level_size-1}])")
                
                if not valid_parent_mask.all():
                    invalid_count = (~valid_parent_mask).sum().item()
                    print(f"  WARNING: {invalid_count} invalid parent indices!")
                    invalid_indices = parent_indices_in_level[~valid_parent_mask]
                    print(f"    Invalid values: min={invalid_indices.min().item()}, max={invalid_indices.max().item()}")
        
        all_parent_indices.append(parent_indices)
        level_start_indices.append(level_start_indices[-1] + num_gaussians)
        print()
    
    # Concatenate all levels (as done in load_hierarchy_multigrid)
    means = torch.cat(all_means, dim=0)  # [N_total, 3]
    levels = torch.cat(all_levels, dim=0)  # [N_total]
    parent_indices = torch.cat(all_parent_indices, dim=0)  # [N_total]
    
    N_total = means.shape[0]
    print(f"{'='*60}")
    print(f"Combined Structure (as loaded by load_hierarchy_multigrid):")
    print(f"{'='*60}")
    print(f"Total Gaussians: {N_total}")
    print(f"Levels range: [{levels.min().item()}, {levels.max().item()}]")
    print(f"Parent indices range: [{parent_indices.min().item()}, {parent_indices.max().item()}]")
    print()
    
    # Verify parent-child relationships
    print(f"Verifying parent-child relationships...")
    for level_idx in range(1, num_levels):
        level_mask = (levels == level_idx)
        level_parent_indices = parent_indices[level_mask]
        
        # Check that parents are in previous level
        parent_levels = levels[level_parent_indices[level_parent_indices >= 0]]
        if len(parent_levels) > 0:
            expected_level = level_idx - 1
            correct_parents = (parent_levels == expected_level).sum().item()
            total_parents = len(parent_levels)
            print(f"  Level {level_idx} -> Level {level_idx - 1}: {correct_parents}/{total_parents} correct parent assignments")
            
            if correct_parents != total_parents:
                wrong_levels = parent_levels[parent_levels != expected_level]
                if len(wrong_levels) > 0:
                    print(f"    WARNING: {len(wrong_levels)} parents are not in Level {level_idx - 1}!")
                    unique_wrong = wrong_levels.unique()
                    print(f"    Wrong parent levels: {unique_wrong.tolist()}")
    
    print()
    
    # Verify set_visible_mask logic compatibility
    print(f"{'='*60}")
    print(f"Compatibility with multigrid_gaussians_v8.py set_visible_mask:")
    print(f"{'='*60}")
    print(f"Expected behavior:")
    print(f"  - Level 0 (finest): All Level 0 gaussians visible")
    print(f"  - Level 1: All Level 0 and Level 1 gaussians visible (Level 0 parents hidden)")
    print(f"  - Level 2: All Level 0, 1, 2 gaussians visible (Level 0, 1 parents hidden)")
    print(f"  - etc.")
    print()
    
    # Simulate set_visible_mask for each level
    for render_level in range(num_levels):
        # Initialize all as visible
        visible_mask = torch.ones(N_total, dtype=torch.bool)
        
        # Step 1: Set gaussians with level > specified level to invisible
        visible_mask[levels > render_level] = False
        
        # Step 2: From level down to 0, set parent nodes to invisible
        for l in range(render_level, -1, -1):
            level_mask = (levels == l)
            if not level_mask.any():
                continue
            
            parent_indices_at_level = parent_indices[level_mask]
            valid_parent_mask = (parent_indices_at_level != -1)
            if not valid_parent_mask.any():
                continue
            
            valid_parent_indices = parent_indices_at_level[valid_parent_mask]
            visible_mask[valid_parent_indices] = False
        
        num_visible = visible_mask.sum().item()
        visible_by_level = {}
        for l in range(num_levels):
            level_mask = (levels == l)
            visible_by_level[l] = (visible_mask & level_mask).sum().item()
        
        print(f"  Render at Level {render_level}:")
        print(f"    Total visible: {num_visible}/{N_total}")
        for l in range(num_levels):
            if visible_by_level[l] > 0:
                total_at_level = (levels == l).sum().item()
                print(f"      Level {l}: {visible_by_level[l]}/{total_at_level} visible")
        print()
    
    print(f"{'='*60}")
    print(f"Verification complete!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify hierarchy structure consistency"
    )
    parser.add_argument(
        "hierarchy_path",
        type=str,
        help="Path to hierarchy.pt file"
    )
    
    args = parser.parse_args()
    verify_hierarchy_structure(args.hierarchy_path)


if __name__ == "__main__":
    main()

