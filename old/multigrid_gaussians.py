"""
Graph-based hierarchical Gaussian Splatting structure for multigrid methods.

This module provides a hierarchical structure where each Gaussian has level and parent_index
attributes to represent the hierarchical graph structure.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gsplat.rendering import rasterization
from utils import knn, rgb_to_sh


@torch.no_grad()
def find_closest_points_chunked(
    query_points: Tensor,
    reference_points: Tensor,
    chunk_size: int = 5000,
) -> Tensor:
    """
    Find closest reference point for each query point, processing in chunks to save memory.
    
    Args:
        query_points: [M, D] tensor of query points
        reference_points: [N, D] tensor of reference points
        chunk_size: Number of query points to process at once
    
    Returns:
        indices: [M] tensor of indices into reference_points for closest point
    """
    M, D = query_points.shape
    N = reference_points.shape[0]
    device = query_points.device
    
    closest_indices = torch.zeros(M, dtype=torch.long, device=device)
    
    # Process query points in chunks
    for i in range(0, M, chunk_size):
        end_idx = min(i + chunk_size, M)
        query_chunk = query_points[i:end_idx]  # [chunk_size, D]
        
        # Compute distances: [chunk_size, N]
        distances = torch.cdist(query_chunk, reference_points, p=2)
        
        # Find closest reference point for each query point in chunk
        closest_indices[i:end_idx] = distances.argmin(dim=1)  # [chunk_size]
    
    return closest_indices


class MultigridGaussians:
    """
    Multigrid Gaussian Splatting structure with hierarchical levels.
    
    Each Gaussian has level and parent_index attributes to represent the hierarchical graph structure.
    """
    @torch.no_grad()
    def __init__(
        self,
        parser,  # ColmapParser or NerfParser
        cfg,
        init_type: str = "sfm",
        init_num_pts: int = 100_000,
        init_extent: float = 3.0,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        means_lr: float = 1.6e-4,
        scales_lr: float = 5e-3,
        opacities_lr: float = 5e-2,
        quats_lr: float = 1e-3,
        sh0_lr: float = 2.5e-3,
        shN_lr: float = 2.5e-3 / 20,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        sparse_grad: bool = False,
        visible_adam: bool = False,
        batch_size: int = 1,
        feature_dim: Optional[int] = None,
        device: str = "cuda",
        world_rank: int = 0,
        world_size: int = 1,
        position_scale_reduction: float = 0.75,
        max_level: Optional[int] = None,
    ):
        """
        Initialize multigrid gaussians with hierarchical structure.
        
        This creates gaussians with the same structure as create_splats_with_optimizers,
        but adds level and parent_index attributes for hierarchical graph structure.
        
        Attributes:
            splats: ParameterDict with gaussian parameters (means, scales, quats, opacities, sh0, shN)
            optimizers: Dict of optimizers for each parameter
            level_indices: Dict mapping level -> list of gaussian indices at that level
            levels: Tensor [N,] with level for each gaussian
            parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent)
        """
        # Initialize points and colors with hierarchical structure using k-means
        # Level 2: Original SFM points
        # Level 1: Level 2 points downsampled using k-means
        if init_type == "sfm":
            points_level2_all = torch.from_numpy(parser.points).float()
            rgbs_level2_all = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif init_type == "random":
            points_level2_all = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs_level2_all = torch.rand((init_num_pts, 3))
        else:
            raise ValueError("Please specify a correct init_type: sfm or random")

        self.cfg = cfg
        
        # Hierarchical initialization using uniform sampling:
        # - Level 2: Original SFM points
        # - Level 1: Uniformly sampled subset of Level 2 points
        init_level1_ratio = getattr(cfg, 'init_level1_ratio', 0.1)
        
        n_level2_all = len(points_level2_all)
        n_level1 = max(1, int(n_level2_all * init_level1_ratio))
        
        # Move to device
        points_level2_device = points_level2_all.to(device)
        rgbs_level2_device = rgbs_level2_all.to(device)
        
        # Check if we should create Level 1 points
        if n_level1 > 0 and n_level1 < n_level2_all:
            # Get sampling method from config
            sampling_method = getattr(cfg, 'parent_sampling_method', 'uniform')
            print(f"Hierarchical initialization: Level 2={n_level2_all} points, Level 1={n_level1} points ({sampling_method} sampling)")
            
            if sampling_method == "fps":
                # Farthest Point Sampling (FPS) using fpsample package
                try:
                    import fpsample
                    # fpsample expects numpy array [N, 3]
                    points_np = points_level2_device.cpu().numpy()  # [n_level2_all, 3]
                    # Sample n_level1 points using FPS
                    # fps_sampling returns indices as numpy array
                    level1_indices_np = fpsample.bucket_fps_kdline_sampling(points_np, n_level1, h=5)
                    # Convert back to torch tensor on the correct device
                    level1_indices = torch.from_numpy(level1_indices_np).to(device=device, dtype=torch.long)
                except ImportError:
                    print("Warning: fpsample not available, falling back to uniform sampling")
                    print("Install with: pip install fpsample")
                    # Fallback to uniform sampling
                    torch.manual_seed(42)  # For reproducibility
                    level1_indices = torch.randperm(n_level2_all, device=device)[:n_level1]
                except Exception as e:
                    print(f"Warning: FPS sampling failed ({e}), falling back to uniform sampling")
                    # Fallback to uniform sampling
                    torch.manual_seed(42)  # For reproducibility
                    level1_indices = torch.randperm(n_level2_all, device=device)[:n_level1]
            else:
                # Uniform sampling: randomly select n_level1 points for Level 1
                torch.manual_seed(42)  # For reproducibility
                level1_indices = torch.randperm(n_level2_all, device=device)[:n_level1]
            
            level2_indices = torch.ones(n_level2_all, dtype=torch.bool, device=device)
            level2_indices[level1_indices] = False  # Mark Level 1 points as not Level 2
            
            # Extract Level 1 and Level 2 points
            points_level1 = points_level2_device[level1_indices]  # [n_level1, 3]
            rgbs_level1 = rgbs_level2_device[level1_indices]  # [n_level1, 3]
            
            points_level2 = points_level2_device[level2_indices]  # [n_level2_all - n_level1, 3]
            rgbs_level2 = rgbs_level2_device[level2_indices]  # [n_level2_all - n_level1, 3]
            
            # Combine Level 1 and Level 2 points
            # Order: Level 1 first, then Level 2
            points_all = torch.cat([points_level1, points_level2], dim=0)  # [n_level1 + n_level2, 3]
            rgbs_all = torch.cat([rgbs_level1, rgbs_level2], dim=0)  # [n_level1 + n_level2, 3]
        else:
            # No Level 1 points: all points become Level 1 (root nodes)
            print(f"Hierarchical initialization: All {n_level2_all} points set to Level 1 (no Level 2)")
            points_all = points_level2_device  # All points
            rgbs_all = rgbs_level2_device  # All points
            n_level1 = n_level2_all  # All points are Level 1
            points_level1 = points_all
            rgbs_level1 = rgbs_all
            points_level2 = torch.empty(0, 3, device=device)  # Empty
            rgbs_level2 = torch.empty(0, 3, device=device)  # Empty
        
        # Initialize scales based on all SfM points (not separated by level)
        # Calculate scale based on all points to get consistent scale across levels
        if len(points_all) > 0:
            # Calculate scale based on all SfM points
            dist2_avg_all = (knn(points_all, 4)[:, 1:] ** 2).mean(dim=-1)  # [n_all,]
            dist_avg_all = torch.sqrt(dist2_avg_all)
            scales_all = torch.log(dist_avg_all * init_scale).unsqueeze(-1).repeat(1, 3)  # [n_all, 3]
            
            # Split scales for Level 1 and Level 2
            if len(points_level1) > 0:
                scales_level1 = scales_all[:len(points_level1)]  # [n_level1, 3]
            else:
                scales_level1 = torch.empty(0, 3, device=device)
        else:
            scales_level1 = torch.empty(0, 3, device=device)
            scales_all = torch.empty(0, 3, device=device)
        
        # Distribute the GSs to different ranks (also works for single rank)
        # Note: For hierarchical structure, we need to be careful about parent-child relationships
        # For simplicity, distribute Level 1 and Level 2 separately
        # Level 1: All ranks get the same Level 1 points (for consistency)
        # Level 2: Distribute across ranks
        points_level1_dist = points_level1  # All ranks get same Level 1
        rgbs_level1_dist = rgbs_level1
        
        n_level2 = len(points_level2)
        if n_level2 > 0:
            # Level 2: Use scales from all points calculation
            scales_level2_all = scales_all[len(points_level1):]  # [n_level2, 3]
            
            # Find closest Level 1 point for ALL Level 2 points (before distribution)
            # This must be done BEFORE converting to relative coordinates and distribution
            if len(points_level2) > 0 and len(points_level1) > 0:
                # Find closest Level 1 point for each Level 2 point (using all Level 2 points)
                closest_level1_indices_all = find_closest_points_chunked(
                    query_points=points_level2,  # All Level 2 points
                    reference_points=points_level1,  # All Level 1 points
                    chunk_size=5000,
                )  # [n_level2] - indices into points_level1 (0 to n_level1-1)
                
                # Validate indices: should be in range [0, n_level1-1]
                assert (closest_level1_indices_all >= 0).all() and (closest_level1_indices_all < n_level1).all(), \
                    f"Invalid parent indices: {closest_level1_indices_all.min().item()} to {closest_level1_indices_all.max().item()}, " \
                    f"expected range [0, {n_level1-1}]"
                
                # Get parent absolute positions for all Level 2 points
                parent_absolute_means_all = points_level1[closest_level1_indices_all]  # [n_level2, 3]
                # Convert all Level 2 absolute positions to relative (residual)
                # Apply position_scale_reduction: Level 2 has scale_factor = position_scale_reduction^(2-1) = position_scale_reduction
                scale_factor_level2 = position_scale_reduction ** (2 - 1)  # Level 2 scale factor
                points_level2_relative_all = (points_level2 - parent_absolute_means_all) / scale_factor_level2  # [n_level2, 3]
            else:
                # No Level 2 points to process
                points_level2_relative_all = points_level2
                closest_level1_indices_all = torch.empty(0, dtype=torch.long, device=device)
            
            # Now distribute Level 2 points (and their parent indices) across ranks
            points_level2_dist = points_level2_relative_all[world_rank::world_size]  # Distributed relative coordinates
            rgbs_level2_dist = rgbs_level2[world_rank::world_size]
            scales_level2 = scales_level2_all[world_rank::world_size]
            closest_level1_indices = closest_level1_indices_all[world_rank::world_size]  # Distributed parent indices
            
            n_level1_distributed = n_level1
            n_level2_distributed = len(points_level2_dist)
            
            # Validate distributed indices: should be in range [0, n_level1_distributed-1]
            if len(closest_level1_indices) > 0:
                assert (closest_level1_indices >= 0).all() and (closest_level1_indices < n_level1_distributed).all(), \
                    f"Invalid distributed parent indices: {closest_level1_indices.min().item()} to {closest_level1_indices.max().item()}, " \
                    f"expected range [0, {n_level1_distributed-1}]"
            
            # Combine distributed points: Level 1 (absolute) + Level 2 (relative)
            points = torch.cat([points_level1_dist, points_level2_dist], dim=0)
            rgbs = torch.cat([rgbs_level1_dist, rgbs_level2_dist], dim=0)
            scales = torch.cat([scales_level1, scales_level2], dim=0)
        else:
            # No Level 2 points: only Level 1 points exist
            points_level2_dist = torch.empty(0, 3, device=device)
            rgbs_level2_dist = torch.empty(0, 3, device=device)
            scales_level2 = torch.empty(0, 3, device=device)
            closest_level1_indices = torch.empty(0, dtype=torch.long, device=device)
            
            # Distribute Level 1 points across ranks
            # Note: scales_level1 is already calculated above with level1_scale_multiplier
            points = points_level1_dist[world_rank::world_size]
            rgbs = rgbs_level1_dist[world_rank::world_size]
            scales = scales_level1[world_rank::world_size]
            
            n_level1_distributed = len(points)
            n_level2_distributed = 0
        
        N = points.shape[0]
        
        quats = torch.rand((N, 4))  # [N, 4]
        # Opacities are independent parameters (not residual)
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

        # Initialize level and parent_index
        if n_level2 > 0:
            # Hierarchical structure: Level 1 and Level 2
            # Level 1: indices [0, n_level1_distributed-1], parent = -1 (root)
            # Level 2: indices [n_level1_distributed, N-1], parent = closest Level 1 point
            levels = torch.ones(N, dtype=torch.long, device=device)
            levels[:n_level1_distributed] = 1  # Level 1
            levels[n_level1_distributed:] = 2  # Level 2
            
            parent_indices = torch.full((N,), -1, dtype=torch.long, device=device)
            # Level 1 has no parent (root nodes)
            # Level 2 points: assign parent to closest Level 1 point
            if n_level2_distributed > 0:
                parent_indices[n_level1_distributed:] = closest_level1_indices
        else:
            # No Level 2 points: all points are Level 1 (root nodes)
            levels = torch.ones(N, dtype=torch.long, device=device)  # All Level 1
            parent_indices = torch.full((N,), -1, dtype=torch.long, device=device)  # All root nodes

        params = [
            # name, value, lr
            # Note: Level 1 means are absolute, Level 2+ means are relative to parent
            ("means", torch.nn.Parameter(points), means_lr * scene_scale),
            ("scales", torch.nn.Parameter(scales), scales_lr),
            ("quats", torch.nn.Parameter(quats), quats_lr),
            ("opacities", torch.nn.Parameter(opacities), opacities_lr),
        ]

        if feature_dim is None:
            # color is SH coefficients.
            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
            colors[:, 0, :] = rgb_to_sh(rgbs)
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
        else:
            # features will be used for appearance and view-dependent shading
            features = torch.rand(N, feature_dim)  # [N, feature_dim]
            params.append(("features", torch.nn.Parameter(features), sh0_lr))
            colors = torch.logit(rgbs)  # [N, 3]
            params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
        
        # Scale learning rate based on batch size
        BS = batch_size * world_size
        optimizer_class = None
        if sparse_grad:
            optimizer_class = torch.optim.SparseAdam
        elif visible_adam:
            from gsplat.optimizers import SelectiveAdam
            optimizer_class = SelectiveAdam
        else:
            optimizer_class = torch.optim.Adam
        
        optimizers = {
            name: optimizer_class(
                [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                fused=True,
            )
            for name, _, lr in params
        }
        
        # Initialize level_indices structure
        # level_indices[level] = list of gaussian indices at that level
        self.level_indices: Dict[int, List[int]] = {}
        level1_indices = torch.where(levels == 1)[0].cpu().tolist()
        level2_indices = torch.where(levels == 2)[0].cpu().tolist()
        self.level_indices[1] = level1_indices
        if len(level2_indices) > 0:
            self.level_indices[2] = level2_indices
        
        # Store as instance attributes
        self.splats = splats
        self.optimizers = optimizers
        self.levels = levels
        self.parent_indices = parent_indices
        # Initialize visible_mask (all True by default)
        self.visible_mask = torch.ones(N, dtype=torch.bool, device=device)
        # Position scale reduction factor for hierarchical structure
        # Higher level gaussians are constrained to stay closer to their parents
        self.position_scale_reduction = position_scale_reduction
        # Maximum level for hierarchical structure
        # If set, gaussians at max_level will only duplicate (not split) even if split conditions are met
        self.max_level = max_level
    
    def get_splats(self, level: Optional[int] = None, detach_parents: bool = False) -> Dict[str, Tensor]:
        """
        Get splats with hierarchical structure applied.
        
        - Level 1 gaussians: Use their own parameters directly
        - Level 2+ gaussians: 
          * means: parent mean + (child residual * scale_factor) - residual with position scaling
          * scales, quats: use child's own parameters directly (independent, not residual)
          * opacities: parent opacity + child residual (residual parameter, like means, sh0, shN)
          * sh0, shN, etc.: parent parameter + child residual - residual
        
        Args:
            level: If specified, only return splats for gaussians at this level.
                  If None, return splats for all gaussians with hierarchical structure applied.
            detach_parents: If True, detach parent parameters from computation graph.
                          This prevents parent parameters from receiving gradients when
                          leaf nodes are being trained. Useful for V-cycle training where
                          we want to train leaf nodes independently.
        
        Returns:
            Dict[str, Tensor] with processed splats where:
            - means: hierarchical structure applied (parent + child residual * scale_factor)
            - scales, quats, opacities: independent parameters (child's own values)
            - sh0, shN: residual parameters (parent + child residual)
            - sh0, shN, etc.: hierarchical structure applied (parent + child residual)
            Note: Returns plain tensors (not ParameterDict) to preserve gradient flow.
        """
        device = self.splats["means"].device
        N = self.splats["means"].shape[0]
        
        # Determine which gaussians to process
        if level is not None:
            # Only process gaussians at the specified level
            mask = (self.levels == level)
            indices = torch.where(mask)[0]
            if len(indices) == 0:
                # Return empty splats if no gaussians at this level
                return {
                    name: torch.empty((0, *param.shape[1:]), device=device, dtype=param.dtype)
                    for name, param in self.splats.items()
                }
        else:
            # Process all gaussians
            indices = torch.arange(N, device=device)
        
        # Process level by level: start from level 2 and go up
        # This ensures that when processing level N, all lower levels are already processed
        max_level = int(self.levels.max().item()) if len(indices) > 0 else 1
        
        # Store final parameters for all gaussians (for efficient lookup)
        # Use v + 0 to preserve gradients and create a view that tracks gradients
        # This creates a new tensor that shares the computation graph with the original
        # Note: v + 0 preserves gradients better than clone()
        # final_params = {k: v + 0 for k, v in self.splats.items()}
        # Initialize with zeros - we'll accumulate parameters level by level
        # This is more memory efficient than cloning all parameters upfront
        final_params = {k: torch.zeros_like(v) for k, v in self.splats.items()}
        
        # Process level by level: start from level 1 and go up
        # Level 1: root nodes, use their own parameters directly
        # Level 2+: accumulate parent's final parameter + child residual
        for current_level in range(1, max_level + 1):
            # Get mask for all gaussians at current level
            curr_mask = (self.levels == current_level)
            
            if not curr_mask.any():
                continue
            
            curr_indices = torch.where(curr_mask)[0]
            
            # 현재 레벨 가우시안들의 부모 인덱스를 한 번에 추출
            parent_ids = self.parent_indices[curr_mask]
            
            # Separate Level 1 (no parent) and Level 2+ (has parent)
            is_level1 = (current_level == 1)
            has_parent_mask = (parent_ids != -1)
            
            if is_level1:
                # Level 1: root nodes - use their own parameters directly
                for name in final_params.keys():
                    if name in ["scales", "quats", "opacities"]:
                        # Independent parameters: use child's own values
                        final_params[name][curr_indices] = self.splats[name][curr_indices]
                    else:
                        # Residual parameters (means, sh0, shN): Level 1 uses own values (no parent)
                        final_params[name][curr_indices] = self.splats[name][curr_indices]
            else:
                # Level 2+: has parent - accumulate parent + child residual
                if not has_parent_mask.any():
                    continue
                
                # Apply valid mask to get valid current indices and parent indices
                valid_curr_indices = curr_indices[has_parent_mask]
                valid_parent_ids = parent_ids[has_parent_mask]
                
                # Get levels for valid current gaussians (for position scaling)
                valid_curr_levels = self.levels[valid_curr_indices]
                
                # Vectorized operation: add parent's final parameter to current residual
                for name in final_params.keys():
                    if name == "means":
                        # For means (positions), apply level-based scale reduction
                        # Higher level = smaller scale, keeping children closer to parents
                        scale_factor = self.position_scale_reduction ** (valid_curr_levels.float() - 1)
                        scale_factor = scale_factor.unsqueeze(-1)  # [N_valid, 1] for broadcasting
                        # Get parent's final parameter (already accumulated from lower levels)
                        parent_means = final_params[name][valid_parent_ids]
                        if detach_parents:
                            parent_means = parent_means.detach()
                        # Child mean = parent mean + (child residual * scale_factor)
                        child_residual = self.splats[name][valid_curr_indices]  # Get from original splats
                        final_params[name][valid_curr_indices] = (
                            parent_means + 
                            child_residual * scale_factor
                        )
                        # Free intermediate tensors
                        del parent_means, child_residual, scale_factor
                    elif name in ["scales", "quats", "opacities"]:
                        # Independent parameters: use child's own values directly
                        final_params[name][valid_curr_indices] = self.splats[name][valid_curr_indices]
                    else:
                        # For residual parameters (means, sh0, shN), add parent's final parameter to current residual
                        # Get parent's final parameter (already accumulated from lower levels)
                        parent_params = final_params[name][valid_parent_ids]
                        if detach_parents:
                            parent_params = parent_params.detach()
                        # Child residual from original splats
                        child_residual = self.splats[name][valid_curr_indices]
                        final_params[name][valid_curr_indices] = (
                            parent_params + child_residual
                        )
                        # Free intermediate tensors
                        del parent_params, child_residual
        
        # Now extract the final parameters for the requested indices
        # This indexing operation preserves gradients
        # IMPORTANT: Return Dict[str, Tensor] instead of ParameterDict to preserve gradient flow
        # If indices is all indices (level=None), return final_params directly to avoid unnecessary indexing
        if level is None and len(indices) == N:
            # Return final_params directly without indexing to save memory
            output_tensors = final_params
        else:
            # Extract only requested indices
            output_tensors = {name: final_params[name][indices] for name in final_params.keys()}
            # Free memory for final_params if we extracted a subset
            del final_params
            torch.cuda.empty_cache()
        
        return output_tensors
    
    def set_visible_mask(self, level: int) -> Tensor:
        """
        Set visible mask for gaussians based on hierarchical level.
        
        - Gaussians with level > specified level are invisible
        - Parent nodes from level down to 0 are invisible
        - Result: Only leaf nodes at or below the specified level are visible
        
        Args:
            level: Target level. All leaf nodes at or below this level will be visible,
                  while parent nodes (those with children) will be invisible.
        
        Returns:
            visible_mask: Boolean tensor [N,] indicating which gaussians are visible
        """
        device = self.levels.device
        N = len(self.levels)
        
        # Initialize all as visible
        visible_mask = torch.ones(N, dtype=torch.bool, device=device)
        
        # Step 1: Set gaussians with level > specified level to invisible
        visible_mask[self.levels > level] = False
        
        # Step 2: From level down to 0, set parent nodes to invisible
        for l in range(level, -1, -1):  # level, level-1, ..., 1, 0
            # Get all gaussians at current level
            level_mask = (self.levels == l)
            
            if not level_mask.any():
                continue
            
            # Get parent indices for all gaussians at current level (vectorized)
            parent_indices_at_level = self.parent_indices[level_mask]
            
            # Filter out invalid parents (parent_idx == -1)
            valid_parent_mask = (parent_indices_at_level != -1)
            if not valid_parent_mask.any():
                continue
            
            # Get valid parent indices (vectorized)
            valid_parent_indices = parent_indices_at_level[valid_parent_mask]
            
            # Set all valid parents to invisible (vectorized)
            visible_mask[valid_parent_indices] = False
        
        # Store the mask as instance attribute
        self.visible_mask = visible_mask
        
        return visible_mask
    
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        level: int = -1,
        masks: Optional[Tensor] = None,
        sh_degree: int = 3,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = "pinhole",
        packed: bool = False,
        sparse_grad: bool = False,
        absgrad: bool = False,
        distributed: bool = False,
        with_ut: bool = False,
        with_eval3d: bool = False,
        backgrounds: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Rasterize splats with level-of-detail (LOD) support.
        
        Args:
            camtoworlds: Camera-to-world matrices [C, 4, 4]
            Ks: Camera intrinsics [C, 3, 3]
            width: Image width
            height: Image height
            level: Level of detail. If -1, use max level (highest LOD).
                   Otherwise, render only visible gaussians at or below this level.
            masks: Optional mask [C, H, W] for masking rendered output
            sh_degree: Spherical harmonics degree
            near_plane: Near clipping plane
            far_plane: Far clipping plane
            rasterize_mode: Rasterization mode ("classic" or "antialiased")
            camera_model: Camera model type
            packed: Use packed mode for rasterization
            sparse_grad: Use sparse gradients
            absgrad: Use absolute gradients
            distributed: Whether using distributed training
            with_ut: Enable uncentered transform
            with_eval3d: Enable 3D evaluation
            **kwargs: Additional arguments passed to rasterization
        
        Returns:
            render_colors: Rendered colors [C, H, W, 3]
            render_alphas: Rendered alphas [C, H, W, 1]
            info: Dictionary with rendering information
        """
        device = self.splats["means"].device
        
        # Determine the level to use
        if level == -1:
            # Use max level for highest LOD
            if len(self.levels) > 0:
                render_level = int(self.levels.max().item())
            else:
                render_level = 1
        else:
            render_level = level
        
        # Fallback logic: If target level has no gaussians, use the highest available level
        # This ensures training continues even when some levels haven't been created yet
        if len(self.levels) > 0:
            max_available_level = int(self.levels.max().item())
            # If target level is higher than available, use the highest available
            if render_level > max_available_level:
                render_level = max_available_level
        
        # Set visible mask based on level
        visible_mask = self.set_visible_mask(render_level)
        
        # Filter by visible mask
        visible_indices = torch.where(visible_mask)[0]

        # print(len(visible_mask), len(visible_indices))
        
        # Fallback: If no gaussians at target level, try lower levels
        if len(visible_indices) == 0:
            # Try progressively lower levels until we find gaussians
            for fallback_level in range(render_level - 1, 0, -1):
                fallback_visible_mask = self.set_visible_mask(fallback_level)
                fallback_visible_indices = torch.where(fallback_visible_mask)[0]
                if len(fallback_visible_indices) > 0:
                    visible_mask = fallback_visible_mask
                    visible_indices = fallback_visible_indices
                    render_level = fallback_level
                    break
        
        # If still no gaussians, return empty render (should not happen in normal training)
        if len(visible_indices) == 0:
            C = camtoworlds.shape[0]
            render_colors = torch.zeros((C, height, width, 3), device=device)
            render_alphas = torch.zeros((C, height, width, 1), device=device)
            info = {
                "gaussian_ids": torch.empty((0,), dtype=torch.long, device=device),
                "radii": torch.empty((C, 0, 2), device=device),
                "means2d": torch.empty((C, 0, 2), device=device),  # Required for step_pre_backward
                "width": width,
                "height": height,
                "n_cameras": C,
                "visible_mask": torch.zeros(len(self.levels), dtype=torch.bool, device=device),
                "render_level": render_level,  # Current target render level
            }
            return render_colors, render_alphas, info
        
        # Get hierarchical splats for visible indices only
        # This ensures gradient flow is preserved by working directly with self.splats
        # When rendering at a specific level, we want to train leaf nodes independently,
        # so we detach parent parameters from the computation graph
        # This prevents parent parameters from receiving gradients when leaf nodes are trained
        visible_splats = self.get_splats(level=None, detach_parents=True)  # Get all splats with hierarchy applied, detach parents
        
        # Extract visible splats - this indexing preserves gradients
        visible_means = visible_splats["means"][visible_indices]  # [N_visible, 3]
        visible_quats = visible_splats["quats"][visible_indices]  # [N_visible, 4]
        visible_scales = visible_splats["scales"][visible_indices]  # [N_visible, 3]
        visible_opacities = visible_splats["opacities"][visible_indices]  # [N_visible,]
        
        # Extract colors before freeing visible_splats
        image_ids = kwargs.pop("image_ids", None)
        try:
            if self.cfg.app_opt:
                visible_features = visible_splats["features"][visible_indices]
                visible_colors = visible_splats["colors"][visible_indices]
            else:
                visible_sh0 = visible_splats["sh0"][visible_indices]
                visible_shN = visible_splats["shN"][visible_indices]
        except:
            visible_sh0 = visible_splats["sh0"][visible_indices]
            visible_shN = visible_splats["shN"][visible_indices]
        
        # Free visible_splats immediately after extracting all needed values
        del visible_splats
        
        # Prepare parameters for rasterization
        means = visible_means  # [N_visible, 3]
        quats = visible_quats  # [N_visible, 4] (rasterization normalizes internally)
        scales = torch.exp(visible_scales)  # [N_visible, 3]
        opacities = torch.sigmoid(visible_opacities)  # [N_visible,]
        
        # Free intermediate tensors
        del visible_means, visible_quats, visible_scales, visible_opacities

        # Handle colors (SH coefficients or features)
        try:
            if self.cfg.app_opt:
                colors = self.app_module(
                    features=visible_features,
                    embed_ids=image_ids,
                    dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                    sh_degree=sh_degree,
                )
                colors = colors + visible_colors
                colors = torch.sigmoid(colors)
                del visible_features, visible_colors
            else:   
                colors = torch.cat([visible_sh0, visible_shN], 1)  # [N_visible, K, 3]
                del visible_sh0, visible_shN
        except: 
            colors = torch.cat([visible_sh0, visible_shN], 1)  # [N_visible, K, 3]
            del visible_sh0, visible_shN
        # Set default rasterize_mode if not provided
        if rasterize_mode is None:
            rasterize_mode = "antialiased" if hasattr(self.cfg, "antialiased") and self.cfg.antialiased else "classic"
        
        # Debug: Check gradient flow before rasterization
        # Uncomment these lines to debug gradient issues
        # print(f"DEBUG: means requires_grad = {means.requires_grad}")
        # print(f"DEBUG: means grad_fn = {means.grad_fn}")
        # print(f"DEBUG: quats requires_grad = {quats.requires_grad}")
        # print(f"DEBUG: quats grad_fn = {quats.grad_fn}")
        # print(f"DEBUG: scales requires_grad = {scales.requires_grad}")
        # print(f"DEBUG: scales grad_fn = {scales.requires_grad}")
        
        # Filter out kwargs that are not supported by rasterization
        # image_ids is used for appearance optimization but not passed to rasterization
        rasterization_kwargs = {k: v for k, v in kwargs.items() if k not in ["image_ids", "render_mode"]}
        
        # Call rasterization
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            sh_degree=sh_degree,  # Must be passed explicitly for SH coefficients
            packed=packed,
            absgrad=absgrad,
            sparse_grad=sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=distributed,
            camera_model=camera_model,
            with_ut=with_ut,
            with_eval3d=with_eval3d,
            # backgrounds=backgrounds,  # Pass backgrounds to rasterization function
            **rasterization_kwargs,
        )
        
        # Add visible_mask and render_level to info for strategy to use
        # visible_mask is [N_total], but we only rendered visible_indices
        # We need to map the rendered gaussians back to their original indices
        N_total = len(self.splats["means"])
        full_visible_mask = torch.zeros(N_total, dtype=torch.bool, device=device)
        full_visible_mask[visible_indices] = True
        info["visible_mask"] = full_visible_mask  # [N_total] mask for all gaussians
        info["render_level"] = render_level  # Current target render level
        
        # Apply masks if provided
        if masks is not None:
            render_colors[~masks] = 0
        
        return render_colors, render_alphas, info

