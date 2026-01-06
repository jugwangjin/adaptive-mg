"""
Graph-based hierarchical Gaussian Splatting structure for multigrid methods.

This module provides a hierarchical structure where each Gaussian has level and parent_index
attributes to represent the hierarchical graph structure.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gsplat.rendering import rasterization
from utils import knn, rgb_to_sh


class MultigridGaussians:
    """
    Multigrid Gaussian Splatting structure with hierarchical levels.
    
    Each Gaussian has level and parent_index attributes to represent the hierarchical graph structure.
    """
    
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
        # Initialize points and colors (same as create_splats_with_optimizers)
        if init_type == "sfm":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif init_type == "random":
            points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
        else:
            raise ValueError("Please specify a correct init_type: sfm or random")

        self.cfg = cfg
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        # Distribute the GSs to different ranks (also works for single rank)
        points = points[world_rank::world_size]
        rgbs = rgbs[world_rank::world_size]
        scales = scales[world_rank::world_size]

        N = points.shape[0]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

        # Initialize level and parent_index as regular variables (not parameters)
        # All initial gaussians are at level 1
        # parent_index = -1 means no parent (root nodes at level 1)
        levels = torch.ones(N, dtype=torch.long, device=device)  # [N,] all level 1
        parent_indices = torch.full((N,), -1, dtype=torch.long, device=device)  # [N,] no parent

        params = [
            # name, value, lr
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
        self.level_indices[1] = list(range(N))  # All initial gaussians are at level 1
        
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
          * scales, quats, opacities: use child's own parameters directly (independent, not residual)
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
        final_params = {k: v.clone() for k, v in self.splats.items()}
        
        # Process level by level using vectorized tensor operations
        for current_level in range(2, max_level + 1):
            # Get mask for all gaussians at current level
            curr_mask = (self.levels == current_level)
            
            if not curr_mask.any():
                continue
            
            # 현재 레벨 가우시안들의 부모 인덱스를 한 번에 추출
            parent_ids = self.parent_indices[curr_mask]
            
            # Filter out invalid parents (parent_idx == -1)
            valid_mask = (parent_ids != -1)
            if not valid_mask.any():
                continue
            
            # Apply valid mask to get valid current indices and parent indices
            valid_curr_indices = torch.where(curr_mask)[0][valid_mask]
            valid_parent_ids = parent_ids[valid_mask]
            
            # Get levels for valid current gaussians (for position scaling)
            valid_curr_levels = self.levels[valid_curr_indices]
            
            # Vectorized operation: add parent's final parameter to current residual
            # In-place operations on tensors created with v + 0 preserve gradients
            for name in final_params.keys():
                if name == "means":
                    # For means (positions), apply level-based scale reduction
                    # Higher level = smaller scale, keeping children closer to parents
                    scale_factor = self.position_scale_reduction ** (valid_curr_levels.float() - 1)
                    scale_factor = scale_factor.unsqueeze(-1)  # [N_valid, 1] for broadcasting
                    # Get parent parameters (detach if requested)
                    parent_means = final_params[name][valid_parent_ids]
                    if detach_parents:
                        parent_means = parent_means.detach()
                    # Child mean = parent mean + (child residual * scale_factor)
                    # Scale factor is applied to child's residual, not parent's mean
                    # This keeps children closer to parents at higher levels
                    child_residual = final_params[name][valid_curr_indices]
                    final_params[name][valid_curr_indices] = (
                        parent_means + 
                        child_residual * scale_factor
                    )
                elif name in ["scales", "quats", "opacities"]:
                    # For scales, quats, and opacities: use child's own parameters directly (independent, not residual)
                    # These parameters are independent and not accumulated from parents
                    # Child's parameters are already in final_params, so no modification needed
                    pass
                else:
                    # For other parameters (sh0, shN, etc.), add parent's final parameter to current residual
                    # Get parent parameters (detach if requested)
                    parent_params = final_params[name][valid_parent_ids]
                    if detach_parents:
                        parent_params = parent_params.detach()
                    # 부모의 '이미 계산된' 절대값을 현재 자식의 잔차에 더함 (Vectorized)
                    final_params[name][valid_curr_indices] = (
                        final_params[name][valid_curr_indices] + parent_params
                    )
        
        # Now extract the final parameters for the requested indices
        # This indexing operation preserves gradients
        # IMPORTANT: Return Dict[str, Tensor] instead of ParameterDict to preserve gradient flow
        output_tensors = {name: final_params[name][indices] for name in final_params.keys()}
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
        
        # Prepare parameters for rasterization
        means = visible_means  # [N_visible, 3]
        quats = visible_quats  # [N_visible, 4] (rasterization normalizes internally)
        scales = torch.exp(visible_scales)  # [N_visible, 3]
        opacities = torch.sigmoid(visible_opacities)  # [N_visible,]
        

        # if self.cfg.app_opt:
        #     colors = self.app_module(
        #         features=self.splats["features"],
        #         embed_ids=image_ids,
        #         dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
        #         sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
        #     )
        #     colors = colors + self.splats["colors"]
        #     colors = torch.sigmoid(colors)
        # else:
        #     colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        # Handle colors (SH coefficients or features)
        image_ids = kwargs.pop("image_ids", None)
        try:
            if self.cfg.app_opt:
                colors = self.app_module(
                    features=visible_splats["features"][visible_indices],
                    embed_ids=image_ids,
                    dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                    sh_degree=sh_degree,
                )
                colors = colors + visible_splats["colors"][visible_indices]
                colors = torch.sigmoid(colors)
            else:   
                colors = torch.cat([visible_splats["sh0"][visible_indices], visible_splats["shN"][visible_indices]], 1)  # [N_visible, K, 3]
        except: 
            colors = torch.cat([visible_splats["sh0"][visible_indices], visible_splats["shN"][visible_indices]], 1)  # [N_visible, K, 3]
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
        
        # Add visible_mask to info for strategy to use
        # visible_mask is [N_total], but we only rendered visible_indices
        # We need to map the rendered gaussians back to their original indices
        N_total = len(self.splats["means"])
        full_visible_mask = torch.zeros(N_total, dtype=torch.bool, device=device)
        full_visible_mask[visible_indices] = True
        info["visible_mask"] = full_visible_mask  # [N_total] mask for all gaussians
        
        # Apply masks if provided
        if masks is not None:
            render_colors[~masks] = 0
        
        return render_colors, render_alphas, info

