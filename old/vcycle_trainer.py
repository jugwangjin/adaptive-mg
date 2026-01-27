import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset as ColmapDataset, Parser as ColmapParser
from datasets.nerf import Dataset as NerfDataset, Parser as NerfParser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.multigrid_v4 import MultigridStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
from multigrid_gaussians import MultigridGaussians

opacity_grad_norms = []

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Dataset type: "colmap" or "nerf"
    dataset_type: Literal["colmap", "nerf"] = "colmap"
    # Path to the dataset directory
    data_dir: Optional[str] = None
    # Directory to save results (if None, auto-generated from data_dir)
    result_dir: Optional[str] = None
    # Downsample factor for the dataset
    data_factor: int = 4
    # For NeRF datasets: use white background for RGBA images
    white_background: bool = False
    # For NeRF datasets: image file extension
    image_extension: str = ".png"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [30_000])
    # Interval for metric measurement (every N steps). -1 means disable metric measurement.
    metric_interval: int = -1
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy, MultigridStrategy] = field(
        default_factory=lambda: MultigridStrategy(verbose=True)
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    
    # Multigrid growth control parameters (passed to MultigridStrategy)
    max_children_per_parent: int = 50  # Maximum number of children a parent can have
    grad_threshold_decay_rate: float = 0.9  # Decay rate for gradient threshold per level
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization (applied only to visible gaussians)
    # Before refine_start_iter, set to 0.0 to match simple_trainer
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # Position scale reduction for hierarchical gaussians
    # Higher level gaussians are constrained to stay closer to their parents
    position_scale_reduction: float = 0.75
    # Maximum level for hierarchical structure
    # If set, gaussians at max_level will only duplicate (not split) even if split conditions are met
    max_level: Optional[int] = 4
    # Ratio of Level 1 points to Level 2 points during hierarchical initialization
    # Level 1 = init_level1_ratio * Level 2 (default: 0.25, i.e., 1/4)
    init_level1_ratio: float = 0.05
    # Parent sampling method: "uniform" or "fps" (farthest point sampling)
    parent_sampling_method: Literal["uniform", "fps"] = "fps"
    
    # Cycle type: "vcycle" or "inv_fcycle"
    cycle_type: Literal["vcycle", "inv_fcycle"] = "inv_fcycle"
    
    # V-cycle parameters
    smoothing_steps: int = 16  # Number of smoothing steps per level (fixed at 100 for V-cycle)
    solving_steps: int = 16  # Number of solving steps at coarsest level (fixed at 100 for V-cycle)
    steps_decaying_per_level: float = 0.5
    
    # Strategy parameters (set based on cycle_type in __post_init__)
    # For V-cycle: reset_every=60, refine_every=2, pause_refine_after_reset=2
    # For inv-F cycle: reset_every=30, refine_every=1, pause_refine_after_reset=1
    reset_every: Optional[int] = None  # Reset opacities every N cycles (auto-set based on cycle_type)
    refine_every: Optional[int] = None  # Refine GSs every N cycles (auto-set based on cycle_type)
    pause_refine_after_reset: Optional[int] = None  # Pause refining for N cycles after reset (auto-set based on cycle_type)
    
    # Gradient scaling parameters
    grad_scale_factor: float = 1.0  # Base factor for level-dependent gradient scaling
    # Gradient scale for level L: grad_scale_factor ** (max_level - L)
    # Coarse levels (low L) get larger scale, fine levels (high L) get smaller scale

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    # Visualization: interval to save visualization images (GT vs render)
    visualization_interval: int = 100
    # Visualization: interval to save hierarchy visualization (level pointclouds and linesets)
    hierarchy_visualization_interval: int = 3000

    def __post_init__(self):
        """Auto-generate result_dir if not provided and set strategy parameters based on cycle_type."""
        from pathlib import Path
        
        if self.data_dir is None:
            if self.dataset_type == "colmap":
                self.data_dir = "/Bean/data/gwangjin/2025/3dgs/garden"
            elif self.dataset_type == "nerf":
                self.data_dir = "/Bean/data/gwangjin/2025/3dgs/lego"
            else:
                raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
        
        if self.result_dir is None:
            # Extract dataset name from data_dir (last directory component)
            dataset_name = Path(self.data_dir).name
            
            # Build settings string
            settings_parts = [
                f"type_{self.dataset_type}",
                f"factor_{self.data_factor}",
                "v1",
            ]
            if self.dataset_type == "nerf" and self.white_background:
                settings_parts.append("whitebg")
            if self.normalize_world_space:
                settings_parts.append("norm")
            if self.patch_size is not None:
                settings_parts.append(f"patch_{self.patch_size}")
            
            settings_str = "_".join(settings_parts)
            self.result_dir = f"/Bean/log/gwangjin/2025/gsplat/multigrid_{self.cycle_type}/{dataset_name}_{settings_str}"
        
        # Set strategy parameters based on cycle_type if not explicitly provided
        if self.reset_every is None:
            if self.cycle_type == "vcycle":
                self.reset_every = 60
            elif self.cycle_type == "inv_fcycle":
                self.reset_every = 30
            else:
                raise ValueError(f"Unknown cycle_type: {self.cycle_type}")
        
        if self.refine_every is None:
            if self.cycle_type == "vcycle":
                self.refine_every = 2
            elif self.cycle_type == "inv_fcycle":
                self.refine_every = 1
            else:
                raise ValueError(f"Unknown cycle_type: {self.cycle_type}")
        
        if self.pause_refine_after_reset is None:
            if self.cycle_type == "vcycle":
                self.pause_refine_after_reset = 2
            elif self.cycle_type == "inv_fcycle":
                self.pause_refine_after_reset = 1
            else:
                raise ValueError(f"Unknown cycle_type: {self.cycle_type}")

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, MultigridStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def vcycle_recursive(
    current_step: int,
    level: int,
    coarsest_level: int,
    max_level: int,
    smoothing_steps: int,
    solving_steps: int,
    total_steps: int,
    runner: "Runner",
    trainloader_iter,
    schedulers: List,
    cfg: Config,
    pbar: Optional[tqdm.tqdm] = None,
    vcycle_idx: int = 0,
    losses: Optional[List[float]] = None,
) -> Tuple[int, List[float]]:
    """
    Recursive V-cycle: max_level -> coarsest_level -> max_level.
    
    V-cycle structure (recursive):
    - Base case: level == coarsest_level -> perform solving
    - Recursive case:
      1. Downward: Smooth at current level -> Restrict -> Recurse to level-1
      2. Upward: Return from recursion -> Smooth at current level
    
    Args:
        current_step: Current training step (will be updated)
        level: Current level in the V-cycle
        coarsest_level: Coarsest LOD level for this V-cycle
        max_level: Maximum LOD level (finest)
        smoothing_steps: Number of smoothing steps per level
        solving_steps: Number of solving steps at coarsest level
        total_steps: Maximum number of steps
        runner: Runner instance for training
        trainloader_iter: Training data iterator
        schedulers: List of learning rate schedulers
        cfg: Config object
        pbar: Optional progress bar
        vcycle_idx: V-cycle index for display
        losses: List of losses (accumulated across recursion)
    
    Returns:
        current_step: Updated current step
        losses: List of losses during this V-cycle
    """
    if losses is None:
        losses = []
    
    # Base case: coarsest level - perform solving
    if level == coarsest_level:
        # ========== Coarsest Level Solving ==========
        for solving_step in range(solving_steps):
            # Continue V-cycle even if max_steps is reached (complete the cycle)
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(runner.trainloader)
                data = next(trainloader_iter)
            
            log_step = min(current_step, total_steps - 1) if total_steps > 0 else current_step
            
            loss = runner._train_step(
                step=log_step,
                data=data,
                target_level=level,
            )
            losses.append(loss)
            
            # Only increment step and scheduler if not past max_steps
            if current_step < total_steps:
                for scheduler in schedulers:
                    scheduler.step()
                current_step += 1
            
            if pbar is not None:
                pbar.update(1)
                avg_loss = np.mean(losses) if losses else 0.0
                desc = f"V-cycle {vcycle_idx}| Solve L{level}| loss={avg_loss:.3f}| step={current_step}/{total_steps}"
                pbar.set_description(desc)
        
        return current_step, losses
    
    # Recursive case: Downward pass (smoothing + restriction) -> recurse -> upward pass (smoothing)
    
    # ========== Downward: Smoothing at current level ==========
    target_level = level
    
    # Save initial values before smoothing
    runner.multigrid_gaussians.set_visible_mask(target_level)
    visible_mask = runner.multigrid_gaussians.visible_mask  # [N,]
    parent_indices = runner.multigrid_gaussians.parent_indices  # [N,]
    has_parent = (parent_indices != -1)  # [N,]
    
    # Get valid indices: visible gaussians with valid parents
    valid_mask = visible_mask & has_parent  # [N,]
    valid_indices = torch.where(valid_mask)[0]  # [K,]
    
    if len(valid_indices) == 0:
        # No valid gaussians, skip restriction and recurse to coarser level
        # Free intermediate variables before recursive call
        del visible_mask, parent_indices, has_parent, valid_mask, valid_indices
        torch.cuda.empty_cache()
        
        current_step, losses = vcycle_recursive(
            current_step=current_step,
            level=level - 1,
            coarsest_level=coarsest_level,
            max_level=max_level,
            smoothing_steps=smoothing_steps,
            solving_steps=solving_steps,
            total_steps=total_steps,
            runner=runner,
            trainloader_iter=trainloader_iter,
            schedulers=schedulers,
            cfg=cfg,
            pbar=pbar,
            vcycle_idx=vcycle_idx,
            losses=losses,
        )
        # Continue with upward pass (will be handled after restriction block)
        return current_step, losses
    
    # Get initial hierarchical values for valid gaussians
    # No gradients needed for restriction (only parameter updates)
    with torch.no_grad():
        initial_splats = runner.multigrid_gaussians.get_splats(level=None, detach_parents=True)
        initial_means = initial_splats["means"][valid_indices].clone().detach()  # [K, 3]
        initial_sh0 = initial_splats["sh0"][valid_indices].clone().detach()  # [K, 1, 3]
        initial_shN = initial_splats["shN"][valid_indices].clone().detach()  # [K, K, 3]
        # Free initial_splats immediately after extracting needed values
        del initial_splats
    
    # Smoothing steps at current level
    for smoothing_step in range(smoothing_steps):
        # Continue V-cycle even if max_steps is reached (complete the cycle)
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(runner.trainloader)
            data = next(trainloader_iter)
        
        log_step = min(current_step, total_steps - 1) if total_steps > 0 else current_step
        
        loss = runner._train_step(
            step=log_step,
            data=data,
            target_level=target_level,
        )
        losses.append(loss)
        
        # Only increment step and scheduler if not past max_steps
        if current_step < total_steps:
            for scheduler in schedulers:
                scheduler.step()
            current_step += 1
        
        if pbar is not None:
            pbar.update(1)
            avg_loss = np.mean(losses) if losses else 0.0
            desc = f"V-cycle {vcycle_idx}| Down L{target_level}| loss={avg_loss:.3f}| step={current_step}/{total_steps}"
            pbar.set_description(desc)
    
    # ========== Restriction: Transfer fine level changes to coarse level ==========
    # This operation updates parameter values only, no gradients needed
    # Use no_grad to prevent computation graph accumulation in recursive calls
    with torch.no_grad():
        # Get final hierarchical values after smoothing
        final_splats = runner.multigrid_gaussians.get_splats(level=None, detach_parents=True)
        final_means = final_splats["means"][valid_indices]  # [K, 3]
        final_sh0 = final_splats["sh0"][valid_indices]  # [K, 1, 3]
        final_shN = final_splats["shN"][valid_indices]  # [K, K, 3]
        # Free final_splats immediately after extracting needed values
        del final_splats
        
        # Calculate changes (delta) for residual parameters only
        # Residual parameters (means, sh0, shN) have prolongation (parent + child residual),
        # so restriction (fine->coarse) is also defined for them.
        # Independent parameters (quats, scales, opacities) have no prolongation,
        # so restriction is not applicable.
        delta_means = final_means - initial_means  # [K, 3]
        delta_sh0 = final_sh0 - initial_sh0  # [K, 1, 3]
        delta_shN = final_shN - initial_shN  # [K, K, 3]
        
        # Get parent indices for valid gaussians
        valid_parent_indices = parent_indices[valid_indices]  # [K,]
        
        N = len(runner.multigrid_gaussians.levels)
        
        # Count number of children per parent (for averaging) - single count for all parameters
        child_count = torch.zeros(N, dtype=torch.float, device=delta_means.device)
        child_count.scatter_add_(0, valid_parent_indices, torch.ones_like(valid_parent_indices, dtype=torch.float))
        
        # Accumulate changes to parents using scatter_add
        parent_delta_means = torch.zeros(N, 3, device=delta_means.device)
        parent_delta_means.scatter_add_(0, valid_parent_indices.unsqueeze(-1).expand(-1, 3), delta_means)
        
        parent_delta_sh0 = torch.zeros_like(runner.splats["sh0"])
        parent_delta_sh0.scatter_add_(0, valid_parent_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3), delta_sh0)
        
        shN_shape = runner.splats["shN"].shape
        parent_delta_shN = torch.zeros_like(runner.splats["shN"])
        parent_delta_shN.scatter_add_(0, valid_parent_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, shN_shape[1], 3), delta_shN)
        
        # Average changes: divide by child count (avoid division by zero)
        child_count = child_count.clamp_min(1.0)  # [N,]
        
        parent_delta_means = parent_delta_means / child_count.unsqueeze(-1)  # [N, 3]
        parent_delta_sh0 = parent_delta_sh0 / child_count.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 3]
        parent_delta_shN = parent_delta_shN / child_count.unsqueeze(-1).unsqueeze(-1)  # [N, K, 3]
        
        # Add averaged changes to parent parameters (residual parameters only)
        # This implements restriction: fine->coarse transfer for parameters with prolongation
        runner.splats["means"].data[valid_parent_indices] += parent_delta_means[valid_parent_indices]
        runner.splats["sh0"].data[valid_parent_indices] += parent_delta_sh0[valid_parent_indices]
        runner.splats["shN"].data[valid_parent_indices] += parent_delta_shN[valid_parent_indices]
        
        # Subtract changes from fine level parameters (to maintain optimized hierarchical values)
        # This ensures that fine level still renders with optimized values after restriction
        runner.splats["means"].data[valid_indices] -= delta_means
        runner.splats["sh0"].data[valid_indices] -= delta_sh0
        runner.splats["shN"].data[valid_indices] -= delta_shN
        
        # Free all intermediate variables after restriction
        del final_means, final_sh0, final_shN
        del delta_means, delta_sh0, delta_shN
        del parent_delta_means, parent_delta_sh0, parent_delta_shN
        del child_count
        # Free initial values after restriction is complete
        del initial_means, initial_sh0, initial_shN
        # Free restriction-related variables before recursive call
        del valid_parent_indices, valid_mask
        torch.cuda.empty_cache()
    
    # ========== Recurse to coarser level ==========
    # Note: visible_mask, parent_indices, has_parent, valid_indices are still needed for upward pass
    # but we can free them after recursive call if not needed
    current_step, losses = vcycle_recursive(
        current_step=current_step,
        level=level - 1,
        coarsest_level=coarsest_level,
        max_level=max_level,
        smoothing_steps=smoothing_steps,
        solving_steps=solving_steps,
        total_steps=total_steps,
        runner=runner,
        trainloader_iter=trainloader_iter,
        schedulers=schedulers,
        cfg=cfg,
        pbar=pbar,
        vcycle_idx=vcycle_idx,
        losses=losses,
    )
    
    # Free variables that are no longer needed after recursive call
    # (visible_mask, parent_indices, has_parent, valid_indices are not used in upward pass)
    del visible_mask, parent_indices, has_parent, valid_indices
    
    # ========== Upward: Smoothing at current level ==========
    # Smoothing steps at current level (after returning from coarser level)
    for smoothing_step in range(smoothing_steps):
        # Continue V-cycle even if max_steps is reached (complete the cycle)
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(runner.trainloader)
            data = next(trainloader_iter)
        
        log_step = min(current_step, total_steps - 1) if total_steps > 0 else current_step
        
        loss = runner._train_step(
            step=log_step,
            data=data,
            target_level=target_level,
        )
        losses.append(loss)
        
        # Only increment step and scheduler if not past max_steps
        if current_step < total_steps:
            for scheduler in schedulers:
                scheduler.step()
            current_step += 1
        
        if pbar is not None:
            pbar.update(1)
            avg_loss = np.mean(losses) if losses else 0.0
            desc = f"V-cycle {vcycle_idx}| Up L{target_level}| loss={avg_loss:.3f}| step={current_step}/{total_steps}"
            pbar.set_description(desc)
    
    return current_step, losses


def perform_vcycle(
    current_step: int,
    max_level: int,
    coarsest_level: int,
    smoothing_steps: int,
    solving_steps: int,
    total_steps: int,
    runner: "Runner",
    trainloader_iter,
    schedulers: List,
    cfg: Config,
    pbar: Optional[tqdm.tqdm] = None,
    vcycle_idx: int = 0,
) -> Tuple[int, List[float]]:
    """
    Perform a single V-cycle: max_level -> coarsest_level -> max_level.
    
    Wrapper function that calls the recursive V-cycle implementation.
    
    Args:
        current_step: Current training step (will be updated)
        max_level: Maximum LOD level (finest)
        coarsest_level: Coarsest LOD level for this V-cycle
        smoothing_steps: Number of smoothing steps per level
        solving_steps: Number of solving steps at coarsest level
        total_steps: Maximum number of steps
        runner: Runner instance for training
        trainloader_iter: Training data iterator
        schedulers: List of learning rate schedulers
        cfg: Config object
        pbar: Optional progress bar
        vcycle_idx: V-cycle index for display
    
    Returns:
        current_step: Updated current step
        losses: List of losses during this V-cycle
    """
    # Ensure coarsest_level is valid
    coarsest_level = max(1, min(coarsest_level, max_level))  # Clamp to [1, max_level]
    
    # Call recursive V-cycle starting from max_level
    return vcycle_recursive(
        current_step=current_step,
        level=max_level,
        coarsest_level=coarsest_level,
        max_level=max_level,
        smoothing_steps=smoothing_steps,
        solving_steps=solving_steps,
        total_steps=total_steps,
        runner=runner,
        trainloader_iter=trainloader_iter,
        schedulers=schedulers,
        cfg=cfg,
        pbar=pbar,
        vcycle_idx=vcycle_idx,
        losses=[],
    )


def perform_inv_fcycle(
    current_step: int,
    max_level: int,
    smoothing_steps: int,
    solving_steps: int,
    total_steps: int,
    runner: "Runner",
    trainloader_iter,
    schedulers: List,
    cfg: Config,
    pbar: Optional[tqdm.tqdm] = None,
    cycle_idx: int = 0,
) -> Tuple[int, List[float]]:
    """
    Perform an inv-F cycle: finest~coarsest -> finest~(coarsest+1) -> ... -> finest~finest.
    
    This performs multiple V-cycles with increasing coarsest_level:
    - Cycle 1: finest -> 1 -> finest
    - Cycle 2: finest -> 2 -> finest
    - ...
    - Cycle N: finest -> finest -> finest (single level)
    
    Args:
        current_step: Current training step (will be updated)
        max_level: Maximum LOD level (finest)
        smoothing_steps: Number of smoothing steps per level
        solving_steps: Number of solving steps at coarsest level
        total_steps: Maximum number of steps
        runner: Runner instance for training
        trainloader_iter: Training data iterator
        schedulers: List of learning rate schedulers
        cfg: Config object
        pbar: Optional progress bar
        cycle_idx: Cycle index for display
    
    Returns:
        current_step: Updated current step
        losses: List of losses during this inv-F cycle
    """
    all_losses = []
    
    # Perform V-cycles with increasing coarsest_level: 1, 2, ..., max_level
    for coarsest_level in range(1, max_level + 1):
        # Perform a single V-cycle with this coarsest_level
        current_step, cycle_losses = perform_vcycle(
            current_step=current_step,
            max_level=max_level,
            coarsest_level=coarsest_level,
            smoothing_steps=smoothing_steps,
            solving_steps=solving_steps,
            total_steps=total_steps,
            runner=runner,
            trainloader_iter=trainloader_iter,
            schedulers=schedulers,
            cfg=cfg,
            pbar=pbar,
            vcycle_idx=cycle_idx,  # Use cycle_idx for display
        )
        all_losses.extend(cycle_losses)
        
        # Stop if we've exceeded max_steps
        if current_step >= total_steps:
            break
    
    return current_step, all_losses


class Runner:
    """Engine for training and testing with hierarchical Gaussians using V-cycle multigrid."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        if cfg.dataset_type == "colmap":
            self.parser = ColmapParser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
            )
            self.trainset = ColmapDataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = ColmapDataset(self.parser, split="val")
        elif cfg.dataset_type == "nerf":
            self.parser = NerfParser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                white_background=cfg.white_background,
                extension=cfg.image_extension,
            )
            self.trainset = NerfDataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = NerfDataset(self.parser, split="val")
        else:
            raise ValueError(f"Unknown dataset type: {cfg.dataset_type}")
        
        # Sample 3 cameras from test set for visualization
        if cfg.visualization_interval > 0:
            val_indices = list(range(len(self.valset)))
            if len(val_indices) >= 3:
                import random
                random.seed(42)  # For reproducibility
                self.viz_camera_indices = random.sample(val_indices, 3)
            else:
                self.viz_camera_indices = val_indices[:3] if len(val_indices) > 0 else []
            print(f"Selected {len(self.viz_camera_indices)} cameras for visualization: {self.viz_camera_indices}")
        else:
            self.viz_camera_indices = []
        
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model: Use MultigridGaussians
        feature_dim = 32 if cfg.app_opt else None
        self.multigrid_gaussians = MultigridGaussians(
            parser=self.parser,
            cfg=cfg,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            position_scale_reduction=cfg.position_scale_reduction,
            max_level=cfg.max_level,
        )
        
        # For compatibility with densification strategy, we need to expose splats and optimizers
        self.splats = self.multigrid_gaussians.splats
        self.optimizers = self.multigrid_gaussians.optimizers
        
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        # If strategy is MultigridStrategy, update growth control parameters from config
        if isinstance(self.cfg.strategy, MultigridStrategy):
            self.cfg.strategy.max_children_per_parent = self.cfg.max_children_per_parent
            self.cfg.strategy.grad_threshold_decay_rate = self.cfg.grad_threshold_decay_rate
            self.cfg.strategy.max_level = self.cfg.max_level if self.cfg.max_level is not None else self.cfg.strategy.max_level
            # Set strategy parameters from config (set in __post_init__ based on cycle_type)
            self.cfg.strategy.reset_every = self.cfg.reset_every
            self.cfg.strategy.refine_every = self.cfg.refine_every
            self.cfg.strategy.pause_refine_after_reset = self.cfg.pause_refine_after_reset
            # If opacity_reg > 0, disable opacity reset (opacity regularization handles opacity control)
            if self.cfg.opacity_reg > 0.0:
                self.cfg.strategy.use_opacity_reset = False
        
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, MultigridStrategy):
            # MultigridStrategy requires hierarchical structure
            # Always use cfg.max_level for strategy, not multigrid_gaussians.max_level
            strategy_max_level = self.cfg.max_level if self.cfg.max_level is not None else self.cfg.strategy.max_level
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale,
                levels=self.multigrid_gaussians.levels,
                parent_indices=self.multigrid_gaussians.parent_indices,
                level_indices=self.multigrid_gaussians.level_indices,
                max_level=strategy_max_level,
                multigrid_gaussians=self.multigrid_gaussians,
            )
        elif isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Debug: Save level1, level2 point clouds and exit
        # from debug_codes import save_initialization_point_clouds
        # save_initialization_point_clouds(self, world_rank)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

            self.multigrid_gaussians.app_module = self.app_module

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        level: int = -1,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Rasterize splats using MultigridGaussians.
        
        Args:
            level: LOD level to render at. -1 means max LOD (default).
        """
        
        # Set rasterize_mode if not provided
        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        
        # Get sh_degree from kwargs or use default
        sh_degree = kwargs.pop("sh_degree", self.cfg.sh_degree)
        
        # Call MultigridGaussians.rasterize_splats
        render_colors, render_alphas, info = self.multigrid_gaussians.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            level=level,
            masks=masks,
            sh_degree=sh_degree,
            near_plane=kwargs.pop("near_plane", self.cfg.near_plane),
            far_plane=kwargs.pop("far_plane", self.cfg.far_plane),
            rasterize_mode=rasterize_mode,
            camera_model=camera_model,
            packed=self.cfg.packed,
            sparse_grad=self.cfg.sparse_grad,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            distributed=self.world_size > 1,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        
        return render_colors, render_alphas, info

    def _train_step(
        self,
        step: int,
        data: Dict,
        target_level: int,
    ) -> float:
        """
        Perform a single training step at a specific LOD level.
        
        Args:
            step: Current training step
            data: Training data dictionary
            target_level: LOD level to render at
        
        Returns:
            loss: Training loss
        """
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        
        camtoworlds_gt = data["camtoworld"].to(device)  # [4, 4] or [1, 4, 4]
        Ks = data["K"].to(device)  # [3, 3] or [1, 3, 3]
        
        # Ensure batch dimension exists for camtoworlds and Ks
        if camtoworlds_gt.dim() == 2:
            camtoworlds_gt = camtoworlds_gt.unsqueeze(0)  # [4, 4] -> [1, 4, 4]
        if Ks.dim() == 2:
            Ks = Ks.unsqueeze(0)  # [3, 3] -> [1, 3, 3]
        
        camtoworlds = camtoworlds_gt
        
        image_data = data["image"].to(device) / 255.0  # [1, H, W, C] or [H, W, C]
        
        # Ensure batch dimension exists
        if image_data.dim() == 3:
            image_data = image_data.unsqueeze(0)  # [H, W, C] -> [1, H, W, C]
        
        # Get max_level for image multigrid downsample
        # Use cfg.max_level (not actual multigrid_gaussians.levels.max())
        # multigrid_gaussians.levels.max() is only used for V-cycle level selection
        image_multigrid_max_level = cfg.max_level if cfg.max_level is not None else 1
        
        # Calculate downsample factor: 2^(max_level - target_level)
        downsample_factor = 2 ** (image_multigrid_max_level - target_level)
        downsample_factor = max(1, downsample_factor)  # Ensure at least 1 (no upsampling)
        
        # Downsample image if needed
        original_height, original_width = image_data.shape[1:3]
        if downsample_factor > 1:
            # Calculate new dimensions
            new_height = max(1, original_height // downsample_factor)
            new_width = max(1, original_width // downsample_factor)
            
            # Downsample image using bilinear interpolation
            # Convert to [B, C, H, W] format for F.interpolate
            image_data_bchw = image_data.permute(0, 3, 1, 2)  # [1, C, H, W]
            image_data_downsampled = F.interpolate(
                image_data_bchw,
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False,
            )
            # Convert back to [B, H, W, C] format
            image_data = image_data_downsampled.permute(0, 2, 3, 1)  # [1, H, W, C]
            
            # Downsample Ks (camera intrinsics)
            # Ks format: [fx, 0, cx; 0, fy, cy; 0, 0, 1]
            # When downsample by factor f, we need to divide fx, fy, cx, cy by f
            Ks = Ks.clone()  # Avoid in-place modification
            Ks[:, 0, 0] = Ks[:, 0, 0] / downsample_factor  # fx
            Ks[:, 1, 1] = Ks[:, 1, 1] / downsample_factor  # fy
            Ks[:, 0, 2] = Ks[:, 0, 2] / downsample_factor  # cx
            Ks[:, 1, 2] = Ks[:, 1, 2] / downsample_factor  # cy
            
            # Downsample masks if provided
            if "mask" in data:
                mask_data = data["mask"].to(device)  # [H, W] or [1, H, W]
                if mask_data.dim() == 2:
                    mask_data = mask_data.unsqueeze(0)  # [1, H, W]
                # Downsample mask using nearest neighbor (preserve binary values)
                mask_data_bchw = mask_data.unsqueeze(1).float()  # [1, 1, H, W]
                mask_data_downsampled = F.interpolate(
                    mask_data_bchw,
                    size=(new_height, new_width),
                    mode='nearest',
                )
                masks = mask_data_downsampled.squeeze(1).bool()  # [1, H, W]
            else:
                masks = None
        else:
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if masks is not None and masks.dim() == 2:
                masks = masks.unsqueeze(0)
        
        # Handle RGBA images: extract RGB and alpha separately
        if image_data.shape[-1] == 4:
            pixels = image_data[..., :3]  # [1, H, W, 3]
            pixels_alpha = image_data[..., 3:4]  # [1, H, W, 1]
            has_alpha = True
        else:
            pixels = image_data  # [1, H, W, 3]
            pixels_alpha = None
            has_alpha = False
        
        num_train_rays_per_step = (
            pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
        )
        # image_id is int from dataset, convert to tensor
        if isinstance(data["image_id"], torch.Tensor):
            image_ids = data["image_id"].to(device)
        else:
            image_ids = torch.tensor(data["image_id"], device=device)

        if cfg.depth_loss:
            points = data["points"].to(device)  # [1, M, 2]
            depths_gt = data["depths"].to(device)  # [1, M]
            # Downsample points if image was downsampled
            if downsample_factor > 1:
                points = points / downsample_factor  # Scale point coordinates

        height, width = pixels.shape[1:3]

        if cfg.pose_noise:
            camtoworlds = self.pose_perturb(camtoworlds, image_ids)

        if cfg.pose_opt:
            camtoworlds = self.pose_adjust(camtoworlds, image_ids)

        # sh schedule
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        # Check if we have any GSs left
        if len(self.splats["means"]) == 0:
            print(f"ERROR: All Gaussian Splats have been pruned at step {step}. Training cannot continue.")
            raise RuntimeError(f"No Gaussian Splats remaining at step {step}")

        # Prepare backgrounds if needed
        backgrounds = None
        if cfg.random_bkgd:
            backgrounds = torch.rand(1, 3, device=device)  # [C=1, 3] for broadcasting
        elif cfg.white_background:
            backgrounds = torch.ones(1, 3, device=device)  # [C=1, 3] for white background
        
        # Render at target_level
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            level=target_level,  # Use target_level for LOD
            sh_degree=sh_degree_to_use,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            masks=masks,
            backgrounds=backgrounds,
        )
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        if cfg.use_bilateral_grid:
            grid_y, grid_x = torch.meshgrid(
                (torch.arange(height, device=self.device) + 0.5) / height,
                (torch.arange(width, device=self.device) + 0.5) / width,
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            colors = slice(
                self.bil_grids,
                grid_xy.expand(colors.shape[0], -1, -1, -1),
                colors,
                image_ids.unsqueeze(-1),
            )["rgb"]


        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors = colors + bkgd * (1.0 - alphas)

        self.cfg.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        # loss
        l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
        
        # Opacity loss: if GT has alpha channel, add opacity loss
        # Disabled: and False prevents activation
        if has_alpha and pixels_alpha is not None and False:
            # alphas is [1, H, W, 1] from rasterize_splats
            opacity_loss = F.l1_loss(alphas, pixels_alpha)
            loss += opacity_loss * (1.0 - cfg.ssim_lambda)
        
        depthloss = torch.tensor(0.0, device=device)
        if cfg.depth_loss:
            # query depths from depth map
            points = torch.stack(
                [
                    points[:, :, 0] / (width - 1) * 2 - 1,
                    points[:, :, 1] / (height - 1) * 2 - 1,
                ],
                dim=-1,
            )  # normalize to [-1, 1]
            grid = points.unsqueeze(2)  # [1, M, 1, 2]
            depths = F.grid_sample(
                depths.permute(0, 3, 1, 2), grid, align_corners=True
            )  # [1, 1, M, 1]
            depths = depths.squeeze(3).squeeze(1)  # [1, M]
            # calculate loss in disparity space
            disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
            disp_gt = 1.0 / depths_gt  # [1, M]
            depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
            loss += depthloss * cfg.depth_lambda
        
        tvloss = torch.tensor(0.0, device=device)
        if cfg.use_bilateral_grid:
            tvloss = 10 * total_variation_loss(self.bil_grids.grids)
            loss += tvloss
        
        if cfg.opacity_reg > 0.0:
            # Opacity regularization: apply to independent opacity values
            loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"][info["visible_mask"]]).mean()
            print("BISANGBISANG")
        # Scale regularization (applied to all gaussians)
        if cfg.scale_reg > 0.0:
            loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

        # Register gradient hooks for level-dependent scaling
        # Gradient scale for level L: grad_scale_factor ** (max_level - L)
        # Coarse levels (low L) get larger scale, fine levels (high L) get smaller scale
        hooks = []
        if cfg.grad_scale_factor != 1.0:
            # Get max level and levels for all gaussians
            if len(self.multigrid_gaussians.levels) > 0:
                actual_max_level = int(self.multigrid_gaussians.levels.max().item())
                levels = self.multigrid_gaussians.levels  # [N,]
            else:
                actual_max_level = 1
                levels = torch.ones(len(self.splats["means"]), dtype=torch.long, device=device)
            
            # Compute scale factors for each gaussian based on its level
            # scale_factor[i] = grad_scale_factor ** (max_level - level[i])
            level_diffs = actual_max_level - levels.float()  # [N,]
            scale_factors = cfg.grad_scale_factor ** level_diffs  # [N,]
            
            # Register hooks for each parameter
            for param_name, param in self.splats.items():
                if param.requires_grad:
                    # Create a closure to capture scale_factors
                    def make_hook(scale_factors_tensor):
                        def hook(grad):
                            if grad is not None:
                                # Scale gradient based on level
                                # grad shape: [N, ...] where first dimension is gaussian index
                                if grad.dim() > 0:
                                    # Apply scaling to first dimension (gaussian index)
                                    scale_shape = [scale_factors_tensor.shape[0]] + [1] * (grad.dim() - 1)
                                    scale = scale_factors_tensor.view(scale_shape)
                                    return grad * scale
                                return grad
                            return grad
                        return hook
                    
                    hook_handle = param.register_hook(make_hook(scale_factors))
                    hooks.append(hook_handle)

        loss.backward()
        
        # Remove hooks after backward
        for hook in hooks:
            hook.remove()

        # Turn Gradients into Sparse Tensor before running optimizer
        if cfg.sparse_grad:
            assert cfg.packed, "Sparse gradients only work with packed mode."
            gaussian_ids = info["gaussian_ids"]
            for k in self.splats.keys():
                grad = self.splats[k].grad
                if grad is None or grad.is_sparse:
                    continue
                self.splats[k].grad = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=grad[gaussian_ids],  # [nnz, ...]
                    size=self.splats[k].size(),  # [N, ...]
                    is_coalesced=len(Ks) == 1,
                )

        if cfg.visible_adam:
            gaussian_cnt = self.splats.means.shape[0]
            if cfg.packed:
                visibility_mask = torch.zeros_like(
                    self.splats["opacities"], dtype=bool
                )
                visibility_mask.scatter_(0, info["gaussian_ids"], 1)
            else:
                visibility_mask = (info["radii"] > 0).all(-1).any(0)

    # optimize
    
        opacity_grad_norms.append(self.splats["opacities"].grad.norm().item())

        for optimizer in self.optimizers.values():
            if cfg.visible_adam:
                optimizer.step(visibility_mask)
            else:
                optimizer.step()
            # print splat means, opacities grad norm
            optimizer.zero_grad(set_to_none=True)
            # print splat means, opacities grad
        for optimizer in self.pose_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.app_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.bil_grid_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Run post-backward steps after backward and optimizer
        # During V-cycle, we only update state (accumulate gradients) but skip densification
        # Densification will be performed after V-cycle completion to maintain consistency
        if isinstance(self.cfg.strategy, MultigridStrategy):
            # Call step_post_backward to update state (accumulate gradients)
            # step_post_backward only updates state, densification is deferred to post_cycle
            # visible_mask is obtained from info["visible_mask"] inside step_post_backward
            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )
            # Note: Densification is deferred until after V-cycle completion (in post_cycle)
        elif isinstance(self.cfg.strategy, DefaultStrategy):
            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                lr=0.0,  # Not used in MCMCStrategy
            )
        else:
            assert_never(self.cfg.strategy)
        
        # Iteration-based operations (similar to simple_trainer.py structure)
        # These are checked after optimizer step, matching simple_trainer.py behavior
        
        # Measure metrics if needed (lightweight, no image saving)
        # Only measure if step hasn't exceeded max_steps
        if step < cfg.max_steps and cfg.metric_interval > 0 and step % cfg.metric_interval == 0:
            self.measure_metric(step, stage="val")
        
        # Eval (full evaluation with image saving)
        # Only eval if step hasn't exceeded max_steps
        if step < cfg.max_steps and step in [i - 1 for i in cfg.eval_steps]:
            self.eval(step)
        
        # Save visualization
        # Only save if step hasn't exceeded max_steps
        if step < cfg.max_steps and cfg.visualization_interval > 0 and step % cfg.visualization_interval == 0:
            self.save_visualization(step)
        
        # Save hierarchy visualization
        # Only save if step hasn't exceeded max_steps
        if step < cfg.max_steps and cfg.hierarchy_visualization_interval > 0 and step % cfg.hierarchy_visualization_interval == 0:
            self.save_hierarchy_visualization(step)
        
        # Run compression
        # Only compress if step hasn't exceeded max_steps
        if step < cfg.max_steps and cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
            self.run_compression(step=step)
        
        return loss.item()

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        self.trainloader = trainloader

        trainloader_iter = iter(trainloader)

        print(len(trainloader))

        # V-cycle training: use config values for smoothing_steps and solving_steps
        smoothing_steps = cfg.smoothing_steps
        solving_steps = cfg.solving_steps
        
        # Training loop with V-cycles
        global_tic = time.time()
        current_step = init_step
        vcycle_idx = 0
        
        pbar = tqdm.tqdm(range(init_step, max_steps))
        
        while current_step < max_steps:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Check if we have any GSs left
            if len(self.multigrid_gaussians.levels) == 0:
                print(f"ERROR: All Gaussian Splats have been pruned at step {current_step}. Training cannot continue.")
                raise RuntimeError(f"No Gaussian Splats remaining at step {current_step}")
            
            # Strategy max_level should always use cfg.max_level (not dynamic)
            if isinstance(self.cfg.strategy, MultigridStrategy):
                # Always use cfg.max_level for strategy
                strategy_max_level = cfg.max_level if cfg.max_level is not None else self.cfg.strategy.max_level
                if "max_level" in self.strategy_state:
                    self.strategy_state["max_level"] = strategy_max_level
            

            # Get max_level for cycle based on cycle_type
            # V-cycle uses actual max level in hierarchical structure
            # inv-F cycle uses cfg.max_level for more regular training
            if cfg.cycle_type == "vcycle":
                # V-cycle: use actual max level from hierarchical structure
                if len(self.multigrid_gaussians.levels) > 0:
                    cycle_max_level = int(self.multigrid_gaussians.levels.max().item())
                else:
                    cycle_max_level = 1
                
                # V-cycle: finest -> 1 -> finest
                cycle_coarsest_level = 1
                current_step, losses = perform_vcycle(
                    current_step=current_step,
                    max_level=cycle_max_level,
                    coarsest_level=cycle_coarsest_level,
                    smoothing_steps=smoothing_steps,
                    solving_steps=solving_steps,
                    total_steps=max_steps,
                    runner=self,
                    trainloader_iter=trainloader_iter,
                    schedulers=schedulers,
                    cfg=cfg,
                    pbar=pbar,
                    vcycle_idx=vcycle_idx,
                )
            elif cfg.cycle_type == "inv_fcycle":
                # inv-F cycle: use cfg.max_level for regular training
                # Rasterization handles cases where actual max level < cfg.max_level
                cycle_max_level = cfg.max_level if cfg.max_level is not None else 1
                
                # inv-F cycle: finest~1 -> finest~2 -> ... -> finest~finest
                current_step, losses = perform_inv_fcycle(
                    current_step=current_step,
                    max_level=cycle_max_level,
                    smoothing_steps=smoothing_steps,
                    solving_steps=solving_steps,
                    total_steps=max_steps,
                    runner=self,
                    trainloader_iter=trainloader_iter,
                    schedulers=schedulers,
                    cfg=cfg,
                    pbar=pbar,
                    cycle_idx=vcycle_idx,
                )
            else:
                raise ValueError(f"Unknown cycle_type: {cfg.cycle_type}")
            
            # After cycle completion, perform densification if needed
            # This ensures hierarchical structure remains consistent during cycle
            if isinstance(self.cfg.strategy, MultigridStrategy):
                # Get the last step in this cycle for reference
                last_step_in_cycle = current_step - 1 if current_step > 0 else 0
                
                # Call post_cycle to perform densification based on cycle count
                # For inv-F cycle, vcycle_idx represents the outer cycle (finest~1, finest~2, etc.)
                # Each inner V-cycle within inv-F cycle should use the same cycle_idx
                self.cfg.strategy.post_cycle(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    cycle=vcycle_idx,
                    step=last_step_in_cycle,
                )
                
                # Update hierarchical structure from strategy state
                self.multigrid_gaussians.levels = self.strategy_state["levels"]
                self.multigrid_gaussians.parent_indices = self.strategy_state["parent_indices"]
                self.multigrid_gaussians.level_indices = self.strategy_state["level_indices"]
            
            # Calculate average loss for this cycle
            avg_loss = np.mean(losses) if losses else 0.0
            
            # Progress bar is already updated inside perform_vcycle/perform_inv_fcycle, but ensure it's synced
            if pbar.n < current_step:
                pbar.update(current_step - pbar.n)
            
            # Update progress bar description
            if isinstance(self.cfg.strategy, MultigridStrategy):
                strategy_max_level = self.strategy_state.get("max_level", cfg.max_level)
            else:
                strategy_max_level = cfg.max_level
            
            cycle_type_str = "V-cycle" if cfg.cycle_type == "vcycle" else "inv-F-cycle"
            if len(self.multigrid_gaussians.levels) > 0:
                actual_max_level = int(self.multigrid_gaussians.levels.max().item())
                desc = f"{cycle_type_str} {vcycle_idx}| strategy_max_level={strategy_max_level}| actual_max_level={actual_max_level}| loss={avg_loss:.3f}| step={current_step}/{max_steps}"
            else:
                desc = f"{cycle_type_str} {vcycle_idx}| strategy_max_level={strategy_max_level}| loss={avg_loss:.3f}| step={current_step}/{max_steps}"
            pbar.set_description(desc)
            
            # Handle eval, save, visualization, etc. for steps completed in this cycle
            # Process each step that was completed in this cycle
            steps_in_cycle = list(range(current_step - len(losses), current_step))
            for step in steps_in_cycle:
                if step >= max_steps:
                    break
                    
                # Tensorboard logging
                if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    # Use the loss from this step if available, otherwise use average
                    step_loss = losses[step - (current_step - len(losses))] if step >= (current_step - len(losses)) else avg_loss
                    self.writer.add_scalar("train/loss", step_loss, step)
                    self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                    self.writer.add_scalar("train/mem", mem, step)
                    if cfg.tb_save_image:
                        # Get a sample image for visualization
                        try:
                            data = next(trainloader_iter)
                        except StopIteration:
                            trainloader_iter = iter(self.trainloader)
                            data = next(trainloader_iter)
                        # Render at max level for visualization
                        camtoworlds = data["camtoworld"].to(device)
                        Ks = data["K"].to(device)
                        # Ensure batch dimension exists
                        if camtoworlds.dim() == 2:
                            camtoworlds = camtoworlds.unsqueeze(0)
                        if Ks.dim() == 2:
                            Ks = Ks.unsqueeze(0)
                        image_data = data["image"].to(device) / 255.0
                        if image_data.dim() == 3:
                            image_data = image_data.unsqueeze(0)
                        if image_data.shape[-1] == 4:
                            pixels = image_data[..., :3]
                        else:
                            pixels = image_data
                        height, width = pixels.shape[1:3]
                        # image_id is int from dataset, convert to tensor
                        if isinstance(data["image_id"], torch.Tensor):
                            image_ids = data["image_id"].to(device)
                        else:
                            image_ids = torch.tensor(data["image_id"], device=device)
                        masks = data["mask"].to(device) if "mask" in data else None
                        backgrounds = None
                        if cfg.white_background:
                            backgrounds = torch.ones(1, 3, device=device)
                        colors, _, _ = self.rasterize_splats(
                            camtoworlds=camtoworlds,
                            Ks=Ks,
                            width=width,
                            height=height,
                            level=-1,
                            sh_degree=min(step // cfg.sh_degree_interval, cfg.sh_degree),
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                            image_ids=image_ids,
                            masks=masks,
                            backgrounds=backgrounds,
                        )
                        canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                        canvas = canvas.reshape(-1, *canvas.shape[2:])
                        self.writer.add_image("train/render", canvas, step)
                    self.writer.flush()
                
                # Save checkpoint
                if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    stats = {
                        "mem": mem,
                        "ellipse_time": time.time() - global_tic,
                        "num_GS": len(self.splats["means"]),
                    }
                    print("Step: ", step, stats)
                    with open(
                        f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                        "w",
                    ) as f:
                        json.dump(stats, f)
                    data = {"step": step, "splats": self.splats.state_dict()}
                    # Also save hierarchical structure
                    data["levels"] = self.multigrid_gaussians.levels.cpu()
                    data["parent_indices"] = self.multigrid_gaussians.parent_indices.cpu()
                    data["level_indices"] = {
                        k: v for k, v in self.multigrid_gaussians.level_indices.items()
                    }
                    if cfg.pose_opt:
                        if world_size > 1:
                            data["pose_adjust"] = self.pose_adjust.module.state_dict()
                        else:
                            data["pose_adjust"] = self.pose_adjust.state_dict()
                    if cfg.app_opt:
                        if world_size > 1:
                            data["app_module"] = self.app_module.module.state_dict()
                        else:
                            data["app_module"] = self.app_module.state_dict()
                    torch.save(
                        data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                    )
                
                # Skip PLY export for multigrid - export_splats doesn't save hierarchical structure
                # Use save_multigrid_checkpoint instead to preserve full hierarchical information
                # PLY export is disabled because it doesn't include parent_indices, levels, etc.
            
            vcycle_idx += 1
            
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_steps_per_sec
                # Update the scene.
                self.viewer.update(current_step, 0)
            
            if current_step >= max_steps:
                # Don't run final V-cycle if we've already exceeded max_steps
                # The current V-cycle should have been stopped at max_steps
                break
        
        pbar.close()
        
        # Save final checkpoint and point clouds at training end
        if world_rank == 0:
            print("\n" + "="*60)
            print("Saving final multigrid checkpoint and point clouds...")
            print("="*60)
            
            from export_multigrid import save_multigrid_checkpoint, save_level_pointclouds
            
            # Save multigrid checkpoint
            final_checkpoint_dir = f"{cfg.result_dir}/final_checkpoint"
            os.makedirs(final_checkpoint_dir, exist_ok=True)
            save_multigrid_checkpoint(
                multigrid_gaussians=self.multigrid_gaussians,
                step=current_step,
                save_dir=final_checkpoint_dir,
                sh_degree=cfg.sh_degree,
            )
            
            # Save level-wise point clouds
            pointcloud_dir = f"{cfg.result_dir}/pointclouds"
            save_level_pointclouds(
                multigrid_gaussians=self.multigrid_gaussians,
                save_dir=pointcloud_dir,
                step=current_step,
            )
            
            print("="*60)
            print("Final checkpoint and point clouds saved successfully!")
            print("="*60)

    @torch.no_grad()
    def measure_metric(self, step: int, stage: str = "val") -> Dict[str, float]:
        """Measure metrics only (no image saving).
        
        Renders at finest level and computes PSNR, SSIM, LPIPS metrics.
        This is a lightweight function for frequent metric measurement during training.
        
        Args:
            step: Current training step
            stage: Evaluation stage name (e.g., "val", "test")
        
        Returns:
            Dictionary with metric values (psnr, ssim, lpips, etc.)
        """
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            # Ensure batch dimension exists
            if camtoworlds.dim() == 2:
                camtoworlds = camtoworlds.unsqueeze(0)
            if Ks.dim() == 2:
                Ks = Ks.unsqueeze(0)
            image_data = data["image"].to(device) / 255.0
            # Ensure batch dimension exists
            if image_data.dim() == 3:
                image_data = image_data.unsqueeze(0)
            masks = data["mask"].to(device) if "mask" in data else None
            # Ensure mask batch dimension exists
            if masks is not None and masks.dim() == 2:
                masks = masks.unsqueeze(0)
            
            # Handle RGBA images: extract RGB and alpha separately
            if image_data.shape[-1] == 4:
                pixels = image_data[..., :3]  # [1, H, W, 3]
            else:
                pixels = image_data  # [1, H, W, 3]
            
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            
            # Prepare backgrounds if needed
            backgrounds = None
            if cfg.white_background:
                backgrounds = torch.ones(1, 3, device=device)  # [C=1, 3] for white background
            
            # Render at highest LOD for metrics (finest level)
            colors, alphas, _ = self.multigrid_gaussians.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                level=-1,  # Highest LOD for metrics (finest level)
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                packed=cfg.packed,
                sparse_grad=cfg.sparse_grad,
                distributed=world_size > 1,
                camera_model=cfg.camera_model,
                backgrounds=backgrounds,
            )  # colors: [1, H, W, 3], alphas: [1, H, W, 1]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            
            # Compute metrics using highest LOD only (finest level)
            if world_rank == 0:
                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))
        
        # Compute average metrics
        if world_rank == 0:
            ellipse_time /= len(valloader)
            
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            
            # Print metrics
            if cfg.use_bilateral_grid:
                print(
                    f"Step {step} - Metrics: PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"Step {step} - Metrics: PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            
            # Save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            
            # Save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()
            
            return stats
        else:
            return {}

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation.
        
        Renders at multiple LOD levels: highest LOD (level=-1) down to level 1,
        decreasing by 2 levels each time. Metrics are computed using highest LOD only.
        Always saves rendered images.
        
        Args:
            step: Current training step
            stage: Evaluation stage name (e.g., "val", "test")
        """
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Get max level for multi-level rendering
        if len(self.multigrid_gaussians.levels) > 0:
            max_level = int(self.multigrid_gaussians.levels.max().item())
            # Generate level sequence: 1, 2, 3, ..., up to max_level (left to right: low to high)
            levels_to_render = list(range(1, max_level + 1))
        else:
            levels_to_render = [1]

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            # Ensure batch dimension exists
            if camtoworlds.dim() == 2:
                camtoworlds = camtoworlds.unsqueeze(0)
            if Ks.dim() == 2:
                Ks = Ks.unsqueeze(0)
            image_data = data["image"].to(device) / 255.0
            # Ensure batch dimension exists
            if image_data.dim() == 3:
                image_data = image_data.unsqueeze(0)
            masks = data["mask"].to(device) if "mask" in data else None
            # Ensure mask batch dimension exists
            if masks is not None and masks.dim() == 2:
                masks = masks.unsqueeze(0)
            
            # Handle RGBA images: extract RGB and alpha separately
            if image_data.shape[-1] == 4:
                pixels = image_data[..., :3]  # [1, H, W, 3]
            else:
                pixels = image_data  # [1, H, W, 3]
            
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            # Prepare backgrounds if needed
            # backgrounds shape should be [..., C, D] where C is number of cameras, D is channels
            # For camtoworlds shape [1, 4, 4], C=1, so backgrounds should be [1, 3]
            backgrounds = None
            if cfg.white_background:
                backgrounds = torch.ones(1, 3, device=device)  # [C=1, 3] for white background
            
            # Render at highest LOD for metrics (finest level)
            colors, alphas, _ = self.multigrid_gaussians.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                level=-1,  # Highest LOD for metrics (finest level)
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                packed=cfg.packed,
                sparse_grad=cfg.sparse_grad,
                distributed=world_size > 1,
                camera_model=cfg.camera_model,
                backgrounds=backgrounds,  # Pass backgrounds to rasterization
            )  # colors: [1, H, W, 3], alphas: [1, H, W, 1]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            
            # Compute metrics using highest LOD only (finest level)
            if world_rank == 0:
                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))
            
            # Render at multiple levels and concatenate
            if world_rank == 0:
                level_images = [pixels[0].cpu().numpy()]  # Start with GT
                
                # Prepare backgrounds if needed
                # backgrounds shape should be [..., C, D] where C is number of cameras, D is channels
                # For camtoworlds shape [1, 4, 4], C=1, so backgrounds should be [1, 3]
                backgrounds = None
                if cfg.white_background:
                    backgrounds = torch.ones(1, 3, device=device)  # [C=1, 3] for white background
                
                for render_level in levels_to_render:
                    colors_level, alphas_level, _ = self.multigrid_gaussians.rasterize_splats(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        level=render_level,
                        sh_degree=cfg.sh_degree,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        masks=masks,
                        packed=cfg.packed,
                        sparse_grad=cfg.sparse_grad,
                        distributed=world_size > 1,
                        camera_model=cfg.camera_model,
                        backgrounds=backgrounds,  # Pass backgrounds to rasterization
                    )  # colors_level: [1, H, W, 3], alphas_level: [1, H, W, 1]
                    colors_level = colors_level[0]  # [H, W, 3]
                    colors_level = torch.clamp(colors_level, 0.0, 1.0).cpu().numpy()
                    level_images.append(colors_level)
                
                # Concatenate horizontally: [H, (num_levels+1)*W, 3]
                canvas = np.concatenate(level_images, axis=1)
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def save_visualization(self, step: int):
        """Save visualization images comparing GT and render from sampled cameras.
        
        Saves concatenated images: GT | Level 1 | Level 2 | ... | Level N
        """
        if self.world_rank != 0:
            return
        
        if len(self.viz_camera_indices) == 0:
            return
        
        cfg = self.cfg
        device = self.device
        
        # Check if we have any GSs
        if len(self.multigrid_gaussians.levels) == 0:
            return
        
        # Get max level
        max_level = int(self.multigrid_gaussians.levels.max().item())
        
        # Generate level sequence: 1, 2, 3, ..., up to max_level (left to right: low to high)
        levels_to_render = list(range(1, max_level + 1))
        
        viz_dir = f"{cfg.result_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Collect all camera images
        all_camera_rows = []
        
        for cam_idx in self.viz_camera_indices:
            # Get data for this camera
            data = self.valset[cam_idx]
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
            
            # Prepare backgrounds if needed
            backgrounds = None
            if cfg.white_background:
                backgrounds = torch.ones(1, 3, device=device)  # [C=1, 3] for white background
            
            # Start with GT image
            level_images = [pixels_gt.cpu().numpy()]  # [H, W, 3]
            
            # Get max_level for downsample calculation
            image_multigrid_max_level = cfg.max_level if cfg.max_level is not None else max_level
            
            # Render at each level and collect images
            for render_level in levels_to_render:
                # Calculate downsample factor: 2^(max_level - render_level)
                downsample_factor = 2 ** (image_multigrid_max_level - render_level)
                downsample_factor = max(1, downsample_factor)  # Ensure at least 1 (no upsampling)
                
                # Downsample for rendering
                if downsample_factor > 1:
                    # Calculate new dimensions
                    render_height = max(1, height // downsample_factor)
                    render_width = max(1, width // downsample_factor)
                    
                    # Downsample Ks (camera intrinsics)
                    # Ks format: [fx, 0, cx; 0, fy, cy; 0, 0, 1]
                    # When downsample by factor f, we need to divide fx, fy, cx, cy by f
                    Ks_downsampled = Ks.clone()  # Avoid in-place modification
                    Ks_downsampled[:, 0, 0] = Ks[:, 0, 0] / downsample_factor  # fx
                    Ks_downsampled[:, 1, 1] = Ks[:, 1, 1] / downsample_factor  # fy
                    Ks_downsampled[:, 0, 2] = Ks[:, 0, 2] / downsample_factor  # cx
                    Ks_downsampled[:, 1, 2] = Ks[:, 1, 2] / downsample_factor  # cy
                    
                    # Downsample masks if provided
                    masks_downsampled = None
                    if masks is not None:
                        # Downsample mask using nearest neighbor (preserve binary values)
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
                
                # Render at downsampled resolution
                colors, alphas, _ = self.multigrid_gaussians.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks_downsampled,
                    width=render_width,
                    height=render_height,
                    level=render_level,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    masks=masks_downsampled,
                    packed=cfg.packed,
                    sparse_grad=cfg.sparse_grad,
                    distributed=self.world_size > 1,
                    camera_model=cfg.camera_model,
                    backgrounds=backgrounds,
                )  # colors: [1, render_H, render_W, 3], alphas: [1, render_H, render_W, 1]
                
                colors = colors[0]  # [render_H, render_W, 3]
                colors = torch.clamp(colors, 0.0, 1.0)
                
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
                
                # Free intermediate tensors
                if downsample_factor > 1:
                    del Ks_downsampled, masks_downsampled, colors_bchw, colors_upsampled
                
                colors = colors.cpu().numpy()
                level_images.append(colors)
                
                # Free rendered tensors
                del colors, alphas
            
            # Concatenate horizontally: [H, (num_levels+1)*W, 3]
            camera_row = np.concatenate(level_images, axis=1)
            all_camera_rows.append(camera_row)
        
        # Concatenate all cameras vertically: [(num_cams)*H, (num_levels+1)*W, 3]
        final_canvas = np.concatenate(all_camera_rows, axis=0)
        final_canvas = (final_canvas * 255).astype(np.uint8)
        
        # Save single concatenated image
        viz_path = os.path.join(viz_dir, f"viz_step_{step:06d}.jpg")
        imageio.imwrite(viz_path, final_canvas)
        
        print(f"  Saved visualization to {viz_dir} (levels: {levels_to_render}, cameras: {self.viz_camera_indices})")

    @torch.no_grad()
    def save_hierarchy_visualization(self, step: int):
        """Save hierarchy visualization: level-wise point clouds and parent-child linesets.
        
        Saves:
        - level_{level}_step_{step}.ply: Point cloud for each level
        - lineset_{level}_{level+1}_step_{step}.ply: LineSet connecting level to level+1
        """
        if self.world_rank != 0:
            return
        
        if len(self.multigrid_gaussians.levels) == 0:
            return
        
        try:
            import open3d as o3d
            import numpy as np
        except ImportError:
            print("Warning: open3d is not installed. Skipping hierarchy visualization.")
            return
        
        cfg = self.cfg
        
        # Create hierarchy visualization directory
        hierarchy_dir = os.path.join(cfg.result_dir, "hierarchy_visualization")
        os.makedirs(hierarchy_dir, exist_ok=True)
        
        # Get means and levels
        means = self.multigrid_gaussians.get_splats(level=None, detach_parents=True)["means"].detach().cpu().numpy()  # [N, 3]
        levels = self.multigrid_gaussians.levels.cpu().numpy()  # [N,]
        parent_indices = self.multigrid_gaussians.parent_indices.cpu().numpy()  # [N,]
        
        # Get unique levels
        unique_levels = sorted(np.unique(levels))
        
        # Save point cloud for each level
        for level in unique_levels:
            level_mask = (levels == level)
            if not level_mask.any():
                continue
            
            level_means = means[level_mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(level_means)
            
            # Color based on level (use a colormap)
            # Simple color scheme: level 1 = red, level 2 = green, level 3 = blue, etc.
            colors = np.zeros((len(level_means), 3))
            level_idx = unique_levels.index(level)
            if level_idx % 3 == 0:
                colors[:, 0] = 1.0  # Red
            elif level_idx % 3 == 1:
                colors[:, 1] = 1.0  # Green
            else:
                colors[:, 2] = 1.0  # Blue
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            filepath = os.path.join(hierarchy_dir, f"level_{level}_step_{step}.ply")
            o3d.io.write_point_cloud(filepath, pcd)
        
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
                    lines.append([parent_idx, next_level_idx])
            
            if len(lines) > 0:
                # Create LineSet
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(means)  # All points
                line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
                
                # Color lines (cyan for visibility)
                line_colors = np.ones((len(lines), 3)) * np.array([0.0, 1.0, 1.0])  # Cyan
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                
                filepath = os.path.join(hierarchy_dir, f"lineset_{level}_{next_level}_step_{step}.ply")
                o3d.io.write_line_set(filepath, line_set)
        
        print(f"  Saved hierarchy visualization to {hierarchy_dir} (step {step})")

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        cfg = self.cfg
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        # Load hierarchical structure if available
        if "levels" in ckpts[0]:
            runner.multigrid_gaussians.levels = ckpts[0]["levels"].to(runner.device)
            runner.multigrid_gaussians.parent_indices = ckpts[0]["parent_indices"].to(runner.device)
            runner.multigrid_gaussians.level_indices = ckpts[0]["level_indices"]
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python vcycle_trainer.py --dataset_type nerf --data_dir /path/to/data

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python vcycle_trainer.py --dataset_type nerf --data_dir /path/to/data --steps_scaler 0.25

    ```
    """

    # Use tyro.cli directly to allow subcommand-free usage
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)

