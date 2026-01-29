"""
*** 망하면 v18 쓰기 

vcycle_trainer_v12.py - Multigrid V-cycle Trainer with Hierarchy Consistency Loss

VERSION HISTORY AND KEY DIFFERENCES:

v12 (Current):
    - **View Sampling Per Step**: Each step samples a new view from trainloader (no fixed views)
    - **Independent Step Counts**: `smoothing_steps` and `solving_steps` are independent config parameters
    - **Hierarchy Consistency Loss**: Uses hierarchy consistency loss to enforce parent-child relationships
    - **Simplified Code**: Removed view caching and fixed view logic for cleaner implementation
    - **No Restriction/Prolongation**: Hierarchy consistency loss replaces render-based restriction/prolongation

v11:
    - Fixed views within V-cycle (removed in v12)
    - View caching for efficiency (removed in v12)
    - Unified step counts based on num_cycle_views (removed in v12)

v10:
    - Render-based restriction/prolongation implementation
    - View sampling per step

Benefits:
---------
- Direct parent-child consistency: Enforces consistency between parent and child Gaussians
- Better multigrid convergence: Coarse level parameters match aggregated fine level parameters
- Reduced GT dependency: Less reliance on GT for defining level relationships
- More principled multigrid: Aligns with hierarchical Gaussian structure

Implementation Notes:
---------------------
- Hierarchy consistency loss is applied during smoothing steps (downward/upward)
- Direction-based regularization:
  - downwards: parents are regularized (children -> parent aggregation)
  - upwards: children are regularized (parent -> children)
- Gradient flow: detach() is crucial to prevent unwanted gradient propagation

VERSION HISTORY AND KEY DIFFERENCES:

v12 (Current):
    - **View Sampling Per Step**: Each step samples a new view from trainloader (no fixed views)
    - **Independent Step Counts**: `smoothing_steps` and `solving_steps` are independent config parameters
    - **Hierarchy Consistency Loss**: Uses hierarchy consistency loss to enforce parent-child relationships
    - **Simplified Code**: Removed view caching and fixed view logic for cleaner implementation
    - **No Restriction/Prolongation**: Hierarchy consistency loss replaces render-based restriction/prolongation
    - **Result Directory**: Updated to `vcycle_trainer_v12_{cycle_type}`

v11:
    - Fixed views within V-cycle (removed in v12)
    - View caching for efficiency (removed in v12)
    - Unified step counts based on num_cycle_views (removed in v12)

v10:
    - Render-based restriction/prolongation implementation
    - View sampling per step
    - Uses `multigrid_v10` and `multigrid_gaussians_v4`
    - `init_level1_ratio: float = 0.9`
    - Result directory: `multigrid_v10_{cycle_type}`

v9:
    - Render-based restriction/prolongation implementation

v8:
    - Cycle-based visualization and evaluation intervals
    - Gradient inheritance for color gradients
    - Integrated `is_grad_high` flag (grad2d OR color_grad)

v7:
    - Various improvements to densification logic

v6:
    - **multigrid_gaussians_v2**: Uses v2 with helper functions for parent<->child conversion
    
    - **Result Directory**: Includes "v6" and "randbkgd" flag in path

v5 (Previous):
    - Basic V-cycle implementation

v4 and earlier:
    - Basic V-cycle implementation
    - May have had different parameter handling
"""

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
from utils import AppearanceOptModule, CameraOptModule, set_random_seed

from gsplat.compression import PngCompression
from gsplat.cuda._wrapper import quat_scale_to_covar_preci, fully_fused_projection
from gsplat.distributed import cli
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
from multigrid_gaussians_v8 import MultigridGaussians
# load_hierarchy_multigrid is now a static method of MultigridGaussians
# Use MultigridGaussians.load_hierarchy_multigrid() instead


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
    max_steps: int = 10_000
    # Cycles to evaluate the model (evaluation happens at the end of each cycle)
    eval_cycles: List[int] = field(default_factory=lambda: [])
    # Interval for metric measurement (every N cycles). -1 means disable metric measurement.
    # Evaluation is performed at the end of each cycle, not during cycle execution.
    metric_interval_cycles: int = 1
    # Whether to measure metrics inside V-cycle (during down/up smoothing). 
    # If False, metrics are only measured at cycle boundaries.
    metric_inside_cycle: bool = False
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [10_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [10_000])
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

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = True
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
    # Hierarchy consistency regularization
    # Enforces that parent gaussians match the aggregation of their children
    # This ensures consistency between fine and coarse levels
    hierarchy_consistency_lambda: float = 0.5
    # Weight one-hot regularization
    # Encourages weights to be one-hot like (sparse) by minimizing entropy
    # This makes aggregation more selective (one child dominates per parent)
    weight_onehot_lambda: float = 0.1
    # Position scale reduction for hierarchical gaussians
    # Higher level gaussians are constrained to stay closer to their parents
    position_scale_reduction: float = 0.75
    # Maximum level for hierarchical structure
    # If set, gaussians at max_level will only duplicate (not split) even if split conditions are met
    max_level: Optional[int] = 4
    # Increase max level every N steps (start from level 2)
    level_increase_interval: int = 500
    # Ratio of Level 1 points to Level 2 points during hierarchical initialization
    # Level 1 = init_level1_ratio * Level 2 (default: 0.9, i.e., 90%)
    # Increased from 0.5 to 0.9 to improve initial representation power
    init_level1_ratio: float = 1
    # Parent sampling method: "uniform" or "fps" (farthest point sampling)
    parent_sampling_method: Literal["uniform", "fps"] = "fps"
    
    # Cycle type: "vcycle" or "inv_fcycle"
    cycle_type: Literal["vcycle", "inv_fcycle"] = "vcycle"
    
    # V-cycle parameters
    smoothing_steps: int = 64  # Number of smoothing steps per level
    solving_steps: int = 64  # Number of solving steps at coarsest level
    steps_decaying_per_level: float = 0.5
    
    
    # Gradient scaling parameters
    grad_scale_factor: float = 1. # Base factor for level-dependent gradient scaling
    # Gradient scale for level L: grad_scale_factor ** (max_level - L)
    # Coarse levels (low L) get larger scale, fine levels (high L) get smaller scale
    
    # Level resolution reduction factor
    # Resolution reduction per level: resolution = original_resolution * (level_resolution_factor ^ (max_level - target_level))
    # Default 0.5 means each level halves the resolution (1/2 per level)
    # Previous default was 1/sqrt(2) ≈ 0.707 (sqrt(2) per level)
    level_resolution_factor: float = 0.75

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

    # Visualization: save visualization images (GT vs render) at the end of each cycle
    # If > 0, saves visualization every N cycles (e.g., 10 = every 10 cycles)
    visualization_interval: int = 2  # 10 = every 10 cycles, 1 = every cycle, 0 = disabled
    # Visualization: save hierarchy visualization (level pointclouds and linesets) at the end of each cycle
    # If > 0, saves hierarchy visualization after each cycle completion
    hierarchy_visualization_interval: int = 0  # 1 = every cycle, 0 = disabled

    # Hierarchy loading (required - loads hierarchy structure)
    hierarchy_path: Optional[str] = None
    
    # Profile timing for rendering vs consistency loss (for debugging)
    profile_timing: bool = False

    def __post_init__(self):
        """Auto-generate result_dir if not provided."""
        from pathlib import Path
        
        if self.data_dir is None:
            if self.dataset_type == "colmap":
                self.data_dir = "./dataset/60_v2/garden"
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
            ]
            if self.dataset_type == "nerf" and self.white_background:
                settings_parts.append("whitebg")
            if self.random_bkgd:
                settings_parts.append("randbkgd")
            if self.normalize_world_space:
                settings_parts.append("norm")
            if self.patch_size is not None:
                settings_parts.append(f"patch_{self.patch_size}")
            
            settings_str = "_".join(settings_parts)
            result_base = f"./results/hierarchy_trainer_vcycle/{dataset_name}_{settings_str}"
            if self.hierarchy_path is not None:
                hierarchy_name = Path(self.hierarchy_path).stem
                result_base += f"_hierarchy_{hierarchy_name}"
            # Add cycle type suffix to distinguish vcycle vs inv_fcycle
            if self.cycle_type == "inv_fcycle":
                result_base += f"_{self.cycle_type}"
            self.result_dir = result_base
        

        # Ensure packed/sparse/visible_adam compatibility
        if self.sparse_grad and not self.packed:
            self.packed = True
        if self.visible_adam and self.sparse_grad:
            self.sparse_grad = False

    def adjust_steps(self, factor: float):
        # Note: eval_cycles and metric_interval_cycles are not scaled by factor
        # as they are cycle-based, not step-based
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)


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
    global_tic: Optional[float] = None,
) -> Tuple[int, List[float]]:
    """
    Recursive V-cycle: max_level -> coarsest_level -> max_level.
    
    V-cycle structure (recursive):
    - Base case: level == coarsest_level -> perform solving
    - Recursive case:
      1. Downward: Smooth at current level -> Recurse to level-1
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
        current_solving_steps = int(solving_steps * (cfg.steps_decaying_per_level ** (max_level - level)))
        current_solving_steps = max(1, current_solving_steps)
        for solving_step in range(current_solving_steps):
            # Continue V-cycle even if max_steps is reached (complete the cycle)
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(runner.trainloader)
                data = next(trainloader_iter)
            
            log_step = min(current_step, total_steps - 1) if total_steps > 0 else current_step
            
            # CHANGED: Solve steps use "upwards" direction (children are regularized)
            loss, losses_dict = runner._train_step(
                step=log_step,
                data=data,
                target_level=level,
                direction="upwards",  # Solve: children at coarsest level are regularized
            )
            losses.append(loss)
            
            # Only increment step and scheduler if not past max_steps
            # if current_step < total_steps:
            for scheduler in schedulers:
                scheduler.step()
            current_step += 1
            
            if pbar is not None:
                pbar.update(1)
                avg_loss = np.mean(losses) if losses else 0.0
                desc = f"V-cycle {vcycle_idx}| Solve L{level}| loss={avg_loss:.3f}| step={current_step}/{total_steps}"
                pbar.set_description(desc)
        
        return current_step, losses
    
    # Recursive case: Downward pass (smoothing) -> recurse -> upward pass (smoothing)
    
    # ========== Downward: Smoothing at current level ==========
    target_level = level
    
    current_smoothing_steps = int(smoothing_steps * (cfg.steps_decaying_per_level ** (max_level - level)))
    current_smoothing_steps = max(1, current_smoothing_steps)
    metric_backgrounds = None
    if cfg.random_bkgd:
        metric_backgrounds = torch.rand(1, 3, device=runner.device)
    elif cfg.white_background:
        metric_backgrounds = torch.ones(1, 3, device=runner.device)

    def log_level_metric(tag: str, print_metric: bool = True):
        if not cfg.metric_inside_cycle:
            return
        metric_step = (
            min(max(current_step - 1, 0), total_steps - 1)
            if total_steps > 0
            else current_step
        )
        runner.measure_metric_on_dataset(
            target_level=target_level,
            step=metric_step,
            tag=tag,
            backgrounds=metric_backgrounds,
            stage="val",
            print_metric=print_metric,
        )

    # Metric before down smoothing
    log_level_metric(tag=f"DownPreSmooth L{target_level}", print_metric=False)
    # Smoothing steps at current level
    for smoothing_step in range(current_smoothing_steps):
        # Continue V-cycle even if max_steps is reached (complete the cycle)
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(runner.trainloader)
            data = next(trainloader_iter)
        
        log_step = min(current_step, total_steps - 1) if total_steps > 0 else current_step
        
        # Downward pass uses "downwards" direction (parents are regularized)
        loss, losses_dict = runner._train_step(
            step=log_step,
            data=data,
            target_level=target_level,
            direction="downwards",  # Downward: parents at target_level are regularized
        )
        losses.append(loss)
        
        for scheduler in schedulers:
            scheduler.step()
        current_step += 1
        
        if pbar is not None:
            pbar.update(1)
            avg_loss = np.mean(losses) if losses else 0.0
            desc = f"V-cycle {vcycle_idx}| Down L{target_level}| loss={avg_loss:.3f}| step={current_step}/{total_steps}"
            pbar.set_description(desc)

    log_level_metric(tag=f"Down L{target_level}")

    # ========== Recurse to coarser level ==========
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
        global_tic=global_tic,
    )

    # Smoothing steps at current level (after returning from coarser level)
    current_smoothing_steps = int(
        smoothing_steps * (cfg.steps_decaying_per_level ** (max_level - level))
    )
    current_smoothing_steps = max(1, current_smoothing_steps)

    log_level_metric(tag=f"UpPreSmooth L{target_level}")

    # ========== Upward: Smoothing at current level ==========
    for smoothing_step in range(current_smoothing_steps):
        # Continue V-cycle even if max_steps is reached (complete the cycle)
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(runner.trainloader)
            data = next(trainloader_iter)
        
        log_step = min(current_step, total_steps - 1) if total_steps > 0 else current_step
        
        # CHANGED: Upward pass uses "upwards" direction (children are regularized)
        loss, losses_dict = runner._train_step(
            step=log_step,
            data=data,
            target_level=target_level,
            direction="upwards",  # Upward: children at target_level are regularized
        )
        losses.append(loss)
        
        # Only increment step and scheduler if not past max_steps
        # if current_step < total_steps:
        for scheduler in schedulers:
            scheduler.step()
        current_step += 1
        
        if pbar is not None:
            pbar.update(1)
            avg_loss = np.mean(losses) if losses else 0.0
            desc = f"V-cycle {vcycle_idx}| Up L{target_level}| loss={avg_loss:.3f}| step={current_step}/{total_steps}"
            pbar.set_description(desc)

    # Metric after up smoothing
    log_level_metric(tag=f"UpPostSmooth L{target_level}", print_metric=False)

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
    global_tic: Optional[float] = None,
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
    
    # Gradient scaling hooks are registered via runner._update_grad_scaling_hooks()
    # which is called after densification. Since densification is disabled
    # in hierarchy_trainer_vcycle.py, hooks are never registered.
    # For vcycle_trainer_v21.py, hooks are updated after densification.

    # Call recursive V-cycle starting from max_level
    current_step, losses = vcycle_recursive(
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
        global_tic=global_tic,
    )

    return current_step, losses


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
    global_tic: Optional[float] = None,
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
            global_tic=global_tic,
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

        # Max level schedule: start from level 1 when init_level1_ratio >= 1, else level 2.
        self.final_max_level = cfg.max_level
        self.current_max_level = self._compute_current_max_level(0)
        if self.current_max_level is not None:
            self.cfg.max_level = self.current_max_level

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

        # V-cycle metric log (jsonl)
        self.metric_log_path = os.path.join(self.stats_dir, "vcycle_metrics.jsonl")

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

        # Model: Load from hierarchy if provided, otherwise create new MultigridGaussians
        feature_dim = 32 if cfg.app_opt else None
        self.use_hierarchy = cfg.hierarchy_path is not None
        
        if self.use_hierarchy:
            # Load hierarchy from file (means and scales from hierarchy, others initialized)
            print(f"Loading hierarchy from {cfg.hierarchy_path}...")
            self.multigrid_gaussians, _ = MultigridGaussians.load_hierarchy_multigrid(
                hierarchy_path=cfg.hierarchy_path,
                parser=self.parser,
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
                device=self.device,
                world_rank=world_rank,
                world_size=world_size,
                position_scale_reduction=cfg.position_scale_reduction,
                max_level=cfg.max_level,
            )
            # Update max_level from hierarchy if not explicitly set
            if cfg.max_level is None:
                # Get max_level from hierarchy
                checkpoint = torch.load(cfg.hierarchy_path, map_location=self.device, weights_only=False)
                hierarchy = checkpoint["hierarchy"]
                num_levels = len(hierarchy["levels"])
                # Multigrid convention: levels are 1..num_levels (1=coarsest, num_levels=finest)
                cfg.max_level = num_levels
                print(f"Using max_level={cfg.max_level} from hierarchy ({num_levels} levels)")
            total_gs = len(self.multigrid_gaussians.splats["means"])
            print(f"Model initialized from hierarchy. Total GS (all levels): {total_gs}")
            
            # Check finest level count (finest = highest multigrid level)
            if len(self.multigrid_gaussians.levels) > 0:
                finest_level = int(self.multigrid_gaussians.levels.max().item())
                finest_mask = (self.multigrid_gaussians.levels == finest_level)
                finest_count = finest_mask.sum().item()
                print(f"  Finest level (level {finest_level}) GS count: {finest_count}")
                print(f"  Other levels GS count: {total_gs - finest_count}")
                
                # Check what would be visible at finest level
                self.multigrid_gaussians.set_visible_mask(finest_level)
                visible_at_finest = self.multigrid_gaussians.visible_mask.sum().item()
                print(f"  Visible GS at finest level (after set_visible_mask): {visible_at_finest}")
            
        else:
            # Create new MultigridGaussians (normal initialization)
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
                render_device=self.device,
                world_rank=world_rank,
                world_size=world_size,
                position_scale_reduction=cfg.position_scale_reduction,
                max_level=cfg.max_level,
            )
            print("Model initialized. Number of GS:", len(self.multigrid_gaussians.splats["means"]))
        
        # Expose splats and optimizers
        self.splats = self.multigrid_gaussians.splats
        self.optimizers = self.multigrid_gaussians.optimizers

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
        
        # Gradient scaling hooks (updated only after densification)
        self.grad_scaling_hooks = []  # List of hook handles
        self.grad_scale_factors = None  # Cached scale factors [N,]
        # Cache for hierarchy index mapping (direction -> level -> (children, parents))
        self._hierarchy_index_cache = {"downwards": {}, "upwards": {}}
    
    @torch.no_grad()
    def render_initial_hierarchy_levels(self, num_cameras: int = 3):
        """
        Render each hierarchy level before training starts (for visual verification).
        Similar to visualize_hierarchy_levels.py but uses already-loaded multigrid_gaussians.
        
        Args:
            num_cameras: Number of cameras to render
        """
        if not self.use_hierarchy:
            print("Skipping initial hierarchy level rendering (not using hierarchy)")
            return
        
        if self.world_rank != 0:
            # Only render on rank 0
            return
        
        cfg = self.cfg
        device = self.device
        
        print("\n" + "="*60)
        print("Rendering initial hierarchy levels for visual verification...")
        print("="*60)
        
        # Create output directory
        output_dir = os.path.join(cfg.result_dir, "initial_hierarchy_levels")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get max level and verify level structure
        if len(self.multigrid_gaussians.levels) > 0:
            actual_max_level = int(self.multigrid_gaussians.levels.max().item())
            actual_min_level = int(self.multigrid_gaussians.levels.min().item())
            unique_levels = sorted(self.multigrid_gaussians.levels.unique().cpu().tolist())
            print(f"Hierarchy level structure:")
            print(f"  Actual levels in data: {unique_levels}")
            print(f"  Min level: {actual_min_level}, Max level: {actual_max_level}")
            for level in unique_levels:
                level_count = (self.multigrid_gaussians.levels == level).sum().item()
                print(f"  Level {level}: {level_count} gaussians")
        else:
            # Multigrid levels start at 1
            actual_max_level = 1
            actual_min_level = 1
            unique_levels = [1]
        
        max_level = int(self.multigrid_gaussians.max_level) if self.multigrid_gaussians.max_level is not None else actual_max_level
        levels_to_render = list(range(1, max_level + 1))
        print(f"Rendering {len(levels_to_render)} levels: {levels_to_render}")
        print(f"  Note: Level 1 = coarsest (lowest resolution), Level {max_level} = finest (highest resolution)")
        
        # Select cameras from validation set
        val_indices = list(range(len(self.valset)))
        if len(val_indices) >= num_cameras:
            import random
            random.seed(42)
            camera_indices = random.sample(val_indices, num_cameras)
        else:
            camera_indices = val_indices[:num_cameras] if len(val_indices) > 0 else []
        
        print(f"Rendering {len(camera_indices)} cameras: {camera_indices}")
        
        # Prepare backgrounds
        backgrounds = None
        if cfg.white_background:
            backgrounds = torch.ones(1, 3, device=device)
        
        # Render each camera and each level
        for cam_idx in camera_indices:
            # Get camera data
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
            
            # Save GT image
            gt_path = os.path.join(output_dir, f"cam_{cam_idx:04d}_GT.png")
            gt_image = (pixels_gt.detach().cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(gt_path, gt_image)
            print(f"  Saved GT to {gt_path}")
            
            # Render each level (multigrid levels: 1=coarsest, max_level=finest)
            for render_level in levels_to_render:
                # Calculate downsample factor:
                # - Level max_level (finest): downsample_factor = 1
                # - Level 1 (coarsest): downsample_factor = (1/level_resolution_factor)^(max_level - 1)
                level_diff = max_level - render_level
                downsample_factor = (1.0 / cfg.level_resolution_factor) ** level_diff
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
                self.multigrid_gaussians.set_visible_mask(render_level)
                visible_count = self.multigrid_gaussians.visible_mask.sum().item()
                total_count = len(self.multigrid_gaussians.levels)
                level_mask = (self.multigrid_gaussians.levels == render_level)
                level_count = level_mask.sum().item()
                print(f"    Level {render_level}: {visible_count}/{total_count} gaussians visible "
                      f"(level {render_level} has {level_count} gaussians)")
                
                # Render at downsampled resolution
                colors, alphas, info = self.multigrid_gaussians.rasterize_splats(
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
                )  # colors: [1, render_H, render_W, 3]
                
                colors = colors[0]  # [render_H, render_W, 3]
                colors = torch.clamp(colors, 0.0, 1.0)
                
                # Debug: Check if rendered image is all black or has very low values
                color_sum = colors.sum().item()
                color_mean = colors.mean().item()
                if color_sum < 1e-6 or color_mean < 1e-6:
                    print(f"    WARNING: Level {render_level} rendered image is all black or very dark!")
                    print(f"      color_sum={color_sum:.6f}, color_mean={color_mean:.6f}")
                    print(f"      visible_count={visible_count}, level_count={level_count}")
                
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
        
        print(f"\nAll initial hierarchy level visualizations saved to {output_dir}")
        print("="*60 + "\n")

    def _compute_current_max_level(self, step: int) -> Optional[int]:
        final_max_level = self.final_max_level
        if final_max_level is None:
            return None

        if final_max_level <= 1:
            return final_max_level
        
        # Start from level 1 and increase gradually based on level_increase_interval
        start_level = 1
        if self.cfg.level_increase_interval <= 0:
            return final_max_level
        
        # Increase max_level every level_increase_interval steps, starting from level 1
        # Step 0-(level_increase_interval-1): level 1
        # Step level_increase_interval-(2*level_increase_interval-1): level 2
        # Step 2*level_increase_interval-(3*level_increase_interval-1): level 3
        # etc.
        increase_steps = step // self.cfg.level_increase_interval
        return min(final_max_level, start_level + increase_steps)

    def _maybe_update_max_level(self, step: int, cycle_idx: int) -> None:
        target_max_level = self._compute_current_max_level(step)
        if target_max_level is None:
            return
        if self.current_max_level == target_max_level:
            return

        self.current_max_level = target_max_level
        self.cfg.max_level = target_max_level
        self.multigrid_gaussians.set_max_level(target_max_level)
    
    def _update_grad_scaling_hooks(self):
        """
        Update gradient scaling hooks based on current hierarchy structure.
        This should be called after densification when hierarchy structure changes.
        
        For hierarchy_trainer_vcycle.py: densification is disabled, so this is never called.
        For vcycle_trainer_v21.py: call this after densification to update hooks.
        """
        cfg = self.cfg
        if cfg.grad_scale_factor == 1.0:
            # No scaling needed, remove any existing hooks
            self._remove_grad_scaling_hooks()
            return
        
        # Remove existing hooks if any
        self._remove_grad_scaling_hooks()
        
        # Compute scale_factors based on current hierarchy structure
        if len(self.multigrid_gaussians.levels) > 0:
            actual_max_level = int(self.multigrid_gaussians.levels.max().item())
            levels = self.multigrid_gaussians.levels  # [N,]
        else:
            actual_max_level = 1
            levels = torch.ones(
                len(self.splats["means"]),
                dtype=torch.long,
                device=self.splats["means"].device,
            )
        
        # Compute scale factors for each gaussian based on its level
        # scale_factor[i] = grad_scale_factor ** (max_level - level[i])
        level_diffs = actual_max_level - levels.float()  # [N,]
        self.grad_scale_factors = cfg.grad_scale_factor ** level_diffs  # [N,]
        
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
                
                hook_handle = param.register_hook(make_hook(self.grad_scale_factors))
                self.grad_scaling_hooks.append(hook_handle)
    
    def _remove_grad_scaling_hooks(self):
        """Remove all gradient scaling hooks."""
        for hook in self.grad_scaling_hooks:
            hook.remove()
        self.grad_scaling_hooks.clear()
        self.grad_scale_factors = None
    
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
            absgrad=False,
            distributed=self.world_size > 1,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        
        return render_colors, render_alphas, info

    def render_at_level(
        self,
        target_level: int,
        pixels_gt: Tensor,  # Original GT image [1, H, W, 3]
        original_height: int,
        original_width: int,
        camtoworlds: Tensor,
        Ks_original: Tensor,  # Original Ks before any downsampling
        image_multigrid_max_level: int,
        sh_degree: int,
        near_plane: float,
        far_plane: float,
        image_ids: Tensor,
        masks: Optional[Tensor] = None,
        render_mode: str = "RGB",
        force_original_resolution: bool = False,
        pixels_downsampled: Optional[Tensor] = None,
        Ks_downsampled: Optional[Tensor] = None,
        masks_downsampled: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        """
        Render at a specific level and return downsampled GT and rendered image.
        
        Args:
            target_level: LOD level to render at
            pixels_gt: Original GT image [1, H, W, 3]
            original_height: Original image height
            original_width: Original image width
            camtoworlds: Camera-to-world matrices [1, 4, 4]
            Ks_original: Original camera intrinsics [1, 3, 3]
            image_multigrid_max_level: Maximum level for image multigrid
            sh_degree: Spherical harmonics degree
            near_plane: Near plane distance
            far_plane: Far plane distance
            image_ids: Image IDs [1,]
            masks: Optional masks [1, H, W]
            render_mode: Render mode ("RGB", "RGB+ED", etc.)
        
        Returns:
            rendered_colors: Rendered image at target_level resolution [1, H, W, 3]
            pixels_downsampled: Downsampled GT at target_level resolution [1, H, W, 3]
            alphas: Alpha channel [1, H, W, 1] or None
            info: Info dict from rasterization
        """
        cfg = self.cfg
        device = self.device
        
        # OPTIMIZED: Fast path for finest level (no downsampling needed)
        if target_level == image_multigrid_max_level and not force_original_resolution:
            # Finest level: no downsampling, use original resolution
            height, width = original_height, original_width
            pixels_downsampled = pixels_gt
            Ks = Ks_original
            masks_downsampled = masks
        elif pixels_downsampled is not None and Ks_downsampled is not None:
            # Use precomputed downsampled inputs (cached on CPU, moved to GPU by caller)
            height, width = pixels_downsampled.shape[1:3]
            Ks = Ks_downsampled
            # masks_downsampled is passed in (may be None)
        else:
            # Calculate downsample factor: (1/level_resolution_factor)^(max_level - target_level)
            # For level_resolution_factor=0.5: 2^(max_level - target_level)
            if force_original_resolution:
                downsample_factor = 1
            else:
                level_diff = image_multigrid_max_level - target_level
                downsample_factor = (1.0 / cfg.level_resolution_factor) ** level_diff
                downsample_factor = max(1, int(downsample_factor))  # Ensure at least 1 (no upsampling) and convert to int
            
            # Calculate target resolution
            height = max(1, int(original_height // downsample_factor))
            width = max(1, int(original_width // downsample_factor))
            
            # Downsample GT image
            if downsample_factor > 1:
                pixels_bchw = pixels_gt.permute(0, 3, 1, 2)  # [1, 3, H, W]
                pixels_downsampled_bchw = F.interpolate(
                    pixels_bchw,
                    size=(height, width),
                    mode='bicubic',
                    align_corners=False,
                )
                pixels_downsampled = pixels_downsampled_bchw.permute(0, 2, 3, 1)  # [1, H, W, 3]
            else:
                pixels_downsampled = pixels_gt  # [1, H, W, 3]
            
            # Downsample Ks (camera intrinsics)
            Ks = Ks_original.clone()  # Avoid in-place modification
            if downsample_factor > 1:
                Ks[:, 0, 0] = Ks_original[:, 0, 0] / downsample_factor  # fx
                Ks[:, 1, 1] = Ks_original[:, 1, 1] / downsample_factor  # fy
                Ks[:, 0, 2] = Ks_original[:, 0, 2] / downsample_factor  # cx
                Ks[:, 1, 2] = Ks_original[:, 1, 2] / downsample_factor  # cy
            
            # Downsample masks if provided
            masks_downsampled = None
            if masks is not None:
                if downsample_factor > 1:
                    masks_bchw = masks.unsqueeze(1).float()  # [1, 1, H, W]
                    masks_downsampled_bchw = F.interpolate(
                        masks_bchw,
                        size=(height, width),
                        mode='nearest',
                    )
                    masks_downsampled = masks_downsampled_bchw.squeeze(1).bool()  # [1, H, W]
                else:
                    masks_downsampled = masks
        
        # Render at target_level
        # Note: backgrounds are handled in trainer after rendering (not in rasterize_splats)
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            level=target_level,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            image_ids=image_ids,
            render_mode=render_mode,
            masks=masks_downsampled,
        )
        
        # Extract colors from renders
        if renders.shape[-1] == 4:
            rendered_colors = renders[..., 0:3]  # [1, H, W, 3]
        else:
            rendered_colors = renders  # [1, H, W, 3]
        
        return rendered_colors, pixels_downsampled, alphas, info

    def _get_projection_only(
        self,
        level: int,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree: int,
        near_plane: float,
        far_plane: float,
        image_ids: Optional[Tensor] = None,
    ) -> Optional[Dict]:
        """
        Get projection data (means2d, conics, opacities, colors) without full rasterization.
        This is more efficient than full rasterization when only projection data is needed.
        
        Args:
            level: Level to project
            camtoworlds: Camera-to-world matrices [1, 4, 4]
            Ks: Camera intrinsics [1, 3, 3]
            width: Image width
            height: Image height
            sh_degree: SH degree for color evaluation
            near_plane: Near plane distance
            far_plane: Far plane distance
            image_ids: Image IDs for appearance optimization
        
        Returns:
            Dictionary with projection data similar to _extract_projection_data format
        """
        device = self.device
        multigrid_gaussians = self.multigrid_gaussians
        
        # Determine render level
        if level == -1:
            if len(multigrid_gaussians.levels) > 0:
                render_level = int(multigrid_gaussians.levels.max().item())
            else:
                render_level = 1
        else:
            render_level = level
        
        # Set visible mask
        if len(multigrid_gaussians.levels) > 0:
            max_available_level = int(multigrid_gaussians.levels.max().item())
            if render_level >= max_available_level:
                visible_mask = torch.ones(len(multigrid_gaussians.levels), dtype=torch.bool, device=device)
                visible_indices = torch.arange(len(multigrid_gaussians.levels), device=device)
            else:
                visible_mask = multigrid_gaussians.set_visible_mask(render_level)
                visible_indices = torch.where(visible_mask)[0]
        else:
            return None
        
        if len(visible_indices) == 0:
            return None
        
        # Get visible splats
        visible_means = multigrid_gaussians.splats["means"][visible_indices]  # [N_visible, 3]
        visible_quats = multigrid_gaussians.splats["quats"][visible_indices]  # [N_visible, 4]
        visible_scales = multigrid_gaussians.splats["scales"][visible_indices]  # [N_visible, 3]
        visible_opacities = multigrid_gaussians.splats["opacities"][visible_indices]  # [N_visible,]
        
        # Prepare for projection
        means = visible_means  # [N_visible, 3]
        quats = visible_quats  # [N_visible, 4]
        scales = torch.exp(visible_scales)  # [N_visible, 3]
        opacities = torch.sigmoid(visible_opacities)  # [N_visible,]
        
        # Convert camtoworlds to viewmats (world-to-camera)
        # camtoworlds: [1, 4, 4] -> viewmats: [1, 4, 4] (inverse)
        viewmats = torch.inverse(camtoworlds)  # [1, 4, 4]
        # Add camera dimension: [1, 4, 4] -> [1, 1, 4, 4] for fully_fused_projection
        viewmats = viewmats.unsqueeze(1)  # [1, 1, 4, 4]
        # Add camera dimension to Ks: [1, 3, 3] -> [1, 1, 3, 3]
        Ks = Ks.unsqueeze(1)  # [1, 1, 3, 3]
        
        # Project to 2D (without rasterization)
        # fully_fused_projection returns: (radii, means2d, depths, conics, compensations)
        radii, means2d, depths, conics, compensations = fully_fused_projection(
            means=means.unsqueeze(0),  # [1, N_visible, 3]
            covars=None,  # Use quats/scales instead
            quats=quats.unsqueeze(0),  # [1, N_visible, 4]
            scales=scales.unsqueeze(0),  # [1, N_visible, 3]
            viewmats=viewmats,  # [1, 1, 4, 4]
            Ks=Ks,  # [1, 1, 3, 3]
            width=width,
            height=height,
            eps2d=0.3,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=0.0,
            packed=False,
            sparse_grad=False,
            calc_compensations=False,
            camera_model=self.cfg.camera_model,
            opacities=opacities.unsqueeze(0),  # [1, N_visible]
        )
        
        # Remove batch and camera dimensions: [1, 1, N_visible, ...] -> [N_visible, ...]
        means2d = means2d[0, 0]  # [N_visible, 2]
        conics = conics[0, 0]  # [N_visible, 3]
        radii = radii[0, 0]  # [N_visible, 2]
        opacities_proj = opacities  # [N_visible,] (already sigmoid)
        
        # Filter by valid radii
        valid = (radii > 0).all(dim=-1)  # [N_visible,]
        if not valid.any():
            return None
        
        valid_indices = visible_indices[valid]
        means2d_valid = means2d[valid]
        conics_valid = conics[valid]
        opacities_proj_valid = opacities_proj[valid]
        means_valid = means[valid]  # [N_valid, 3] - valid 3D means for SH evaluation
        
        # Evaluate SH colors
        try:
            if multigrid_gaussians.cfg.app_opt:
                visible_features = multigrid_gaussians.splats["features"][valid_indices]
                visible_colors = multigrid_gaussians.splats["colors"][valid_indices]
                # Evaluate colors using app_module
                if image_ids is not None:
                    image_ids = image_ids.to(device)
                    dirs = means_valid - camtoworlds[0, :3, 3]  # [N_valid, 3]
                    colors = multigrid_gaussians.app_module(
                        features=visible_features,
                        embed_ids=image_ids,
                        dirs=dirs,
                        sh_degree=sh_degree,
                    )
                    colors = colors + visible_colors
                    colors = torch.sigmoid(colors)
                else:
                    colors = torch.sigmoid(visible_colors)
            else:
                visible_sh0 = multigrid_gaussians.splats["sh0"][valid_indices]  # [N_valid, 1, 3]
                visible_shN = multigrid_gaussians.splats["shN"][valid_indices]  # [N_valid, K-1, 3]
                
                # Evaluate SH colors
                from gsplat.sh import eval_sh
                dirs = means_valid - camtoworlds[0, :3, 3]  # [N_valid, 3]
                dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-10)  # Normalize
                sh_coeffs = torch.cat([visible_sh0, visible_shN], dim=1)  # [N_valid, K, 3]
                colors = eval_sh(sh_degree, sh_coeffs, dirs)  # [N_valid, 3]
                colors = torch.sigmoid(colors)
        except:
            # Fallback: use sh0 only
            visible_sh0 = multigrid_gaussians.splats["sh0"][valid_indices]  # [N_valid, 1, 3]
            colors = torch.sigmoid(visible_sh0.squeeze(1))  # [N_valid, 3]
        
        # Use filtered values
        means2d = means2d_valid
        conics = conics_valid
        opacities_proj = opacities_proj_valid
        
        # Convert conics to covars2d
        a = conics[:, 0]
        b = conics[:, 1]
        c = conics[:, 2]
        det_inv = (a * c - b * b).clamp_min(1e-12)
        inv_det = 1.0 / det_inv
        covar00 = c * inv_det
        covar01 = -b * inv_det
        covar11 = a * inv_det
        covars2d = torch.stack([covar00, covar01, covar01, covar11], dim=-1).view(-1, 2, 2)
        
        # Create index map
        index_map = torch.full(
            (len(multigrid_gaussians.splats["means"]),), -1, device=device, dtype=torch.long
        )
        index_map[valid_indices] = torch.arange(len(valid_indices), device=device)
        
        return {
            "indices": valid_indices,
            "index_map": index_map,
            "means2d": means2d,
            "covars2d": covars2d,
            "det_inv": det_inv,
            "opacities": opacities_proj,
            "colors": colors,
            "render_level": render_level,
        }

    def _extract_projection_data(self, info: Dict) -> Optional[Dict]:
        """Extract per-gaussian 2D projection data from rasterization info or projection data."""
        if info is None:
            return None
        
        # If projection_data is already available (from _get_projection_only), return it directly
        if "projection_data" in info:
            return info["projection_data"]
        
        required_keys = ["means2d", "conics", "opacities", "colors", "radii", "visible_mask"]
        for key in required_keys:
            if info.get(key) is None:
                return None

        visible_mask = info["visible_mask"]
        if visible_mask is None or visible_mask.numel() == 0:
            return None
        visible_indices = torch.where(visible_mask)[0]
        if visible_indices.numel() == 0:
            return None

        means2d = info["means2d"]
        conics = info["conics"]
        opacities = info["opacities"]
        colors = info["colors"]
        radii = info["radii"]
        gaussian_ids = info.get("gaussian_ids", None)

        device = means2d.device

        if gaussian_ids is not None:
            # Packed: per-visible gaussian indices with nnz entries
            valid = (radii > 0).all(dim=-1)
            gaussian_ids = gaussian_ids[valid]
            if gaussian_ids.numel() == 0:
                return None

            means2d = means2d[valid]
            conics = conics[valid]
            opacities = opacities[valid]
            colors = colors[valid]

            num_visible = visible_indices.numel()
            counts = torch.zeros(num_visible, device=device, dtype=torch.float32)
            ones = torch.ones_like(gaussian_ids, dtype=torch.float32)
            counts.scatter_add_(0, gaussian_ids, ones)

            sums_means = torch.zeros(num_visible, 2, device=device, dtype=means2d.dtype)
            sums_means.scatter_add_(0, gaussian_ids.unsqueeze(-1).expand(-1, 2), means2d)
            means2d = sums_means / counts.clamp_min(1).unsqueeze(-1)

            sums_conics = torch.zeros(num_visible, 3, device=device, dtype=conics.dtype)
            sums_conics.scatter_add_(0, gaussian_ids.unsqueeze(-1).expand(-1, 3), conics)
            conics = sums_conics / counts.clamp_min(1).unsqueeze(-1)

            sums_opacities = torch.zeros(num_visible, device=device, dtype=opacities.dtype)
            sums_opacities.scatter_add_(0, gaussian_ids, opacities)
            opacities = sums_opacities / counts.clamp_min(1)

            color_dim = colors.shape[-1]
            sums_colors = torch.zeros(num_visible, color_dim, device=device, dtype=colors.dtype)
            sums_colors.scatter_add_(
                0, gaussian_ids.unsqueeze(-1).expand(-1, color_dim), colors
            )
            colors = sums_colors / counts.clamp_min(1).unsqueeze(-1)

            valid_visible = counts > 0
            visible_indices = visible_indices[valid_visible]
            means2d = means2d[valid_visible]
            conics = conics[valid_visible]
            opacities = opacities[valid_visible]
            colors = colors[valid_visible]
        else:
            # Not packed: per-camera dense buffers
            if means2d.dim() == 3:
                means2d = means2d.mean(dim=0)
                conics = conics.mean(dim=0)
                opacities = opacities.mean(dim=0)
                colors = colors.mean(dim=0)
                valid_visible = (radii > 0).all(dim=-1).any(dim=0)
            else:
                valid_visible = (radii > 0).all(dim=-1)

            if valid_visible.numel() != visible_indices.numel():
                return None
            if not valid_visible.any():
                return None

            visible_indices = visible_indices[valid_visible]
            means2d = means2d[valid_visible]
            conics = conics[valid_visible]
            opacities = opacities[valid_visible]
            colors = colors[valid_visible]

        # Convert conics (inverse covariances) to covariances
        a = conics[:, 0]
        b = conics[:, 1]
        c = conics[:, 2]
        det_inv = (a * c - b * b).clamp_min(1e-12)
        inv_det = 1.0 / det_inv
        covar00 = c * inv_det
        covar01 = -b * inv_det
        covar11 = a * inv_det
        covars2d = torch.stack([covar00, covar01, covar01, covar11], dim=-1).view(-1, 2, 2)

        index_map = torch.full(
            (len(self.splats["means"]),), -1, device=device, dtype=torch.long
        )
        index_map[visible_indices] = torch.arange(
            len(visible_indices), device=device
        )

        return {
            "indices": visible_indices,
            "index_map": index_map,
            "means2d": means2d,
            "covars2d": covars2d,
            "det_inv": det_inv,
            "opacities": opacities,
            "colors": colors,
        }

    def _compute_hierarchy_consistency_loss(
        self,
        target_level: int,
        image_multigrid_max_level: int,
        direction: Literal["downwards", "upwards"],
        info_current: Optional[Dict] = None,
        info_other: Optional[Dict] = None,
        children_indices: Optional[torch.Tensor] = None,
        parent_ids: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute hierarchy consistency loss based on direction.
        
        CHANGED: Direction-based regularization:
        - downwards (fine->coarse): Parents are regularized
          * Compute expected parent parameters from children's actual parameters
          * Compare with parent's actual parameters
          * Children are detached (gradient cut), parents receive gradients
        - upwards (coarse->fine, solve): Children are regularized
          * Compute expected children parameters from parent's actual parameters
          * Compare with children's actual parameters
          * Parents are detached (gradient cut), children receive gradients
        
        Only processes gaussians at target_level to match current training level.
        
        Args:
            target_level: Current training level (only gaussians at this level are processed)
            image_multigrid_max_level: Maximum level for image multigrid
            direction: "downwards" or "upwards" - determines which nodes are regularized
            children_indices: Optional precomputed children indices [K,]. If None, will be computed.
            parent_ids: Optional precomputed parent IDs for children [K,]. If None, will be computed.
        
        Returns:
            hierarchy_loss: Scalar loss value
            losses_dict: Dictionary of individual loss components
        """
        device = self.device
        empty_losses_dict = {
                        "m": torch.tensor(0.0, device=device),
                        "c": torch.tensor(0.0, device=device),
                        "o": torch.tensor(0.0, device=device),
                        "s0": torch.tensor(0.0, device=device),
                        "sN": torch.tensor(0.0, device=device),
                        "w_onehot": torch.tensor(0.0, device=device),
                    }
        # if upwards and target level is max level, return 0
        if direction == "upwards" and target_level == image_multigrid_max_level:

            return torch.tensor(0.0, device=device), empty_losses_dict

        # Profile timing for detailed breakdown
        cfg = self.cfg
        profile_detail = hasattr(cfg, 'profile_timing') and cfg.profile_timing
        timings = {}
        
        if profile_detail:
            torch.cuda.synchronize()
            t0 = time.time()

        multigrid_gaussians = self.multigrid_gaussians
        
        # Get hierarchical structure
        levels = multigrid_gaussians.levels  # [N,]
        parent_indices = multigrid_gaussians.parent_indices  # [N,]
        
        N = len(levels)
        if N == 0:

            return torch.tensor(0.0, device=device), empty_losses_dict
        
        # Compute children_indices and parent_ids if not provided
        if profile_detail:
            torch.cuda.synchronize()
            t1 = time.time()
            timings['init'] = (t1 - t0) * 1000
        if children_indices is None or parent_ids is None:
            cache_for_direction = self._hierarchy_index_cache[direction]
            cached_data = cache_for_direction.get(target_level)
            if cached_data is not None:
                # Use cached indices and mappings
                if isinstance(cached_data, tuple) and len(cached_data) == 5:
                    # Extended cache: (children_indices, parent_ids, valid_parent_ids, parent_id_to_valid_idx, mapped_parent_ids)
                    children_indices, parent_ids, valid_parent_ids, parent_id_to_valid_idx, mapped_parent_ids = cached_data
                    num_valid_parents = len(valid_parent_ids)
                else:
                    # Legacy cache: (children_indices, parent_ids)
                    children_indices, parent_ids = cached_data
                    valid_parent_ids = None
                    parent_id_to_valid_idx = None
                    mapped_parent_ids = None
                    num_valid_parents = None
            else:
                # CHANGED: Filter by target_level - only process gaussians at target_level
                target_level_mask = (levels == target_level)
                if not target_level_mask.any():
                    return torch.tensor(0.0, device=device), empty_losses_dict
                
                if direction == "downwards":
                    # Downwards: target_level-1 (parents)와 target_level (children) 이용
                    # target_level-1의 gaussians가 parents, target_level의 gaussians가 children
                    parent_level = target_level - 1
                    if parent_level < 0:
                        return torch.tensor(0.0, device=device), empty_losses_dict
                    
                    # Find gaussians at parent_level and target_level
                    parent_level_mask = (levels == parent_level)
                    if not parent_level_mask.any():
                        return torch.tensor(0.0, device=device), empty_losses_dict
                    
                    # Children are at target_level
                    children_mask = target_level_mask  # [N,]
                    children_indices = torch.where(children_mask)[0]  # [K,] - children at target_level
                    if len(children_indices) == 0:
                        return torch.tensor(0.0, device=device), empty_losses_dict
                    
                    # Get parent_ids for children
                    parent_ids = parent_indices[children_indices]  # [K,] - parent index for each child
                    
                    # Filter: only keep children whose parents are at parent_level
                    parent_levels_of_children = levels[parent_ids]  # [K,]
                    valid_children_mask = (parent_ids >= 0) & (parent_ids < N) & (parent_levels_of_children == parent_level)
                    if not valid_children_mask.all():
                        children_indices = children_indices[valid_children_mask]
                        parent_ids = parent_ids[valid_children_mask]
                        if len(children_indices) == 0:
                            return torch.tensor(0.0, device=device), empty_losses_dict
                    
                else:  # upwards
                    # Upwards: target_level (parents)와 target_level+1 (children) 이용
                    # target_level의 gaussians가 parents, target_level+1의 gaussians가 children
                    child_level = target_level + 1
                    
                    # Find gaussians at target_level and child_level
                    child_level_mask = (levels == child_level)
                    if not child_level_mask.any():
                        return torch.tensor(0.0, device=device), empty_losses_dict
                    
                    # Children are at target_level+1
                    children_mask = child_level_mask  # [N,]
                    children_indices = torch.where(children_mask)[0]  # [K,] - children at target_level+1
                    if len(children_indices) == 0:
                        return torch.tensor(0.0, device=device), empty_losses_dict
                    
                    # Get parent_ids for children
                    parent_ids = parent_indices[children_indices]  # [K,] - parent index for each child
                    
                    # Filter: only keep children whose parents are at target_level
                    parent_levels_of_children = levels[parent_ids]  # [K,]
                    valid_children_mask = (parent_ids >= 0) & (parent_ids < N) & (parent_levels_of_children == target_level)
                    if not valid_children_mask.all():
                        children_indices = children_indices[valid_children_mask]
                        parent_ids = parent_ids[valid_children_mask]
                        if len(children_indices) == 0:
                            return torch.tensor(0.0, device=device), empty_losses_dict
                
                # Compute mappings for caching
                unique_parent_ids = torch.unique(parent_ids)  # [M,] where M <= K
                unique_parent_ids = unique_parent_ids[unique_parent_ids >= 0]  # Filter out -1
                if len(unique_parent_ids) == 0:
                    return torch.tensor(0.0, device=device), empty_losses_dict
                
                max_parent_id = unique_parent_ids.max().item()
                parent_id_to_valid_idx = torch.full((max_parent_id + 1,), -1, dtype=torch.long, device=device)
                # Will be filled after weight computation
                valid_parent_ids = None
                mapped_parent_ids = None
                num_valid_parents = None
                
                # Store basic cache (will be extended after weight computation)
                cache_for_direction[target_level] = (children_indices, parent_ids, valid_parent_ids, parent_id_to_valid_idx, mapped_parent_ids)
        
        if profile_detail:
            torch.cuda.synchronize()
            t2 = time.time()
            timings['index_compute'] = (t2 - t1) * 1000
        
        # Validate provided indices
        if len(children_indices) == 0:
            return torch.tensor(0.0, device=device), empty_losses_dict
        
        # CHANGED: Use individual parameters directly (no need for get_splats)
        # Individual parameters are stored directly in self.splats
        actual_splats = multigrid_gaussians.splats
        
        # COMMON: Extract children parameters (detach will be applied later based on direction)
        children_means = actual_splats["means"][children_indices]  # [K, 3]
        children_scales = actual_splats["scales"][children_indices]  # [K, 3]
        children_quats = actual_splats["quats"][children_indices]  # [K, 4]
        children_opacities = actual_splats["opacities"][children_indices]  # [K,]
        children_sh0 = actual_splats["sh0"][children_indices]  # [K, 1, 3] - keep shape for consistency
        
        if profile_detail:
            torch.cuda.synchronize()
            t3 = time.time()
            timings['extract_children'] = (t3 - t2) * 1000
        
        # COMMON: Children -> Parent aggregation (same for both directions)
        # Compute weights for aggregation using 2D projection (WITH gradients for one-hot regularization)
        if profile_detail:
            torch.cuda.synchronize()
            t_weight_start = time.time()
        
        # Always compute children_scales_exp for 3D covariance calculation
        children_scales_exp = torch.exp(children_scales).clamp_min(1e-6).clamp_max(1e6)  # [K, 3]
        
        if profile_detail:
            torch.cuda.synchronize()
            t_scale_exp = time.time()
            timings['weight_scale_exp'] = (t_scale_exp - t_weight_start) * 1000
        
        # Extract 2D projection data for children
        # downwards: children are at target_level, use info_current
        # upwards: children are at target_level+1, use info_other
        if direction == "downwards":
            proj_data = self._extract_projection_data(info_current) if info_current is not None else None
        else:  # upwards
            proj_data = self._extract_projection_data(info_other) if info_other is not None else None
        
        if proj_data is not None:
            # Use 2D projection data for weight calculation
            index_map = proj_data["index_map"]  # [N,] maps gaussian index to projection data index
            children_proj_indices = index_map[children_indices]  # [K,]
            valid_proj_mask = children_proj_indices >= 0  # [K,]
            
            if valid_proj_mask.any():
                # Get 2D covariance determinants for valid children
                det_inv = proj_data["det_inv"]  # [N_proj,]
                children_det_inv = det_inv[children_proj_indices[valid_proj_mask]]  # [K_valid,]
                # 2D area = 1/sqrt(det_inv) = sqrt(det)
                children_areas_2d = torch.rsqrt(children_det_inv.clamp_min(1e-12))  # [K_valid,]
                
                # Get opacities from projection data (already in linear space)
                opacities_proj = proj_data["opacities"]  # [N_proj,]
                children_opacities_proj = opacities_proj[children_proj_indices[valid_proj_mask]]  # [K_valid,]
                
                # Compute unnormalized weights: o * area_2d
                unnormalized_weights_valid = children_opacities_proj * children_areas_2d  # [K_valid,]
                
                # Fill in weights for all children (invalid ones use 3D fallback)
                unnormalized_weights = torch.zeros(len(children_indices), device=device, dtype=children_opacities.dtype)
                unnormalized_weights[valid_proj_mask] = unnormalized_weights_valid
                
                # Fallback to 3D for invalid children
                if not valid_proj_mask.all():
                    scale_prod = children_scales_exp[~valid_proj_mask].prod(dim=-1).clamp_min(1e-12)  # [K_invalid,]
                    children_covar_det_pow13 = torch.pow(scale_prod, 2.0 / 3.0)  # [K_invalid,]
                    children_opacities_clamped = torch.sigmoid(children_opacities[~valid_proj_mask]).clamp_min(1e-8)  # [K_invalid,]
                    unnormalized_weights[~valid_proj_mask] = children_opacities_clamped * children_covar_det_pow13
            else:
                # Fallback to 3D if no valid projection data
                scale_prod = children_scales_exp.prod(dim=-1).clamp_min(1e-12)  # [K,]
                children_covar_det_pow13 = torch.pow(scale_prod, 2.0 / 3.0)  # [K,] - (det)^(1/3) for 3D Gaussian
                children_opacities_clamped = torch.sigmoid(children_opacities).clamp_min(1e-8)  # [K,]
                unnormalized_weights = children_opacities_clamped * children_covar_det_pow13  # [K,]
        else:
            # Fallback to 3D if no projection data available
            scale_prod = children_scales_exp.prod(dim=-1).clamp_min(1e-12)  # [K,]
            children_covar_det_pow13 = torch.pow(scale_prod, 2.0 / 3.0)  # [K,] - (det)^(1/3) for 3D Gaussian
            children_opacities_clamped = torch.sigmoid(children_opacities).clamp_min(1e-8)  # [K,]
            unnormalized_weights = children_opacities_clamped * children_covar_det_pow13  # [K,]
        
        if profile_detail:
            torch.cuda.synchronize()
            t_weight_calc = time.time()
            timings['weight_calc'] = (t_weight_calc - t_scale_exp) * 1000
        
        # OPTIMIZED: Use cached mappings if available, otherwise compute
        if valid_parent_ids is None or mapped_parent_ids is None:
            if profile_detail:
                torch.cuda.synchronize()
                t_mapping_start = time.time()
            
            # Need to compute mappings
            unique_parent_ids = torch.unique(parent_ids)  # [M,] where M <= K
            unique_parent_ids = unique_parent_ids[unique_parent_ids >= 0]  # Filter out -1
            if len(unique_parent_ids) == 0:
                return torch.tensor(0.0, device=device), empty_losses_dict
            
            if profile_detail:
                torch.cuda.synchronize()
                t_unique = time.time()
                timings['weight_unique'] = (t_unique - t_mapping_start) * 1000
            
            # OPTIMIZED: Compute sum of weights per parent using scatter_add (vectorized)
            # Only allocate for max(parent_ids) + 1 instead of full N
            max_parent_id = unique_parent_ids.max().item()
            parent_weight_sums = torch.zeros(max_parent_id + 1, device=device, dtype=unnormalized_weights.dtype)
            parent_weight_sums.scatter_add_(0, parent_ids, unnormalized_weights)
            
            if profile_detail:
                torch.cuda.synchronize()
                t_weight_sum = time.time()
                timings['weight_sum'] = (t_weight_sum - t_unique) * 1000
            
            # Filter out parents with no children (weight_sum = 0)
            valid_parent_ids = unique_parent_ids[parent_weight_sums[unique_parent_ids] > 1e-8]
            if len(valid_parent_ids) == 0:
                return torch.tensor(0.0, device=device), empty_losses_dict
            
            num_valid_parents = len(valid_parent_ids)
            
            if profile_detail:
                torch.cuda.synchronize()
                t_filter = time.time()
                timings['weight_filter'] = (t_filter - t_weight_sum) * 1000
            
            # OPTIMIZED: Create mapping from parent_id to index in valid_parent_indices
            if parent_id_to_valid_idx is None or parent_id_to_valid_idx.shape[0] <= max_parent_id:
                parent_id_to_valid_idx = torch.full((max_parent_id + 1,), -1, dtype=torch.long, device=device)
            else:
                parent_id_to_valid_idx.fill_(-1)
            parent_id_to_valid_idx[valid_parent_ids] = torch.arange(num_valid_parents, device=device)
            
            # Map parent_ids to valid indices
            valid_parent_ids_mask = parent_id_to_valid_idx[parent_ids] >= 0  # [K,]
            if not valid_parent_ids_mask.all():
                # Filter children to only those with valid parents
                children_indices = children_indices[valid_parent_ids_mask]
                parent_ids = parent_ids[valid_parent_ids_mask]
                children_means = children_means[valid_parent_ids_mask]
                children_scales = children_scales[valid_parent_ids_mask]
                children_scales_exp = children_scales_exp[valid_parent_ids_mask]
                children_quats = children_quats[valid_parent_ids_mask]
                children_opacities = children_opacities[valid_parent_ids_mask]
                children_sh0 = children_sh0[valid_parent_ids_mask]
                unnormalized_weights = unnormalized_weights[valid_parent_ids_mask]
                K = len(children_indices)
                if K == 0:
                    return torch.tensor(0.0, device=device), empty_losses_dict
            
            # Map parent_ids to valid indices (0..num_valid_parents-1)
            mapped_parent_ids = parent_id_to_valid_idx[parent_ids]  # [K,]
            
            if profile_detail:
                torch.cuda.synchronize()
                t_mapping_end = time.time()
                timings['weight_mapping'] = (t_mapping_end - t_filter) * 1000
            
            # Update cache with computed mappings
            cache_for_direction[target_level] = (children_indices, parent_ids, valid_parent_ids, parent_id_to_valid_idx, mapped_parent_ids)
        else:
            # Use cached mappings - need to compute parent_weight_sums for opacity calculation
            num_valid_parents = len(valid_parent_ids)
            max_parent_id = valid_parent_ids.max().item()
            parent_weight_sums = torch.zeros(max_parent_id + 1, device=device, dtype=unnormalized_weights.dtype)
            parent_weight_sums.scatter_add_(0, parent_ids, unnormalized_weights)
        
        # OPTIMIZED: Normalize weights per parent using precomputed sums
        if profile_detail:
            torch.cuda.synchronize()
            t_normalize_start = time.time()
        
        parent_weight_sum_per_child = parent_weight_sums[parent_ids]  # [K,]
        weights = unnormalized_weights / (parent_weight_sum_per_child + 1e-20)  # [K,]
        
        if profile_detail:
            torch.cuda.synchronize()
            t_weight_end = time.time()
            timings['weight_normalize'] = (t_weight_end - t_normalize_start) * 1000
            if 'weight_compute' not in timings:
                timings['weight_compute'] = (t_weight_end - t_weight_start) * 1000
        
        # Compute children's covariance matrices after filtering
        children_covars, _ = quat_scale_to_covar_preci(
            quats=children_quats,
            scales=children_scales_exp,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )  # [K, 3, 3]
        
        if profile_detail:
            torch.cuda.synchronize()
            t5 = time.time()
            timings['children_covar'] = (t5 - t_weight_end) * 1000
            
        # OPTIMIZED: Use num_valid_parents-sized tensors instead of N-sized
        # Aggregate children to get expected parent parameters
        if profile_detail:
            torch.cuda.synchronize()
            t_agg_start = time.time()
        
        expected_means = torch.zeros(num_valid_parents, 3, device=device, dtype=children_means.dtype)
        expected_covars = torch.zeros(num_valid_parents, 3, 3, device=device, dtype=children_covars.dtype)
        expected_opacities = torch.zeros(num_valid_parents, device=device, dtype=children_opacities.dtype)
        expected_sh0 = torch.zeros(num_valid_parents, 1, 3, device=device, dtype=children_sh0.dtype)
        
        # Mean: μ^(l+1) = Σ w_i^(l) μ_i^(l) (Equation 3)
        # weights are already normalized per parent: sum(weights for parent p) = 1
        # So scatter_add(weighted_means) gives the correct weighted average
        weighted_means = children_means * weights.unsqueeze(-1)  # [K, 3]
        expected_means.scatter_add_(0, mapped_parent_ids.unsqueeze(-1).expand(-1, 3), weighted_means)
        
        if profile_detail:
            torch.cuda.synchronize()
            t_means = time.time()
            timings['agg_means'] = (t_means - t_agg_start) * 1000
        
        # Covariance: Σ^(l+1) = Σ w_i^(l) (Σ_i^(l) + (μ_i^(l) - μ^(l+1)) (μ_i^(l) - μ^(l+1))^T) (Equation 4)
        # Get parent means per child (after scatter_add is complete)
        parent_means_per_child = expected_means[mapped_parent_ids]  # [K, 3]
        
        mean_diffs = children_means - parent_means_per_child  # [K, 3]
        mean_diff_outer = torch.bmm(
            mean_diffs.unsqueeze(2), mean_diffs.unsqueeze(1)
        )  # [K, 3, 3]
        
        if profile_detail:
            torch.cuda.synchronize()
            t_mean_diff = time.time()
            timings['agg_mean_diff'] = (t_mean_diff - t_means) * 1000
        
        children_covars_with_mean_diff = children_covars + mean_diff_outer  # [K, 3, 3]
        weighted_covars = children_covars_with_mean_diff * weights.view(-1, 1, 1)  # [K, 3, 3]
        
        # OPTIMIZED: Flatten covars and concat with sh0 for unified scatter_add (means already computed)
        # covars: [K, 9] (flattened), sh0: [K, 3]
        # Total: [K, 12] (covars 9 + sh0 3)
        weighted_covars_flat = weighted_covars.flatten(start_dim=1)  # [K, 9]
        weighted_sh0_flat = children_sh0.squeeze(1) * weights.unsqueeze(-1)  # [K, 3] (sh0 is [K, 1, 3])
        
        # Concat: [K, 9+3] = [K, 12] (means already computed separately)
        weighted_params_concat = torch.cat([weighted_covars_flat, weighted_sh0_flat], dim=1)  # [K, 12]
        
        # Unified scatter_add for covars and sh0 (means already done)
        expected_params_concat = torch.zeros(num_valid_parents, 12, device=device, dtype=weighted_params_concat.dtype)
        expected_params_concat.scatter_add_(0, mapped_parent_ids.unsqueeze(-1).expand(-1, 12), weighted_params_concat)
        
        # Split back
        expected_covars_flat = expected_params_concat[:, :9]  # [num_valid_parents, 9]
        expected_sh0_flat = expected_params_concat[:, 9:12]  # [num_valid_parents, 3]
        expected_covars = expected_covars_flat.reshape(num_valid_parents, 3, 3)  # [num_valid_parents, 3, 3]
        expected_sh0 = expected_sh0_flat.unsqueeze(1)  # [num_valid_parents, 1, 3]
        
        if profile_detail:
            torch.cuda.synchronize()
            t_scatter = time.time()
            timings['agg_scatter'] = (t_scatter - t_mean_diff) * 1000
        
        # Make symmetric (covariance matrices must be symmetric)
        expected_covars = (expected_covars + expected_covars.transpose(-2, -1)) / 2.0
        
        # Opacity: Use build_hierarchy.py method - falloff = (∑_i w'_i) / S_p
        # Convert falloff to opacity in linear space: o = falloff / (1 + falloff)
        # This is more stable than log space for loss computation
        # OPTIMIZED: Reuse parent_weight_sums instead of recomputing
        parent_unnormalized_weight_sums = parent_weight_sums[valid_parent_ids]  # [num_valid_parents,]
        
        # Compute merged surface area per parent using 2D projection
        # downwards: parents are at target_level-1, use info_other
        # upwards: parents are at target_level, use info_current
        if direction == "downwards":
            parent_proj_data = self._extract_projection_data(info_other) if info_other is not None else None
        else:  # upwards
            parent_proj_data = self._extract_projection_data(info_current) if info_current is not None else None
        
        if parent_proj_data is not None:
            # Use 2D projection data for parent surface area
            index_map = parent_proj_data["index_map"]  # [N,]
            parent_proj_indices = index_map[valid_parent_ids]  # [num_valid_parents,]
            valid_parent_proj_mask = parent_proj_indices >= 0  # [num_valid_parents,]
            
            if valid_parent_proj_mask.any():
                # Get 2D covariance determinants for valid parents
                det_inv = parent_proj_data["det_inv"]  # [N_proj,]
                parent_det_inv = det_inv[parent_proj_indices[valid_parent_proj_mask]]  # [num_valid_parents_valid,]
                # 2D area = 1/sqrt(det_inv) = sqrt(det)
                parent_areas_2d_valid = torch.rsqrt(parent_det_inv.clamp_min(1e-12))  # [num_valid_parents_valid,]
                
                # Fill in areas for all parents (invalid ones use 3D fallback)
                expected_surface_areas = torch.zeros(num_valid_parents, device=device, dtype=parent_unnormalized_weight_sums.dtype)
                expected_surface_areas[valid_parent_proj_mask] = parent_areas_2d_valid
                
                # For invalid parents, fallback to 3D
                if not valid_parent_proj_mask.all():
                    expected_covar_dets_3d = torch.det(expected_covars[~valid_parent_proj_mask]).clamp_min(1e-8)  # [num_invalid,]
                    expected_surface_areas_3d = torch.pow(expected_covar_dets_3d, 1.0/3.0)  # [num_invalid,]
                    expected_surface_areas[~valid_parent_proj_mask] = expected_surface_areas_3d
            else:
                # Fallback to 3D if no valid projection data
                expected_covar_dets = torch.det(expected_covars).clamp_min(1e-8)  # [num_valid_parents,]
                expected_surface_areas = torch.pow(expected_covar_dets, 1.0/3.0)  # [num_valid_parents,]
        else:
            # Fallback to 3D if no projection data available
            expected_covar_dets = torch.det(expected_covars).clamp_min(1e-8)  # [num_valid_parents,]
            expected_surface_areas = torch.pow(expected_covar_dets, 1.0/3.0)  # [num_valid_parents,]
        
        # Falloff = (∑_i w'_i) / S_p per parent
        falloff = parent_unnormalized_weight_sums / expected_surface_areas.clamp_min(1e-12)  # [num_valid_parents,]
        # Convert falloff to opacity in linear space: o = falloff / (1 + falloff)
        # This is more stable than log space for loss computation
        expected_opacities_linear = falloff / (falloff + 1.0)  # [num_valid_parents,]
        
        if profile_detail:
            torch.cuda.synchronize()
            t_agg_end = time.time()
            timings['agg_opacity'] = (t_agg_end - t_scatter) * 1000
            timings['aggregation'] = (t_agg_end - t_agg_start) * 1000
        
        # COMMON: Extract parent parameters for valid parents only
        parent_means_actual = actual_splats["means"][valid_parent_ids]  # [num_valid_parents, 3]
        parent_scales_actual = actual_splats["scales"][valid_parent_ids]  # [num_valid_parents, 3]
        parent_quats_actual = actual_splats["quats"][valid_parent_ids]  # [num_valid_parents, 4]
        parent_opacities_actual = actual_splats["opacities"][valid_parent_ids]  # [num_valid_parents,] (log space)
        parent_sh0_actual = actual_splats["sh0"][valid_parent_ids]  # [num_valid_parents, 1, 3]
        
        # Convert parent opacities to linear space for loss computation
        parent_opacities_linear_actual = torch.sigmoid(parent_opacities_actual)  # [num_valid_parents,] (linear space)
        
        if profile_detail:
            torch.cuda.synchronize()
            t7 = time.time()
            timings['extract_parent'] = (t7 - t_agg_end) * 1000
        
        # COMMON: Expected values are already in num_valid_parents size
        expected_means_parents = expected_means  # [num_valid_parents, 3]
        expected_covars_parents = expected_covars  # [num_valid_parents, 3, 3]
        expected_opacities_linear_parents = expected_opacities_linear  # [num_valid_parents,] (linear space)
        expected_sh0_parents = expected_sh0  # [num_valid_parents, 1, 3]

        # COMMON: Compute parent's actual covariance matrices
        parent_scales_exp = torch.exp(parent_scales_actual).clamp_min(1e-6).clamp_max(1e6)
        parent_covars_actual, _ = quat_scale_to_covar_preci(
            quats=parent_quats_actual,
            scales=parent_scales_exp,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )  # [num_valid_parents, 3, 3]
        
        if profile_detail:
            torch.cuda.synchronize()
            t8 = time.time()
            timings['parent_covar'] = (t8 - t7) * 1000
        
        # DIFFERENT: Apply detach based on direction (only difference between downwards/upwards)
        # IMPORTANT: Gradient flow analysis:
        # - downwards: expected is computed from children (children -> expected), so expected has gradients from children.
        #   We detach expected to prevent gradients from flowing back to children, so only parent receives gradients.
        # - upwards: expected is computed from children (children -> expected), so expected has gradients from children.
        #   We detach parent to prevent gradients from flowing to parent, so only children (via expected) receive gradients.
        if direction == "downwards":
            # Downwards: expected.detach() prevents gradients from flowing back to children
            # Only parent receives gradients (parent is updated to match expected from children)
            expected_means_parents = expected_means_parents.detach()
            expected_covars_parents = expected_covars_parents.detach()
            expected_opacities_linear_parents = expected_opacities_linear_parents.detach()
            expected_sh0_parents = expected_sh0_parents.detach()
        else:  # upwards
            # Upwards: parent.detach() prevents gradients from flowing to parent
            # Only children (via expected) receive gradients (children are updated to match expected from parent)
            parent_means_actual = parent_means_actual.detach()
            parent_covars_actual = parent_covars_actual.detach()
            parent_opacities_linear_actual = parent_opacities_linear_actual.detach()
            parent_sh0_actual = parent_sh0_actual.detach()
        
        # COMMON: Compute losses: expected (from children) vs actual (parent)
        # Note: shN loss is excluded as requested
        mean_loss = F.mse_loss(expected_means_parents, parent_means_actual)
        covar_diff = expected_covars_parents - parent_covars_actual
        covar_loss = (covar_diff ** 2).sum(dim=(1, 2)).mean()
        # Compare opacities in linear space (more stable than log space)
        opacity_loss = F.mse_loss(expected_opacities_linear_parents, parent_opacities_linear_actual)
        sh0_loss = F.mse_loss(expected_sh0_parents, parent_sh0_actual)
        shN_loss = torch.tensor(0.0, device=device)  # Excluded from loss computation
        
        # Weight one-hot regularization: minimize entropy to encourage sparse weights
        # For each parent, compute entropy of its children's weights
        # Lower entropy = more one-hot like (one child dominates)
        weight_onehot_loss = torch.tensor(0.0, device=device)
        if cfg.weight_onehot_lambda > 0.0:
            # VECTORIZED: Compute entropy for all parents at once
            # weights: [K,], mapped_parent_ids: [K,] where each value is in [0, num_valid_parents-1]
            # Entropy for parent p: -sum(w_i * log(w_i + eps)) where mapped_parent_ids[i] == p
            
            eps = 1e-10
            # Compute w * log(w + eps) for all weights: [K,]
            weight_log_weights = weights * torch.log(weights + eps)  # [K,]
            
            # Sum per parent using scatter_add: [num_valid_parents,]
            parent_entropies = torch.zeros(num_valid_parents, device=device, dtype=weights.dtype)
            parent_entropies.scatter_add_(0, mapped_parent_ids, weight_log_weights)
            parent_entropies = -parent_entropies  # Negate to get entropy
            
            # Count number of children per parent: [num_valid_parents,]
            parent_child_counts = torch.zeros(num_valid_parents, device=device, dtype=torch.long)
            parent_child_counts.scatter_add_(0, mapped_parent_ids, torch.ones_like(mapped_parent_ids))
            
            # Only consider parents with multiple children (entropy is 0 for single child)
            # Filter: keep only parents with parent_child_counts > 1
            multi_child_mask = parent_child_counts > 1
            if multi_child_mask.any():
                # Average entropy across parents with multiple children
                weight_onehot_loss = parent_entropies[multi_child_mask].mean()
        
        hierarchy_loss = mean_loss + covar_loss + opacity_loss + sh0_loss + cfg.weight_onehot_lambda * weight_onehot_loss
        
        if profile_detail:
            torch.cuda.synchronize()
            t9 = time.time()
            timings['loss_compute'] = (t9 - t8) * 1000
            timings['total'] = (t9 - t0) * 1000
            
            # Store timings for periodic reporting
            if not hasattr(self, '_consistency_timings'):
                self._consistency_timings = []
            self._consistency_timings.append(timings)
            
            # Print detailed breakdown every 100 steps
            if len(self._consistency_timings) % 100 == 0:
                # Average over last 100 steps
                avg_timings = {}
                for key in timings.keys():
                    avg_timings[key] = np.mean([t[key] for t in self._consistency_timings[-100:]])
                
                print(f"[Consistency Loss Profile] Step {len(self._consistency_timings)}:")
                print(f"  Init: {avg_timings.get('init', 0):.2f}ms")
                print(f"  Index compute: {avg_timings.get('index_compute', 0):.2f}ms")
                print(f"  Extract children: {avg_timings.get('extract_children', 0):.2f}ms")
                print(f"  Weight compute (total): {avg_timings.get('weight_compute', 0):.2f}ms")
                if 'weight_scale_exp' in avg_timings:
                    print(f"    - scale_exp: {avg_timings.get('weight_scale_exp', 0):.2f}ms")
                    print(f"    - weight_calc: {avg_timings.get('weight_calc', 0):.2f}ms")
                    print(f"    - unique: {avg_timings.get('weight_unique', 0):.2f}ms")
                    print(f"    - weight_sum: {avg_timings.get('weight_sum', 0):.2f}ms")
                    print(f"    - filter: {avg_timings.get('weight_filter', 0):.2f}ms")
                    print(f"    - mapping: {avg_timings.get('weight_mapping', 0):.2f}ms")
                    print(f"    - normalize: {avg_timings.get('weight_normalize', 0):.2f}ms")
                print(f"  Children covar: {avg_timings.get('children_covar', 0):.2f}ms")
                print(f"  Aggregation (total): {avg_timings.get('aggregation', 0):.2f}ms")
                if 'agg_means' in avg_timings:
                    print(f"    - means: {avg_timings.get('agg_means', 0):.2f}ms")
                    print(f"    - mean_diff: {avg_timings.get('agg_mean_diff', 0):.2f}ms")
                    print(f"    - scatter (unified): {avg_timings.get('agg_scatter', 0):.2f}ms")
                    print(f"    - opacity: {avg_timings.get('agg_opacity', 0):.2f}ms")
                print(f"  Extract parent: {avg_timings.get('extract_parent', 0):.2f}ms")
                print(f"  Parent covar: {avg_timings.get('parent_covar', 0):.2f}ms")
                print(f"  Loss compute: {avg_timings.get('loss_compute', 0):.2f}ms")
                print(f"  Total: {avg_timings.get('total', 0):.2f}ms")
                print(f"  K={len(children_indices)}, M={num_valid_parents}")
        
        losses_dict = {
            "m": mean_loss,
            "c": covar_loss,
            "o": opacity_loss,
            "s0": sh0_loss,
            "sN": shN_loss,
            "w_onehot": weight_onehot_loss,
        }


        return hierarchy_loss, losses_dict

    @torch.no_grad()
    def measure_metric_on_batch(
        self,
        data: Dict,
        target_level: int,
        step: int,
        tag: str,
        backgrounds: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Measure PSNR/SSIM/LPIPS on a single batch at a specific level."""
        cfg = self.cfg
        device = self.device

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
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

        if isinstance(data["image_id"], torch.Tensor):
            image_ids = data["image_id"].to(device)
        else:
            image_ids = torch.tensor(data["image_id"], device=device)

        masks = data["mask"].to(device) if "mask" in data else None
        if masks is not None and masks.dim() == 2:
            masks = masks.unsqueeze(0)

        original_height, original_width = pixels.shape[1:3]
        image_multigrid_max_level = cfg.max_level if cfg.max_level is not None else 1
        sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        colors, pixels_downsampled, alphas, _ = self.render_at_level(
            target_level=target_level,
            pixels_gt=pixels,
            original_height=original_height,
            original_width=original_width,
            camtoworlds=camtoworlds,
            Ks_original=Ks,
            image_multigrid_max_level=image_multigrid_max_level,
            sh_degree=sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            masks=masks,
            render_mode="RGB",
        )

        if backgrounds is None:
            if cfg.random_bkgd:
                backgrounds = torch.rand(1, 3, device=device)
            elif cfg.white_background:
                backgrounds = torch.ones(1, 3, device=device)
        if backgrounds is not None and alphas is not None:
            colors = colors + backgrounds * (1.0 - alphas)

        colors = torch.clamp(colors, 0.0, 1.0)
        pixels_downsampled = torch.clamp(pixels_downsampled, 0.0, 1.0)

        colors_p = colors.permute(0, 3, 1, 2)
        pixels_p = pixels_downsampled.permute(0, 3, 1, 2)

        psnr = self.psnr(colors_p, pixels_p).item()
        ssim = self.ssim(colors_p, pixels_p).item()
        lpips = self.lpips(colors_p, pixels_p).item()

        if self.world_rank == 0:
            print(
                f"[V-cycle Metric] {tag} | L{target_level} | step={step} | "
                f"PSNR={psnr:.3f} | SSIM={ssim:.4f} | LPIPS={lpips:.3f}"
            )

        return {"psnr": psnr, "ssim": ssim, "lpips": lpips}

    @torch.no_grad()
    def measure_metric_on_dataset(
        self,
        target_level: int,
        step: int,
        tag: str,
        backgrounds: Optional[Tensor] = None,
        stage: str = "val",
        print_metric: bool = True,
    ) -> Dict[str, float]:
        """Measure PSNR/SSIM/LPIPS on the full validation set at a specific level."""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )

        if backgrounds is None:
            if cfg.random_bkgd:
                backgrounds = torch.rand(1, 3, device=device)
            elif cfg.white_background:
                backgrounds = torch.ones(1, 3, device=device)

        metrics = defaultdict(list)
        for data in valloader:
            camtoworlds = data["camtoworld"].to(device)
            Ks_original = data["K"].to(device)
            image_data = data["image"].to(device) / 255.0

            if camtoworlds.dim() == 2:
                camtoworlds = camtoworlds.unsqueeze(0)
            if Ks_original.dim() == 2:
                Ks_original = Ks_original.unsqueeze(0)
            if image_data.dim() == 3:
                image_data = image_data.unsqueeze(0)

            if image_data.shape[-1] == 4:
                pixels_gt = image_data[..., :3]
            else:
                pixels_gt = image_data

            if isinstance(data["image_id"], torch.Tensor):
                image_ids = data["image_id"].to(device)
            else:
                image_ids = torch.tensor(data["image_id"], device=device)

            masks = data["mask"].to(device) if "mask" in data else None
            if masks is not None and masks.dim() == 2:
                masks = masks.unsqueeze(0)

            original_height, original_width = pixels_gt.shape[1:3]
            image_multigrid_max_level = cfg.max_level if cfg.max_level is not None else 1
            sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            colors, pixels_downsampled, alphas, _ = self.render_at_level(
                target_level=target_level,
                pixels_gt=pixels_gt,
                original_height=original_height,
                original_width=original_width,
                camtoworlds=camtoworlds,
                Ks_original=Ks_original,
                image_multigrid_max_level=image_multigrid_max_level,
                sh_degree=sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                masks=masks,
                render_mode="RGB",
            )

            if backgrounds is not None and alphas is not None:
                colors = colors + backgrounds * (1.0 - alphas)

            colors = torch.clamp(colors, 0.0, 1.0)
            pixels_downsampled = torch.clamp(pixels_downsampled, 0.0, 1.0)

            if world_rank == 0:
                colors_p = colors.permute(0, 3, 1, 2)
                pixels_p = pixels_downsampled.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if world_rank == 0:
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            if print_metric:
                print(
                    f"[V-cycle Metric] {tag} | L{target_level} | step={step} | "
                    f"PSNR={stats['psnr']:.3f} | SSIM={stats['ssim']:.4f} | LPIPS={stats['lpips']:.3f}"
                )
            record = {
                "tag": tag,
                "level": int(target_level),
                "step": int(step),
                "stage": stage,
                "psnr": stats["psnr"],
                "ssim": stats["ssim"],
                "lpips": stats["lpips"],
                "time": time.time(),
            }
            with open(self.metric_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            return stats
        return {}

    def _train_step(
        self,
        step: int,
        data: Dict,
        target_level: int,
        direction: Optional[Literal["downwards", "upwards"]] = None,
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
        camtoworlds_gt = data["camtoworld"].to(device)  # [4, 4] or [1, 4, 4]
        Ks = data["K"].to(device)  # [3, 3] or [1, 3, 3]
        
        # Ensure batch dimension exists for camtoworlds and Ks
        if camtoworlds_gt.dim() == 2:
            camtoworlds_gt = camtoworlds_gt.unsqueeze(0)  # [4, 4] -> [1, 4, 4]
        if Ks.dim() == 2:
            Ks = Ks.unsqueeze(0)  # [3, 3] -> [1, 3, 3]
        
        camtoworlds = camtoworlds_gt
        
        image_data_cpu = data["image"]  # [1, H, W, C] or [H, W, C] on CPU
        
        # Ensure batch dimension exists
        if image_data_cpu.dim() == 3:
            image_data_cpu = image_data_cpu.unsqueeze(0)  # [H, W, C] -> [1, H, W, C]
        
        image_data = image_data_cpu.to(device) / 255.0  # [1, H, W, C]
        
        # Get max_level for image multigrid downsample
        # Use cfg.max_level (not actual multigrid_gaussians.levels.max())
        # multigrid_gaussians.levels.max() is only used for V-cycle level selection
        image_multigrid_max_level = cfg.max_level if cfg.max_level is not None else 1
        
        # Store original dimensions before any downsampling
        original_height, original_width = image_data.shape[1:3]
        
        # Handle RGBA images: extract RGB and alpha separately
        if image_data.shape[-1] == 4:
            pixels = image_data[..., :3]  # [1, H, W, 3]
            pixels_alpha = image_data[..., 3:4]  # [1, H, W, 1]
            has_alpha = True
        else:
            pixels = image_data  # [1, H, W, 3]
            pixels_alpha = None
            has_alpha = False
        
        # image_id is int from dataset, convert to tensor
        if isinstance(data["image_id"], torch.Tensor):
            image_ids = data["image_id"].to(device)
        else:
            image_ids = torch.tensor(data["image_id"], device=device)
        
        # Prepare masks (will be downsampled in render_at_level if needed)
        masks_cpu = data["mask"] if "mask" in data else None
        masks = masks_cpu.to(device) if masks_cpu is not None else None
        if masks is not None and masks.dim() == 2:
            masks = masks.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        # Calculate downsample factor for current target_level (for points downsampling)
        # (1/level_resolution_factor)^(max_level - target_level)
        level_diff = image_multigrid_max_level - target_level
        downsample_factor = (1.0 / cfg.level_resolution_factor) ** level_diff
        downsample_factor = max(1, int(downsample_factor))
        
        if cfg.depth_loss:
            points = data["points"].to(device)  # [1, M, 2]
            depths_gt = data["depths"].to(device)  # [1, M]
            # Downsample points if image was downsampled
            if downsample_factor > 1:
                points = points / downsample_factor  # Scale point coordinates

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
        
        # Store original Ks before any downsampling (needed for render_at_level)
        Ks_original = data["K"].to(device)  # [3, 3] or [1, 3, 3]
        if Ks_original.dim() == 2:
            Ks_original = Ks_original.unsqueeze(0)  # [3, 3] -> [1, 3, 3]
        
        # Optional CPU-side downsample cache (avoid GPU F.interpolate for coarse levels)
        cached_pixels = None
        cached_masks = None
        if downsample_factor > 1 and self.trainset is not None:
            image_id_value = None
            if isinstance(data["image_id"], torch.Tensor):
                if data["image_id"].numel() == 1:
                    image_id_value = int(data["image_id"].item())
            else:
                image_id_value = int(data["image_id"])
            
            if image_id_value is not None and self.trainset.patch_size is None:
                mask_cpu_single = None
                if masks_cpu is not None:
                    mask_cpu_single = masks_cpu
                    if mask_cpu_single.dim() == 3:
                        mask_cpu_single = mask_cpu_single[0]
                image_cpu_single = image_data_cpu[0]
                if hasattr(self.trainset, "cache_image") and hasattr(self.trainset, "get_downsampled"):
                    self.trainset.cache_image(image_id_value, image_cpu_single, mask_cpu_single)
                    cached_pixels, cached_masks = self.trainset.get_downsampled(
                        image_id_value, downsample_factor
                    )
        
        # OPTIMIZED: For finest level, skip render_at_level overhead and call rasterize_splats directly
        # This avoids unnecessary downsampling calculations and F.interpolate calls
        if target_level == image_multigrid_max_level:
            # Finest level: no downsampling needed, call rasterize_splats directly
            height, width = original_height, original_width
            Ks = Ks_original
            pixels_downsampled = pixels
            
            # Profile rendering time (optional, controlled by cfg)
            if hasattr(cfg, 'profile_timing') and cfg.profile_timing:
                torch.cuda.synchronize()
                render_start = time.time()
            
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                level=target_level,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            
            if hasattr(cfg, 'profile_timing') and cfg.profile_timing:
                torch.cuda.synchronize()
                render_time = time.time() - render_start
                if not hasattr(self, '_render_times'):
                    self._render_times = []
                self._render_times.append(render_time)
            
            if renders.shape[-1] == 4:
                colors = renders[..., 0:3]  # [1, H, W, 3]
            else:
                colors = renders  # [1, H, W, 3]
        else:
            cached_pixels_gpu = None
            cached_masks_gpu = None
            cached_Ks = None
            if cached_pixels is not None:
                cached_pixels_gpu = cached_pixels.unsqueeze(0).to(device) / 255.0
                if cached_masks is not None:
                    cached_masks_gpu = cached_masks.unsqueeze(0).to(device)
                cached_Ks = Ks_original.clone()
                cached_Ks[:, 0, 0] = Ks_original[:, 0, 0] / downsample_factor  # fx
                cached_Ks[:, 1, 1] = Ks_original[:, 1, 1] / downsample_factor  # fy
                cached_Ks[:, 0, 2] = Ks_original[:, 0, 2] / downsample_factor  # cx
                cached_Ks[:, 1, 2] = Ks_original[:, 1, 2] / downsample_factor  # cy

            # Coarser levels: use render_at_level for downsampling
            if hasattr(cfg, 'profile_timing') and cfg.profile_timing:
                torch.cuda.synchronize()
                render_start = time.time()
            
            colors, pixels_downsampled, alphas, info = self.render_at_level(
                target_level=target_level,
                pixels_gt=pixels,
                original_height=original_height,
                original_width=original_width,
                camtoworlds=camtoworlds,
                Ks_original=Ks_original,
                image_multigrid_max_level=image_multigrid_max_level,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                masks=masks,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                pixels_downsampled=cached_pixels_gpu,
                Ks_downsampled=cached_Ks,
                masks_downsampled=cached_masks_gpu,
            )
            
            if hasattr(cfg, 'profile_timing') and cfg.profile_timing:
                torch.cuda.synchronize()
                render_time = time.time() - render_start
                if not hasattr(self, '_render_times'):
                    self._render_times = []
                self._render_times.append(render_time)
        
        # Extract depths if available
        if cfg.depth_loss and info.get("depths") is not None:
            depths = info["depths"]
        else:
            depths = None
        
        # Update pixels to use downsampled version for loss calculation
        pixels = pixels_downsampled
        height, width = pixels.shape[1:3]

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

        # Apply random background if enabled
        # Note: multigrid_gaussians_v2.rasterize_splats has backgrounds commented out,
        # so we need to apply it manually here
        if cfg.random_bkgd or cfg.white_background:
            colors = colors + backgrounds * (1.0 - alphas)
        


        # Standard loss: render should match GT
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
            # Opacity regularization: apply to opacity values
            opacity_reg = cfg.opacity_reg * torch.sigmoid(
                self.splats["opacities"][info["visible_mask"]]
            ).mean()
            loss = loss + opacity_reg.to(device)
            del opacity_reg
        # Scale regularization (applied to all gaussians)
        if cfg.scale_reg > 0.0:
            scale_reg = cfg.scale_reg * torch.exp(self.splats["scales"]).mean()
            loss = loss + scale_reg.to(device)
            del scale_reg
        
        # Hierarchy consistency regularization
        # Enforces hierarchy consistency based on V-cycle direction:
        # - downwards (fine->coarse): parents are regularized (children -> parent aggregation)
        #   * children are detached (gradient cut), parents receive gradients
        # - upwards (coarse->fine, solve): children are regularized (parent -> children)
        #   * parents are detached (gradient cut), children receive gradients
        # Note: children_indices and parent_ids are None here (will be computed inside).
        # For vcycle_trainer_v21.py, these can be precomputed and cached when hierarchy structure changes.
        losses_dict = {}
        if cfg.hierarchy_consistency_lambda > 0.0 and direction is not None:
            # Profile consistency loss time (optional, controlled by cfg)
            if hasattr(cfg, 'profile_timing') and cfg.profile_timing:
                torch.cuda.synchronize()
                consistency_start = time.time()
            
            # Get projection data for other level (parent/child level)
            info_other = None
            if direction == "downwards":
                # downwards: children = target_level (info_current), parent = target_level-1 (info_other)
                other_level = target_level - 1
                if other_level >= 1:
                    # Calculate resolution for other level
                    other_level_diff = image_multigrid_max_level - other_level
                    other_downsample_factor = (1.0 / cfg.level_resolution_factor) ** other_level_diff
                    other_downsample_factor = max(1, int(other_downsample_factor))
                    other_height = max(1, int(original_height // other_downsample_factor))
                    other_width = max(1, int(original_width // other_downsample_factor))
                    other_Ks = Ks_original.clone()
                    if other_downsample_factor > 1:
                        other_Ks[:, 0, 0] = Ks_original[:, 0, 0] / other_downsample_factor
                        other_Ks[:, 1, 1] = Ks_original[:, 1, 1] / other_downsample_factor
                        other_Ks[:, 0, 2] = Ks_original[:, 0, 2] / other_downsample_factor
                        other_Ks[:, 1, 2] = Ks_original[:, 1, 2] / other_downsample_factor
                    info_other = self._get_projection_only(
                        level=other_level,
                        camtoworlds=camtoworlds,
                        Ks=other_Ks,
                        width=other_width,
                        height=other_height,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=image_ids,
                    )
            else:  # upwards
                # upwards: children = target_level+1 (info_other), parent = target_level (info_current)
                other_level = target_level + 1
                if other_level <= image_multigrid_max_level:
                    # Calculate resolution for other level
                    other_level_diff = image_multigrid_max_level - other_level
                    other_downsample_factor = (1.0 / cfg.level_resolution_factor) ** other_level_diff
                    other_downsample_factor = max(1, int(other_downsample_factor))
                    other_height = max(1, int(original_height // other_downsample_factor))
                    other_width = max(1, int(original_width // other_downsample_factor))
                    other_Ks = Ks_original.clone()
                    if other_downsample_factor > 1:
                        other_Ks[:, 0, 0] = Ks_original[:, 0, 0] / other_downsample_factor
                        other_Ks[:, 1, 1] = Ks_original[:, 1, 1] / other_downsample_factor
                        other_Ks[:, 0, 2] = Ks_original[:, 0, 2] / other_downsample_factor
                        other_Ks[:, 1, 2] = Ks_original[:, 1, 2] / other_downsample_factor
                    info_other = self._get_projection_only(
                        level=other_level,
                        camtoworlds=camtoworlds,
                        Ks=other_Ks,
                        width=other_width,
                        height=other_height,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=image_ids,
                    )
            
            hierarchy_loss, losses_dict = self._compute_hierarchy_consistency_loss(
                target_level=target_level,
                image_multigrid_max_level=image_multigrid_max_level,
                direction=direction,
                info_current=info,  # Target level rasterization info
                info_other=info_other,  # Parent/child level projection data
                children_indices=None,  # Will be computed inside (hierarchy structure is fixed here)
                parent_ids=None,  # Will be computed inside (hierarchy structure is fixed here)
            )
            
            if hasattr(cfg, 'profile_timing') and cfg.profile_timing:
                torch.cuda.synchronize()
                consistency_time = time.time() - consistency_start
                if not hasattr(self, '_consistency_times'):
                    self._consistency_times = []
                self._consistency_times.append(consistency_time)
                # Print comparison every 100 steps
                if step % 100 == 0 and hasattr(self, '_render_times') and len(self._render_times) > 0:
                    avg_render = np.mean(self._render_times[-100:])
                    avg_consistency = np.mean(self._consistency_times[-100:])
                    ratio = avg_consistency / avg_render if avg_render > 0 else float('inf')
                    print(f"[Profile] Step {step}: Render={avg_render*1000:.2f}ms, "
                          f"Consistency={avg_consistency*1000:.2f}ms, Ratio={ratio:.2f}x")
            
            loss = loss + cfg.hierarchy_consistency_lambda * hierarchy_loss
            
            # Debug: Print detailed loss breakdown
            if step % 10 == 0:  # Print every 10 steps
                print(f"[Loss Debug] Step {step} | direction={direction} | level={target_level}")
                print(f"  Main loss: {loss.item():.6f}")
                print(f"  L1 loss: {l1loss.item():.6f}")
                print(f"  SSIM loss: {ssimloss.item():.6f}")
                if cfg.hierarchy_consistency_lambda > 0.0:
                    print(f"  Hierarchy loss (raw): {hierarchy_loss.item():.6f}")
                    print(f"  Hierarchy loss (scaled): {(cfg.hierarchy_consistency_lambda * hierarchy_loss).item():.6f}")
                    print(f"    - mean_loss: {losses_dict.get('m', torch.tensor(0.0)).item():.6f}")
                    print(f"    - covar_loss: {losses_dict.get('c', torch.tensor(0.0)).item():.6f}")
                    print(f"    - opacity_loss: {losses_dict.get('o', torch.tensor(0.0)).item():.6f}")
                    print(f"    - sh0_loss: {losses_dict.get('s0', torch.tensor(0.0)).item():.6f}")
                    print(f"    - shN_loss: {losses_dict.get('sN', torch.tensor(0.0)).item():.6f}")
                    print(f"    - weight_onehot_loss: {losses_dict.get('w_onehot', torch.tensor(0.0)).item():.6f}")

        # Gradient scaling hooks are registered via _update_grad_scaling_hooks()
        # which is called after densification. Since densification is disabled
        # in hierarchy_trainer_vcycle.py, hooks are never registered.
        # For vcycle_trainer_v21.py, hooks are updated after densification.

        loss.backward()

        # Turn Gradients into Sparse Tensor before running optimizer
        if cfg.sparse_grad:
            assert cfg.packed, "Sparse gradients only work with packed mode."
            visible_indices = torch.where(info["visible_mask"])[0]
            gaussian_ids = info["gaussian_ids"]
            if len(visible_indices) > 0 and len(gaussian_ids) > 0:
                full_ids = visible_indices[gaussian_ids]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=full_ids[None],  # [1, nnz]
                        values=grad[full_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )
                del full_ids
            del visible_indices, gaussian_ids

        if cfg.visible_adam:
            if cfg.packed:
                visible_indices = torch.where(info["visible_mask"])[0]
                gaussian_ids = info["gaussian_ids"]
                visibility_mask = torch.zeros_like(
                    self.splats["opacities"], dtype=torch.bool
                )
                if len(visible_indices) > 0 and len(gaussian_ids) > 0:
                    full_ids = visible_indices[gaussian_ids]
                    visibility_mask.scatter_(0, full_ids, True)
                    del full_ids
                del visible_indices, gaussian_ids
            else:
                radii = info["radii"]
                if radii.dim() == 3:
                    visibility_local = (radii > 0).all(-1).any(0)
                else:
                    visibility_local = (radii > 0).all(-1)
                visibility_mask = torch.zeros_like(
                    self.splats["opacities"], dtype=torch.bool
                )
                visible_indices = torch.where(info["visible_mask"])[0]
                if len(visible_indices) > 0:
                    visibility_mask[visible_indices] = visibility_local
                del radii, visibility_local, visible_indices

        # optimize    
        # # Debug: Check parameter and optimizer state dtype/device/layout
        # for k, v in self.splats.items():
        #     print(f"Parameter {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}, contiguous={v.is_contiguous()}")
        #     if v.grad is not None:
        #         print(f"  grad: shape={v.grad.shape}, dtype={v.grad.dtype}, device={v.grad.device}, contiguous={v.grad.is_contiguous()}")
        #     else:
        #         print(f"  grad: None")
            
        #     # Check optimizer state for this parameter
        #     if k in self.optimizers:
        #         optimizer = self.optimizers[k]
        #         for param_group in optimizer.param_groups:
        #             for param in param_group["params"]:
        #                 if param is v and param in optimizer.state:
        #                     state = optimizer.state[param]
        #                     print(f"  optimizer state:")
        #                     for state_key, state_value in state.items():
        #                         if isinstance(state_value, torch.Tensor):
        #                             print(f"    {state_key}: shape={state_value.shape}, dtype={state_value.dtype}, device={state_value.device}, contiguous={state_value.is_contiguous()}")
        #                         else:
        #                             print(f"    {state_key}: {type(state_value)}")
        #     print()

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

        # Iteration-based operations (similar to simple_trainer.py structure)
        # These are checked after optimizer step, matching simple_trainer.py behavior
        
        # Evaluation is now performed at the end of each cycle (not per step)
        
        # Free large tensors to reduce peak memory
        if cfg.visible_adam:
            del visibility_mask
        del camtoworlds_gt, Ks, camtoworlds, image_data, pixels, image_ids, masks, Ks_original
        del colors, pixels_downsampled, alphas, backgrounds, depths
        del info
        if cfg.depth_loss:
            del points, depths_gt
        return loss.item(), losses_dict

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)
        
        # Render initial hierarchy levels for visual verification (before training starts)
        if self.use_hierarchy:
            self.render_initial_hierarchy_levels(num_cameras=3)

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
        
        # self.save_visualization(-1)
        while current_step < max_steps:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Update max level based on the schedule
            self._maybe_update_max_level(current_step, vcycle_idx)

            # Check if we have any GSs left
            if len(self.multigrid_gaussians.levels) == 0:
                print(f"ERROR: All Gaussian Splats have been pruned at step {current_step}. Training cannot continue.")
                raise RuntimeError(f"No Gaussian Splats remaining at step {current_step}")
            
            # Get max_level for cycle based on cycle_type
            # V-cycle uses current_max_level (gradually increased based on level_increase_interval)
            # inv-F cycle uses cfg.max_level for more regular training
            if cfg.cycle_type == "vcycle":
                # V-cycle: use current_max_level (gradually increased from 1 to final_max_level)
                # Clamp to actual available levels in hierarchy
                if len(self.multigrid_gaussians.levels) > 0:
                    actual_max_available = int(self.multigrid_gaussians.levels.max().item())
                    cycle_max_level = min(self.current_max_level, actual_max_available)
                else:
                    cycle_max_level = self.current_max_level
                
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
                    global_tic=global_tic,
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
                    global_tic=global_tic,
                )
            else:
                raise ValueError(f"Unknown cycle_type: {cfg.cycle_type}")
            
            # Save visualization at the end of each cycle (if interval matches)
            # Use last_step_in_cycle for visualization step number
            last_step_in_cycle = current_step - 1 if current_step > 0 else 0
            # Only save visualization when vcycle_idx is a multiple of visualization_interval
            # vcycle_idx is 0-indexed, so we check (vcycle_idx + 1) to make it 1-indexed
            if cfg.visualization_interval > 0 and (vcycle_idx + 1) % cfg.visualization_interval == 0:
                self.save_visualization(last_step_in_cycle)
            
            # Save hierarchy visualization at the end of each cycle (if interval matches)
            if cfg.hierarchy_visualization_interval > 0 and vcycle_idx % cfg.hierarchy_visualization_interval == 0:
                self.save_hierarchy_visualization(last_step_in_cycle)
            
            # Measure metrics at the end of each cycle (lightweight, no image saving)
            if cfg.metric_interval_cycles > 0 and vcycle_idx % cfg.metric_interval_cycles == 0:
                self.measure_metric(last_step_in_cycle, stage="val", global_tic=global_tic)
                torch.cuda.empty_cache()
            
            # Eval (full evaluation with image saving) at the end of each cycle
            if vcycle_idx in cfg.eval_cycles:
                self.eval(last_step_in_cycle)
                torch.cuda.empty_cache()
            
            # Run compression at the end of each cycle (if specified in eval_cycles)
            if cfg.compression is not None and vcycle_idx in cfg.eval_cycles:
                self.run_compression(step=last_step_in_cycle)
                torch.cuda.empty_cache()
            
            # Calculate average loss for this cycle
            avg_loss = np.mean(losses) if losses else 0.0
            
            # Progress bar is already updated inside perform_vcycle/perform_inv_fcycle, but ensure it's synced
            if pbar.n < current_step:
                pbar.update(current_step - pbar.n)
            
            # Update progress bar description
            cycle_type_str = "V-cycle" if cfg.cycle_type == "vcycle" else "inv-F-cycle"
            if len(self.multigrid_gaussians.levels) > 0:
                actual_max_level = int(self.multigrid_gaussians.levels.max().item())
                desc = f"{cycle_type_str} {vcycle_idx}| max_level={actual_max_level}| loss={avg_loss:.3f}| step={current_step}/{max_steps}"
            else:
                desc = f"{cycle_type_str} {vcycle_idx}| loss={avg_loss:.3f}| step={current_step}/{max_steps}"
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
                        del camtoworlds, Ks, image_data, pixels, image_ids, masks, backgrounds, colors, canvas
                    self.writer.flush()
                
                # Save checkpoint
                if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    stats = {
                        "mem": mem,
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
            
            del steps_in_cycle, losses, avg_loss
            torch.cuda.empty_cache()

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
    def measure_metric(self, step: int, stage: str = "val", global_tic = None) -> Dict[str, float]:
        """Measure metrics only (no image saving).
        
        Renders at current max level and computes PSNR, SSIM, LPIPS metrics.
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
            
            # Render at current max level for metrics
            metric_level = self.current_max_level if self.current_max_level is not None else -1
            colors, alphas, _ = self.multigrid_gaussians.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                level=metric_level,  # Use current_max_level for metrics
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
            
            # Compute metrics using current max level
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
            
            # Calculate current max level num_GS (rendered GS count) for direct comparison with baseline
            # This counts only the gaussians that are actually rendered at the current max level
            if len(self.multigrid_gaussians.levels) > 0:
                metric_level = self.current_max_level if self.current_max_level is not None else int(self.multigrid_gaussians.levels.max().item())
                # Set visible mask and count visible gaussians at current max level
                self.multigrid_gaussians.set_visible_mask(metric_level)
                num_GS_finest = self.multigrid_gaussians.visible_mask.sum().item()
            else:
                num_GS_finest = len(self.splats["means"])
            
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                    "num_GS_finest": num_GS_finest,
                }
            )
            if global_tic is not None:
                stats["ellipse_time"] = time.time() - global_tic
            # Print metrics
            if cfg.use_bilateral_grid:
                print(
                    f"Step {step} - Metrics: PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS (total): {stats['num_GS']}, (finest): {stats['num_GS_finest']}"
                )
            else:
                print(
                    f"Step {step} - Metrics: PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS (total): {stats['num_GS']}, (finest): {stats['num_GS_finest']}"
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
            
            # Calculate finest level num_GS (rendered GS count) for direct comparison with baseline
            if len(self.multigrid_gaussians.levels) > 0:
                highest_level = int(self.multigrid_gaussians.levels.max().item())
                # Count gaussians at each level (for debugging)
                levels = self.multigrid_gaussians.levels
                unique_levels = sorted(levels.unique().cpu().tolist())
                level_counts = {l: (levels == l).sum().item() for l in unique_levels}
                num_GS_at_finest_level = level_counts.get(highest_level, 0)
                # Set visible mask and count visible gaussians
                self.multigrid_gaussians.set_visible_mask(highest_level)
                num_GS_finest = self.multigrid_gaussians.visible_mask.sum().item()
                # Debug: Check if masking is working correctly
                if num_GS_finest == len(self.splats["means"]) and num_GS_at_finest_level < len(self.splats["means"]):
                    # This means all gaussians are visible, but not all are at finest level
                    # This suggests masking might not be working correctly
                    print(f"[DEBUG] Step {step}: num_GS_total={len(self.splats['means'])}, "
                          f"num_GS_at_finest_level={num_GS_at_finest_level}, "
                          f"num_GS_visible={num_GS_finest}, highest_level={highest_level}")
                    print(f"[DEBUG] Level counts: {level_counts}")
            else:
                num_GS_finest = len(self.splats["means"])
            
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                    "num_GS_finest": num_GS_finest,
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS (total): {stats['num_GS']}, (finest): {stats['num_GS_finest']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS (total): {stats['num_GS']}, (finest): {stats['num_GS_finest']}"
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
        max_level = int(cfg.max_level)
        
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
                # Calculate downsample factor: (1/level_resolution_factor)^(max_level - render_level)
                level_diff = image_multigrid_max_level - render_level
                downsample_factor = (1.0 / cfg.level_resolution_factor) ** level_diff
                downsample_factor = max(1, int(downsample_factor))  # Ensure at least 1 (no upsampling) and convert to int
                
                # Downsample for rendering
                if downsample_factor > 1:
                    # Calculate new dimensions
                    render_height = max(1, int(height // downsample_factor))
                    render_width = max(1, int(width // downsample_factor))
                    
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
                        mode='bicubic',
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
        means = self.multigrid_gaussians.splats["means"].detach().cpu().numpy()  # [N, 3]
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
            
            # Create mapping: level index -> position in level_indices (vectorized)
            level_index_set = set(level_indices)  # Use set for O(1) lookup instead of dict
            
            # Build lines: each next_level point connects to its parent level point (vectorized)
            # Get parent indices for all next_level points at once
            parent_indices_array = parent_indices[next_level_indices]  # [M,]
            
            # Filter: keep only valid parent connections (parent_idx >= 0 and parent_idx in level_index_set)
            valid_mask = (parent_indices_array >= 0) & np.isin(parent_indices_array, level_indices)
            
            if valid_mask.any():
                # Vectorized line creation: [parent_idx, next_level_idx] pairs
                valid_next_level_indices = next_level_indices[valid_mask]
                valid_parent_indices = parent_indices_array[valid_mask]
                lines = np.stack([valid_parent_indices, valid_next_level_indices], axis=1)  # [K, 2]
                lines = lines.tolist()  # Convert to list for o3d compatibility
            else:
                lines = []
            
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
            # Normalize depths safely (handle empty or constant depth values)
            depth_min = depths.min()
            depth_max = depths.max()
            depth_range = depth_max - depth_min
            if depth_range > 1e-8:  # Avoid division by zero
                depths = (depths - depth_min) / depth_range
            else:
                # If all depths are the same or empty, set to zero
                depths = torch.zeros_like(depths)
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
        param_device = self.splats["means"].device
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(param_device)
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
        ckpts = [torch.load(file, map_location="cpu", weights_only=True) for file in cfg.ckpt]
        param_device = runner.splats["means"].device
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts]).to(param_device)
        # Load hierarchical structure if available
        if "levels" in ckpts[0]:
            runner.multigrid_gaussians.levels = ckpts[0]["levels"].to(param_device)
            runner.multigrid_gaussians.parent_indices = ckpts[0]["parent_indices"].to(param_device)
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

