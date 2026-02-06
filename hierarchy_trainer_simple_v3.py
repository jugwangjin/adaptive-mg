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
import faiss
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
from gsplat.cuda._wrapper import quat_scale_to_covar_preci, fully_fused_projection
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
from multigrid_gaussians_v8 import MultigridGaussians
# load_hierarchy_multigrid is now a static method of MultigridGaussians

# Import consistency loss functions from vcycle trainer
from hierarchy_trainer_vcycle_v3 import Runner as VCycleRunnerV3


@dataclass
class Config:
    # Hierarchy loading (required)
    hierarchy_path: Optional[str] = None
    
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
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    # Interval for metric measurement (every N steps). -1 means disable metric measurement.
    metric_interval: int = 200
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Visualization: interval to save visualization images (GT vs render)
    visualization_interval: int = 100

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
    init_opa: float = 0.9
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
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

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # Hierarchy consistency regularization
    # Enforces that parent gaussians match the aggregation of their children
    # This ensures consistency between fine and coarse levels
    hierarchy_consistency_lambda: float = 0.25
    # Level resolution reduction factor (for coarse-to-fine)
    # Resolution reduction per level: resolution = original_resolution * (level_resolution_factor ^ (max_level - target_level))
    # Default 0.5 means each level halves the resolution (1/2 per level)
    level_resolution_factor: float = 0.5
    # Increase max level every N steps (for coarse-to-fine training)
    level_increase_interval: int = 1000
    # Steps decaying per level (for coarse-to-fine training)
    # Number of steps per level is scaled by steps_decaying_per_level^(max_level - target_level)
    steps_decaying_per_level: float = 0.5

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

    # Use coarse-to-fine training (like simple_trainer_c2f.py)
    use_coarse_to_fine: bool = False
    # Coarse-to-fine: interval for switching to next level (every N steps)
    level_interval: int = 1500

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
        
        # Always regenerate result_dir to ensure it matches current CLI arguments
        # (e.g., if data_factor was changed via CLI, result_dir should reflect that)
        # Extract dataset name from data_dir (last directory component)
        dataset_name = Path(self.data_dir).name
        
        # Build settings string
        settings_parts = [
            f"type_{self.dataset_type}",
            f"factor_{self.data_factor}",
        ]
        if self.dataset_type == "nerf" and self.white_background:
            settings_parts.append("whitebg")
        if self.normalize_world_space:
            settings_parts.append("norm")
        if self.patch_size is not None:
            settings_parts.append(f"patch_{self.patch_size}")
        
        settings_str = "_".join(settings_parts)
        # hierarchy_path may be None during config initialization, will be validated later
        if self.hierarchy_path is not None:
            hierarchy_name = Path(self.hierarchy_path).stem
            result_base = f"./results/hierarchy_trainer_simple/{dataset_name}_{settings_str}_hierarchy_{hierarchy_name}"
        else:
            result_base = f"./results/hierarchy_trainer_simple/{dataset_name}_{settings_str}_hierarchy_unknown"
        if self.use_coarse_to_fine:
            result_base += "_c2f"
        self.result_dir = result_base

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        if self.use_coarse_to_fine:
            self.level_interval = int(self.level_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
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


def create_splats_with_optimizers(
    parser: Union[ColmapParser, NerfParser],
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
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    # NOTE: slicing with step can produce non-contiguous views; make them contiguous to keep
    # optimizer (including fused Adam) happy.
    points = points[world_rank::world_size].contiguous()
    rgbs = rgbs[world_rank::world_size].contiguous()
    scales = scales[world_rank::world_size].contiguous()

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # SH coefficients (initialize DC from RGB, higher bands to 0).
        # IMPORTANT: avoid non-contiguous views like `colors[:, :1, :]` because fused Adam
        # can error on non-standard strides.
        sh0 = rgb_to_sh(rgbs).unsqueeze(1)  # [N, 1, 3]
        shN = torch.zeros(
            (N, (sh_degree + 1) ** 2 - 1, 3),
            device=sh0.device,
            dtype=sh0.dtype,
        )  # [N, K-1, 3]
        params.append(("sh0", torch.nn.Parameter(sh0), sh0_lr))
        params.append(("shN", torch.nn.Parameter(shN), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {}
    for name, _, lr in params:
        optimizer_kwargs = {
            "params": [splats[name]],
            "lr": lr * math.sqrt(BS),
            "name": name,
        }
        optimizer_init_kwargs = {
            "eps": 1e-15 / math.sqrt(BS),
            "betas": (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        }
        if optimizer_class != SelectiveAdam:
            optimizer_init_kwargs["fused"] = True
        optimizers[name] = optimizer_class(
            [optimizer_kwargs],
            **optimizer_init_kwargs,
        )
    return splats, optimizers


def load_hierarchy_leaf_nodes(
    hierarchy_path: str,
    parser: Union[ColmapParser, NerfParser],
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
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer], int]:
    """
    Load leaf nodes (level 0) from hierarchy and initialize parameters.
    
    Position (means) is kept from hierarchy, but other parameters are re-initialized
    like in simple_trainer_original.py.
    
    Args:
        hierarchy_path: Path to hierarchy.pt file
        parser: Dataset parser (for getting RGB colors from SfM points)
        init_opacity: Initial opacity value
        init_scale: Initial scale multiplier
        ... (other parameters same as create_splats_with_optimizers)
        
    Returns:
        splats: ParameterDict with leaf node Gaussians
        optimizers: Dictionary of optimizers
        num_levels: Number of levels in hierarchy
    """
    print(f"Loading hierarchy from {hierarchy_path}...")
    checkpoint = torch.load(hierarchy_path, map_location=device, weights_only=False)
    hierarchy = checkpoint["hierarchy"]
    
    # Get number of levels from hierarchy
    num_levels = len(hierarchy["levels"])
    
    # Extract leaf nodes (level 0) - use means and scales from hierarchy
    # Other parameters will be initialized like simple_trainer_original.py
    level_0 = hierarchy["levels"][0]
    leaf_means = level_0["means"].to(device).float().contiguous()  # [N, 3]
    leaf_scales = level_0["scales"].to(device).float().contiguous()  # [N, 3]
    
    N = leaf_means.shape[0]
    print(f"Loaded {N} leaf nodes from hierarchy (total {num_levels} levels)")
    print("Using means and scales from hierarchy, initializing other parameters like simple_trainer_original.py")
    
    # Get RGB colors from SfM points (if available) for SH initialization.
    have_sfm = (
        hasattr(parser, "points")
        and hasattr(parser, "points_rgb")
        and parser.points is not None
        and parser.points_rgb is not None
        and len(parser.points) > 0
        and len(parser.points_rgb) > 0
    )

    if have_sfm:
        # Find nearest SfM points for each leaf node using FAISS (CPU index).
        print(f"Finding nearest SfM points for {N} leaf nodes using FAISS...")
        sfm_points_np = np.asarray(parser.points, dtype=np.float32)  # [M, 3]
        sfm_rgbs = torch.from_numpy(np.asarray(parser.points_rgb, dtype=np.float32) / 255.0).to(device)  # [M, 3]
        leaf_means_np = leaf_means.detach().cpu().numpy().astype(np.float32)  # [N, 3]

        index = faiss.IndexFlatL2(3)
        index.add(sfm_points_np)
        _, nearest_indices_np = index.search(leaf_means_np, 1)  # [N, 1]
        nearest_indices = torch.from_numpy(nearest_indices_np[:, 0].astype(np.int64)).to(device)
        print("FAISS search completed")

        rgbs = sfm_rgbs[nearest_indices]  # [N, 3]
    else:
        print("[init] parser has no SfM points; initializing colors randomly")
        rgbs = torch.rand((N, 3), device=device)
    
    # Use scales from hierarchy (no need to initialize)
    
    # Distribute to different ranks
    leaf_means = leaf_means[world_rank::world_size].contiguous()
    leaf_scales = leaf_scales[world_rank::world_size].contiguous()
    rgbs = rgbs[world_rank::world_size].contiguous()
    
    N = leaf_means.shape[0]
    
    # Initialize other parameters (like simple_trainer_original.py)
    quats = torch.rand((N, 4), device=device)  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity, device=device))  # [N,]
    
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(leaf_means), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(leaf_scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]
    
    if feature_dim is None:
        # Initialize SH coefficients from RGB (DC) and 0 for higher bands.
        # IMPORTANT: avoid non-contiguous views because fused Adam can error.
        sh0 = rgb_to_sh(rgbs).unsqueeze(1)  # [N, 1, 3]
        shN = torch.zeros(
            (N, (sh_degree + 1) ** 2 - 1, 3),
            device=device,
            dtype=sh0.dtype,
        )  # [N, K-1, 3]
        params.append(("sh0", torch.nn.Parameter(sh0), sh0_lr))
        params.append(("shN", torch.nn.Parameter(shN), shN_lr))
    else:
        # Initialize features/colors for appearance optimization (like simple_trainer_original.py)
        features = torch.rand(N, feature_dim, device=device)
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(torch.clamp(rgbs, 0.0, 1.0))  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))
    
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    
    # Create optimizers (same as create_splats_with_optimizers)
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {}
    for name, _, lr in params:
        optimizer_kwargs = {
            "params": [splats[name]],
            "lr": lr * math.sqrt(BS),
            "name": name,
        }
        optimizer_init_kwargs = {
            "eps": 1e-15 / math.sqrt(BS),
            "betas": (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        }
        if optimizer_class != SelectiveAdam:
            optimizer_init_kwargs["fused"] = True
        optimizers[name] = optimizer_class(
            [optimizer_kwargs],
            **optimizer_init_kwargs,
        )
    
    return splats, optimizers, num_levels


# load_hierarchy_multigrid is now a static method of MultigridGaussians
# Use MultigridGaussians.load_hierarchy_multigrid() instead


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)
        print(cfg)
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

        # Validate hierarchy_path is provided
        if cfg.hierarchy_path is None:
            raise ValueError("hierarchy_path is required. Please provide --hierarchy_path <path>")

        # Model: Load from hierarchy (required)
        print(f"Loading leaf nodes from hierarchy: {cfg.hierarchy_path}")
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers, self.hierarchy_num_levels = load_hierarchy_leaf_nodes(
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
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        # Multigrid convention (matches MultigridGaussians / visualize_hierarchy_levels.py):
        # levels are 1..num_levels where 1=coarsest and num_levels=finest.
        self.max_level = self.hierarchy_num_levels
        print("Model initialized from hierarchy. Number of GS:", len(self.splats["means"]))
        print(f"Hierarchy has {self.hierarchy_num_levels} levels (max_level={self.max_level})")

        # Load MultigridGaussians for consistency loss (full hierarchy structure)
        self.use_hierarchy = cfg.hierarchy_path is not None
        if self.use_hierarchy and cfg.hierarchy_consistency_lambda > 0.0:
            print(f"Loading full hierarchy from {cfg.hierarchy_path} for consistency loss...")
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
                position_scale_reduction=0.75,
                max_level=cfg.max_level,
            )
            # Initialize hierarchy index cache for consistency loss
            self._hierarchy_index_cache = {
                "downwards": {},
                "upwards": {},
            }
        else:
            self.multigrid_gaussians = None
            self._hierarchy_index_cache = None

        # Densification Strategy (disabled when using hierarchy)
        print("Densification disabled (using hierarchy leaf nodes)")
        # Create a dummy strategy state that does nothing
        class DummyStrategyState:
            pass
        self.strategy_state = DummyStrategyState()

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
    
    @torch.no_grad()
    def render_initial_hierarchy_levels(self, num_cameras: int = 3):
        """
        Render each hierarchy level before training starts (for visual verification).
        Loads full hierarchy temporarily for rendering (training uses leaf nodes only).
        
        Args:
            num_cameras: Number of cameras to render
        """
        if self.world_rank != 0:
            # Only render on rank 0
            return
        
        cfg = self.cfg
        device = self.device
        
        print("\n" + "="*60)
        print("Rendering initial hierarchy levels for visual verification...")
        print("="*60)
        print("Note: Training uses leaf nodes only, but rendering uses full hierarchy")
        
        # Load full hierarchy temporarily for rendering
        print(f"Loading full hierarchy from {cfg.hierarchy_path} for rendering...")
        multigrid_gaussians, _ = MultigridGaussians.load_hierarchy_multigrid(
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
            device=device,
            world_rank=0,
            world_size=1,
            position_scale_reduction=0.75,
            max_level=None,
        )
        
        # Create output directory
        output_dir = os.path.join(cfg.result_dir, "initial_hierarchy_levels")
        os.makedirs(output_dir, exist_ok=True)
        
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
            # Multigrid levels start at 1
            actual_max_level = 1
            actual_min_level = 1
            unique_levels = [1]
        
        max_level = int(multigrid_gaussians.max_level) if multigrid_gaussians.max_level is not None else actual_max_level
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
                # Simple downsampling: each coarser level halves resolution.
                # Finest (max_level): downsample_factor=1
                # Coarsest (1): downsample_factor=2^(max_level-1)
                downsample_factor = 2 ** (max_level - render_level)
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
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    masks=masks_downsampled,
                    packed=cfg.packed,
                    sparse_grad=cfg.sparse_grad,
                    distributed=False,
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
        
        # Free temporary multigrid_gaussians
        del multigrid_gaussians
        torch.cuda.empty_cache()
        
        print(f"\nAll initial hierarchy level visualizations saved to {output_dir}")
        print("="*60 + "\n")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]


        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model

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
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def _sync_leaf_to_multigrid(self):
        """Sync leaf node parameters from self.splats to multigrid_gaussians."""
        if not hasattr(self, 'multigrid_gaussians') or self.multigrid_gaussians is None:
            return
        
        # Find leaf nodes (at max_level)
        leaf_mask = self.multigrid_gaussians.levels == self.max_level
        leaf_indices = torch.where(leaf_mask)[0]
        
        if len(leaf_indices) != len(self.splats["means"]):
            # This should not happen if hierarchy was loaded correctly
            print(f"Warning: Leaf node count mismatch: {len(leaf_indices)} vs {len(self.splats['means'])}")
            return
        
        # Update multigrid_gaussians with current leaf node parameters
        self.multigrid_gaussians.splats["means"][leaf_indices] = self.splats["means"].detach()
        self.multigrid_gaussians.splats["scales"][leaf_indices] = self.splats["scales"].detach()
        self.multigrid_gaussians.splats["quats"][leaf_indices] = self.splats["quats"].detach()
        self.multigrid_gaussians.splats["opacities"][leaf_indices] = self.splats["opacities"].detach()
        self.multigrid_gaussians.splats["sh0"][leaf_indices] = self.splats["sh0"].detach()
        if "shN" in self.splats:
            self.multigrid_gaussians.splats["shN"][leaf_indices] = self.splats["shN"].detach()
        if "colors" in self.splats:
            self.multigrid_gaussians.splats["colors"][leaf_indices] = self.splats["colors"].detach()
        if "features" in self.splats:
            self.multigrid_gaussians.splats["features"][leaf_indices] = self.splats["features"].detach()

    # Import consistency loss functions from vcycle trainer
    _get_projection_only = VCycleRunnerV3._get_projection_only
    _extract_projection_data = VCycleRunnerV3._extract_projection_data
    _compute_hierarchy_consistency_loss = VCycleRunnerV3._compute_hierarchy_consistency_loss

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
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()  # Start time for total training
        training_start_time = time.time()  # Start time for actual training (excluding setup)
        pbar = tqdm.tqdm(range(init_step, max_steps))
        
        # Training time tracking
        training_times = []
        
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)


            # Measure training time per step
            step_start_time = time.time()
            
            camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks_original = data["K"].to(device)  # [1, 3, 3]
            image_data = data["image"].to(device) / 255.0  # [1, H, W, 3]
            
            # Ensure batch dimension exists
            if camtoworlds_gt.dim() == 2:
                camtoworlds_gt = camtoworlds_gt.unsqueeze(0)
            if Ks_original.dim() == 2:
                Ks_original = Ks_original.unsqueeze(0)
            if image_data.dim() == 3:
                image_data = image_data.unsqueeze(0)
            
            # Handle RGBA images
            if image_data.shape[-1] == 4:
                pixels_gt = image_data[..., :3]  # [1, H, W, 3]
            else:
                pixels_gt = image_data  # [1, H, W, 3]
            
            original_height, original_width = pixels_gt.shape[1:3]
            
            # Coarse-to-fine: calculate target level and downsample
            target_level = None  # Initialize for later use in description
            if cfg.use_coarse_to_fine:
                max_level = self.max_level
                target_level = min(1 + step // cfg.level_interval, max_level)
                image_multigrid_max_level = self.max_level
                
                # Downsample for coarse-to-fine
                downsample_factor = 2 ** ((image_multigrid_max_level - target_level) / 2.0)
                downsample_factor = max(1, downsample_factor)
                
                height = int(max(1, original_height / downsample_factor))
                width = int(max(1, original_width / downsample_factor))
                
                # Downsample GT image
                if downsample_factor > 1:
                    pixels_bchw = pixels_gt.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    pixels_downsampled_bchw = F.interpolate(
                        pixels_bchw,
                        size=(height, width),
                        mode='bicubic',
                        align_corners=False,
                    )
                    pixels = pixels_downsampled_bchw.permute(0, 2, 3, 1)  # [1, H, W, 3]
                else:
                    pixels = pixels_gt
                
                # Downsample Ks
                Ks = Ks_original.clone()
                if downsample_factor > 1:
                    Ks[:, 0, 0] = Ks_original[:, 0, 0] / downsample_factor  # fx
                    Ks[:, 1, 1] = Ks_original[:, 1, 1] / downsample_factor  # fy
                    Ks[:, 0, 2] = Ks_original[:, 0, 2] / downsample_factor  # cx
                    Ks[:, 1, 2] = Ks_original[:, 1, 2] / downsample_factor  # cy
                else:
                    Ks = Ks_original
            else:
                # Normal training (no coarse-to-fine)
                pixels = pixels_gt
                Ks = Ks_original
                height, width = pixels.shape[1:3]
            
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if masks is not None and masks.dim() == 2:
                masks = masks.unsqueeze(0)
            if cfg.use_coarse_to_fine and masks is not None:
                # Downsample masks
                if downsample_factor > 1:
                    masks_bchw = masks.unsqueeze(1).float()  # [1, 1, H, W]
                    masks_downsampled_bchw = F.interpolate(
                        masks_bchw,
                        size=(height, width),
                        mode='nearest',
                    )
                    masks = masks_downsampled_bchw.squeeze(1).bool()  # [1, H, W]
            
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            # Initialize camtoworlds
            camtoworlds = camtoworlds_gt
            
            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
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

            # Densification disabled (using hierarchy)

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
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
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()
            
            # Hierarchy consistency regularization
            # For simple trainers, we only train leaf nodes, so we use "downwards" direction
            # where leaf nodes (children) are compared to their parents
            affected_gaussians = None
            if cfg.hierarchy_consistency_lambda > 0.0 and self.use_hierarchy and cfg.use_coarse_to_fine and target_level is not None and target_level > 1:
                # Sync leaf node parameters to multigrid_gaussians before computing consistency loss
                self._sync_leaf_to_multigrid()
                
                # Get projection data for current level (children) and parent level
                parent_level = target_level - 1
                image_multigrid_max_level = self.max_level
                
                # Get projection data for parent level
                proj_data_parent = self._get_projection_only(
                    level=parent_level,
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                )
                
                # Create info dicts for consistency loss
                info_current = info  # Current level (children) from rasterization
                info_other = None
                if proj_data_parent is not None:
                    info_other = {
                        "means2d": proj_data_parent["means2d"],
                        "conics": None,
                        "opacities": proj_data_parent["opacities"],
                        "colors": proj_data_parent["colors"],
                        "radii": None,
                        "visible_mask": None,
                        "gaussian_ids": None,
                        "render_level": proj_data_parent["render_level"],
                        "projection_data": proj_data_parent,
                    }
                
                # Compute consistency loss (downwards: children -> parent)
                if info_other is not None:
                    hierarchy_loss, losses_dict, affected_gaussians = self._compute_hierarchy_consistency_loss(
                        target_level=target_level,
                        image_multigrid_max_level=image_multigrid_max_level,
                        direction="downwards",
                        info_current=info_current,
                        info_other=info_other,
                        children_indices=None,
                        parent_ids=None,
                    )
                    loss = loss + cfg.hierarchy_consistency_lambda * hierarchy_loss
            
            loss.backward()

            # Measure step time
            step_time = time.time() - step_start_time
            training_times.append(step_time)
            
            # Build description
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.use_coarse_to_fine and target_level is not None:
                desc += f"level={target_level}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            desc += f"time={step_time:.3f}s| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
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
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

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
                
                # Add hierarchy consistency loss affected gaussians to visibility mask
                # These gaussians receive gradients from hierarchy consistency loss and should be updated
                if cfg.hierarchy_consistency_lambda > 0.0 and affected_gaussians is not None and len(affected_gaussians) > 0:
                    # Ensure affected_gaussians are within valid range
                    valid_affected = (affected_gaussians >= 0) & (affected_gaussians < len(visibility_mask))
                    if valid_affected.any():
                        visibility_mask[affected_gaussians[valid_affected]] = True

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Densification disabled (using hierarchy)

            # Measure metrics (lightweight, no image saving)
            if cfg.metric_interval > 0 and step % cfg.metric_interval == 0:
                self.measure_metric(step, stage="val", global_tic=global_tic)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # Save visualization images
            if cfg.visualization_interval > 0 and step % cfg.visualization_interval == 0:
                self.save_visualization(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)
        
        # Training completed - save training time statistics
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        if world_rank == 0:
            training_time_stats = {
                "total_training_time_seconds": total_training_time,
                "total_training_time_minutes": total_training_time / 60.0,
                "total_training_time_hours": total_training_time / 3600.0,
                "average_step_time_seconds": np.mean(training_times) if training_times else 0.0,
                "median_step_time_seconds": np.median(training_times) if training_times else 0.0,
                "min_step_time_seconds": np.min(training_times) if training_times else 0.0,
                "max_step_time_seconds": np.max(training_times) if training_times else 0.0,
                "total_steps": max_steps,
                "num_gaussians": len(self.splats["means"]),
                "hierarchy_path": cfg.hierarchy_path if hasattr(cfg, 'hierarchy_path') else None,
                "use_coarse_to_fine": cfg.use_coarse_to_fine,
            }
            
            training_time_path = os.path.join(cfg.result_dir, "training_time_stats.json")
            with open(training_time_path, "w") as f:
                json.dump(training_time_stats, f, indent=2)
            
            print("\n" + "="*80)
            print("TRAINING TIME STATISTICS")
            print("="*80)
            print(f"Total training time: {total_training_time / 60.0:.2f} minutes ({total_training_time / 3600.0:.2f} hours)")
            print(f"Average step time: {np.mean(training_times):.4f} seconds")
            print(f"Median step time: {np.median(training_times):.4f} seconds")
            print(f"Min step time: {np.min(training_times):.4f} seconds")
            print(f"Max step time: {np.max(training_times):.4f} seconds")
            print(f"Total steps: {max_steps}")
            print(f"Number of Gaussians: {len(self.splats['means'])}")
            print(f"Hierarchy path: {cfg.hierarchy_path}")
            print(f"Using leaf nodes only: True")
            if cfg.use_coarse_to_fine:
                print(f"Coarse-to-fine training: True (max_level={self.max_level}, level_interval={cfg.level_interval})")
            print(f"Training time stats saved to: {training_time_path}")
            print("="*80)

    @torch.no_grad()
    def measure_metric(self, step: int, stage: str = "val", global_tic: Optional[float] = None) -> Dict[str, float]:
        """Measure metrics only (no image saving).
        
        Renders and computes PSNR, SSIM, LPIPS metrics.
        This is a lightweight function for frequent metric measurement during training.
        
        Args:
            step: Current training step
            stage: Evaluation stage name (e.g., "val", "test")
            global_tic: Global start time for training (to measure total training time)
        
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
        
        # Coarse-to-fine: calculate target level (same as training)
        if cfg.use_coarse_to_fine:
            max_level = self.max_level
            target_level = min(1 + step // cfg.level_interval, max_level)
            image_multigrid_max_level = self.max_level
        else:
            target_level = None
            image_multigrid_max_level = None
        
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks_original = data["K"].to(device)
            image_data = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            
            # Handle RGBA images
            if image_data.shape[-1] == 4:
                pixels_gt = image_data[..., :3]  # [1, H, W, 3]
            else:
                pixels_gt = image_data  # [1, H, W, 3]
            
            original_height, original_width = pixels_gt.shape[1:3]
            
            # Coarse-to-fine: downsample if needed
            if cfg.use_coarse_to_fine and target_level is not None:
                # Downsample for coarse-to-fine (same logic as training)
                downsample_factor = 2 ** ((image_multigrid_max_level - target_level) / 2.0)
                downsample_factor = max(1, downsample_factor)
                
                height = int(max(1, original_height / downsample_factor))
                width = int(max(1, original_width / downsample_factor))
                
                # Downsample GT image
                if downsample_factor > 1:
                    pixels_bchw = pixels_gt.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    pixels_downsampled_bchw = F.interpolate(
                        pixels_bchw,
                        size=(height, width),
                        mode='bicubic',
                        align_corners=False,
                    )
                    pixels = pixels_downsampled_bchw.permute(0, 2, 3, 1)  # [1, H, W, 3]
                else:
                    pixels = pixels_gt
                
                # Downsample Ks
                Ks = Ks_original.clone()
                if downsample_factor > 1:
                    Ks[:, 0, 0] = Ks_original[:, 0, 0] / downsample_factor  # fx
                    Ks[:, 1, 1] = Ks_original[:, 1, 1] / downsample_factor  # fy
                    Ks[:, 0, 2] = Ks_original[:, 0, 2] / downsample_factor  # cx
                    Ks[:, 1, 2] = Ks_original[:, 1, 2] / downsample_factor  # cy
                else:
                    Ks = Ks_original
                
                # Downsample masks if provided
                if masks is not None and downsample_factor > 1:
                    masks_bchw = masks.unsqueeze(1).float()  # [1, 1, H, W]
                    masks_downsampled_bchw = F.interpolate(
                        masks_bchw,
                        size=(height, width),
                        mode='nearest',
                    )
                    masks = masks_downsampled_bchw.squeeze(1).bool()  # [1, H, W]
            else:
                # Normal metric measurement (no coarse-to-fine)
                pixels = pixels_gt
                Ks = Ks_original
                height, width = original_height, original_width

            torch.cuda.synchronize()
            tic = time.time()
            
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            # Clamp pixels to [0, 1] range (may go out of range due to downsampling/upsampling)
            pixels = torch.clamp(pixels, 0.0, 1.0)

            if world_rank == 0:
                # Compute metrics
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
            if global_tic is not None:
                stats["ellipse_time"] = time.time() - global_tic
            
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
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        
        # Coarse-to-fine: calculate target level (same as training)
        if cfg.use_coarse_to_fine:
            max_level = self.max_level
            target_level = min(1 + step // cfg.level_interval, max_level)
            image_multigrid_max_level = self.max_level
        else:
            target_level = None
            image_multigrid_max_level = None
        
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks_original = data["K"].to(device)
            image_data = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            
            # Handle RGBA images
            if image_data.shape[-1] == 4:
                pixels_gt = image_data[..., :3]  # [1, H, W, 3]
            else:
                pixels_gt = image_data  # [1, H, W, 3]
            
            original_height, original_width = pixels_gt.shape[1:3]
            
            # Coarse-to-fine: downsample if needed
            if cfg.use_coarse_to_fine and target_level is not None:
                # Downsample for coarse-to-fine (same logic as training)
                downsample_factor = 2 ** ((image_multigrid_max_level - target_level) / 2.0)
                downsample_factor = max(1, downsample_factor)
                
                height = int(max(1, original_height / downsample_factor))
                width = int(max(1, original_width / downsample_factor))
                
                # Downsample GT image
                if downsample_factor > 1:
                    pixels_bchw = pixels_gt.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    pixels_downsampled_bchw = F.interpolate(
                        pixels_bchw,
                        size=(height, width),
                        mode='bicubic',
                        align_corners=False,
                    )
                    pixels = pixels_downsampled_bchw.permute(0, 2, 3, 1)  # [1, H, W, 3]
                else:
                    pixels = pixels_gt
                
                # Downsample Ks
                Ks = Ks_original.clone()
                if downsample_factor > 1:
                    Ks[:, 0, 0] = Ks_original[:, 0, 0] / downsample_factor  # fx
                    Ks[:, 1, 1] = Ks_original[:, 1, 1] / downsample_factor  # fy
                    Ks[:, 0, 2] = Ks_original[:, 0, 2] / downsample_factor  # cx
                    Ks[:, 1, 2] = Ks_original[:, 1, 2] / downsample_factor  # cy
                else:
                    Ks = Ks_original
                
                # Downsample masks if provided
                if masks is not None and downsample_factor > 1:
                    masks_bchw = masks.unsqueeze(1).float()  # [1, 1, H, W]
                    masks_downsampled_bchw = F.interpolate(
                        masks_bchw,
                        size=(height, width),
                        mode='nearest',
                    )
                    masks = masks_downsampled_bchw.squeeze(1).bool()  # [1, H, W]
            else:
                # Normal evaluation (no coarse-to-fine)
                pixels = pixels_gt
                Ks = Ks_original
                height, width = original_height, original_width

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            # Clamp pixels to [0, 1] range (may go out of range due to downsampling/upsampling)
            pixels = torch.clamp(pixels, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                level_suffix = f"_level{target_level}" if cfg.use_coarse_to_fine and target_level is not None else ""
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}{level_suffix}_{i:04d}.png",
                    canvas,
                )

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
        """Save visualization images comparing GT and render from sampled cameras."""
        if self.world_rank != 0:
            return
        
        if len(self.viz_camera_indices) == 0:
            return
        
        cfg = self.cfg
        device = self.device
        
        viz_dir = f"{cfg.result_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Coarse-to-fine: calculate target level and downsample (same as training)
        if cfg.use_coarse_to_fine:
            max_level = cfg.max_level
            target_level = min(1 + step // cfg.level_interval, max_level)
            image_multigrid_max_level = cfg.max_level
        else:
            target_level = None
            image_multigrid_max_level = None
        
        images_list = []
        for cam_idx in self.viz_camera_indices:
            # Get data for this camera
            data = self.valset[cam_idx]
            camtoworlds = data["camtoworld"].unsqueeze(0).to(device)  # [1, 4, 4]
            Ks_original = data["K"].unsqueeze(0).to(device)  # [1, 3, 3]
            image_data = data["image"].to(device) / 255.0  # [H, W, C]
            masks = data["mask"].to(device).unsqueeze(0) if "mask" in data else None  # [1, H, W]
            
            # Handle RGBA images
            if image_data.shape[-1] == 4:
                pixels_gt = image_data[..., :3]  # [H, W, 3]
            else:
                pixels_gt = image_data  # [H, W, 3]
            
            original_height, original_width = pixels_gt.shape[:2]
            
            # Coarse-to-fine: downsample if needed
            if cfg.use_coarse_to_fine and target_level is not None:
                # Downsample for coarse-to-fine (same logic as training)
                downsample_factor = 2 ** ((image_multigrid_max_level - target_level) / 2.0)
                downsample_factor = max(1, downsample_factor)
                
                height = int(max(1, original_height / downsample_factor))
                width = int(max(1, original_width / downsample_factor))
                
                # Downsample GT image
                if downsample_factor > 1:
                    pixels_gt_bchw = pixels_gt.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                    pixels_gt_downsampled_bchw = F.interpolate(
                        pixels_gt_bchw,
                        size=(height, width),
                        mode='bicubic',
                        align_corners=False,
                    )
                    pixels_gt_downsampled = pixels_gt_downsampled_bchw.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
                else:
                    pixels_gt_downsampled = pixels_gt
                
                # Downsample Ks
                Ks = Ks_original.clone()
                if downsample_factor > 1:
                    Ks[:, 0, 0] = Ks_original[:, 0, 0] / downsample_factor  # fx
                    Ks[:, 1, 1] = Ks_original[:, 1, 1] / downsample_factor  # fy
                    Ks[:, 0, 2] = Ks_original[:, 0, 2] / downsample_factor  # cx
                    Ks[:, 1, 2] = Ks_original[:, 1, 2] / downsample_factor  # cy
                else:
                    Ks = Ks_original
                
                # Downsample masks if provided
                if masks is not None and downsample_factor > 1:
                    masks_bchw = masks.unsqueeze(1).float()  # [1, 1, H, W]
                    masks_downsampled_bchw = F.interpolate(
                        masks_bchw,
                        size=(height, width),
                        mode='nearest',
                    )
                    masks = masks_downsampled_bchw.squeeze(1).bool()  # [1, H, W]
            else:
                # Normal visualization (no coarse-to-fine)
                height, width = original_height, original_width
                Ks = Ks_original
                pixels_gt_downsampled = pixels_gt
            
            # Render
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            colors = colors[0]  # [H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            
            # Upsample to original resolution for visualization
            if cfg.use_coarse_to_fine and target_level is not None and downsample_factor > 1:
                # Upsample rendered image to original resolution
                colors_bchw = colors.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                colors_upsampled_bchw = F.interpolate(
                    colors_bchw,
                    size=(original_height, original_width),
                    mode='bicubic',
                    align_corners=False,
                )
                colors_upsampled = colors_upsampled_bchw.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
                colors = colors_upsampled
                # Also upsample GT for consistency
                pixels_gt_downsampled_bchw = pixels_gt_downsampled.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                pixels_gt_upsampled_bchw = F.interpolate(
                    pixels_gt_downsampled_bchw,
                    size=(original_height, original_width),
                    mode='bicubic',
                    align_corners=False,
                )
                pixels_gt_downsampled = pixels_gt_upsampled_bchw.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
            
            # Convert to numpy and stack: [H, W, 3] for GT, [H, W, 3] for render
            pixels_gt_np = pixels_gt_downsampled.cpu().numpy()
            colors_np = colors.cpu().numpy()
            
            # Stack horizontally: [H, 2*W, 3]
            combined = np.concatenate([pixels_gt_np, colors_np], axis=1)
            images_list.append(combined)
        
        # Stack vertically: [3*H, 2*W, 3]
        if len(images_list) > 0:
            final_image = np.concatenate(images_list, axis=0)
            
            # Convert to uint8 and save as JPG
            final_image_uint8 = (final_image * 255).astype(np.uint8)
            level_suffix = f"_level{target_level}" if cfg.use_coarse_to_fine and target_level is not None else ""
            viz_path = f"{viz_dir}/viz_step_{step:06d}{level_suffix}.jpg"
            imageio.imwrite(viz_path, final_image_uint8, format='jpg', quality=95)
            print(f"  Saved visualization to {viz_path} (target_level={target_level if cfg.use_coarse_to_fine else 'N/A'})")

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
    print(cfg)
    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
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
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using hierarchy leaf nodes (no densification).",
            Config(
                hierarchy_path=None,  # Must be provided via CLI
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using hierarchy leaf nodes with MCMC strategy (no densification).",
            Config(
                hierarchy_path=None,  # Must be provided via CLI
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    # Manually call __post_init__ after CLI parsing to ensure result_dir and data_dir
    # are auto-generated with the correct CLI-provided values (e.g., data_factor)
    cfg.__post_init__()
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
