"""
Evaluate multigrid Gaussian Splatting model at different levels.

Usage:
    python eval_levels.py --ckpt_file /path/to/checkpoint.pt --data_dir /path/to/data --output_dir /path/to/output
    python eval_levels.py --ckpt_dir /path/to/ckpt/dir --data_dir /path/to/data --output_dir /path/to/output
"""

import argparse
import json
import os
import glob
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import imageio
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.colmap import Dataset as ColmapDataset, Parser as ColmapParser
from datasets.nerf import Dataset as NerfDataset, Parser as NerfParser
from multigrid_gaussians_v9 import MultigridGaussians


def load_multigrid_from_checkpoint(
    ckpt_path: str,
    cfg: dict,
    device: str = "cuda",
):
    """
    Load MultigridGaussians from checkpoint file.
    
    Args:
        ckpt_path: Path to checkpoint file (.pt)
        device: Device to load on
    
    Returns:
        multigrid_gaussians: MultigridGaussians instance
        step: Training step number
        config: Dictionary with checkpoint config
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract checkpoint information
    step = ckpt.get("step", 0)
    splats_dict = ckpt["splats"]
    levels = ckpt["levels"].to(device)
    parent_indices = ckpt["parent_indices"].to(device)
    level_indices = ckpt.get("level_indices", {})
    position_scale_reduction = ckpt.get("position_scale_reduction", 0.75)
    max_level = ckpt.get("max_level", None)
    sh_degree = ckpt.get("sh_degree", 3)
    
    # Get number of gaussians
    N = len(levels)
    print(f"Loaded {N} gaussians from step {step}")
    
    # Create a minimal MultigridGaussians instance
    # We need to create a dummy parser for initialization
    class DummyParser:
        def __init__(self):
            self.points = torch.zeros((N, 3)).numpy()
            self.points_rgb = torch.zeros((N, 3)).numpy()
            self.scene_scale = 1.0
    
    dummy_parser = DummyParser()
    
    # Initialize MultigridGaussians with dummy data
    multigrid_gaussians = MultigridGaussians(
        parser=dummy_parser,
        cfg=cfg,
        init_type="random",
        init_num_pts=N,
        init_extent=1.0,
        init_opacity=0.1,
        init_scale=1.0,
        scene_scale=1.0,
        sh_degree=sh_degree,
        device=device,
        position_scale_reduction=position_scale_reduction,
        max_level=max_level,
    )
    
    # Replace splats with loaded ones
    for key, value in splats_dict.items():
        if key in multigrid_gaussians.splats:
            multigrid_gaussians.splats[key].data = value.to(device)
    
    # Set hierarchical structure
    multigrid_gaussians.levels = levels
    multigrid_gaussians.parent_indices = parent_indices
    multigrid_gaussians.level_indices = level_indices
    multigrid_gaussians.max_level = max_level
    
    print(f"Loaded hierarchical structure: {len(level_indices)} levels")
    for level, indices in level_indices.items():
        print(f"  Level {level}: {len(indices)} gaussians")
    
    config = {
        "sh_degree": sh_degree,
        "position_scale_reduction": position_scale_reduction,
        "max_level": max_level,
    }
    
    return multigrid_gaussians, step, config

@torch.no_grad()
def evaluate_levels(
    multigrid_gaussians: MultigridGaussians,
    valloader: DataLoader,
    valset,
    device: str,
    output_dir: Optional[str] = None,
    sh_degree: int = 3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    white_background: bool = False,
    camera_model: str = "pinhole",
    packed: bool = False,
    sparse_grad: bool = False,
    use_bilateral_grid: bool = False,
    lpips_net: str = "alex",
):
    """
    Evaluate model at different levels.
    Uses the same logic as vcycle_trainer.py Runner.eval() but without image saving.
    
    Args:
        valset: Validation dataset (for sampling camera indices)
        output_dir: Directory to save visualization images (first 2 eval images)
    
    Returns:
        summary: Dictionary mapping level -> average metrics
        detailed_results: Dictionary mapping level -> list of metrics per image
    """
    
    # Sample camera indices for visualization (same as vcycle_trainer.py)
    val_indices = list(range(len(valset)))
    if len(val_indices) >= 3:
        random.seed(42)  # For reproducibility (same as trainer)
        viz_camera_indices = random.sample(val_indices, 3)
    else:
        viz_camera_indices = val_indices[:3] if len(val_indices) > 0 else []
    # Use first 2 indices for saving images
    save_image_indices = viz_camera_indices[:2] if len(viz_camera_indices) >= 2 else viz_camera_indices
    print(f"Selected {len(viz_camera_indices)} cameras for visualization: {viz_camera_indices}")
    if len(save_image_indices) > 0:
        print(f"Will save images for first {len(save_image_indices)} cameras: {save_image_indices}")
    
    # Get available levels
    if len(multigrid_gaussians.levels) > 0:
        max_level = int(multigrid_gaussians.levels.max().item())
        levels_to_render = list(range(1, max_level + 1))
    else:
        levels_to_render = [1]
    
    # Initialize metrics (same as trainer)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    if lpips_net == "alex":
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    elif lpips_net == "vgg":
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)
    else:
        raise ValueError(f"Unknown LPIPS network: {lpips_net}")
    
    # Store results for each level
    results = {level: defaultdict(list) for level in levels_to_render + [-1]}
    render_times = {level: [] for level in levels_to_render + [-1]}
    
    ellipse_time = 0
    metrics = defaultdict(list)
    
    print(f"\nEvaluating {len(valloader)} images at levels: {levels_to_render + [-1]}")
    
    # Get step from multigrid_gaussians or use 0 as default
    step = getattr(multigrid_gaussians, 'step', 0)
    
    for i, data in enumerate(valloader):
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
        
        masks = data["mask"].to(device) if "mask" in data else None
        if masks is not None and masks.dim() == 2:
            masks = masks.unsqueeze(0)
        
        # Handle RGBA images
        if image_data.shape[-1] == 4:
            pixels = image_data[..., :3]  # [1, H, W, 3]
        else:
            pixels = image_data  # [1, H, W, 3]
        
        height, width = pixels.shape[1:3]
        
        torch.cuda.synchronize()
        tic = time.time()
        
        # Prepare backgrounds
        backgrounds = None
        if white_background:
            backgrounds = torch.ones(1, 3, device=device)
        
        # Render at highest LOD for metrics (finest level) - same as trainer
        colors, alphas, _ = multigrid_gaussians.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            level=-1,  # Highest LOD for metrics (finest level)
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            masks=masks,
            packed=packed,
            sparse_grad=sparse_grad,
            distributed=False,
            camera_model=camera_model,
            backgrounds=backgrounds,
        )
        
        torch.cuda.synchronize()
        ellipse_time += max(time.time() - tic, 1e-10)
        
        colors = torch.clamp(colors, 0.0, 1.0)
        
        # Compute metrics using highest LOD only (finest level) - same as trainer
        pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        metrics["psnr"].append(psnr_fn(colors_p, pixels_p))
        metrics["ssim"].append(ssim_fn(colors_p, pixels_p))
        metrics["lpips"].append(lpips_fn(colors_p, pixels_p))
        
        if use_bilateral_grid:
            from utils import color_correct
            cc_colors = color_correct(colors, pixels)
            cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["cc_psnr"].append(psnr_fn(cc_colors_p, pixels_p))
            metrics["cc_ssim"].append(ssim_fn(cc_colors_p, pixels_p))
            metrics["cc_lpips"].append(lpips_fn(cc_colors_p, pixels_p))
        
        # Evaluate at each level (for level-wise metrics)
        for level in levels_to_render:
            torch.cuda.synchronize()
            tic_level = time.time()
            
            colors_level, alphas_level, _ = multigrid_gaussians.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                level=level,
                sh_degree=sh_degree,
                near_plane=near_plane,
                far_plane=far_plane,
                masks=masks,
                packed=packed,
                sparse_grad=sparse_grad,
                distributed=False,
                camera_model=camera_model,
                backgrounds=backgrounds,
            )
            
            torch.cuda.synchronize()
            render_time = time.time() - tic_level
            render_times[level].append(render_time)
            
            colors_level = torch.clamp(colors_level, 0.0, 1.0)
            
            # Compute metrics for this level
            colors_level_p = colors_level.permute(0, 3, 1, 2)  # [1, 3, H, W]
            psnr_val = psnr_fn(colors_level_p, pixels_p).item()
            ssim_val = ssim_fn(colors_level_p, pixels_p).item()
            lpips_val = lpips_fn(colors_level_p, pixels_p).item()
            
            results[level]["psnr"].append(psnr_val)
            results[level]["ssim"].append(ssim_val)
            results[level]["lpips"].append(lpips_val)
            
            if i == 0:  # Print first image for each level
                print(f"  L{level}: PSNR={psnr_val:.3f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.3f}, Time={render_time:.3f}s")
        
        # Also evaluate finest level
        render_times[-1].append(max(time.time() - tic, 1e-10))
        results[-1]["psnr"].append(metrics["psnr"][-1].item())
        results[-1]["ssim"].append(metrics["ssim"][-1].item())
        results[-1]["lpips"].append(metrics["lpips"][-1].item())
        
        # Save visualization images for first 2 eval images (level-wise renderings and diff maps)
        if output_dir is not None and i in save_image_indices:
            pixels_gt_np = pixels[0].cpu().numpy()  # [H, W, 3]
            
            # Prepare backgrounds if needed
            backgrounds = None
            if white_background:
                backgrounds = torch.ones(1, 3, device=device)
            
            # Store previous level rendering for diff calculation
            prev_colors_np = None
            
            # Render at each level and save separately
            for render_level in levels_to_render:
                colors_level, alphas_level, _ = multigrid_gaussians.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    level=render_level,
                    sh_degree=sh_degree,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    masks=masks,
                    packed=packed,
                    sparse_grad=sparse_grad,
                    distributed=False,
                    camera_model=camera_model,
                    backgrounds=backgrounds,
                )  # colors_level: [1, H, W, 3], alphas_level: [1, H, W, 1]
                colors_level = colors_level[0]  # [H, W, 3]
                colors_level = torch.clamp(colors_level, 0.0, 1.0).cpu().numpy()
                
                # Save level rendering
                os.makedirs(output_dir, exist_ok=True)
                render_path = os.path.join(output_dir, f"render_level_{render_level}_cam_{i:04d}_step_{step:04d}.png")
                render_uint8 = (colors_level * 255).astype(np.uint8)
                imageio.imwrite(render_path, render_uint8)
                
                # Calculate and save diff vs GT
                diff_vs_gt = np.abs(colors_level - pixels_gt_np)  # [H, W, 3]
                # Normalize to [0, 1] and apply colormap-like visualization (multiply by 3 for better visibility)
                diff_vs_gt_vis = np.clip(diff_vs_gt * 3.0, 0.0, 1.0)
                diff_gt_path = os.path.join(output_dir, f"diff_level_{render_level}_vs_GT_cam_{i:04d}_step_{step:04d}.png")
                diff_gt_uint8 = (diff_vs_gt_vis * 255).astype(np.uint8)
                imageio.imwrite(diff_gt_path, diff_gt_uint8)
                
                # Calculate and save diff vs previous level (if exists)
                if prev_colors_np is not None:
                    diff_vs_prev = np.abs(colors_level - prev_colors_np)  # [H, W, 3]
                    # Normalize to [0, 1] and apply colormap-like visualization
                    diff_vs_prev_vis = np.clip(diff_vs_prev * 3.0, 0.0, 1.0)
                    diff_prev_path = os.path.join(output_dir, f"diff_level_{render_level}_vs_{render_level-1}_cam_{i:04d}_step_{step:04d}.png")
                    diff_prev_uint8 = (diff_vs_prev_vis * 255).astype(np.uint8)
                    imageio.imwrite(diff_prev_path, diff_prev_uint8)
                
                # Update previous level for next iteration
                prev_colors_np = colors_level.copy()
            
            print(f"  Saved level-wise visualizations for camera {i} (levels: {levels_to_render})")
    
    # Compute average metrics for each level (same as trainer)
    ellipse_time /= len(valloader)
    
    stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
    stats.update({
        "ellipse_time": ellipse_time,
    })
    
    # Compute level-wise summary
    summary = {}
    for level in levels_to_render + [-1]:
        level_name = "finest" if level == -1 else f"L{level}"
        summary[level_name] = {
            "psnr": sum(results[level]["psnr"]) / len(results[level]["psnr"]),
            "ssim": sum(results[level]["ssim"]) / len(results[level]["ssim"]),
            "lpips": sum(results[level]["lpips"]) / len(results[level]["lpips"]),
            "render_time": sum(render_times[level]) / len(render_times[level]),
            "num_images": len(results[level]["psnr"]),
        }
    
    return summary, results, stats


def load_config_from_yaml(cfg_path: str) -> dict:
    """Load config from yaml file.
    
    Note: Uses unsafe_load because training configs may contain Python objects
    (tuples, dataclass instances) that require unsafe loading.
    """
    with open(cfg_path, "r") as f:
        try:
            cfg = yaml.unsafe_load(f)
        except AttributeError:
            # Fallback for older PyYAML versions
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate multigrid model at different levels")
    parser.add_argument("--cfg", type=str, default=None, help="Path to cfg.yml file from training (if not provided, will look for cfg.yml in ckpt_dir)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory containing checkpoint files (or result_dir from training)")
    parser.add_argument("--ckpt_file", type=str, default=None, help="Path to specific checkpoint file")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory (overrides cfg.yml)")
    parser.add_argument("--dataset_type", type=str, default=None, choices=["nerf", "colmap"], help="Dataset type (overrides cfg.yml)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results (overrides cfg.yml)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--step", type=int, default=None, help="Specific step to evaluate (if multiple checkpoints)")
    parser.add_argument("--data_factor", type=int, default=None, help="Downsample factor for dataset (overrides cfg.yml)")
    parser.add_argument("--white_background", action="store_true", help="Use white background for RGBA images (overrides cfg.yml)")
    parser.add_argument("--image_extension", type=str, default=None, help="Image file extension for NeRF datasets (overrides cfg.yml)")
    
    args = parser.parse_args()
    
    # Load config from yaml if available
    cfg = {}
    cfg_path = None
    if args.cfg:
        cfg_path = args.cfg
        if os.path.exists(cfg_path):
            cfg = load_config_from_yaml(cfg_path)
            print(f"Loaded config from: {cfg_path}")
            # If cfg is provided, try final_checkpoint first, then ckpts subdirectory
            cfg_dir = os.path.dirname(cfg_path)
            final_checkpoint_dir = os.path.join(cfg_dir, "final_checkpoint")
            default_ckpt_dir = os.path.join(cfg_dir, "ckpts")
            if args.ckpt_dir is None:
                # Prefer final_checkpoint if it exists, otherwise use ckpts
                if os.path.exists(final_checkpoint_dir):
                    args.ckpt_dir = final_checkpoint_dir
                    print(f"Using final_checkpoint from cfg location: {final_checkpoint_dir}")
                else:
                    args.ckpt_dir = default_ckpt_dir
                    print(f"Using ckpt_dir from cfg location: {default_ckpt_dir}")
    elif args.ckpt_dir:
        # Try to find cfg.yml in ckpt_dir (result_dir)
        cfg_path = os.path.join(args.ckpt_dir, "cfg.yml")
        if os.path.exists(cfg_path):
            cfg = load_config_from_yaml(cfg_path)
            print(f"Loaded config from: {cfg_path}")
    elif args.ckpt_file:
        # Try to find cfg.yml in the same directory as checkpoint
        ckpt_dir = os.path.dirname(args.ckpt_file)
        cfg_path = os.path.join(ckpt_dir, "cfg.yml")
        if os.path.exists(cfg_path):
            cfg = load_config_from_yaml(cfg_path)
            print(f"Loaded config from: {cfg_path}")
    # Use config values with command line args as overrides
    data_dir = args.data_dir if args.data_dir is not None else cfg.get("data_dir")
    dataset_type = args.dataset_type if args.dataset_type is not None else cfg.get("dataset_type", "nerf")
    result_dir = cfg.get("result_dir") if cfg.get("result_dir") else (os.path.dirname(cfg_path) if cfg_path else None)
    data_factor = args.data_factor if args.data_factor is not None else cfg.get("data_factor", 4)
    white_background = args.white_background if args.white_background else cfg.get("white_background", False)
    image_extension = args.image_extension if args.image_extension is not None else cfg.get("image_extension", ".png")
    sh_degree = cfg.get("sh_degree", 3)
    near_plane = cfg.get("near_plane", 0.01)
    far_plane = cfg.get("far_plane", 1e10)
    camera_model = cfg.get("camera_model", "pinhole")
    packed = cfg.get("packed", False)
    sparse_grad = cfg.get("sparse_grad", False)
    
    # Set output directory: use result_dir/levelwise_eval if cfg is available, otherwise use args.output_dir
    if args.output_dir:
        output_dir = args.output_dir
    elif result_dir:
        output_dir = os.path.join(result_dir, "levelwise_eval")
    else:
        raise ValueError("output_dir must be provided either via --output_dir or in cfg.yml (as result_dir)")
    
    # Validate required arguments
    if data_dir is None:
        raise ValueError("data_dir must be provided either via --data_dir or in cfg.yml")
    
    # Determine checkpoint file(s)
    if args.ckpt_file:
        ckpt_files = [args.ckpt_file]
    elif args.ckpt_dir:
        # Check if this is result_dir (contains final_checkpoint or ckpts subdirectories)
        final_checkpoint_dir = os.path.join(args.ckpt_dir, "final_checkpoint")
        ckpts_dir = os.path.join(args.ckpt_dir, "ckpts")
        
        ckpt_files = []
        
        # Prefer final_checkpoint if it exists
        if os.path.exists(final_checkpoint_dir):
            final_ckpt_pattern = os.path.join(final_checkpoint_dir, "*.pt")
            final_ckpt_files = glob.glob(final_ckpt_pattern)
            if not final_ckpt_files:
                final_ckpt_pattern = os.path.join(final_checkpoint_dir, "**", "*.pt")
                final_ckpt_files = glob.glob(final_ckpt_pattern, recursive=True)
            if final_ckpt_files:
                ckpt_files.extend(final_ckpt_files)
                print(f"Found {len(final_ckpt_files)} checkpoint file(s) in final_checkpoint")
        
        # Also check ckpts directory if no files found in final_checkpoint
        if not ckpt_files and os.path.exists(ckpts_dir):
            ckpt_pattern = os.path.join(ckpts_dir, "*.pt")
            ckpts_files = glob.glob(ckpt_pattern)
            if not ckpts_files:
                ckpt_pattern = os.path.join(ckpts_dir, "**", "*.pt")
                ckpts_files = glob.glob(ckpt_pattern, recursive=True)
            if ckpts_files:
                ckpt_files.extend(ckpts_files)
                print(f"Found {len(ckpts_files)} checkpoint file(s) in ckpts")
        
        # If still no files, check the ckpt_dir itself directly
        if not ckpt_files:
            ckpt_pattern = os.path.join(args.ckpt_dir, "*.pt")
            ckpt_files = glob.glob(ckpt_pattern)
            if not ckpt_files:
                ckpt_pattern = os.path.join(args.ckpt_dir, "**", "*.pt")
                ckpt_files = glob.glob(ckpt_pattern, recursive=True)
        
        if not ckpt_files:
            raise ValueError(f"No checkpoint files found in {args.ckpt_dir} (checked final_checkpoint, ckpts, and direct)")
        
        # Filter by step if specified
        if args.step is not None:
            ckpt_files = [f for f in ckpt_files if f"step_{args.step}" in f]
        
        # Sort by step number
        def extract_step(f):
            try:
                parts = Path(f).stem.split("_")
                for i, part in enumerate(parts):
                    if part == "step" and i + 1 < len(parts):
                        return int(parts[i + 1])
            except:
                return 0
            return 0
        
        ckpt_files.sort(key=extract_step, reverse=True)
        print(f"Found {len(ckpt_files)} checkpoint file(s) total")
    else:
        raise ValueError("Either --ckpt_dir or --ckpt_file must be specified")
    
    # Load dataset
    print(f"\nLoading dataset from: {data_dir}")
    if dataset_type == "colmap":
        dataset_parser = ColmapParser(
            data_dir=data_dir,
            factor=data_factor,
            normalize=cfg.get("normalize_world_space", True),
            test_every=cfg.get("test_every", 8),
        )
        valset = ColmapDataset(dataset_parser, split="val")
    elif dataset_type == "nerf":
        dataset_parser = NerfParser(
            data_dir=data_dir,
            factor=data_factor,
            normalize=cfg.get("normalize_world_space", True),
            test_every=cfg.get("test_every", 8),
            white_background=white_background,
            extension=image_extension,
        )
        valset = NerfDataset(dataset_parser, split="val")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
    print(f"Loaded {len(valset)} validation images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each checkpoint
    for ckpt_file in ckpt_files:
        print(f"\n{'='*60}")
        print(f"Processing: {ckpt_file}")
        print(f"{'='*60}")
        
        try:
            # Load checkpoint
            multigrid_gaussians, step, config = load_multigrid_from_checkpoint(
                ckpt_file,
                cfg=cfg,
                device=args.device,
            )
            
            # Store step in multigrid_gaussians for use in evaluate_levels
            multigrid_gaussians.step = step
            
            # Evaluate at different levels
            summary, detailed_results, stats = evaluate_levels(
                multigrid_gaussians=multigrid_gaussians,
                valloader=valloader,
                valset=valset,
                device=args.device,
                output_dir=output_dir,
                sh_degree=sh_degree,
                near_plane=near_plane,
                far_plane=far_plane,
                white_background=white_background,
                camera_model=camera_model,
                packed=packed,
                sparse_grad=sparse_grad,
                use_bilateral_grid=cfg.get("use_bilateral_grid", False),
                lpips_net=cfg.get("lpips_net", "alex"),
            )
            
            # Print summary (same format as trainer)
            print(f"\n{'='*60}")
            print(f"Evaluation Summary (Step {step}):")
            print(f"{'='*60}")
            
            # Print finest level metrics (same as trainer)
            if cfg.get("use_bilateral_grid", False):
                print(
                    f"Finest: PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {len(multigrid_gaussians.splats['means'])}"
                )
            else:
                print(
                    f"Finest: PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {len(multigrid_gaussians.splats['means'])}"
                )
            
            # Print level-wise metrics
            print("\nLevel-wise metrics:")
            for level_name, metrics in sorted(summary.items()):
                if level_name != "finest":  # Already printed above
                    print(
                        f"  {level_name:10s}: "
                        f"PSNR={metrics['psnr']:.3f}, "
                        f"SSIM={metrics['ssim']:.4f}, "
                        f"LPIPS={metrics['lpips']:.3f}, "
                        f"Time={metrics['render_time']:.3f}s/image"
                    )
            
            # Save results (same format as trainer)
            output_file = os.path.join(output_dir, f"val_step{step:04d}.json")
            stats_to_save = stats.copy()
            stats_to_save["num_GS"] = len(multigrid_gaussians.splats["means"])
            stats_to_save["step"] = step
            stats_to_save["levelwise"] = summary
            
            with open(output_file, "w") as f:
                json.dump(stats_to_save, f, indent=2)
            
            print(f"\nSaved results to: {output_file}")
            
        except Exception as e:
            print(f"Error processing {ckpt_file}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

