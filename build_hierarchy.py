"""
Build hierarchical 3D Gaussian representation from trained point cloud.

This script implements the hierarchical structure building method from:
"A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets"
(https://arxiv.org/pdf/2406.12080)

The hierarchy is built by:
1. Loading trained point cloud from PLY file
2. Building hierarchical levels using spatial clustering
3. Merging Gaussians based on geometric and volumetric properties
4. Saving the hierarchical structure
"""

import argparse
import struct
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import json
from tqdm import tqdm
import faiss
import fpsample
from gsplat.cuda._wrapper import quat_scale_to_covar_preci


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: [3, 3] rotation matrix
        
    Returns:
        quat: [4] quaternion (w, x, y, z)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    quat = torch.stack([w, x, y, z])
    return F.normalize(quat, p=2, dim=-1)


def rotmat_to_quat_batch(R: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of rotation matrices to quaternions (vectorized).
    
    Args:
        R: [N, 3, 3] batch of rotation matrices
        
    Returns:
        quats: [N, 4] batch of quaternions (w, x, y, z)
    """
    N = R.shape[0]
    device = R.device
    dtype = R.dtype
    
    # Compute trace for each rotation matrix
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # [N,]
    
    # Initialize output quaternions
    quats = torch.zeros(N, 4, device=device, dtype=dtype)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # [M,]
        quats[mask1, 0] = 0.25 * s  # w
        quats[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # x
        quats[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # y
        quats[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # z
    
    # Case 2: R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # [M,]
        quats[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # w
        quats[mask2, 1] = 0.25 * s  # x
        quats[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # y
        quats[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # z
    
    # Case 3: R[1, 1] > R[2, 2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # [M,]
        quats[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # w
        quats[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # x
        quats[mask3, 2] = 0.25 * s  # y
        quats[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # z
    
    # Case 4: else (R[2, 2] is largest)
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # [M,]
        quats[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # w
        quats[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # x
        quats[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # y
        quats[mask4, 3] = 0.25 * s  # z
    
    # Normalize quaternions
    return F.normalize(quats, p=2, dim=-1)


def load_ply(ply_path: str) -> Dict[str, torch.Tensor]:
    """
    Load 3D Gaussian point cloud from PLY file.
    
    This function reads PLY files saved by export_splats() which stores:
    - x, y, z (means)
    - f_dc_* (SH band 0, 3 values)
    - f_rest_* (SH higher bands, K*3 values)
    - opacity
    - scale_0, scale_1, scale_2
    - rot_0, rot_1, rot_2, rot_3 (quaternions)
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        Dictionary containing:
            - means: [N, 3] position
            - scales: [N, 3] scale (in log space)
            - quats: [N, 4] quaternion rotation
            - opacities: [N] opacity (in logit space)
            - sh0: [N, 1, 3] spherical harmonics band 0
            - shN: [N, K, 3] spherical harmonics higher bands
    """
    with open(ply_path, "rb") as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if b"end_header" in line:
                break
        
        # Parse header to get property names and counts
        num_vertices = 0
        properties = []
        for line in header_lines:
            line_str = line.decode("ascii", errors="ignore").strip()
            if line_str.startswith("element vertex"):
                num_vertices = int(line_str.split()[-1])
            elif line_str.startswith("property float"):
                prop_name = line_str.split()[-1]
                properties.append(prop_name)
        
        # Determine SH dimensions
        f_dc_count = sum(1 for p in properties if p.startswith("f_dc_"))
        f_rest_count = sum(1 for p in properties if p.startswith("f_rest_"))
        
        # Calculate K (number of SH bands beyond DC)
        # f_rest has K*3 values, so K = f_rest_count / 3
        K = f_rest_count // 3 if f_rest_count > 0 else 0
        
        # Read binary data
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape(num_vertices, -1)
        
        # Extract properties based on PLY format from export_splats
        idx = 0
        means = data[:, idx:idx+3]  # x, y, z
        idx += 3
        
        sh0 = data[:, idx:idx+f_dc_count]  # f_dc_* (should be 3 values)
        idx += f_dc_count
        
        shN = data[:, idx:idx+f_rest_count]  # f_rest_* (K*3 values)
        idx += f_rest_count
        
        opacities = data[:, idx]  # opacity
        idx += 1
        
        scales = data[:, idx:idx+3]  # scale_0, scale_1, scale_2
        idx += 3
        
        quats = data[:, idx:idx+4]  # rot_0, rot_1, rot_2, rot_3
        idx += 4
        
        # Convert to torch tensors
        # IMPORTANT: export_splats() with format="ply" stores scales and opacities in log/logit space (not actual values)
        # The splat2ply_bytes function in gsplat/exporter.py saves them directly without conversion.
        # However, simple_trainer_original.py calls export_splats with log/logit space values,
        # so PLY files contain log/logit space values.
        # Therefore, we should NOT convert them again - use them as-is.
        scales_log = torch.from_numpy(scales).float()
        opacities_logit = torch.from_numpy(opacities).float()
        
        # Reshape SH coefficients
        sh0_tensor = torch.from_numpy(sh0.copy()).float()  # [N, 3] - copy to make writable
        sh0_tensor = sh0_tensor.reshape(-1, 1, 3)  # [N, 1, 3]
        
        shN_tensor = torch.from_numpy(shN.copy()).float()  # [N, K*3] - copy to make writable
        shN_tensor = shN_tensor.reshape(-1, K, 3)  # [N, K, 3]
        
        return {
            "means": torch.from_numpy(means).float(),
            "scales": scales_log,
            "quats": torch.from_numpy(quats).float(),
            "opacities": opacities_logit,
            "sh0": sh0_tensor,
            "shN": shN_tensor,
        }


def merge_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge multiple Gaussians into a single Gaussian based on geometric and volumetric properties.
    
    This implements the merging method from the paper (Section 4.1), which combines Gaussians based on:
    - Local geometric properties: position (mean), scale, orientation (quaternion)
    - Volumetric properties: opacity, color (SH coefficients)
    
    The merging uses weighted averages with weights computed from opacity and surface area:
    - Weight formula (Eq. 8): w'_i = o_i * S_i
    - where o_i is opacity and S_i is volume measure (approximated as (det(Σ_3D_i))^(1/3) for 3D Gaussian)
    - Normalized weights: w_i = w'_i / sum(w'_i)
    
    All parameters are merged using: merged_param = sum_i(w_i * param_i)
    
    Args:
        means: [N, 3] positions
        scales: [N, 3] scales (in log space)
        quats: [N, 4] quaternions
        opacities: [N] opacities (in logit space)
        sh0: [N, 1, 3] SH band 0
        shN: [N, K, 3] SH higher bands
        weights: [N] optional weights for weighted averaging (e.g., based on opacity)
        
    Returns:
        Merged Gaussian parameters (single values, not batched)
    """
    N = means.shape[0]
    
    if weights is None:
        # Weight formula from paper (Section 4.1, Eq. 8):
        # w'_i = o_i * S_i
        # where o_i is opacity and S_i is the surface area
        # 
        # According to the paper: "In practice, since the square root of the determinant 
        # of a Gaussian's 2D covariance is proportional to the (projected) surface of the 
        # corresponding 3D ellipsoid, we compute the surface S_i"
        # 
        # For 3D Gaussian, we use the cube root of the determinant (volume measure):
        # S_i ≈ (det(Σ_3D_i))^(1/3)
        # 
        # This is equivalent to: w'_i = o_i * (det(Σ_3D_i))^(1/3)
        device = means.device
        
        # Compute 3D covariance matrices from scale and quaternion
        scales_exp = torch.exp(scales).clamp_min(1e-6).clamp_max(1e6)  # [N, 3]
        covars, _ = quat_scale_to_covar_preci(
            quats=quats,
            scales=scales_exp,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )  # [N, 3, 3]
        
        # Compute volume measure S_i ≈ (det(Σ_3D_i))^(1/3) for 3D Gaussian
        # For 3D Gaussian, we use power of 1/3 (volume measure) instead of sqrt (2D area measure)
        opacities_clamped = torch.sigmoid(opacities).clamp_min(1e-8)  # [N,]
        covar_dets = torch.det(covars).clamp_min(1e-8)  # [N,]
        surface_area = torch.pow(covar_dets, 1.0/3.0)  # [N,] - S_i = (det)^(1/3) for 3D Gaussian
        
        # Unnormalized weights: w'_i = o_i * S_i
        unnormalized_weights = opacities_clamped * surface_area  # [N,]
        
        # Normalize weights: w_i = w'_i / sum(w'_i)
        weights = unnormalized_weights / (unnormalized_weights.sum() + 1e-8)
    else:
        weights = weights / (weights.sum() + 1e-8)
    
    # Merging formulas from paper (Section 4.1):
    # All parameters are merged using weighted average with normalized weights w_i
    
    # Merge means: μ^(l+1) = ∑_i^N w_i * μ_i^(l) (Eq. 3)
    merged_mean = (means * weights.unsqueeze(-1)).sum(dim=0)
    
    # Merge covariance: Σ^(l+1) = ∑_i^N w_i * (Σ_i^(l) + (μ_i^(l) - μ^(l+1)) * (μ_i^(l) - μ^(l+1))^T) (Eq. 4)
    # First compute individual covariance matrices
    scales_exp = torch.exp(scales).clamp_min(1e-6).clamp_max(1e6)  # [N, 3]
    covars, _ = quat_scale_to_covar_preci(
        quats=quats,
        scales=scales_exp,
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )  # [N, 3, 3]
    
    # Compute mean differences: (μ_i^(l) - μ^(l+1))
    mean_diffs = means - merged_mean.unsqueeze(0)  # [N, 3]
    
    # Compute outer products: (μ_i^(l) - μ^(l+1)) * (μ_i^(l) - μ^(l+1))^T
    mean_diff_outer = torch.einsum('ni,nj->nij', mean_diffs, mean_diffs)  # [N, 3, 3]
    
    # Compute weighted sum: Σ^(l+1) = ∑_i^N w_i * (Σ_i^(l) + mean_diff_outer)
    weighted_covars = (covars + mean_diff_outer) * weights.view(-1, 1, 1)  # [N, 3, 3]
    merged_covar = weighted_covars.sum(dim=0)  # [3, 3]
    
    # Decompose merged covariance back to quaternion and scale using eigendecomposition
    # Σ = R * S^2 * R^T, where R is rotation matrix and S is diagonal scale matrix
    eigenvals, eigenvecs = torch.linalg.eigh(merged_covar)  # eigenvals: [3], eigenvecs: [3, 3]
    eigenvals = eigenvals.clamp_min(1e-6)  # Ensure positive
    
    # Extract scale: S = sqrt(eigenvals)
    merged_scale = torch.log(torch.sqrt(eigenvals))  # [3] in log space
    
    # Extract rotation matrix and convert to quaternion
    # eigenvecs is already orthonormal (rotation matrix)
    merged_quat = rotmat_to_quat(eigenvecs)
    
    # Merge opacities: compute per-node falloff then store in log-space
    # We preserve the integrated opacity mass by matching:
    #   o_p * S_p  ≈  ∑_i (o_i * S_i)
    # where o_i = sigmoid(opacity_i) and S_i is the volume/surface proxy.
    # This yields:
    #   o_p ≈ (∑_i w'_i) / S_p   with w'_i = o_i * S_i
    # We store logit(o_p) as log(falloff) where falloff = (∑_i w'_i) / S_p,
    # because sigmoid(log(falloff)) = falloff / (1 + falloff).
    # Compute volume measure of merged Gaussian: S_p ≈ (det(Σ_merged))^(1/3) for 3D Gaussian
    merged_covar_det = torch.det(merged_covar).clamp_min(1e-8)
    merged_surface_area = torch.pow(merged_covar_det, 1.0/3.0)
    
    # Falloff = ∑_i^N w'_i * S_p
    # Note: w'_i are unnormalized weights (o_i * S_i), not normalized w_i
    # We need to use unnormalized_weights from weight calculation
    if weights is None:
        # This shouldn't happen since weights are computed above, but for safety
        opacities_clamped = torch.sigmoid(opacities).clamp_min(1e-8)  # [N,]
        covar_dets = torch.det(covars).clamp_min(1e-8)  # [N,]
        surface_area = torch.pow(covar_dets, 1.0/3.0)  # [N,] - (det)^(1/3) for 3D Gaussian
        unnormalized_weights = opacities_clamped * surface_area  # [N,]
    else:
        # Recompute unnormalized weights for falloff calculation
        opacities_clamped = torch.sigmoid(opacities).clamp_min(1e-8)  # [N,]
        covar_dets = torch.det(covars).clamp_min(1e-8)  # [N,]
        surface_area = torch.pow(covar_dets, 1.0/3.0)  # [N,] - (det)^(1/3) for 3D Gaussian
        unnormalized_weights = opacities_clamped * surface_area  # [N,]
    
    # Falloff = (∑_i w'_i) / S_p
    falloff = unnormalized_weights.sum() / merged_surface_area.clamp_min(1e-12)
    merged_opacity = falloff.clamp_min(1e-12).log()  # logit(o_p) where o_p = falloff / (1 + falloff)
    
    # Merge SH coefficients: weighted average using the same weights
    # From paper: "We can similarly use the weighted average of the SH coefficients for the merged node using the same weight"
    merged_sh0 = (sh0 * weights.view(-1, 1, 1)).sum(dim=0)  # [1, 3]
    merged_shN = (shN * weights.view(-1, 1, 1)).sum(dim=0)  # [K, 3]
    
    return merged_mean, merged_scale, merged_quat, merged_opacity, merged_sh0, merged_shN


def build_hierarchy_level(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    target_num: int,
    method: str = "fps_faiss",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build one level of hierarchy by clustering and merging Gaussians.
    
    Args:
        means: [N, 3] positions
        scales: [N, 3] scales (in log space)
        quats: [N, 4] quaternions
        opacities: [N] opacities (in logit space)
        sh0: [N, 1, 3] SH band 0
        shN: [N, K, 3] SH higher bands
        target_num: Target number of Gaussians for this level
        method: Clustering method (ignored; FPS + closest points is always used)
        
    Returns:
        Merged parameters and cluster assignments [M] where M is the number of clusters
    """
    N = means.shape[0]
    device = means.device
    
    # Ensure all input tensors are on the correct device
    means = means.to(device)
    scales = scales.to(device)
    quats = quats.to(device)
    opacities = opacities.to(device)
    sh0 = sh0.to(device)
    shN = shN.to(device)

    # Sanitize inputs to prevent NaN/inf propagation into covariance/weights
    means = torch.where(torch.isfinite(means), means, torch.zeros_like(means))
    scales = torch.where(torch.isfinite(scales), scales, torch.zeros_like(scales))
    quats = torch.where(torch.isfinite(quats), quats, torch.zeros_like(quats))
    opacities = torch.where(torch.isfinite(opacities), opacities, torch.zeros_like(opacities))
    sh0 = torch.where(torch.isfinite(sh0), sh0, torch.zeros_like(sh0))
    shN = torch.where(torch.isfinite(shN), shN, torch.zeros_like(shN))
    # Normalize quaternions for stable covariance computation
    quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True).clamp_min(1e-8)
    
    if N == 0:
        empty = torch.zeros((0,), device=device, dtype=torch.long)
        return means, scales, quats, opacities, sh0, shN, empty, empty

    target_num = min(max(1, int(target_num)), N)
    if target_num >= N:
        # No need to merge
        cluster_ids = torch.arange(N, device=device)
        sampled_indices = torch.arange(N, device=device)
        return means, scales, quats, opacities, sh0, shN, cluster_ids, sampled_indices
    
    # Use FPS (Farthest Point Sampling) + FAISS for clustering
    # 1. Sample target_num points using FPS (these become coarse level candidates)
    # 2. Keep fine level gaussians (copy sampled points)
    # 3. For non-sampled gaussians, find closest sampled points using FAISS
    # 4. Each cluster = sampled point + closest non-sampled gaussians
    
    # Step 1: FPS sampling to get target_num candidate points
    # fpsample expects [N, 3] numpy array and returns indices
    means_np = means.cpu().numpy().astype(np.float32)  # [N, 3]
    print("starting fpsample")
    # Use bucket_fps_kdline_sampling (same as multigrid_gaussians_v8.py)
    sampled_indices_np = fpsample.bucket_fps_kdline_sampling(means_np, target_num, h=9)
    if sampled_indices_np.size == 0:
        sampled_indices_np = np.array([0], dtype=np.int64)
    sampled_indices_np = np.clip(sampled_indices_np, 0, N - 1)
    # Convert back to torch tensor on the correct device
    sampled_indices = torch.from_numpy(sampled_indices_np).to(device=device, dtype=torch.long)
    actual_num_samples = len(sampled_indices)
    print(f"fpsample done: {actual_num_samples} samples (requested: {target_num})")
    
    if actual_num_samples < target_num:
        print(f"  Warning: FPS returned fewer samples than requested ({actual_num_samples} < {target_num})")
        print(f"  Using actual number of samples: {actual_num_samples}")
        # Update target_num to match actual samples
        target_num = actual_num_samples
    
    # Step 2: Get sampled points (coarse level candidates - these are kept as-is)
    # CRITICAL: Use means_np (already computed) to ensure exact match
    # sampled_means_np must be exactly means_np[sampled_indices] to guarantee
    # that means_np[sampled_indices[i]] == sampled_means_np[i] (exact match, distance = 0)
    sampled_means_np = means_np[sampled_indices_np]  # [actual_num_samples, 3]
    print("finding closest sampled points using FAISS")
    
    # Step 3: Assign each gaussian to its closest sampled point using FAISS (GPU).
    # IMPORTANT:
    # - We ONLY run nearest-neighbor search for NON-sampled gaussians.
    # - Sampled gaussians are forced to map to their own cluster (self assignment).
    #
    # This guarantees:
    # - cluster_ids is always in [0, actual_num_samples - 1]
    # - num_clusters == actual_num_samples (each sampled point defines exactly one cluster)
    dimension = 3

    # Build a boolean mask for non-sampled points (search queries)
    non_sample_mask = np.ones(N, dtype=np.bool_)
    non_sample_mask[sampled_indices_np] = False
    query_means_np = means_np[non_sample_mask]  # [N_non_sample, 3]

    # Choose GPU id from torch device (use same device as tensors)
    gpu_id = 0
    if device.type == "cuda" and device.index is not None:
        gpu_id = int(device.index)

    use_faiss_gpu = (
        device.type == "cuda"
        and hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "get_num_gpus")
        and faiss.get_num_gpus() > 0
    )

    if use_faiss_gpu:
        res = faiss.StandardGpuResources()
        if hasattr(faiss, "GpuIndexFlatL2"):
            config = faiss.GpuIndexFlatConfig()
            config.device = gpu_id
            index = faiss.GpuIndexFlatL2(res, dimension, config)
        else:
            index_cpu = faiss.IndexFlatL2(dimension)
            index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)
        print(f"creating FAISS GPU index (gpu_id={gpu_id})")
    else:
        index = faiss.IndexFlatL2(dimension)
        print("creating FAISS CPU index (GPU not available)")

    # Add sampled points to index (FAISS expects float32 numpy)
    print("adding to index")
    index.add(sampled_means_np)
    print("adding done")

    # Search only non-sampled gaussians
    print("searching (non-sampled only)")
    distances, nearest_indices = index.search(query_means_np, 1)  # [N_non_sample, 1]
    nearest_indices = nearest_indices[:, 0].astype(np.int64)  # [N_non_sample]
    print("searching done")

    # Build cluster ids:
    # - sampled points -> their own cluster id (0..actual_num_samples-1)
    # - non-sampled points -> nearest sampled index
    cluster_ids_np = np.empty(N, dtype=np.int64)
    cluster_ids_np[sampled_indices_np] = np.arange(actual_num_samples, dtype=np.int64)
    cluster_ids_np[non_sample_mask] = nearest_indices

    # Convert to torch tensor on the correct device
    cluster_ids = torch.from_numpy(cluster_ids_np).long().to(device)
    cluster_ids = cluster_ids.clamp(0, actual_num_samples - 1)
    num_clusters = int(actual_num_samples)

    # Cluster->sample mapping is identity by construction
    unique_cluster_ids = torch.arange(num_clusters, device=device, dtype=torch.long)

    # Merge Gaussians within each cluster using scatter_add (parallel processing)
    # Similar to vcycle_trainer_v21.py hierarchy consistency loss
    
    # Weight formula from paper (Section 4.1, Eq. 8):
    # w'_i = o_i * S_i
    # where o_i is opacity and S_i is the volume measure
    # 
    # For 3D Gaussian, we use the cube root of the determinant (volume measure):
    # S_i ≈ (det(Σ_3D_i))^(1/3)
    # 
    # This is equivalent to: w'_i = o_i * (det(Σ_3D_i))^(1/3)
    
    # Compute 3D covariance matrices from scale and quaternion
    scales_exp = torch.exp(scales).clamp_min(1e-6).clamp_max(1e6)  # [N, 3]
    # Ensure scales_exp are valid
    scales_exp = torch.where(torch.isfinite(scales_exp), scales_exp, torch.ones_like(scales_exp) * 1e-6)
    # Ensure quats are normalized and valid
    quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True).clamp_min(1e-8)
    quats = torch.where(torch.isfinite(quats), quats, torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=quats.dtype).unsqueeze(0).expand_as(quats))
    
    covars, _ = quat_scale_to_covar_preci(
        quats=quats,
        scales=scales_exp,
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )  # [N, 3, 3]
    
    # Ensure covars are valid immediately after computation
    covars = torch.where(torch.isfinite(covars), covars, torch.zeros_like(covars))
    covars = torch.where(torch.isfinite(covars), covars, torch.zeros_like(covars))
    
    # Compute volume measure S_i ≈ (det(Σ_3D_i))^(1/3) for 3D Gaussian
    # For 3D Gaussian, we use power of 1/3 (volume measure) instead of sqrt (2D area measure)
    opacities_clamped = torch.sigmoid(opacities).clamp_min(1e-8)  # [N,]
    covar_dets = torch.det(covars).clamp_min(1e-8)  # [N,]
    surface_area = torch.pow(covar_dets, 1.0/3.0)  # [N,] - S_i = (det)^(1/3) for 3D Gaussian
    
    # Unnormalized weights: w'_i = o_i * S_i
    unnormalized_weights = opacities_clamped * surface_area  # [N,]
    
    # Compute sum of weights per cluster using scatter_add
    cluster_weight_sums = torch.zeros(num_clusters, device=device, dtype=unnormalized_weights.dtype)
    cluster_weight_sums.scatter_add_(0, cluster_ids, unnormalized_weights)
    
    # Count number of gaussians per cluster (for debugging and filtering)
    cluster_counts = torch.zeros(num_clusters, device=device, dtype=torch.long)
    cluster_counts.scatter_add_(0, cluster_ids, torch.ones_like(cluster_ids))
    
    # Filter out empty clusters: only keep clusters with at least one gaussian
    valid_cluster_mask = cluster_counts > 0  # [num_clusters,]
    if not valid_cluster_mask.all():
        # Remap cluster_ids to only valid clusters
        # Create mapping: old_cluster_id -> new_cluster_id (only for valid clusters)
        valid_cluster_indices = torch.where(valid_cluster_mask)[0]  # [M,] where M is number of valid clusters
        old_to_new = torch.full((num_clusters,), -1, dtype=torch.long, device=device)
        old_to_new[valid_cluster_indices] = torch.arange(len(valid_cluster_indices), device=device)
        
        # Remap cluster_ids: invalid clusters will have -1
        cluster_ids_remapped = old_to_new[cluster_ids]  # [N,]
        valid_gaussian_mask = cluster_ids_remapped >= 0  # [N,]
        
        # Filter to only valid gaussians
        cluster_ids = cluster_ids_remapped[valid_gaussian_mask]  # [K,] where K <= N
        means = means[valid_gaussian_mask]
        scales = scales[valid_gaussian_mask]
        quats = quats[valid_gaussian_mask]
        opacities = opacities[valid_gaussian_mask]
        sh0 = sh0[valid_gaussian_mask]
        shN = shN[valid_gaussian_mask]
        unnormalized_weights = unnormalized_weights[valid_gaussian_mask]
        
        # Update num_clusters to number of valid clusters
        num_clusters = len(valid_cluster_indices)
        cluster_weight_sums = cluster_weight_sums[valid_cluster_mask]
        cluster_counts = cluster_counts[valid_cluster_mask]
        unique_cluster_ids = torch.arange(num_clusters, device=device, dtype=torch.long)
        sampled_indices = sampled_indices[valid_cluster_mask]
        
        # Recompute covars for filtered gaussians
        scales_exp = torch.exp(scales).clamp_min(1e-6).clamp_max(1e6)
        covars, _ = quat_scale_to_covar_preci(
            quats=quats,
            scales=scales_exp,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )  # [K, 3, 3]
    
    # Normalize weights per cluster: w_i = w'_i / sum(w'_i) for each cluster
    #
    # IMPORTANT:
    # Do NOT clamp sum(w'_i) to a large epsilon (e.g. 1e-8) here.
    # At coarse levels, sum(w'_i) can legitimately be much smaller than 1e-8
    # (because opacity is clamped to 1e-8 and det(covar)^(1/3) can also be small).
    # Clamping would make sum_i w_i < 1 and scales down merged_means toward 0.
    cluster_weight_sum_per_gaussian = cluster_weight_sums[cluster_ids]  # [K,]
    weights = unnormalized_weights / (cluster_weight_sum_per_gaussian + 1e-20)  # [K,]
    
    # Ensure weights are valid (no NaN/inf)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    
    # Initialize merged tensors for all clusters
    merged_means = torch.zeros(num_clusters, 3, device=device, dtype=means.dtype)
    merged_scales = torch.zeros(num_clusters, 3, device=device, dtype=scales.dtype)
    merged_quats = torch.zeros(num_clusters, 4, device=device, dtype=quats.dtype)
    merged_opacities = torch.zeros(num_clusters, device=device, dtype=opacities.dtype)
    merged_sh0 = torch.zeros(num_clusters, 1, 3, device=device, dtype=sh0.dtype)
    shN_dims = shN.shape[1:]  # [K, 3]
    merged_shN = torch.zeros(num_clusters, *shN_dims, device=device, dtype=shN.dtype)
    
    # Merging formulas from paper (Section 4.1) - all use weighted average with normalized weights w_i
    
    # Merge means: μ^(l+1) = ∑_i^N w_i * μ_i^(l) (Eq. 3)
    # weights are already normalized per cluster: sum(weights for cluster c) = 1
    # So scatter_add(weighted_means) gives the correct weighted average
    weighted_means = means * weights.unsqueeze(-1)  # [K, 3]
    # Ensure weighted_means are valid before scatter_add
    weighted_means = torch.where(torch.isfinite(weighted_means), weighted_means, torch.zeros_like(weighted_means))
    
    # Debug: Print input means range before scatter_add
    if num_clusters > 1000000:  # Only print for large levels to avoid spam
        print(f"    Debug: Input means range: min={means.min().item():.6f}, max={means.max().item():.6f}, mean={means.mean().item():.6f}, std={means.std().item():.6f}")
        print(f"    Debug: weights range: min={weights.min().item():.6f}, max={weights.max().item():.6f}, mean={weights.mean().item():.6f}")
        print(f"    Debug: weighted_means range: min={weighted_means.min().item():.6f}, max={weighted_means.max().item():.6f}, mean={weighted_means.mean().item():.6f}, std={weighted_means.std().item():.6f}")
        print(f"    Debug: cluster_ids range: min={cluster_ids.min().item()}, max={cluster_ids.max().item()}, num_clusters={num_clusters}")
    
    merged_means.scatter_add_(0, cluster_ids.unsqueeze(-1).expand(-1, 3), weighted_means)
    
    # Ensure merged_means are valid (no NaN/inf)
    merged_means = torch.where(torch.isfinite(merged_means), merged_means, torch.zeros_like(merged_means))
    
    # Debug: Check if merged_means is being scaled incorrectly
    if num_clusters > 1000000:  # Only print for large levels
        print(f"    Debug: After scatter_add, merged_means range: min={merged_means.min().item():.6f}, max={merged_means.max().item():.6f}, mean={merged_means.mean().item():.6f}, std={merged_means.std().item():.6f}")
    
    # Merge covariance: Σ^(l+1) = ∑_i^N w_i * (Σ_i^(l) + (μ_i^(l) - μ^(l+1)) * (μ_i^(l) - μ^(l+1))^T) (Eq. 4)
    # Compute mean differences per cluster: (μ_i^(l) - μ^(l+1))
    mean_diffs = means - merged_means[cluster_ids]  # [K, 3]
    
    # Ensure mean_diffs are valid
    mean_diffs = torch.where(torch.isfinite(mean_diffs), mean_diffs, torch.zeros_like(mean_diffs))
    
    # Compute outer products: (μ_i^(l) - μ^(l+1)) * (μ_i^(l) - μ^(l+1))^T
    mean_diff_outer = torch.einsum('ni,nj->nij', mean_diffs, mean_diffs)  # [K, 3, 3]
    
    # Ensure mean_diff_outer is valid
    mean_diff_outer = torch.where(torch.isfinite(mean_diff_outer), mean_diff_outer, torch.zeros_like(mean_diff_outer))
    
    # Ensure covars are valid
    covars = torch.where(torch.isfinite(covars), covars, torch.zeros_like(covars))
    
    # Compute weighted sum per cluster: Σ^(l+1) = ∑_i^N w_i * (Σ_i^(l) + mean_diff_outer)
    weighted_covars = (covars + mean_diff_outer) * weights.view(-1, 1, 1)  # [K, 3, 3]
    
    # Ensure weighted_covars are valid
    weighted_covars = torch.where(torch.isfinite(weighted_covars), weighted_covars, torch.zeros_like(weighted_covars))
    
    # Scatter add to accumulate merged covariance per cluster (vectorized)
    merged_covars = torch.zeros(num_clusters, 3, 3, device=device, dtype=covars.dtype)
    # Expand cluster_ids to match weighted_covars shape: [K, 3, 3]
    cluster_ids_expanded = cluster_ids.view(-1, 1, 1).expand(-1, 3, 3)  # [K, 3, 3]
    merged_covars.scatter_add_(0, cluster_ids_expanded, weighted_covars)  # [num_clusters, 3, 3]
    
    # Make symmetric (covariance matrices must be symmetric)
    merged_covars = (merged_covars + merged_covars.transpose(-2, -1)) / 2.0
    
    # Ensure merged_covars are valid and positive semi-definite
    merged_covars = torch.where(torch.isfinite(merged_covars), merged_covars, torch.zeros_like(merged_covars))
    
    # For empty clusters (should not happen after filtering, but add small identity as safety)
    cluster_covar_norms = merged_covars.abs().sum(dim=(1, 2))  # [num_clusters,]
    empty_mask = cluster_covar_norms < 1e-8  # [num_clusters,]
    if empty_mask.any():
        identity = torch.eye(3, device=device, dtype=covars.dtype) * 1e-6
        merged_covars[empty_mask] = identity.unsqueeze(0).expand(empty_mask.sum(), -1, -1)
    
    # Decompose merged covariance back to quaternion and scale per cluster.
    # NOTE: cuSOLVER batched eigh can fail for very large batch sizes or invalid matrices, so we run it in chunks.
    # Also check for NaN/inf in each chunk and replace invalid matrices with identity.
    merged_covars = merged_covars.contiguous()
    
    # Final validation: check for NaN/inf in merged_covars and replace with identity
    has_nan_inf = torch.isnan(merged_covars).any(dim=(1, 2)) | torch.isinf(merged_covars).any(dim=(1, 2))  # [num_clusters,]
    if has_nan_inf.any():
        identity = torch.eye(3, device=device, dtype=merged_covars.dtype) * 1e-6
        merged_covars[has_nan_inf] = identity.unsqueeze(0).expand(has_nan_inf.sum(), -1, -1)
    
    eigenvals = torch.empty((num_clusters, 3), device=device, dtype=merged_covars.dtype)
    eigenvecs = torch.empty((num_clusters, 3, 3), device=device, dtype=merged_covars.dtype)
    chunk_size = 65535  # safe batch size for cuSOLVER batched ops
    
    for start in range(0, num_clusters, chunk_size):
        end = min(start + chunk_size, num_clusters)
        chunk_covars = merged_covars[start:end]  # [chunk_size, 3, 3]
        
        # Validate chunk: check for NaN/inf and replace invalid matrices
        chunk_has_nan_inf = torch.isnan(chunk_covars).any(dim=(1, 2)) | torch.isinf(chunk_covars).any(dim=(1, 2))  # [chunk_size,]
        if chunk_has_nan_inf.any():
            identity = torch.eye(3, device=device, dtype=chunk_covars.dtype) * 1e-6
            chunk_covars[chunk_has_nan_inf] = identity.unsqueeze(0).expand(chunk_has_nan_inf.sum(), -1, -1)
        
        # Try GPU eigh first, fallback to CPU if it fails
        try:
            e_vals, e_vecs = torch.linalg.eigh(chunk_covars)
        except RuntimeError:
            # Fallback to CPU if GPU fails
            chunk_covars_cpu = chunk_covars.cpu()
            e_vals_cpu, e_vecs_cpu = torch.linalg.eigh(chunk_covars_cpu)
            e_vals = e_vals_cpu.to(device)
            e_vecs = e_vecs_cpu.to(device)
        
        eigenvals[start:end] = e_vals
        eigenvecs[start:end] = e_vecs
    
    eigenvals = eigenvals.clamp_min(1e-6)  # Ensure positive
    
    # Extract scale: S = sqrt(eigenvals) - vectorized
    merged_scales = torch.log(torch.sqrt(eigenvals))  # [num_clusters, 3] in log space
    
    # Debug: Print scale statistics before and after merge
    input_scales_actual = torch.exp(scales)
    merged_scales_actual = torch.exp(merged_scales)
    print(f"    Debug scale merge:")
    print(f"      Input scales (actual): min={input_scales_actual.min().item():.6f}, max={input_scales_actual.max().item():.6f}, mean={input_scales_actual.mean().item():.6f}, std={input_scales_actual.std().item():.6f}")
    print(f"      Merged scales (actual): min={merged_scales_actual.min().item():.6f}, max={merged_scales_actual.max().item():.6f}, mean={merged_scales_actual.mean().item():.6f}, std={merged_scales_actual.std().item():.6f}")
    if input_scales_actual.mean().item() > 1e-8:
        print(f"      Scale ratio (merged/input mean): {merged_scales_actual.mean().item() / input_scales_actual.mean().item():.6f}")
    
    # Extract rotation matrix and convert to quaternion - vectorized
    # rotmat_to_quat_batch handles batch operations
    merged_quats = rotmat_to_quat_batch(eigenvecs)  # [num_clusters, 4]
    
    # Merge opacities:
    # Preserve integrated opacity mass by matching:
    #   o_p * S_p  ≈  ∑_i (o_i * S_i)  =  ∑_i w'_i
    # => falloff = (∑_i w'_i) / S_p
    # Store logit(o_p) as log(falloff) where o_p = falloff / (1 + falloff).
    # Compute volume measure of merged Gaussians: S_p ≈ (det(Σ_merged))^(1/3) for 3D Gaussian
    merged_covar_dets = torch.det(merged_covars).clamp_min(1e-8)  # [num_clusters,]
    merged_surface_areas = torch.pow(merged_covar_dets, 1.0/3.0)  # [num_clusters,] - (det)^(1/3) for 3D Gaussian
    
    # Falloff per cluster = (∑_i w'_i) / S_p (vectorized)
    # cluster_weight_sums is already computed above, reuse it
    cluster_falloff = cluster_weight_sums / merged_surface_areas.clamp_min(1e-12)  # [num_clusters,]
    
    # Convert to log space (same as merge_gaussians function, line 371)
    # Note: The original merge_gaussians uses .log() which gives log space, not logit space.
    # However, opacity should be in logit space for storage. The issue is that if falloff is very small,
    # log(falloff) will be very negative, making sigmoid(opacity) very small (black rendering).
    # We need to ensure falloff is in a reasonable range before taking log.
    cluster_falloff_clamped = cluster_falloff.clamp_min(1e-12)
    merged_opacities = cluster_falloff_clamped.log()  # logit(o_p) where o_p = falloff / (1 + falloff)
    
    # Merge SH coefficients: weighted average using the same weights
    # From paper: "We can similarly use the weighted average of the SH coefficients for the merged node using the same weight"
    # Weights are already normalized per cluster (sum to 1), so scatter_add gives correct weighted average
    weighted_sh0 = sh0 * weights.view(-1, 1, 1)  # [N, 1, 3]
    cluster_ids_sh0 = cluster_ids.view(-1, 1, 1).expand(-1, 1, 3)  # [N, 1, 3]
    merged_sh0.scatter_add_(0, cluster_ids_sh0, weighted_sh0)
    
    # Merge SHN: weighted average using scatter_add
    weighted_shN = shN * weights.view(-1, *([1] * len(shN_dims)))  # [N, K, 3]
    cluster_ids_expanded = cluster_ids.view(-1, *([1] * len(shN_dims))).expand_as(weighted_shN)  # [N, K, 3]
    merged_shN.scatter_add_(0, cluster_ids_expanded, weighted_shN)
    
    # Ensure SH coefficients are valid (no NaN/inf)
    merged_sh0 = torch.where(torch.isfinite(merged_sh0), merged_sh0, torch.zeros_like(merged_sh0))
    merged_shN = torch.where(torch.isfinite(merged_shN), merged_shN, torch.zeros_like(merged_shN))
    
    # Map cluster IDs back to original sampled indices in the input means array
    sampled_indices_for_clusters = sampled_indices[unique_cluster_ids]
    
    return merged_means, merged_scales, merged_quats, merged_opacities, merged_sh0, merged_shN, cluster_ids, sampled_indices_for_clusters


def build_hierarchy(
    ply_path: str,
    num_levels: int = 4,
    reduction_factor: float = 0.5,
    clustering_method: str = "fps_faiss",
    output_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict:
    """
    Build hierarchical 3D Gaussian representation.
    
    Args:
        ply_path: Path to input PLY file
        num_levels: Number of hierarchy levels (default: 4)
        reduction_factor: Factor to reduce number of Gaussians per level (default: 0.5)
        clustering_method: Clustering method (ignored; FPS + closest points only)
        output_dir: Directory to save hierarchy files
        
    Returns:
        Dictionary containing hierarchy structure and parameters
    """
    # Force FPS + closest points clustering regardless of input
    clustering_method = "fps_faiss"

    print(f"Loading point cloud from {ply_path}...")
    gaussians = load_ply(ply_path)
    
    # Move all tensors to the specified device
    means = gaussians["means"].to(device)
    scales = gaussians["scales"].to(device)
    quats = gaussians["quats"].to(device)
    opacities = gaussians["opacities"].to(device)
    sh0 = gaussians["sh0"].to(device)
    shN = gaussians["shN"].to(device)
    
    N = means.shape[0]
    print(f"Loaded {N} Gaussians")
    print(f"Using device: {device}")
    
    # Debug: Print statistics for Level 0
    print(f"Level 0 statistics:")
    print(f"  means: min={means.min().item():.6f}, max={means.max().item():.6f}, mean={means.mean().item():.6f}, std={means.std().item():.6f}")
    scales_actual = torch.exp(scales)
    print(f"  scales (log): min={scales.min().item():.6f}, max={scales.max().item():.6f}, mean={scales.mean().item():.6f}, std={scales.std().item():.6f}")
    print(f"  scales (actual): min={scales_actual.min().item():.6f}, max={scales_actual.max().item():.6f}, mean={scales_actual.mean().item():.6f}, std={scales_actual.std().item():.6f}")
    print(f"  quats: min={quats.min().item():.6f}, max={quats.max().item():.6f}, mean={quats.mean().item():.6f}, std={quats.std().item():.6f}")
    opacities_actual = torch.sigmoid(opacities)
    print(f"  opacities (logit): min={opacities.min().item():.6f}, max={opacities.max().item():.6f}, mean={opacities.mean().item():.6f}, std={opacities.std().item():.6f}")
    print(f"  opacities (actual): min={opacities_actual.min().item():.6f}, max={opacities_actual.max().item():.6f}, mean={opacities_actual.mean().item():.6f}, std={opacities_actual.std().item():.6f}")
    print(f"  sh0: min={sh0.min().item():.6f}, max={sh0.max().item():.6f}, mean={sh0.mean().item():.6f}, std={sh0.std().item():.6f}")
    print(f"  shN: min={shN.min().item():.6f}, max={shN.max().item():.6f}, mean={shN.mean().item():.6f}, std={shN.std().item():.6f}")
    
    # Normalize quaternions
    quats = F.normalize(quats, p=2, dim=-1)
    
    # Build hierarchy levels
    hierarchy = {
        "levels": [],
        # For each level (build_hierarchy level indexing: 0=finest -> N-1=coarsest),
        # parent_indices[level_idx] maps each gaussian at this level to its parent gaussian
        # at the NEXT (coarser) level (level_idx + 1). Coarsest level has parent -1.
        "parent_indices": [],
        "num_levels": num_levels,
    }
    
    current_means = means
    current_scales = scales
    current_quats = quats
    current_opacities = opacities
    current_sh0 = sh0
    current_shN = shN
    current_num = N
    
    # Level 0 is the original (finest level)
    hierarchy["levels"].append({
        "means": current_means.clone(),
        "scales": current_scales.clone(),
        "quats": current_quats.clone(),
        "opacities": current_opacities.clone(),
        "sh0": current_sh0.clone(),
        "shN": current_shN.clone(),
        "num_gaussians": current_num,
    })
    hierarchy["parent_indices"].append(None)  # Will be filled after building level 1
    
    print(f"\nBuilding hierarchy with {num_levels} levels...")
    print(f"Level 0: {current_num} Gaussians (original)")
    
    # Build coarser levels
    for level in range(1, num_levels):
        # 클러스터 개수 결정: 현재 Gaussian 수에 reduction_factor를 곱함
        # 예: 1,000,000개 * 0.5 = 500,000개 (각 레벨마다 50% 감소)
        target_num = max(1, int(current_num * reduction_factor))
        print(f"Building level {level}: target {target_num} Gaussians (from {current_num})...")
        
        merged_means, merged_scales, merged_quats, merged_opacities, merged_sh0, merged_shN, cluster_ids, sampled_indices = build_hierarchy_level(
            current_means, current_scales, current_quats, current_opacities, current_sh0, current_shN,
            target_num, method=clustering_method
        )
        print(f"completed clustering")
        
        num_merged = merged_means.shape[0]
        
        # Debug: Print statistics for merged level
        print(f"  Level {level} statistics:")
        print(f"    means: min={merged_means.min().item():.6f}, max={merged_means.max().item():.6f}, mean={merged_means.mean().item():.6f}, std={merged_means.std().item():.6f}")
        merged_scales_actual = torch.exp(merged_scales)
        print(f"    scales (log): min={merged_scales.min().item():.6f}, max={merged_scales.max().item():.6f}, mean={merged_scales.mean().item():.6f}, std={merged_scales.std().item():.6f}")
        print(f"    scales (actual): min={merged_scales_actual.min().item():.6f}, max={merged_scales_actual.max().item():.6f}, mean={merged_scales_actual.mean().item():.6f}, std={merged_scales_actual.std().item():.6f}")
        print(f"    quats: min={merged_quats.min().item():.6f}, max={merged_quats.max().item():.6f}, mean={merged_quats.mean().item():.6f}, std={merged_quats.std().item():.6f}")
        merged_opacities_actual = torch.sigmoid(merged_opacities)
        print(f"    opacities (logit): min={merged_opacities.min().item():.6f}, max={merged_opacities.max().item():.6f}, mean={merged_opacities.mean().item():.6f}, std={merged_opacities.std().item():.6f}")
        print(f"    opacities (actual): min={merged_opacities_actual.min().item():.6f}, max={merged_opacities_actual.max().item():.6f}, mean={merged_opacities_actual.mean().item():.6f}, std={merged_opacities_actual.std().item():.6f}")
        print(f"    sh0: min={merged_sh0.min().item():.6f}, max={merged_sh0.max().item():.6f}, mean={merged_sh0.mean().item():.6f}, std={merged_sh0.std().item():.6f}")
        print(f"    shN: min={merged_shN.min().item():.6f}, max={merged_shN.max().item():.6f}, mean={merged_shN.mean().item():.6f}, std={merged_shN.std().item():.6f}")
        print(f"  Level {level-1} (input) statistics:")
        print(f"    means: min={current_means.min().item():.6f}, max={current_means.max().item():.6f}, mean={current_means.mean().item():.6f}, std={current_means.std().item():.6f}")
        current_scales_actual = torch.exp(current_scales)
        print(f"    scales (log): min={current_scales.min().item():.6f}, max={current_scales.max().item():.6f}, mean={current_scales.mean().item():.6f}, std={current_scales.std().item():.6f}")
        print(f"    scales (actual): min={current_scales_actual.min().item():.6f}, max={current_scales_actual.max().item():.6f}, mean={current_scales_actual.mean().item():.6f}, std={current_scales_actual.std().item():.6f}")
        print(f"    quats: min={current_quats.min().item():.6f}, max={current_quats.max().item():.6f}, mean={current_quats.mean().item():.6f}, std={current_quats.std().item():.6f}")
        current_opacities_actual = torch.sigmoid(current_opacities)
        print(f"    opacities (logit): min={current_opacities.min().item():.6f}, max={current_opacities.max().item():.6f}, mean={current_opacities.mean().item():.6f}, std={current_opacities.std().item():.6f}")
        print(f"    opacities (actual): min={current_opacities_actual.min().item():.6f}, max={current_opacities_actual.max().item():.6f}, mean={current_opacities_actual.mean().item():.6f}, std={current_opacities_actual.std().item():.6f}")
        print(f"    sh0: min={current_sh0.min().item():.6f}, max={current_sh0.max().item():.6f}, mean={current_sh0.mean().item():.6f}, std={current_sh0.std().item():.6f}")
        print(f"    shN: min={current_shN.min().item():.6f}, max={current_shN.max().item():.6f}, mean={current_shN.mean().item():.6f}, std={current_shN.std().item():.6f}")
        
        # 실제 merged 가우시안 수와 target_num 비교
        if num_merged != target_num:
            print(f"  Warning: Actual merged Gaussians ({num_merged}) != target ({target_num})")
            print(f"  Reduction ratio: {num_merged / current_num:.4f} (expected: {reduction_factor:.4f})")
        
        # Store parent mapping for the PREVIOUS (finer) level:
        # cluster_ids has shape [N_fine] and values in [0, num_merged-1], mapping each fine gaussian to its parent (cluster) in this merged level.
        hierarchy["parent_indices"][level - 1] = cluster_ids.clone()
        
        hierarchy["levels"].append({
            "means": merged_means.clone(),
            "scales": merged_scales.clone(),
            "quats": merged_quats.clone(),
            "opacities": merged_opacities.clone(),
            "sh0": merged_sh0.clone(),
            "shN": merged_shN.clone(),
            "num_gaussians": merged_means.shape[0],
        })
        # Placeholder for this level's parent indices (filled when building next level)
        hierarchy["parent_indices"].append(None)
        
        # Update for next level
        current_means = merged_means
        current_scales = merged_scales
        current_quats = merged_quats
        current_opacities = merged_opacities
        current_sh0 = merged_sh0
        current_shN = merged_shN
        current_num = merged_means.shape[0]
        
        print(f"Level {level}: {current_num} Gaussians")

    # Coarsest level has no parent
    if hierarchy["parent_indices"][-1] is None:
        hierarchy["parent_indices"][-1] = torch.full(
            (current_num,),
            -1,
            dtype=torch.long,
            device=device if isinstance(device, torch.device) else current_means.device,
        )
    
    # Save hierarchy
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as PyTorch checkpoint
        checkpoint = {
            "hierarchy": hierarchy,
            "num_levels": num_levels,
            "reduction_factor": reduction_factor,
            "clustering_method": clustering_method,
        }
        
        checkpoint_path = os.path.join(output_dir, "hierarchy.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"\nSaved hierarchy to {checkpoint_path}")
        
        # Save statistics
        stats = {
            "num_levels": num_levels,
            "reduction_factor": reduction_factor,
            "clustering_method": clustering_method,
            "level_stats": [
                {
                    "level": i,
                    "num_gaussians": level_data["num_gaussians"]
                }
                for i, level_data in enumerate(hierarchy["levels"])
            ]
        }
        
        stats_path = os.path.join(output_dir, "hierarchy_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")
        
        # Save each level as separate PLY file for visualization
        from gsplat import export_splats
        import open3d as o3d
        
        # Collect all means for lineset visualization
        all_means_list = []
        all_levels_list = []
        for level_idx, level_data in enumerate(hierarchy["levels"]):
            all_means_list.append(level_data["means"].cpu().numpy())
            all_levels_list.append(np.full(level_data["means"].shape[0], level_idx))
        
        all_means = np.concatenate(all_means_list, axis=0)  # [N_total, 3]
        all_levels = np.concatenate(all_levels_list, axis=0)  # [N_total,]
        level_start_indices = [0]
        for level_data in hierarchy["levels"]:
            level_start_indices.append(level_start_indices[-1] + level_data["means"].shape[0])
        
        for level_idx, level_data in enumerate(hierarchy["levels"]):
            # IMPORTANT: export_splats() with format="ply" expects log/logit space values (not actual values)
            # This matches how simple_trainer_original.py calls it (with log/logit space values)
            # The splat2ply_bytes function saves them directly without conversion.
            # So we pass log/logit space values directly (no conversion needed).
            ply_level_path = os.path.join(output_dir, f"level_{level_idx}.ply")
            export_splats(
                means=level_data["means"],
                scales=level_data["scales"],  # Already in log space
                quats=level_data["quats"],
                opacities=level_data["opacities"],  # Already in logit space
                sh0=level_data["sh0"],
                shN=level_data["shN"],
                format="ply",
                save_to=ply_level_path,
            )
            print(f"Saved level {level_idx} PLY to {ply_level_path}")
        
        # Save linesets for level -> level+1 connections (child -> parent in the next coarser level)
        for level_idx in range(len(hierarchy["levels"]) - 1):
            child_level_data = hierarchy["levels"][level_idx]
            parent_level_data = hierarchy["levels"][level_idx + 1]
            parent_indices = hierarchy["parent_indices"][level_idx]  # [N_child] indices into parent level (level_idx+1)
            
            # Convert indices to global indices in the concatenated all_means array
            child_start_idx = level_start_indices[level_idx]
            parent_start_idx = level_start_indices[level_idx + 1]
            child_global_indices = np.arange(child_start_idx, child_start_idx + child_level_data["means"].shape[0])
            parent_global_indices = parent_indices.cpu().numpy() + parent_start_idx
            
            # Build lines: each child connects to its parent
            lines = []
            for child_idx, parent_idx in zip(child_global_indices, parent_global_indices):
                if parent_idx >= 0 and parent_idx < len(all_means):
                    lines.append([int(parent_idx), int(child_idx)])
            
            if len(lines) > 0:
                # Create LineSet
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(all_means)
                line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
                
                # Color lines (cyan for visibility)
                line_colors = np.ones((len(lines), 3)) * np.array([0.0, 1.0, 1.0])  # Cyan
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                
                lineset_path = os.path.join(output_dir, f"lineset_{level_idx}_{level_idx + 1}.ply")
                o3d.io.write_line_set(lineset_path, line_set)
                print(f"Saved lineset {level_idx}->{level_idx + 1}: {lineset_path} ({len(lines)} lines)")
    
    return hierarchy


def main():
    parser = argparse.ArgumentParser(
        description="Build hierarchical 3D Gaussian representation from trained point cloud"
    )
    parser.add_argument(
        "ply_path",
        type=str,
        help="Path to input PLY file (trained point cloud)"
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        default=4,
        help="Number of hierarchy levels (default: 4)"
    )
    parser.add_argument(
        "--reduction_factor",
        type=float,
        default=0.5,
        help="Factor to reduce number of Gaussians per level (default: 0.5)"
    )
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="fps_faiss",
        help="Clustering method (ignored; uses FPS + closest points only)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for hierarchy files (default: same as PLY file directory)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        ply_dir = Path(args.ply_path).parent
        ply_name = Path(args.ply_path).stem
        args.output_dir = str(ply_dir / f"{ply_name}_hierarchy")
    
    # Build hierarchy
    hierarchy = build_hierarchy(
        ply_path=args.ply_path,
        num_levels=args.num_levels,
        reduction_factor=args.reduction_factor,
        clustering_method=args.clustering_method,
        output_dir=args.output_dir,
    )
    
    print("\nHierarchy building completed!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

