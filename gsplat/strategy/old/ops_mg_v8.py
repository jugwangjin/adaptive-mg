from typing import Any, Callable, Dict, List, Optional, Union

import math
import torch
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat

SPLIT_SCALE_LOG = math.log(1.6)


@torch.no_grad()
def _elu_plus_one_inverse(weight_positive: Tensor) -> Tensor:
    """Inverse of elu(x) + 1: convert positive weight back to raw weight.
    
    Args:
        weight_positive: Positive weight (elu(x) + 1) [N,]
        
    Returns:
        Raw weight [N,]
    """
    weight_raw = weight_positive - 1.0
    return torch.where(weight_raw >= 0, weight_raw, torch.log(weight_raw + 1.0 + 1e-8))


@torch.no_grad()
def _adjust_sibling_weights_preserve_ratio(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    siblings: Tensor,  # [K,] - new indices of siblings
    sibling_old_indices: Tensor,  # [K,] - old indices of siblings (matching order)
    original_parent_weights: Tensor,  # [N_old,] - original parent weights
) -> float:
    """Adjust parent_weights for siblings to preserve weight_ratio.
    
    Adjusts weights so that weight_ratio (child_weight / sum_of_sibling_weights) remains constant.
    
    Args:
        params: Parameter dictionary containing parent_weight
        siblings: New sibling indices [K,]
        sibling_old_indices: Old indices of siblings [K,] (must match siblings order)
        original_parent_weights: Original parent weights before operation [N_old,]
        
    Returns:
        New parent weight sum after adjustment
    """
    if len(siblings) == 0:
        return 0.0
    
    device = params["parent_weight"].device
    
    # Compute original sum
    original_sibling_weights = original_parent_weights[sibling_old_indices]
    original_sum = (torch.nn.functional.elu(original_sibling_weights) + 1.0).sum().item()
    
    if original_sum < 1e-8:
        return 0.0
    
    # Get current sum (before adjustment)
    current_parent_weights = params["parent_weight"].data[siblings]
    current_sum = (torch.nn.functional.elu(current_parent_weights) + 1.0).sum().item()
    
    # Adjust each sibling's weight to preserve ratio: new_weight = old_weight * (new_sum / old_sum)
    # Vectorized: process all siblings at once
    old_weights = original_parent_weights[sibling_old_indices]
    old_weights_positive = torch.nn.functional.elu(old_weights) + 1.0
    new_weights_positive = old_weights_positive * (current_sum / original_sum)
    new_weights_raw = _elu_plus_one_inverse(new_weights_positive)
    params["parent_weight"].data[siblings] = new_weights_raw
    
    # Recompute sum after adjustment
    adjusted_parent_weights = params["parent_weight"].data[siblings]
    new_sum = (torch.nn.functional.elu(adjusted_parent_weights) + 1.0).sum().item()
    return new_sum


@torch.no_grad()
def _compute_residual_from_actual(
    name: str,
    orig_actual_val: Tensor,
    parent_actual_val: Tensor,
    level: int,
    multigrid_gaussians: Any,
    weight_ratio: Optional[float] = None,
) -> Tensor:
    """Compute residual parameter from actual parameter to preserve it.
    
    Args:
        name: Parameter name ("means", "scales", "opacities", "quats", "sh0", "shN", etc.)
        orig_actual_val: Original actual parameter value
        parent_actual_val: Parent actual parameter value
        level: Level of child
        multigrid_gaussians: MultigridGaussians object
        weight_ratio: Optional weight ratio (child_weight / sum_of_sibling_weights)
        
    Returns:
        New residual parameter value
    """
    from multigrid_gaussians_v9 import CHILD_SCALE_LOG
    
    if name == "means":
        # means: actual = parent + residual * scale_factor (no weight_ratio)
        scale_factor = multigrid_gaussians.position_scale_reduction ** (level - 1)
        scale_factor = scale_factor.unsqueeze(-1) if orig_actual_val.dim() > 0 else scale_factor
        return (orig_actual_val - parent_actual_val) / scale_factor.clamp_min(1e-8)
    elif name == "scales":
        # scales: actual = parent * weight_ratio + residual - CHILD_SCALE_LOG
        if weight_ratio is not None:
            if parent_actual_val.dim() > 0:
                weight_ratio_tensor = torch.tensor(weight_ratio, device=parent_actual_val.device, dtype=parent_actual_val.dtype).unsqueeze(-1)
            else:
                weight_ratio_tensor = torch.tensor(weight_ratio, device=parent_actual_val.device, dtype=parent_actual_val.dtype)
            return orig_actual_val - parent_actual_val * weight_ratio_tensor + CHILD_SCALE_LOG
        else:
            return orig_actual_val - parent_actual_val + CHILD_SCALE_LOG
    else:
        # opacities, quats, sh0, shN, etc.: actual = parent * weight_ratio + residual
        if weight_ratio is not None:
            if parent_actual_val.dim() > 0:
                weight_ratio_tensor = torch.tensor(weight_ratio, device=parent_actual_val.device, dtype=parent_actual_val.dtype).view(-1, *([1] * (parent_actual_val.dim() - 1)))
            else:
                weight_ratio_tensor = torch.tensor(weight_ratio, device=parent_actual_val.device, dtype=parent_actual_val.dtype)
            return orig_actual_val - parent_actual_val * weight_ratio_tensor
        else:
            return orig_actual_val - parent_actual_val


@torch.no_grad()
def _update_hierarchical_structure(
    state: Dict[str, Any],
    new_levels: Tensor,
    new_parent_indices: Tensor,
) -> Dict[int, List[int]]:
    """Update hierarchical structure in state and multigrid_gaussians object.
    
    Args:
        state: State dictionary containing levels, parent_indices, level_indices
        new_levels: Updated levels tensor [N,]
        new_parent_indices: Updated parent_indices tensor [N,]
        
    Returns:
        Updated level_indices dictionary
    """
    # Update level_indices
    level_indices_new = {}
    for level_val in new_levels.unique():
        level_val_int = level_val.item()
        mask_level = (new_levels == level_val_int)
        level_indices_new[level_val_int] = torch.where(mask_level)[0].tolist()
    
    # Update state
    state["levels"] = new_levels
    state["parent_indices"] = new_parent_indices
    state["level_indices"] = level_indices_new
    
    # Update multigrid_gaussians object if present
    if "multigrid_gaussians" in state and state["multigrid_gaussians"] is not None:
        multigrid_gaussians = state["multigrid_gaussians"]
        multigrid_gaussians.levels = new_levels
        multigrid_gaussians.parent_indices = new_parent_indices
        multigrid_gaussians.level_indices = level_indices_new
        # Invalidate cache after densification
        multigrid_gaussians.invalidate_splats_cache()
    
    return level_indices_new


@torch.no_grad()
def _update_param_with_optimizer(
    param_fn: Callable[[str, Tensor], Tensor],
    optimizer_fn: Callable[[str, Tensor], Tensor],
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    names: Union[List[str], None] = None,
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    if names is None:
        # If names is not provided, update all parameters
        names = list(params.keys())

    for name in names:
        param = params[name]
        new_param = param_fn(name, param)
        params[name] = new_param
        if name not in optimizers:
            assert not param.requires_grad, (
                f"Optimizer for {name} is not found, but the parameter is trainable."
                f"Got requires_grad={param.requires_grad}"
            )
            continue
        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state[param]
            del optimizer.state[param]
            for key in param_state.keys():
                if key != "step":
                    v = param_state[key]
                    param_state[key] = optimizer_fn(key, v)
            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state


@torch.no_grad()
def duplicate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    levels: Tensor,
    parent_indices: Tensor,
    level_indices: Dict[int, List[int]],
    max_level: Optional[int] = None,
):
    """Inplace duplicate the Gaussian with the given mask (multigrid version, default.py style).
    
    Same as default.py duplicate but with hierarchical structure updates.
    After duplication, adds zero-parameter children to duplicated gaussians that need them
    (if max_level is provided and they are not at max_level).

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that may contain level_indices, levels, parent_indices.
        mask: A boolean mask to duplicate the Gaussians.
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent).
        level_indices: Dict mapping level -> list of gaussian indices at that level.
        max_level: Optional maximum level in the hierarchy. If provided, adds zero-parameter
                  children to duplicated gaussians that need them.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    
    # Get levels and parent indices of selected gaussians
    sel_levels = levels[sel]
    sel_parents = parent_indices[sel]
    
    # Store actual parameters and parent_weights BEFORE duplication to preserve them
    multigrid_gaussians = state.get("multigrid_gaussians", None)
    original_actual_params = None
    original_parent_weights = None
    if multigrid_gaussians is not None and sel.numel() > 0 and "parent_weight" in params:
        with torch.no_grad():
            # Store ALL gaussians' actual parameters (needed for adjusting original siblings too)
            original_actual_params = multigrid_gaussians.get_splats(
                level=None, detach_parents=False, current_splats=None
            )
            # Store original parent_weights before duplication
            original_parent_weights = params["parent_weight"].data.clone()

    def param_fn(name: str, p: Tensor) -> Tensor:
        # Vanilla-style duplicate: copy parameters as-is (matches ops.py)
        p_new = torch.cat([p, p[sel]])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    # update the extra running state (exclude hierarchical structure, handled separately)
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k not in ["levels", "parent_indices", "level_indices"]:
            state[k] = torch.cat((v, v[sel]))
    
    # Update hierarchical structure
    N_old = len(levels)
    N_new = N_old + len(sel)
    
    # Validate: sel_parents should be within bounds or -1
    valid_parent_mask = (sel_parents == -1) | ((sel_parents >= 0) & (sel_parents < N_old))
    if not valid_parent_mask.all():
        # Fix invalid parent indices (set to -1, making them root nodes)
        sel_parents = sel_parents.clone()
        sel_parents[~valid_parent_mask] = -1
    
    # Append duplicated levels and parent_indices
    new_levels = torch.cat([levels, sel_levels])
    new_parent_indices = torch.cat([parent_indices, sel_parents])
    
    # Update hierarchical structure
    _update_hierarchical_structure(state, new_levels, new_parent_indices)
    
    # Adjust parent_weight and residual parameters to preserve actual parameters
    # This is necessary because duplicating parent_weight changes the sum for siblings,
    # which changes weight_ratio and thus actual parameters
    if original_actual_params is not None and multigrid_gaussians is not None and "parent_weight" in params:
        with torch.no_grad():
            # Get new indices for duplicated gaussians
            new_indices = torch.arange(N_old, N_new, device=device)
            
            # Recompute actual parameters with new structure
            new_actual_splats = multigrid_gaussians.get_splats(
                level=None, detach_parents=False, current_splats=None
            )
            
            # Group duplicated gaussians by (parent, level) to avoid duplicate processing
            # Process each (parent, level) group only once
            processed_groups = set()
            
            # For each duplicated gaussian, adjust parent_weight and residuals to preserve actual parameters
            for i, old_idx in enumerate(sel):
                new_idx = new_indices[i]
                old_parent = sel_parents[i]
                new_parent = new_parent_indices[new_idx]
                level = sel_levels[i]
                
                # Only adjust if this is a child (has parent)
                if new_parent >= 0 and new_parent < N_new:
                    # Create unique key for (parent, level) group
                    group_key = (new_parent.item(), level.item())
                    
                    # Skip if this group was already processed
                    if group_key in processed_groups:
                        continue
                    
                    # Mark this group as processed
                    processed_groups.add(group_key)
                    
                    # Find all siblings (children of same parent) AFTER duplication
                    siblings_mask = (new_parent_indices == new_parent) & (new_levels == level)
                    siblings = torch.where(siblings_mask)[0]
                    
                    if len(siblings) > 0 and original_parent_weights is not None:
                        # Get original siblings (before duplication) - these are old indices
                        original_siblings_old = siblings[siblings < N_old]
                        
                        if len(original_siblings_old) > 0:
                            # Map new sibling indices to old indices (vectorized)
                            # siblings contains both old (siblings < N_old) and new (siblings >= N_old) indices
                            is_old_sibling = siblings < N_old
                            
                            # Build mapping: new_idx -> old_idx for duplicated gaussians
                            sibling_old_indices = torch.zeros_like(siblings)
                            sibling_old_indices[is_old_sibling] = siblings[is_old_sibling]  # Original siblings
                            
                            # For duplicated gaussians, find their old indices
                            is_dup = ~is_old_sibling
                            if is_dup.any():
                                dup_indices = siblings[is_dup]
                                # new_idx = N_old + i, so i = new_idx - N_old
                                dup_positions = dup_indices - N_old
                                valid_dup_mask = (dup_positions >= 0) & (dup_positions < len(sel))
                                if valid_dup_mask.any():
                                    sibling_old_indices[is_dup][valid_dup_mask] = sel[dup_positions[valid_dup_mask]]
                            
                            # Adjust parent_weights to preserve weight_ratio (process all siblings at once)
                            new_sum = _adjust_sibling_weights_preserve_ratio(
                                params=params,
                                siblings=siblings,
                                sibling_old_indices=sibling_old_indices,
                                original_parent_weights=original_parent_weights,
                            )
                            
                            if new_sum > 1e-8:
                                # Adjust residuals for all siblings in this group (both original and duplicated)
                                # All siblings need adjustment because parent_weight changed, which changes weight_ratio
                                for sib_new_idx in siblings:
                                    # Find original index
                                    sib_is_dup = sib_new_idx >= N_old
                                    if sib_is_dup:
                                        # This is a duplicated gaussian
                                        sib_dup_pos = sib_new_idx - N_old
                                        if sib_dup_pos >= 0 and sib_dup_pos < len(sel):
                                            sib_old_idx = sel[sib_dup_pos]
                                        else:
                                            continue
                                    else:
                                        # This is an original sibling - also needs adjustment!
                                        sib_old_idx = sib_new_idx
                                    
                                    # Get original actual parameters (original_actual_params contains all gaussians)
                                    original_actual = {k: v[sib_old_idx] for k, v in original_actual_params.items()}
                                    
                                    # Compute weight_ratio for this child
                                    child_weight = torch.nn.functional.elu(params["parent_weight"].data[sib_new_idx]) + 1.0
                                    weight_ratio = (child_weight / new_sum).item()
                                    
                                    new_parent_actual = {k: v[new_parent] for k, v in new_actual_splats.items()}
                                    
                                    if level > 1:
                                        for name in original_actual.keys():
                                            if name not in params or name == "parent_weight":
                                                continue
                                            new_residual = _compute_residual_from_actual(
                                                name=name,
                                                orig_actual_val=original_actual[name],
                                                parent_actual_val=new_parent_actual[name],
                                                level=level,
                                                multigrid_gaussians=multigrid_gaussians,
                                                weight_ratio=weight_ratio,
                                            )
                                            params[name].data[sib_new_idx] = new_residual
    
    # Note: _add_zero_parameter_children is now called once after all grow operations
    # in _grow_gs to avoid redundant processing


@torch.no_grad()
def split(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    levels: Tensor,
    parent_indices: Tensor,
    level_indices: Dict[int, List[int]],
    revised_opacity: bool = False,
    max_level: Optional[int] = None,
):
    """Inplace split the Gaussian with the given mask (multigrid version).
    
    Same as duplicate but with scale reduced by -0.5.
    Opacity is copied as-is (residual parameter, same level so same parent).
    After splitting, adds zero-parameter children to split gaussians that need them
    (if max_level is provided and they are not at max_level).

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that may contain level_indices, levels, parent_indices.
        mask: A boolean mask to split the Gaussians.
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent).
        level_indices: Dict mapping level -> list of gaussian indices at that level.
        revised_opacity: Whether to use revised opacity formulation from arXiv:2404.06109.
        max_level: Optional maximum level in the hierarchy. If provided, adds zero-parameter
                  children to split gaussians that need them.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    
    # Get levels and parent indices of selected gaussians
    sel_levels = levels[sel]
    sel_parents = parent_indices[sel]

    N_old = len(levels)
    multigrid_gaussians = state.get("multigrid_gaussians", None)
    mean_residuals = None
    new_opacity_residuals = None
    original_actual_params = None
    original_parent_weights = None
    if multigrid_gaussians is not None and sel.numel() > 0:
        with torch.no_grad():
            actual_splats = multigrid_gaussians.get_splats(
                level=None, detach_parents=False, current_splats=None
            )
            # Store ALL gaussians' actual parameters (needed for adjusting original siblings too)
            if "parent_weight" in params:
                original_actual_params = actual_splats  # Store all, not just selected
                # Store original parent_weights before splitting
                original_parent_weights = params["parent_weight"].data.clone()
            
            actual_means = actual_splats["means"][sel]
            actual_scales = torch.exp(actual_splats["scales"][sel])
            actual_quats = actual_splats["quats"][sel]
            rotmats = normalized_quat_to_rotmat(actual_quats)  # [N, 3, 3]
            samples = torch.einsum(
                "nij,nj,bnj->bni",
                rotmats,
                actual_scales,
                torch.randn(2, len(actual_scales), 3, device=device),
            ) * 0.01  # [2, N, 3]

            parent_means = torch.zeros_like(actual_means)
            valid_parent_mask = (sel_parents >= 0) & (sel_parents < len(parent_indices))
            if valid_parent_mask.any():
                parent_means[valid_parent_mask] = actual_splats["means"][
                    sel_parents[valid_parent_mask]
                ]

            scale_factor = multigrid_gaussians.position_scale_reduction ** (
                sel_levels.float() - 1.0
            )
            scale_factor = scale_factor.unsqueeze(-1)
            mean_residuals = (actual_means.unsqueeze(0) + samples - parent_means.unsqueeze(0)) / scale_factor.unsqueeze(0)

            if revised_opacity:
                actual_logits = actual_splats["opacities"][sel]
                actual_alpha = torch.sigmoid(actual_logits)
                target_alpha = 1.0 - torch.sqrt(
                    (1.0 - actual_alpha).clamp_min(1e-6)
                )
                target_alpha = target_alpha.clamp(1e-6, 1.0 - 1e-6)
                target_logits = torch.logit(target_alpha)
                parent_logits = torch.zeros_like(target_logits)
                if valid_parent_mask.any():
                    parent_logits[valid_parent_mask] = actual_splats["opacities"][
                        sel_parents[valid_parent_mask]
                    ]
                new_opacity_residuals = target_logits - parent_logits
                del actual_logits, actual_alpha, target_alpha, target_logits, parent_logits

            del actual_splats, actual_means, actual_scales, actual_quats, parent_means, scale_factor

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "means":
            if mean_residuals is not None:
                p_updated = p.clone()
                p_updated[sel] = mean_residuals[0]
                p_split = mean_residuals[1]
                p_new = torch.cat([p_updated, p_split])
                return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
            # Fallback: copy as-is if we cannot compute residual offsets
            p_split = p[sel]
        elif name == "scales":
            p_updated = p.clone()
            p_updated[sel] = p_updated[sel] - SPLIT_SCALE_LOG
            p_split = p_updated[sel]
            p_new = torch.cat([p_updated, p_split])
            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        elif name == "opacities" and revised_opacity and new_opacity_residuals is not None:
            p_updated = p.clone()
            p_updated[sel] = new_opacity_residuals
            p_split = new_opacity_residuals
            p_new = torch.cat([p_updated, p_split])
            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        else:
            # Other parameters: same as duplicate (copy as is)
            p_split = p[sel]
        # Keep all original gaussians (now modified) and append split copies
        p_new = torch.cat([p, p_split])
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    # update the extra running state (exclude hierarchical structure, handled separately)
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k not in ["levels", "parent_indices", "level_indices"]:
            state[k] = torch.cat((v, v[sel]))
    
    # Update hierarchical structure (same as duplicate)
    N_old = len(levels)
    N_new = N_old + len(sel)
    
    # Validate: sel_parents should be within bounds or -1
    valid_parent_mask = (sel_parents == -1) | ((sel_parents >= 0) & (sel_parents < N_old))
    if not valid_parent_mask.all():
        # Fix invalid parent indices (set to -1, making them root nodes)
        sel_parents = sel_parents.clone()
        sel_parents[~valid_parent_mask] = -1
    
    # Append split levels and parent_indices (same level and parent as original)
    new_levels = torch.cat([levels, sel_levels])
    new_parent_indices = torch.cat([parent_indices, sel_parents])
    
    # Update hierarchical structure
    _update_hierarchical_structure(state, new_levels, new_parent_indices)
    
    # Adjust parent_weight and residual parameters to preserve actual parameters
    # This is necessary because splitting parent_weight changes the sum for siblings,
    # which changes weight_ratio and thus actual parameters
    if original_actual_params is not None and multigrid_gaussians is not None and "parent_weight" in params:
        with torch.no_grad():
            # Get new indices for split gaussians
            new_indices = torch.arange(N_old, N_new, device=device)
            
            # Recompute actual parameters with new structure
            new_actual_splats = multigrid_gaussians.get_splats(
                level=None, detach_parents=False, current_splats=None
            )
            
            # Group split gaussians by (parent, level) to avoid duplicate processing
            # Process each (parent, level) group only once
            processed_groups = set()
            
            # For each split gaussian, adjust parent_weight and residuals to preserve actual parameters
            for i, old_idx in enumerate(sel):
                new_idx = new_indices[i]
                old_parent = sel_parents[i]
                new_parent = new_parent_indices[new_idx]
                level = sel_levels[i]
                
                # Only adjust if this is a child (has parent)
                if new_parent >= 0 and new_parent < N_new:
                    # Create unique key for (parent, level) group
                    group_key = (new_parent.item(), level.item())
                    
                    # Skip if this group was already processed
                    if group_key in processed_groups:
                        continue
                    
                    # Mark this group as processed
                    processed_groups.add(group_key)
                    
                    # Find all siblings (children of same parent) AFTER splitting
                    siblings_mask = (new_parent_indices == new_parent) & (new_levels == level)
                    siblings = torch.where(siblings_mask)[0]
                    
                    if len(siblings) > 0 and original_parent_weights is not None:
                        # Get original siblings (before splitting) - these are old indices
                        original_siblings_old = siblings[siblings < N_old]
                        
                        if len(original_siblings_old) > 0:
                            # Map new sibling indices to old indices (vectorized)
                            # siblings contains both old (siblings < N_old) and new (siblings >= N_old) indices
                            is_old_sibling = siblings < N_old
                            
                            # Build mapping: new_idx -> old_idx for split gaussians
                            sibling_old_indices = torch.zeros_like(siblings)
                            sibling_old_indices[is_old_sibling] = siblings[is_old_sibling]  # Original siblings
                            
                            # For split gaussians, find their old indices
                            is_split = ~is_old_sibling
                            if is_split.any():
                                split_indices = siblings[is_split]
                                # new_idx = N_old + i, so i = new_idx - N_old
                                split_positions = split_indices - N_old
                                valid_split_mask = (split_positions >= 0) & (split_positions < len(sel))
                                if valid_split_mask.any():
                                    sibling_old_indices[is_split][valid_split_mask] = sel[split_positions[valid_split_mask]]
                            
                            # Adjust parent_weights to preserve weight_ratio (process all siblings at once)
                            new_sum = _adjust_sibling_weights_preserve_ratio(
                                params=params,
                                siblings=siblings,
                                sibling_old_indices=sibling_old_indices,
                                original_parent_weights=original_parent_weights,
                            )
                            
                            if new_sum > 1e-8:
                                # Adjust residuals for all siblings in this group (both original and split)
                                # All siblings need adjustment because parent_weight changed, which changes weight_ratio
                                for sib_new_idx in siblings:
                                    # Find original index
                                    sib_is_split = sib_new_idx >= N_old
                                    if sib_is_split:
                                        # This is a split gaussian
                                        sib_split_pos = sib_new_idx - N_old
                                        if sib_split_pos >= 0 and sib_split_pos < len(sel):
                                            sib_old_idx = sel[sib_split_pos]
                                        else:
                                            continue
                                    else:
                                        # This is an original sibling - also needs adjustment!
                                        sib_old_idx = sib_new_idx
                                    
                                    # Get original actual parameters (original_actual_params contains all gaussians)
                                    original_actual = {k: v[sib_old_idx] for k, v in original_actual_params.items()}
                                    
                                    # Compute weight_ratio for this child
                                    child_weight = torch.nn.functional.elu(params["parent_weight"].data[sib_new_idx]) + 1.0
                                    weight_ratio = (child_weight / new_sum).item()
                                    
                                    new_parent_actual = {k: v[new_parent] for k, v in new_actual_splats.items()}
                                    
                                    if level > 1:
                                        for name in original_actual.keys():
                                            if name not in params or name == "parent_weight":
                                                continue
                                            new_residual = _compute_residual_from_actual(
                                                name=name,
                                                orig_actual_val=original_actual[name],
                                                parent_actual_val=new_parent_actual[name],
                                                level=level,
                                                multigrid_gaussians=multigrid_gaussians,
                                                weight_ratio=weight_ratio,
                                            )
                                            params[name].data[sib_new_idx] = new_residual
    
    # Note: _add_zero_parameter_children is now called once after all grow operations
    # in _grow_gs to avoid redundant processing


@torch.no_grad()
def _add_zero_parameter_children(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    levels: Tensor,  # Current levels tensor
    parent_indices_tensor: Tensor,  # Current parent_indices tensor
    level_indices: Dict[int, List[int]],  # Current level_indices dict
    max_level: Optional[int] = None,  # Maximum level (if None, use state)
    min_children_per_parent: int = 1,  # Minimum children per parent
):
    """Add zero-parameter children to parents that need them, processing level by level from coarse to fine.
    
    This function processes all levels from 1 to max_level-1, ensuring that each parent
    has at least min_children_per_parent children. Similar to initialization logic in
    MultigridGaussians.__init__.
    
    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers.
        state: State dictionary containing hierarchical structure.
        levels: Current levels tensor [N,].
        parent_indices_tensor: Current parent_indices tensor [N,].
        level_indices: Current level_indices dict.
        max_level: Maximum level in hierarchy. If None, retrieved from state.
        min_children_per_parent: Minimum number of children to add per parent. Default is 1.
    """
    device = levels.device
    N_old = len(levels)
    
    # Get max_level from state if not provided
    if max_level is None:
        max_level = state.get("max_level", None)
        if max_level is None:
            return  # Cannot proceed without max_level
    
    if max_level <= 1:
        return  # No children to add
    
    # Process level by level from coarsest to finest
    # This ensures parent nodes exist before adding their children
    for current_level in range(1, max_level):
        child_level = current_level + 1
        
        # Get all nodes at current_level
        current_level_mask = (levels == current_level)
        current_level_indices = torch.where(current_level_mask)[0]
        
        if len(current_level_indices) == 0:
            continue
        
        # Count children for each node at current_level (vectorized)
        # Use current length of levels to handle children added in previous iterations
        current_N = len(levels)
        valid_parent_mask = (parent_indices_tensor >= 0) & (parent_indices_tensor < current_N)
        n_children_all = torch.bincount(
            parent_indices_tensor[valid_parent_mask],
            minlength=current_N
        )  # [current_N,] - number of children for each node
        n_children_per_node = n_children_all[current_level_indices]  # [M,] where M = len(current_level_indices)
        
        # Find nodes that need children (num_children < min_children_per_parent)
        needs_children_mask = n_children_per_node < min_children_per_parent  # [M,]
        parents_to_add = current_level_indices[needs_children_mask]  # [K,]
        
        if len(parents_to_add) == 0:
            continue
        
        # For each parent, add exactly min_children_per_parent children (vectorized)
        n_parents = len(parents_to_add)
        n_children_per_parent = min_children_per_parent
        total_new_children = n_parents * n_children_per_parent
        
        # Create parent indices and levels for new children
        new_children_parent_indices = parents_to_add.repeat_interleave(n_children_per_parent)  # [total_new_children,]
        new_children_levels = torch.full((total_new_children,), child_level, device=device, dtype=torch.long)

        # Initialize child opacity residuals to conserve parent's actual opacity
        child_opacity_residuals = None
        multigrid_gaussians = state.get("multigrid_gaussians", None)
        if multigrid_gaussians is not None and total_new_children > 0:
            with torch.no_grad():
                actual_splats = multigrid_gaussians.get_splats(
                    level=None, detach_parents=False, current_splats=None
                )
                parent_opacity_logits = actual_splats["opacities"][parents_to_add]
                parent_opacity = torch.sigmoid(parent_opacity_logits)
                # Each child should satisfy: 1 - (1 - a')^K = a (K = n_children_per_parent)
                target_opacity = 1.0 - torch.pow(
                    (1.0 - parent_opacity).clamp_min(1e-6),
                    1.0 / float(n_children_per_parent),
                )
                target_opacity = target_opacity.clamp(1e-6, 1.0 - 1e-6)
                target_logits = torch.logit(target_opacity)
                child_opacity_residuals = target_logits - parent_opacity_logits
                child_opacity_residuals = child_opacity_residuals.repeat_interleave(
                    n_children_per_parent
                )
                del actual_splats, parent_opacity_logits, parent_opacity, target_opacity, target_logits
        
        # Initialize all children with zero residuals (vectorized)
        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "means":
                p_children = torch.zeros((total_new_children, 3), device=device, dtype=p.dtype)
            elif name == "scales":
                # Set child residual scale to SPLIT_SCALE_LOG so that actual scale equals parent scale
                # In get_splats: child_scale = parent_scale + child_residual - SPLIT_SCALE_LOG
                # So child_residual = SPLIT_SCALE_LOG gives child_scale = parent_scale
                # p_children = torch.full((total_new_children, 3), SPLIT_SCALE_LOG, device=device, dtype=p.dtype)
                p_children = torch.zeros((total_new_children, 3), device=device, dtype=p.dtype)
            elif name == "quats":
                # Zero quaternions: [0, 0, 0, 0] (residual)
                p_children = torch.zeros((total_new_children, 4), device=device, dtype=p.dtype)
            elif name == "opacities":
                if child_opacity_residuals is not None:
                    p_children = child_opacity_residuals.to(device=device, dtype=p.dtype)
                else:
                    p_children = torch.zeros((total_new_children,), device=device, dtype=p.dtype)
            elif name == "parent_weight":
                # Initialize parent_weight to 1.0 (uniform distribution initially)
                p_children = torch.ones((total_new_children,), device=device, dtype=p.dtype)
            else:
                # For colors (sh0, shN, features, colors): initialize with zero residuals
                p_children = torch.zeros((total_new_children, *p.shape[1:]), device=device, dtype=p.dtype)
            p_new = torch.cat([p, p_children])
            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        
        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v_children = torch.zeros((total_new_children, *v.shape[1:]), device=device)
            return torch.cat([v, v_children])
        
        # Update parameters and optimizers
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        
        # Update state (exclude hierarchical structure, handled separately)
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and k not in ["levels", "parent_indices", "level_indices"]:
                v_children = torch.zeros((total_new_children, *v.shape[1:]), device=device, dtype=v.dtype)
                state[k] = torch.cat([v, v_children])
        
        # Update hierarchical structure
        levels = torch.cat([levels, new_children_levels])
        parent_indices_tensor = torch.cat([parent_indices_tensor, new_children_parent_indices])
        
        # Update N_old for next iteration
        N_old = len(levels)
    
    # Update hierarchical structure in state
    _update_hierarchical_structure(state, levels, parent_indices_tensor)


@torch.no_grad()
def remove_mg(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Inplace remove the Gaussian with the given mask (multigrid version).
    
    Safety-checked version that handles out-of-bounds parent indices gracefully.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that contains level_indices, levels, parent_indices.
        mask: A boolean mask to remove the Gaussians (True = keep, False = remove).
    """
    # Get hierarchical structure from state
    levels = state["levels"]  # [N,] with level for each gaussian
    parent_indices = state["parent_indices"]  # [N,] with parent index for each gaussian (-1 means no parent)
    level_indices = state["level_indices"]  # Dict mapping level -> list of gaussian indices at that level
    
    # 1. 살아남을 인덱스 선택
    sel = torch.where(~mask)[0]  # Keep these indices
    
    # Early return if nothing to remove (all kept)
    if len(sel) == len(mask):
        return  # Nothing to remove
    
    # Early return if everything is removed (should not happen in normal usage)
    if len(sel) == 0:
        # This would remove all gaussians, which is likely an error
        # But we handle it gracefully by keeping the structure
        return
    
    # Store actual parameters and parent_weight BEFORE pruning to preserve them
    multigrid_gaussians = state.get("multigrid_gaussians", None)
    original_actual_params = None
    original_parent_weights = None
    if multigrid_gaussians is not None and "parent_weight" in params:
        with torch.no_grad():
            original_actual_params = multigrid_gaussians.get_splats(
                level=None, detach_parents=False, current_splats=None
            )
            # Extract only surviving gaussians' actual parameters
            original_actual_params = {k: v[sel] for k, v in original_actual_params.items()}
            # Store original parent_weight before pruning (needed for computing original sum)
            original_parent_weights = params["parent_weight"].data.clone()

    # 2. Parameter 및 Optimizer 상태 업데이트
    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[sel]

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    # 3. State 업데이트 (levels, parent_indices 제외)
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k not in ["levels", "parent_indices", "level_indices"]:
            state[k] = v[sel]
    
    # 4. 계층 구조 업데이트 준비
    current_size = len(levels)
    new_levels = levels[sel]
    # 살아남은 자식들이 가리키고 있는 '구' 부모 인덱스
    current_parent_indices_of_survivors = parent_indices[sel]
    
    # 5. Index Remapping (Old Index -> New Index)
    # 삭제된 노드는 -1, 살아남은 노드는 0부터 순차적으로 매핑
    old_to_new = torch.full((current_size,), -1, dtype=torch.long, device=levels.device)
    old_to_new[sel] = torch.arange(len(sel), device=levels.device)
    
    # 6. 부모 인덱스 갱신: old_to_new를 사용하여 직접 remapping
    # 초기화: 모든 부모를 -1로 설정
    new_parent_indices = torch.full_like(current_parent_indices_of_survivors, -1)
    
    # 유효한 부모 인덱스만 remapping (범위 체크 포함)
    valid_mask = (current_parent_indices_of_survivors >= 0) & (current_parent_indices_of_survivors < current_size)
    # print invalid mask size
    invalid_mask = ~valid_mask
    # print(f"Invalid mask size: {invalid_mask.sum().item()}")
    if valid_mask.any():
        # old_to_new를 사용하여 부모 인덱스 remapping
        # 삭제된 부모는 old_to_new[parent] = -1이 되어 자동으로 고아가 됨
        new_parent_indices[valid_mask] = old_to_new[current_parent_indices_of_survivors[valid_mask]]

    # 7. Level Indices 재구성
    # new_levels는 이미 필터링된 것이므로, 새로운 인덱스 공간(0부터 len(sel)-1)에서 재구성
    level_indices_new = {
        level_val.item(): torch.where(new_levels == level_val)[0].tolist()
        for level_val in new_levels.unique()
    }
    
    # 8. 최종 State 반영
    state["levels"] = new_levels
    state["parent_indices"] = new_parent_indices
    state["level_indices"] = level_indices_new
    
    # 9. multigrid_gaussians 객체도 업데이트 (params와 동기화)
    if "multigrid_gaussians" in state and state["multigrid_gaussians"] is not None:
        multigrid_gaussians = state["multigrid_gaussians"]
        multigrid_gaussians.levels = new_levels
        multigrid_gaussians.parent_indices = new_parent_indices
        multigrid_gaussians.level_indices = level_indices_new
        # Invalidate cache after densification
        multigrid_gaussians.invalidate_splats_cache()
    
    # 10. Adjust parent_weight and residual parameters to preserve actual parameters
    # This is necessary because pruning parent_weight changes the sum for siblings,
    # which changes weight_ratio and thus actual parameters
    if original_actual_params is not None and original_parent_weights is not None and multigrid_gaussians is not None and "parent_weight" in params:
        with torch.no_grad():
            N_new = len(new_levels)
            
            # Recompute actual parameters with new structure
            new_actual_splats = multigrid_gaussians.get_splats(
                level=None, detach_parents=False, current_splats=None
            )
            
            # Group surviving gaussians by (parent, level) to avoid duplicate processing
            # Process each (parent, level) group only once
            processed_groups = set()
            
            # For each surviving gaussian, adjust parent_weight and residuals to preserve actual parameters
            for new_idx in range(N_new):
                new_parent = new_parent_indices[new_idx]
                level = new_levels[new_idx]
                
                # Only adjust if this is a child (has parent)
                if new_parent >= 0 and new_parent < N_new:
                    # Create unique key for (parent, level) group
                    group_key = (new_parent.item(), level.item())
                    
                    # Skip if this group was already processed
                    if group_key in processed_groups:
                        continue
                    
                    # Mark this group as processed
                    processed_groups.add(group_key)
                    
                    # Find all siblings (children of same parent) AFTER pruning
                    siblings_mask = (new_parent_indices == new_parent) & (new_levels == level)
                    siblings = torch.where(siblings_mask)[0]
                    
                    if len(siblings) > 0:
                        # Get original parent index
                        old_parent_idx = sel[new_parent]  # Map new parent index back to old parent index
                        
                        # Find all original siblings (before pruning)
                        original_siblings_mask = (parent_indices == old_parent_idx) & (levels == level)
                        original_siblings = torch.where(original_siblings_mask)[0]
                        
                        if len(original_siblings) > 0 and original_parent_weights is not None:
                            # Map surviving siblings to their old indices
                            siblings_old_indices = sel[siblings]
                            
                            # Adjust all siblings' weights at once to preserve weight_ratio
                            new_sum = _adjust_sibling_weights_preserve_ratio(
                                params=params,
                                siblings=siblings,
                                sibling_old_indices=siblings_old_indices,
                                original_parent_weights=original_parent_weights,
                            )
                            
                            if new_sum > 1e-8:
                                # Adjust residuals for all siblings in this group
                                for sib_new_idx in siblings:
                                    # Get original actual parameters for this sibling
                                    sib_old_idx = sel[sib_new_idx]
                                    sib_level = new_levels[sib_new_idx]
                                    
                                    # Find position in sel (original_actual_params is indexed by sel)
                                    sib_pos_in_sel = (sel == sib_old_idx).nonzero(as_tuple=True)[0]
                                    if len(sib_pos_in_sel) > 0:
                                        sib_pos = sib_pos_in_sel[0].item()
                                        
                                        # Compute weight_ratio for this child
                                        child_weight = torch.nn.functional.elu(params["parent_weight"].data[sib_new_idx]) + 1.0
                                        weight_ratio = (child_weight / new_sum).item()
                                        
                                        # Get original actual parameters
                                        original_actual = {k: v[sib_pos] for k, v in original_actual_params.items()}
                                        
                                        # Get new parent actual parameters
                                        new_parent_actual = {k: v[new_parent] for k, v in new_actual_splats.items()}
                                        
                                        # Compute new residuals to preserve actual parameters
                                        if sib_level > 1:
                                            for name in original_actual.keys():
                                                if name not in params or name == "parent_weight":
                                                    continue
                                                new_residual = _compute_residual_from_actual(
                                                    name=name,
                                                    orig_actual_val=original_actual[name],
                                                    parent_actual_val=new_parent_actual[name],
                                                    level=sib_level,
                                                    multigrid_gaussians=multigrid_gaussians,
                                                    weight_ratio=weight_ratio,
                                                )
                                                params[name].data[sib_new_idx] = new_residual


@torch.no_grad()
def create_children_mg(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    levels: Tensor,
    parent_indices: Tensor,
    level_indices: Dict[int, List[int]],
    n_children_per_split: int = 4,
):
    """Inplace create children for the Gaussians with the given mask (multigrid version).
    
    Creates children at the next level with independent/residual parameters:
    - scales: parent_actual_scale - 0.5 (in log space, independent parameter)
    - quats: parent_actual_quats (same as parent, independent parameter)
    - opacities: revised_opacity formula (independent parameter, like scales, quats)
    - means: small residual offsets (parent mean + child residual * scale_factor in get_splats)
    - sh0, shN, etc.: zero residuals (parent + child residual in get_splats)

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that may contain level_indices, levels, parent_indices.
        mask: A boolean mask to create children for the Gaussians.
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent).
        level_indices: Dict mapping level -> list of gaussian indices at that level.
        n_children_per_split: Number of children to create per parent. Default is 4.
    """
    device = mask.device
    sel = torch.where(mask)[0]  # Parents to create children for (these will be kept)
    
    # Validate: sel should be within bounds BEFORE processing
    N_old = len(levels)
    valid_sel_mask = (sel >= 0) & (sel < N_old)
    if not valid_sel_mask.all():
        # Filter out invalid indices (should not happen, but safety check)
        sel = sel[valid_sel_mask]
        if len(sel) == 0:
            return  # No valid parents to create children for
    
    # Now process with validated sel
    # All parameters are now residual, so we need parent's actual values for initialization
    multigrid_gaussians = state.get("multigrid_gaussians", None)

    # Get actual splats for parents to get their actual scales, quats, and opacities
    # All parameters are residual, so we use parent's actual values for initialization
    # No gradients needed for initialization
    # Always use current splats (no caching)
    with torch.no_grad():
        parent_actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False, current_splats=None)
        parent_actual_scales = parent_actual_splats["scales"][sel]  # [len(sel), 3] - actual scales in log space
        parent_actual_quats = parent_actual_splats["quats"][sel]  # [len(sel), 4] - actual quaternions
        parent_actual_opacities = parent_actual_splats["opacities"][sel]  # [len(sel),] - actual opacities in logit space
        # Free parent_actual_splats immediately after extracting needed values
        del parent_actual_splats
    
    # For means initialization: use parent's actual scales and quats
    scales_exp = torch.exp(parent_actual_scales)  # [len(sel), 3] - actual scales in exp space
    rotmats = normalized_quat_to_rotmat(parent_actual_quats)  # [len(sel), 3, 3] - actual rotation matrices
    
    # Generate small random samples for means residual initialization
    # These are small offsets relative to parent's scale
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales_exp,
        torch.randn(n_children_per_split, len(scales_exp), 3, device=device),
    )  # [n_children_per_split, N, 3]
    # Scale down samples to be small residual offsets
    # Use small scale factor to keep children close to parent initially

    # Get parent levels and compute child levels (after sel validation)
    parent_levels = levels[sel]
    child_levels = parent_levels + 1

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [n_children_per_split] + [1] * (p.dim() - 1)
        if name == "means":
            # Initialize means as small residual offsets (not absolute positions)
            # Parent position will be added in get_splats() with hierarchical structure
            # Small random offset helps children explore nearby space while staying close to parent
            p_split = samples.reshape(-1, 3)  # [n_children_per_split*N, 3] - residual only
        elif name == "scales":
            # Initialize as residual: 0 (in log space)
            # Child scale = parent_scale + 0 - SPLIT_SCALE_LOG = parent_scale - SPLIT_SCALE_LOG
            # This makes child scale approximately parent/1.6 (exp(-log(1.6)) ≈ 0.625)
            p_split = (p[sel] - torch.sqrt(n_children_per_split) + SPLIT_SCALE_LOG).repeat_interleave(n_children_per_split, 1)
            # p_split = torch.zeros((n_children_per_split * len(sel), 3), device=device, dtype=p.dtype)
        elif name == "quats":
            # Initialize as residual: identity quaternion [1, 0, 0, 0]
            # Child quat = parent_quat * identity = parent_quat (no rotation change)
            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=p.dtype)  # [4] - identity quaternion
            p_split = identity_quat.unsqueeze(0).repeat(n_children_per_split * len(sel), 1)  # [n_children_per_split * len(sel), 4]
        elif name == "opacities":
            # Initialize as residual: 0
            # Child opacity = parent_opacity + 0 = parent_opacity
            p_split = torch.zeros((n_children_per_split * len(sel),), device=device, dtype=p.dtype)
        else:
            # For other parameters (sh0, shN, etc.), initialize to zero (residual)
            p_split = torch.zeros((n_children_per_split * len(sel), *p.shape[1:]), device=device, dtype=p.dtype)
        # Keep all original gaussians (including parents) and append children
        p_new = torch.cat([p, p_split])
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros((n_children_per_split * len(sel), *v.shape[1:]), device=device)
        # Keep all original optimizer states and append zeros for children
        return torch.cat([v, v_split])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    # update the extra running state
    # For state variables (grad2d, count, radii, etc.), initialize children to zero
    # Note: scales, quats, opacities are independent parameters
    # State variables are always initialized to zero regardless of parameter type
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k not in ["levels", "parent_indices", "level_indices"]:
            # Initialize children state to zero (state variables like grad2d, count, radii)
            v_split = torch.zeros((n_children_per_split * len(sel), *v.shape[1:]), device=device, dtype=v.dtype)
            state[k] = torch.cat([v, v_split])
    
    # Update hierarchical structure
    # Note: sel has already been validated and filtered above
    N_new = N_old + n_children_per_split * len(sel)
    
    # Create child parent indices (pointing to original parent indices)
    # Each parent index should be repeated n_children_per_split times
    # Example: sel=[0, 5, 10], n_children_per_split=4 -> [0,0,0,0, 5,5,5,5, 10,10,10,10]
    child_parent_indices = sel.repeat_interleave(n_children_per_split)  # [n_children_per_split*len(sel)]
    
    # Append child levels and parent_indices
    # child_levels should also be repeated with repeat_interleave to match child_parent_indices
    # Example: child_levels=[2, 3, 2], n_children_per_split=4 -> [2,2,2,2, 3,3,3,3, 2,2,2,2]
    new_levels = torch.cat([levels, child_levels.repeat_interleave(n_children_per_split)])
    new_parent_indices = torch.cat([parent_indices, child_parent_indices])
    
    # Update level_indices
    level_indices_new = {}
    for level_val in new_levels.unique():
        level_val_int = level_val.item()
        mask_level = (new_levels == level_val_int)
        level_indices_new[level_val_int] = torch.where(mask_level)[0].tolist()
    
    # Update state
    state["levels"] = new_levels
    state["parent_indices"] = new_parent_indices
    state["level_indices"] = level_indices_new
    
    # Update multigrid_gaussians object if present
    if "multigrid_gaussians" in state and state["multigrid_gaussians"] is not None:
        multigrid_gaussians = state["multigrid_gaussians"]
        multigrid_gaussians.levels = new_levels
        multigrid_gaussians.parent_indices = new_parent_indices
        multigrid_gaussians.level_indices = level_indices_new
        # Invalidate cache after densification
        multigrid_gaussians.invalidate_splats_cache()


@torch.no_grad()
def clone_hierarchy_block(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    parent_indices_to_clone: Tensor,  # [M,] - parent indices to clone (with their children)
    levels: Tensor,
    parent_indices: Tensor,
    level_indices: Dict[int, List[int]],
    signal_indices: Optional[Tensor] = None,  # [N,] bool - only clone children that are in signal_indices
    is_small_mask: Optional[Tensor] = None,  # [N,] bool - True for duplicate-like scale adjustment, False for split-like
    return_sources: bool = False,  # If True, return (new_indices, source_indices)
):
    """Clone hierarchy blocks: for each parent, clone the parent and specified children.
    
    This is used when a child needs duplication but its parent is full.
    Instead of just duplicating the parent, we clone the parent and the children that need duplication
    to maintain the hierarchy structure.
    
    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers.
        state: State dictionary containing hierarchical structure.
        parent_indices_to_clone: Tensor [M,] with parent indices to clone.
        children_to_clone: Optional dict mapping parent_idx -> Tensor of child indices to clone.
                         If None, clones all children of each parent.
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian.
        level_indices: Dict mapping level -> list of gaussian indices.
    """
    if len(parent_indices_to_clone) == 0:
        return (None, None) if return_sources else None
    
    device = parent_indices_to_clone.device
    N_old = len(levels)
    
    # Remove duplicates and validate
    parent_indices_to_clone = torch.unique(parent_indices_to_clone)
    valid_mask = (parent_indices_to_clone >= 0) & (parent_indices_to_clone < N_old)
    parent_indices_to_clone = parent_indices_to_clone[valid_mask]
    
    if len(parent_indices_to_clone) == 0:
        return (None, None) if return_sources else None
    
    if signal_indices is not None:
        parents_mask = torch.zeros(N_old, dtype=torch.bool, device=device)
        parents_mask[parent_indices_to_clone] = True
        descendant_mask = parents_mask.clone()
        max_level = int(levels.max().item()) if N_old > 0 else 1

        # Mark descendants of parents_to_clone
        for level in range(1, max_level + 1):
            level_mask = levels == level
            if not level_mask.any():
                continue
            level_indices_tensor = torch.where(level_mask)[0]
            level_parents = parent_indices[level_indices_tensor]
            valid_mask = (level_parents >= 0) & (level_parents < N_old)
            if valid_mask.any():
                level_children = level_indices_tensor[valid_mask]
                descendant_mask[level_children] = (
                    descendant_mask[level_children] | descendant_mask[level_parents[valid_mask]]
                )

        # Include signal descendants and connect them to parents_to_clone
        signal_descendants = signal_indices & descendant_mask
        include_mask = parents_mask | signal_descendants
        for level in range(max_level, 1, -1):
            level_mask = (levels == level) & include_mask
            if not level_mask.any():
                continue
            level_indices_tensor = torch.where(level_mask)[0]
            level_parents = parent_indices[level_indices_tensor]
            valid_mask = (level_parents >= 0) & (level_parents < N_old)
            if valid_mask.any():
                include_mask[level_parents[valid_mask]] = True

        include_mask = include_mask & descendant_mask
        all_indices_to_clone = torch.where(include_mask)[0]
    else:
        # Clone all children of all parents (fully vectorized)
        # Find all children for all parents at once using torch.isin
        valid_parent_mask = (parent_indices >= 0) & (parent_indices < N_old)
        children_of_any_parent = torch.isin(parent_indices, parent_indices_to_clone) & valid_parent_mask
        all_children_indices = torch.where(children_of_any_parent)[0]  # All children of any parent in parent_indices_to_clone
        
        # Group children by their parent (vectorized)
        if len(all_children_indices) > 0:
            children_parents = parent_indices[all_children_indices]  # Parent index for each child
            
            # Build blocks: for each parent, collect its children (vectorized filtering)
            # Use argsort to group children by parent efficiently
            sort_indices = torch.argsort(children_parents)
            sorted_children = all_children_indices[sort_indices]
            sorted_parents = children_parents[sort_indices]
            
            # Find unique parents and their boundaries in sorted array
            unique_parents, counts = torch.unique_consecutive(sorted_parents, return_counts=True)
            
            # Build blocks using cumulative sum for indexing
            cumsum = torch.cat([torch.tensor([0], device=device), counts.cumsum(0)])
            
            # Vectorized: map parent_indices_to_clone to unique indices
            # Create a tensor mapping from parent_idx to unique_idx (or -1 if not found)
            max_parent_idx = parent_indices_to_clone.max().item() if len(parent_indices_to_clone) > 0 else -1
            parent_to_unique_tensor = torch.full((max_parent_idx + 1,), -1, dtype=torch.long, device=device)
            if len(unique_parents) > 0:
                parent_to_unique_tensor[unique_parents] = torch.arange(len(unique_parents), device=device)
            
            # Vectorized: find unique indices for all parents to clone
            parent_unique_indices = parent_to_unique_tensor[parent_indices_to_clone]  # [M,]
            has_children_mask = parent_unique_indices >= 0  # [M,] - True if parent has children
            
            # Build blocks (partially vectorized: dictionary creation still needs loop, but indexing is vectorized)
            blocks_to_clone = {}
            for i, parent_idx in enumerate(parent_indices_to_clone):
                parent_idx_int = parent_idx.item()
                if has_children_mask[i]:
                    unique_idx = parent_unique_indices[i].item()
                    start_idx = cumsum[unique_idx].item()
                    end_idx = cumsum[unique_idx + 1].item()
                    children = sorted_children[start_idx:end_idx]
                else:
                    children = torch.empty(0, dtype=torch.long, device=device)
                # Include parent itself and its children
                blocks_to_clone[parent_idx_int] = [parent_idx_int] + children.tolist()
        else:
            # No children found, only clone parents
            blocks_to_clone = {parent_idx.item(): [parent_idx.item()] for parent_idx in parent_indices_to_clone}
    
        # Collect all gaussians to clone (flatten blocks) - vectorized where possible
        # Since blocks have different sizes, we still need to iterate, but we can pre-allocate
        block_sizes = [len(block) for block in blocks_to_clone.values()]
        total_size = sum(block_sizes)
        if total_size > 0:
            all_indices_to_clone = torch.empty(total_size, dtype=torch.long, device=device)
            idx = 0
            for block in blocks_to_clone.values():
                # Optimize: if block is already a tensor, use it directly; otherwise convert once
                if isinstance(block, torch.Tensor):
                    block_tensor = block
                else:
                    block_tensor = torch.tensor(block, dtype=torch.long, device=device)
                block_len = len(block_tensor)
                all_indices_to_clone[idx:idx + block_len] = block_tensor
                idx += block_len
        else:
            all_indices_to_clone = torch.empty(0, dtype=torch.long, device=device)

    # Remove duplicates (a gaussian might be both a parent and a child) - vectorized
    all_indices_to_clone = torch.unique(all_indices_to_clone)
    
    if len(all_indices_to_clone) == 0:
        return (None, None) if return_sources else None
    
    # Sort by level (top-down: parents before children)
    all_levels = levels[all_indices_to_clone]
    sort_indices = torch.argsort(all_levels)
    all_indices_to_clone = all_indices_to_clone[sort_indices]
    
    # Clone all gaussians in the blocks
    sel = all_indices_to_clone
    
    # Get levels and parent indices of selected gaussians
    sel_levels = levels[sel]
    sel_parents = parent_indices[sel]
    
    # Compute actual parameters for original gaussians BEFORE cloning
    # This preserves actual parameters when parent changes after cloning
    multigrid_gaussians = state.get("multigrid_gaussians", None)
    original_actual_params = None
    if multigrid_gaussians is not None and sel.numel() > 0:
        with torch.no_grad():
            # Store ALL gaussians' actual parameters (needed for adjusting original siblings too)
            original_actual_params = multigrid_gaussians.get_splats(
                level=None, detach_parents=False, current_splats=None
            )
    
    def param_fn(name: str, p: Tensor) -> Tensor:
        # Vanilla-style clone: copy parameters as-is (matches ops.py duplicate)
        # Note: We'll adjust residuals after parent_indices are updated
        p_new = torch.cat([p, p[sel]])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
    
    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])
    
    # Update parameters and optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    # Update state (exclude hierarchical structure, handled separately)
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k not in ["levels", "parent_indices", "level_indices"]:
            state[k] = torch.cat((v, v[sel]))
    
    # Update hierarchical structure
    N_new = N_old + len(sel)
    
    # Build mapping: old_idx -> new_idx for cloned gaussians
    old_to_new = torch.full((N_old,), -1, dtype=torch.long, device=device)
    new_indices = torch.arange(N_old, N_new, device=device)
    old_to_new[sel] = new_indices
    
    # For cloned gaussians, update parent indices:
    # - If parent was also cloned, point to new parent index
    # - If parent was not cloned, keep original parent index
    # Vectorized: avoid Python loop
    new_parent_indices_cloned = torch.full_like(sel_parents, -1)
    valid_parent_mask = (sel_parents >= 0) & (sel_parents < N_old)
    if valid_parent_mask.any():
        old_parents = sel_parents[valid_parent_mask]  # [K,]
        # Check which parents were cloned
        cloned_parent_new_indices = old_to_new[old_parents]  # [K,]
        parent_was_cloned = cloned_parent_new_indices >= 0  # [K,]
        
        # Use new parent index if cloned, otherwise keep original
        new_parent_indices_cloned[valid_parent_mask] = torch.where(
            parent_was_cloned,
            cloned_parent_new_indices,
            old_parents
        )
    
    # Validate: sel_parents should be within bounds or -1
    valid_parent_mask = (sel_parents == -1) | ((sel_parents >= 0) & (sel_parents < N_old))
    if not valid_parent_mask.all():
        # Fix invalid parent indices (set to -1, making them root nodes)
        sel_parents = sel_parents.clone()
        sel_parents[~valid_parent_mask] = -1
    
    # Append cloned levels and parent_indices
    new_levels = torch.cat([levels, sel_levels])
    new_parent_indices = torch.cat([parent_indices, new_parent_indices_cloned])
    
    # Update hierarchical structure FIRST (needed for get_splats)
    _update_hierarchical_structure(state, new_levels, new_parent_indices)
    
    # Recompute residual parameters for cloned gaussians to preserve actual parameters
    # This is necessary because cloned gaussians may have different parents
    # Also adjust parent_weight to preserve weight_ratio when siblings change
    if original_actual_params is not None and multigrid_gaussians is not None:
        with torch.no_grad():
            # Get new parent actual parameters for cloned gaussians
            new_actual_splats = multigrid_gaussians.get_splats(
                level=None, detach_parents=False, current_splats=None
            )
            
            # First, adjust parent_weight for all cloned children to preserve weight_ratio
            # Group by (parent, level) to avoid duplicate processing
            processed_groups = set()
            original_parent_weights_stored = None
            if "parent_weight" in params:
                original_parent_weights_stored = params["parent_weight"].data.clone()
                
                for i, old_idx in enumerate(sel):
                    new_idx = new_indices[i]
                    new_parent = new_parent_indices_cloned[i]
                    level = sel_levels[i]
                    
                    # Only adjust if this is a child (has parent)
                    if new_parent >= 0 and new_parent < N_new:
                        # Create unique key for (parent, level) group
                        group_key = (new_parent.item(), level.item())
                        
                        # Skip if this group was already processed
                        if group_key in processed_groups:
                            continue
                        
                        # Mark this group as processed
                        processed_groups.add(group_key)
                        
                        # Find all siblings (children of same parent) AFTER cloning
                        siblings_mask = (new_parent_indices == new_parent) & (new_levels == level)
                        siblings = torch.where(siblings_mask)[0]
                        
                        if len(siblings) > 0:
                            # Map sibling indices to old indices
                            is_old_sibling = siblings < N_old
                            sibling_old_indices = torch.zeros_like(siblings)
                            sibling_old_indices[is_old_sibling] = siblings[is_old_sibling]
                            
                            # For cloned gaussians, find their old indices
                            is_cloned = ~is_old_sibling
                            if is_cloned.any():
                                cloned_indices = siblings[is_cloned]
                                cloned_positions = cloned_indices - N_old
                                valid_cloned_mask = (cloned_positions >= 0) & (cloned_positions < len(sel))
                                if valid_cloned_mask.any():
                                    sibling_old_indices[is_cloned][valid_cloned_mask] = sel[cloned_positions[valid_cloned_mask]]
                            
                            # Adjust parent_weights to preserve weight_ratio (process all siblings at once)
                            new_sum = _adjust_sibling_weights_preserve_ratio(
                                params=params,
                                siblings=siblings,
                                sibling_old_indices=sibling_old_indices,
                                original_parent_weights=original_parent_weights_stored,
                            )
            
            # Then, adjust residual parameters to preserve actual parameters
            # For each cloned gaussian, compute residual from new parent
            for name in original_actual_params.keys():
                if name not in params or name == "parent_weight":
                    continue
                
                cloned_actual = original_actual_params[name]  # [len(sel), ...]
                cloned_levels = sel_levels  # [len(sel),]
                cloned_parents = new_parent_indices_cloned  # [len(sel),]
                
                # Level 1: absolute parameters, no adjustment needed
                level1_mask = (cloned_levels == 1)
                if level1_mask.any():
                    # Level 1: actual = residual, so just copy
                    params[name].data[new_indices[level1_mask]] = cloned_actual[level1_mask]
                
                # Level 2+: residual = actual - parent_actual * weight_ratio + offset
                level2plus_mask = (cloned_levels > 1)
                if level2plus_mask.any():
                    level2plus_indices = new_indices[level2plus_mask]
                    level2plus_parents = cloned_parents[level2plus_mask]
                    valid_parent_mask = (level2plus_parents >= 0) & (level2plus_parents < N_new)
                    
                    if valid_parent_mask.any():
                        valid_indices = level2plus_indices[valid_parent_mask]
                        valid_parents = level2plus_parents[valid_parent_mask]
                        valid_actual = cloned_actual[level2plus_mask][valid_parent_mask]
                        parent_actual = new_actual_splats[name][valid_parents]
                        valid_levels = cloned_levels[level2plus_mask][valid_parent_mask]
                        
                        # Compute weight_ratio for each child (vectorized)
                        if "parent_weight" in params:
                            child_weights = torch.nn.functional.elu(params["parent_weight"].data[valid_indices]) + 1.0
                            # Compute parent weight sum for each child using scatter_add
                            # Group children by parent and level
                            device = valid_indices.device
                            max_parent_idx = valid_parents.max().item() if len(valid_parents) > 0 else -1
                            max_level = valid_levels.max().item() if len(valid_levels) > 0 else 1
                            
                            # Create unique keys for (parent, level) pairs
                            # Use a simple encoding: key = parent_idx * (max_level + 1) + level
                            parent_level_keys = valid_parents * (max_level + 1) + valid_levels
                            
                            # Compute weight sums per (parent, level) group using scatter_add
                            unique_keys, inverse_indices = torch.unique(parent_level_keys, return_inverse=True)
                            weight_sums_per_group = torch.zeros(len(unique_keys), device=device, dtype=child_weights.dtype)
                            weight_sums_per_group.scatter_add_(0, inverse_indices, child_weights)
                            
                            # Map back to children
                            parent_weight_sums = weight_sums_per_group[inverse_indices]
                            weight_ratios = child_weights / parent_weight_sums.clamp_min(1e-8)
                        else:
                            weight_ratios = None
                        
                        if name == "means":
                            # means: residual = (actual - parent_actual) / scale_factor (no weight_ratio)
                            scale_factor = multigrid_gaussians.position_scale_reduction ** (valid_levels.float() - 1)
                            scale_factor = scale_factor.unsqueeze(-1) if valid_actual.dim() > 1 else scale_factor
                            new_residual = (valid_actual - parent_actual) / scale_factor.clamp_min(1e-8)
                            params[name].data[valid_indices] = new_residual
                        elif name == "scales":
                            # scales: residual = actual - parent_actual * weight_ratio + CHILD_SCALE_LOG
                            from multigrid_gaussians_v9 import CHILD_SCALE_LOG
                            if weight_ratios is not None:
                                if parent_actual.dim() > 1:
                                    weight_ratios = weight_ratios.unsqueeze(-1)
                                new_residual = valid_actual - parent_actual * weight_ratios + CHILD_SCALE_LOG
                            else:
                                new_residual = valid_actual - parent_actual + CHILD_SCALE_LOG
                            params[name].data[valid_indices] = new_residual
                        else:
                            # opacities, sh0, shN, etc.: residual = actual - parent_actual * weight_ratio
                            if weight_ratios is not None:
                                if parent_actual.dim() > 1:
                                    weight_ratios_expanded = weight_ratios.view(-1, *([1] * (parent_actual.dim() - 1)))
                                else:
                                    weight_ratios_expanded = weight_ratios
                                new_residual = valid_actual - parent_actual * weight_ratios_expanded
                            else:
                                new_residual = valid_actual - parent_actual
                            params[name].data[valid_indices] = new_residual
                    
                    # Handle invalid parents (shouldn't happen, but safety check)
                    invalid_parent_mask = ~valid_parent_mask
                    if invalid_parent_mask.any():
                        invalid_indices = level2plus_indices[invalid_parent_mask]
                        # Set to zero residual (will become root-like)
                        params[name].data[invalid_indices] = 0.0
    
    # Note: _add_zero_parameter_children is now called once after all grow operations
    # in _grow_gs to avoid redundant processing
    
    # Return cloned indices (and sources if requested)
    if return_sources:
        return new_indices, all_indices_to_clone  # [len(sel),], [num_source]
    return new_indices


@torch.no_grad()
def reset_opa_mg(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    value: float,
    max_level: Optional[int] = None,
):
    """Inplace reset the opacities to the given post-sigmoid value (multigrid version).
    
    For multigrid_v6.py, opacities are residual parameters.
    - Only resets opacities at the current max_level.
    - Other levels maintain their current opacity values.
    - If max_level is 1: Reset to target value (absolute value)
    - If max_level > 1: Compute residual so that actual opacity equals target value.
                       actual_opacity = parent_actual_opacity + child_residual
                       child_residual = target_opacity - parent_actual_opacity

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary containing hierarchical structure (levels, parent_indices, level_indices).
        value: The target opacity value (post-sigmoid, e.g., 0.01).
        max_level: Maximum level in the hierarchy. If None, retrieved from state.
                   Only gaussians at this level will be reset.
    """
    levels = state.get("levels", None)
    if levels is None:
        raise ValueError("levels not found in state. Cannot reset opacities by level.")
    
    # Get max_level from state if not provided
    if max_level is None:
        max_level = state.get("max_level", None)
        if max_level is None:
            # Fallback: use maximum level in current structure
            if len(levels) > 0:
                max_level = int(levels.max().item())
            else:
                return  # No gaussians to reset
    
    max_level = int(max_level)
    
    # Only reset gaussians at max_level
    max_level_mask = (levels == max_level)
    if not max_level_mask.any():
        return  # No gaussians at max_level to reset
    
    # Get multigrid_gaussians to compute actual opacities
    multigrid_gaussians = state.get("multigrid_gaussians", None)
    if multigrid_gaussians is None:
        raise ValueError("multigrid_gaussians not found in state. Cannot compute actual opacities.")
    
    # Compute current actual opacities for max_level gaussians
    with torch.no_grad():
        actual_splats = multigrid_gaussians.get_splats(
            level=None, detach_parents=False, current_splats=None
        )
        max_level_indices = torch.where(max_level_mask)[0]
        current_actual_opacities = torch.sigmoid(actual_splats["opacities"][max_level_indices])
        
        # Target opacity in logit space
        target_opacity = torch.tensor(value, device=params["opacities"].device, dtype=params["opacities"].dtype)
        target_logit = torch.logit(target_opacity)
        
        if max_level == 1:
            # Level 1: absolute parameter, reset directly to target
            new_residuals = target_logit
        else:
            # Level 2+: residual parameter, compute residual to achieve target actual opacity
            parent_indices = state.get("parent_indices", None)
            if parent_indices is None:
                raise ValueError("parent_indices not found in state.")
            
            max_level_parents = parent_indices[max_level_indices]
            valid_parent_mask = (max_level_parents >= 0) & (max_level_parents < len(levels))
            
            if not valid_parent_mask.any():
                # No valid parents, treat as level 1
                new_residuals = target_logit
            else:
                # Get parent actual opacities
                parent_actual_opacities = torch.sigmoid(actual_splats["opacities"][max_level_parents[valid_parent_mask]])
                
                # Compute residual: target_opacity = parent_actual_opacity + residual
                # In logit space: logit(target) = logit(parent + residual)
                # We need: sigmoid(parent_logit + residual) = target_opacity
                # residual = logit(target_opacity) - parent_logit
                parent_logits = actual_splats["opacities"][max_level_parents[valid_parent_mask]]
                target_residuals = target_logit - parent_logits
                
                # Initialize with current residuals
                new_residuals = params["opacities"].data[max_level_indices].clone()
                new_residuals[valid_parent_mask] = target_residuals
                
                # For invalid parents, set to target (treat as level 1)
                invalid_parent_mask = ~valid_parent_mask
                if invalid_parent_mask.any():
                    new_residuals[invalid_parent_mask] = target_logit
        
        del actual_splats, current_actual_opacities
    
    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            # Create new opacities tensor
            new_opacities = p.clone()
            # Only update max_level gaussians
            new_opacities[max_level_indices] = new_residuals
            return torch.nn.Parameter(new_opacities, requires_grad=p.requires_grad)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )

