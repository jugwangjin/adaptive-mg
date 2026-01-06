import numpy as np
from typing import Callable, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat import quat_scale_to_covar_preci
from gsplat.relocation import compute_relocation
from gsplat.utils import normalized_quat_to_rotmat


@torch.no_grad()
def _multinomial_sample(weights: Tensor, n: int, replacement: bool = True) -> Tensor:
    """Sample from a distribution using torch.multinomial or numpy.random.choice.

    This function adaptively chooses between `torch.multinomial` and `numpy.random.choice`
    based on the number of elements in `weights`. If the number of elements exceeds
    the torch.multinomial limit (2^24), it falls back to using `numpy.random.choice`.

    Args:
        weights (Tensor): A 1D tensor of weights for each element.
        n (int): The number of samples to draw.
        replacement (bool): Whether to sample with replacement. Default is True.

    Returns:
        Tensor: A 1D tensor of sampled indices.
    """
    num_elements = weights.size(0)

    if num_elements <= 2**24:
        # Use torch.multinomial for elements within the limit
        return torch.multinomial(weights, n, replacement=replacement)
    else:
        # Fallback to numpy.random.choice for larger element spaces
        weights = weights / weights.sum()
        weights_np = weights.detach().cpu().numpy()
        sampled_idxs_np = np.random.choice(
            num_elements, size=n, p=weights_np, replace=replacement
        )
        sampled_idxs = torch.from_numpy(sampled_idxs_np)

        # Return the sampled indices on the original device
        return sampled_idxs.to(weights.device)


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
):
    """Inplace duplicate the Gaussian with the given mask (multigrid version, default.py style).
    
    Same as default.py duplicate but with hierarchical structure updates.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that may contain level_indices, levels, parent_indices.
        mask: A boolean mask to duplicate the Gaussians.
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent).
        level_indices: Dict mapping level -> list of gaussian indices at that level.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    
    # Get levels and parent indices of selected gaussians
    sel_levels = levels[sel]
    sel_parents = parent_indices[sel]

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            # Opacity reduced by -0.5 (modify original and copy)
            p[sel] -= 0.25
            p_split = p[sel]
            p_new = torch.cat([p, p_split])
        else:
            # Other parameters: copy as is
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
):
    """Inplace split the Gaussian with the given mask (multigrid version).
    
    Same as duplicate but with scale and opacity reduced by -0.5.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that may contain level_indices, levels, parent_indices.
        mask: A boolean mask to split the Gaussians.
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent).
        level_indices: Dict mapping level -> list of gaussian indices at that level.
        revised_opacity: Whether to use revised opacity formulation from arXiv:2404.06109.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    
    # Get levels and parent indices of selected gaussians
    sel_levels = levels[sel]
    sel_parents = parent_indices[sel]

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "scales":
            # Scale reduced by -0.5 (modify original and copy)
            p[sel] -= 0.5
            # p[sel] = p[sel].mean(dim=-1, keepdim=True).repeat(1, 3)
            p_split = p[sel]
        elif name == "opacities":
            # Opacity reduced by -0.5 (modify original and copy)
            p[sel] -= 0.25
            p_split = p[sel]
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


@torch.no_grad()
def remove_mg(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    levels: Tensor,
    parent_indices: Tensor,
    level_indices: Dict[int, List[int]],
):
    """Inplace remove the Gaussian with the given mask (multigrid version).
    
    Safety-checked version that handles out-of-bounds parent indices gracefully.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that may contain level_indices, levels, parent_indices.
        mask: A boolean mask to remove the Gaussians (True = keep, False = remove).
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent).
        level_indices: Dict mapping level -> list of gaussian indices at that level.
    """
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
    print(f"Invalid mask size: {invalid_mask.sum().item()}")
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


@torch.no_grad()
def create_children_mg(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    levels: Tensor,
    parent_indices: Tensor,
    level_indices: Dict[int, List[int]],
    n_children: int = 4,
):
    """Inplace create children for the Gaussians with the given mask (multigrid version).
    
    Creates children at the next level with independent parameters:
    - scales: parent_actual_scale - 1.4 (in log space, independent parameter)
    - quats: parent_actual_quats (same as parent, independent parameter)
    - opacities: parent_actual_opacity - 0.25 (in logit space, independent parameter)
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
        n_children: Number of children to create per parent. Default is 4.
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
    # Get actual scales (with hierarchical structure applied) instead of residual scales
    # This ensures children are initialized relative to parent's actual scale, not residual scale
    multigrid_gaussians = state.get("multigrid_gaussians", None)

    # Get actual splats for parents to get their actual scales, quats, and opacities
    # Since scales, quats, and opacities are now independent (not residual), we use parent's actual values
    parent_actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False)
    parent_actual_scales = parent_actual_splats["scales"][sel]  # [len(sel), 3] - actual scales in log space
    parent_actual_quats = parent_actual_splats["quats"][sel]  # [len(sel), 4] - actual quaternions
    parent_actual_opacities = parent_actual_splats["opacities"][sel]  # [len(sel),] - actual opacities in logit space
    
    # For means initialization: use parent's actual scales and quats
    scales_exp = torch.exp(parent_actual_scales)  # [len(sel), 3] - actual scales in exp space
    rotmats = normalized_quat_to_rotmat(parent_actual_quats)  # [len(sel), 3, 3] - actual rotation matrices
    
    # Generate small random samples for means residual initialization
    # These are small offsets relative to parent's scale
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales_exp,
        torch.randn(n_children, len(scales_exp), 3, device=device),
    )  # [n_children, N, 3]
    # Scale down samples to be small residual offsets
    # Use small scale factor to keep children close to parent initially
    small_scale_factor = 0.01  # Small scale for residual means
    samples = samples * small_scale_factor  # [n_children, N, 3]

    # Get parent levels and compute child levels (after sel validation)
    parent_levels = levels[sel]
    child_levels = parent_levels + 1

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [n_children] + [1] * (p.dim() - 1)
        if name == "means":
            # Initialize means as small residual offsets (not absolute positions)
            # Parent position will be added in get_splats() with hierarchical structure
            # Small random offset helps children explore nearby space while staying close to parent
            p_split = samples.reshape(-1, 3)  # [n_children*N, 3] - residual only
        elif name == "scales":
            # Initialize as independent parameter: parent_actual_scale - 1.4 (in log space)
            # Child scale = parent_actual_scale_log - 1.4
            parent_scales_log = parent_actual_scales  # [len(sel), 3] - already in log space
            child_scales_log = parent_scales_log - 0.5  # [len(sel), 3]
            # child_scales_log = child_scales_log.mean(dim=-1, keepdim=True).repeat(1, 3)
            # Expand to [n_children * len(sel), 3] - same for all children of each parent
            p_split = child_scales_log.repeat_interleave(n_children, dim=0)  # [n_children * len(sel), 3]
        elif name == "quats":
            # Initialize as independent parameter: same as parent (parent's actual quats)
            # Child quats = parent_actual_quats
            # Expand to [n_children * len(sel), 4] - same for all children of each parent
            p_split = parent_actual_quats.repeat_interleave(n_children, dim=0)  # [n_children * len(sel), 4]
        elif name == "opacities":
            # Initialize as independent parameter: parent_actual_opacity - 0.25 (in logit space)
            # Child opacity = parent_actual_opacity - 0.25
            parent_opacities_logit = parent_actual_opacities  # [len(sel),] - already in logit space
            child_opacities_logit = parent_opacities_logit - 0.25  # [len(sel),]
            # Expand to [n_children * len(sel),] - same for all children of each parent
            p_split = child_opacities_logit.repeat_interleave(n_children, dim=0)  # [n_children * len(sel),]
        else:
            # For other parameters (sh0, shN, etc.), initialize to zero (residual)
            p_split = torch.zeros((n_children * len(sel), *p.shape[1:]), device=device, dtype=p.dtype)
        # Keep all original gaussians (including parents) and append children
        p_new = torch.cat([p, p_split])
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros((n_children * len(sel), *v.shape[1:]), device=device)
        # Keep all original optimizer states and append zeros for children
        return torch.cat([v, v_split])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    # update the extra running state
    # For state variables (grad2d, count, radii, etc.), initialize children to zero
    # Note: scales, quats, opacities are independent parameters (not residual), but state variables are still initialized to zero
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k not in ["levels", "parent_indices", "level_indices"]:
            # Initialize children state to zero (state variables like grad2d, count, radii)
            v_split = torch.zeros((n_children * len(sel), *v.shape[1:]), device=device, dtype=v.dtype)
            state[k] = torch.cat([v, v_split])
    
    # Update hierarchical structure
    # Note: sel has already been validated and filtered above
    N_new = N_old + n_children * len(sel)
    
    # Create child parent indices (pointing to original parent indices)
    # Each parent index should be repeated n_children times
    # Example: sel=[0, 5, 10], n_children=4 -> [0,0,0,0, 5,5,5,5, 10,10,10,10]
    child_parent_indices = sel.repeat_interleave(n_children)  # [n_children*len(sel)]
    
    # Append child levels and parent_indices
    # child_levels should also be repeated with repeat_interleave to match child_parent_indices
    # Example: child_levels=[2, 3, 2], n_children=4 -> [2,2,2,2, 3,3,3,3, 2,2,2,2]
    new_levels = torch.cat([levels, child_levels.repeat_interleave(n_children)])
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


@torch.no_grad()
def reset_opa_mg(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    value: float,
):
    """Inplace reset the opacities to the given post-sigmoid value (multigrid version).
    
    For multigrid_v3.py, opacities are independent parameters (not residual).
    Therefore, all levels are reset to the same value, regardless of hierarchical level.
    
    This is safe because:
    - Opacities are independent parameters (not accumulated from parents)
    - get_splats() returns child's own opacity directly (no parent inheritance)
    - Resetting all levels to the same value does not affect hierarchical structure

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary containing hierarchical structure (levels, parent_indices, level_indices).
        value: The value to reset the opacities (post-sigmoid, e.g., 0.01).
    """
    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            # Target value in logit space
            target_val = torch.logit(torch.tensor(value)).item()
            
            # For independent parameters (multigrid_v3), reset all levels to the same value
            # This is safe because opacities are not residual and don't inherit from parents
            new_opacities = torch.clamp(p, max=target_val)
            
            return torch.nn.Parameter(new_opacities, requires_grad=p.requires_grad)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )

