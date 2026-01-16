from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat


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

    def param_fn(name: str, p: Tensor) -> Tensor:
        # if name == "opacities":
        #     # Opacity reduced by -0.5 (modify original and copy)
        #     p[sel] -= 0.25
        #     p_split = p[sel]
        #     p_new = torch.cat([p, p_split])
        # else:
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
    
    # Update hierarchical structure
    _update_hierarchical_structure(state, new_levels, new_parent_indices)
    
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

    rest = torch.where(~mask)[0]
    N_old = len(levels)

    # Find children of selected gaussians that are parents (vectorized)
    # For each sel[i], find all gaussians where parent_indices == sel[i]
    # Using torch.isin for memory efficiency: O(N_old) memory instead of O(N_old * M)
    valid_parent_mask = (parent_indices >= 0) & (parent_indices < N_old)
    
    if sel.numel() > 0:
        # Check if parent_indices[i] is in sel
        in_sel = torch.isin(parent_indices, sel)  # [N_old] bool
        children_mask = in_sel & valid_parent_mask
    else:
        children_mask = torch.zeros(N_old, dtype=torch.bool, device=device)
    
    # Get indices of children
    children_indices = torch.where(children_mask)[0]  # [M,] - indices of children

    splats = state["multigrid_gaussians"].get_splats(level=None, detach_parents=False, current_splats=None)
    scales = torch.exp(splats["scales"][sel])

    quats = splats["quats"][sel]
    
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni", 
        rotmats, 
        scales, 
        torch.randn(2, len(scales), 3, device=device),
    ) * 0.01 # [1, N, 3]

    del splats

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "means":
            p_split = (p[sel]).reshape(-1, 3)  # [2N, 3]
            p[sel] += samples[1]
        elif name == "scales":
            # Scale adjustment for split:
            # - Split 대상의 scale을 -1.6 (normal split behavior)
            # - 만약 split 대상이 parent라면, 그 children들의 residual scale을 +1.6 해서
            #   children의 actual scale이 유지되도록 함
            #   (children's actual_scale = parent_scale + child_residual)
            #   parent_scale -= 1.6이면, child_residual += 1.6 해야 actual_scale 유지
            
            # Split 대상의 scale 감소
            p[sel] -= 1.6
            
            # Parent인 경우, children의 residual scale 증가 (actual scale 유지)
            if len(children_indices) > 0:
                p[children_indices] += 1.6
            
            p_split = p[sel]
        # elif name == "opacities":
        #     # Opacity reduced by -0.5 (modify original and copy)
        #     p[sel] -= 0.25
        #     p_split = p[sel]
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
        
        # Initialize all children with zero residuals (vectorized)
        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "means":
                p_children = torch.zeros((total_new_children, 3), device=device, dtype=p.dtype)
            elif name == "scales":
                # Set child residual scale to 0.6931 so that actual scale equals parent scale
                # In get_splats: child_scale = parent_scale + child_residual - 0.6931
                # So child_residual = 0.6931 gives child_scale = parent_scale
                p_children = torch.full((total_new_children, 3), 0.6931, device=device, dtype=p.dtype)
                # p_children = torch.zeros((total_new_children, 3), device=device, dtype=p.dtype)
            elif name == "quats":
                # Zero quaternions: [0, 0, 0, 0] (residual)
                p_children = torch.zeros((total_new_children, 4), device=device, dtype=p.dtype)
            elif name == "opacities":
                p_children = torch.zeros((total_new_children,), device=device, dtype=p.dtype)
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
            # Child scale = parent_scale + 0 - 0.6931 = parent_scale - 0.6931
            # This makes child scale approximately 50% of parent (exp(-0.6931) ≈ 0.5)
            p_split = (p[sel] - torch.sqrt(n_children_per_split) + 0.6931).repeat_interleave(n_children_per_split, 1)
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
        return
    
    device = parent_indices_to_clone.device
    N_old = len(levels)
    
    # Remove duplicates and validate
    parent_indices_to_clone = torch.unique(parent_indices_to_clone)
    valid_mask = (parent_indices_to_clone >= 0) & (parent_indices_to_clone < N_old)
    parent_indices_to_clone = parent_indices_to_clone[valid_mask]
    
    if len(parent_indices_to_clone) == 0:
        return
    
    # Vectorized: collect children for all parents at once
    # Build a mapping: parent_idx -> [parent_idx, child1, child2, ...]
    
    if signal_indices is not None:
        # Only clone children that are in signal_indices
        # Find all children of parents that are in signal_indices (vectorized)
        valid_parent_mask = (parent_indices >= 0) & (parent_indices < N_old)
        children_of_any_parent = torch.isin(parent_indices, parent_indices_to_clone) & valid_parent_mask & signal_indices
        all_children_indices = torch.where(children_of_any_parent)[0]  # All signal children of any parent in parent_indices_to_clone
        
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
            parent_to_unique_idx = {p.item(): i for i, p in enumerate(unique_parents)}
            
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
                # Include parent itself and its signal children
                blocks_to_clone[parent_idx_int] = [parent_idx_int] + children.tolist()
        else:
            # No signal children found, only clone parents
            blocks_to_clone = {parent_idx.item(): [parent_idx.item()] for parent_idx in parent_indices_to_clone}
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
        return
    
    # Sort by level (top-down: parents before children)
    all_levels = levels[all_indices_to_clone]
    sort_indices = torch.argsort(all_levels)
    all_indices_to_clone = all_indices_to_clone[sort_indices]
    
    # Clone all gaussians in the blocks
    sel = all_indices_to_clone
    
    # Get levels and parent indices of selected gaussians
    sel_levels = levels[sel]
    sel_parents = parent_indices[sel]
    
    # Determine which cloned gaussians need split-like scale adjustment
    # is_small_mask is for original indices, need to map to sel indices
    sel_is_small = None
    if is_small_mask is not None:
        sel_is_small = is_small_mask[sel]  # [len(sel),] - True for duplicate-like, False for split-like
    
    # Find children of cloned gaussians (for scale adjustment)
    valid_parent_mask = (parent_indices >= 0) & (parent_indices < N_old)
    if sel.numel() > 0:
        in_sel = torch.isin(parent_indices, sel)
        cloned_children_mask = in_sel & valid_parent_mask
        cloned_children_indices = torch.where(cloned_children_mask)[0]
    else:
        cloned_children_indices = torch.empty(0, dtype=torch.long, device=device)
    
    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "scales" and is_small_mask is not None and sel_is_small is not None:
            # Apply scale adjustment based on is_small_mask (similar to split operation)
            # Clone된 gaussians의 scale을 조절
            p_cloned = p[sel].clone()  # Clone할 scale
            
            # Split-like adjustment (is_small=False): scale 감소
            split_mask = ~sel_is_small  # [len(sel),] - True for split-like cloned gaussians
            if split_mask.any():
                # Split-like: clone된 gaussians의 scale을 -1.6 감소
                p_cloned[split_mask] -= 1.6
                
                # Clone된 children의 residual scale 증가 (actual scale 유지)
                # Find which cloned gaussians are children of split-like cloned parents
                # Vectorized: avoid Python loop
                if len(cloned_children_indices) > 0:
                    # Create mapping tensor instead of dict for vectorized lookup
                    old_to_sel_tensor = torch.full((N_old,), -1, dtype=torch.long, device=device)
                    old_to_sel_tensor[sel] = torch.arange(len(sel), device=device)
                    
                    # Get parent indices for cloned children
                    cloned_children_parents = parent_indices[cloned_children_indices]  # [K,]
                    
                    # Check which cloned children have cloned parents
                    cloned_parent_mask = (cloned_children_parents >= 0) & (cloned_children_parents < N_old)
                    cloned_parent_sel_indices = old_to_sel_tensor[cloned_children_parents[cloned_parent_mask]]  # [K_valid,]
                    valid_cloned_parent_mask = cloned_parent_sel_indices >= 0  # [K_valid,] - parent is also cloned
                    
                    if valid_cloned_parent_mask.any():
                        # Get children sel indices
                        cloned_children_sel_indices = old_to_sel_tensor[cloned_children_indices[cloned_parent_mask][valid_cloned_parent_mask]]  # [M,]
                        valid_parent_sel_indices = cloned_parent_sel_indices[valid_cloned_parent_mask]  # [M,]
                        
                        # Check if parents are split-like
                        parent_split_mask = split_mask[valid_parent_sel_indices]  # [M,]
                        if parent_split_mask.any():
                            # Increase residual scale for children of split-like cloned parents
                            children_to_adjust = cloned_children_sel_indices[parent_split_mask]
                            p_cloned[children_to_adjust] += 1.6
            
            # Duplicate-like (is_small=True): scale 조절 없음 (그대로 복사)
            p_new = torch.cat([p, p_cloned])
        else:
            # Other parameters: clone as-is
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
    valid_parent_mask = sel_parents >= 0
    if valid_parent_mask.any():
        old_parents = parent_indices[sel[valid_parent_mask]]  # [K,]
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
    
    # Update hierarchical structure
    _update_hierarchical_structure(state, new_levels, new_parent_indices)
    
    # Note: _add_zero_parameter_children is now called once after all grow operations
    # in _grow_gs to avoid redundant processing
    
    # Return cloned indices for further processing (duplicate/split)
    return new_indices  # [len(sel),] - indices of cloned gaussians in new structure


@torch.no_grad()
def reset_opa_mg(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    value: float,
):
    """Inplace reset the opacities to the given post-sigmoid value (multigrid version).
    
    For multigrid_v6.py, opacities are residual parameters.
    - Level 1: Reset to target value (absolute value)
    - Level 2+: Reset to 0 (residual, so they inherit parent's opacity)

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary containing hierarchical structure (levels, parent_indices, level_indices).
        value: The value to reset Level 1 opacities (post-sigmoid, e.g., 0.01).
    """
    levels = state.get("levels", None)
    if levels is None:
        raise ValueError("levels not found in state. Cannot reset opacities by level.")
    
    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            # Target value in logit space for Level 1
            target_val = torch.logit(torch.tensor(value, device=p.device, dtype=p.dtype))
            
            # Create new opacities tensor
            new_opacities = p.clone()
            
            # Level 1: Reset to target value (absolute value)
            level1_mask = (levels == 1)
            if level1_mask.any():
                new_opacities[level1_mask] = torch.clamp(p[level1_mask], max=target_val)
            
            # Level 2+: Reset to 0 (residual, so they inherit parent's opacity)
            level2plus_mask = (levels > 1)
            if level2plus_mask.any():
                new_opacities[level2plus_mask] = 0.0
            
            return torch.nn.Parameter(new_opacities, requires_grad=p.requires_grad)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )

