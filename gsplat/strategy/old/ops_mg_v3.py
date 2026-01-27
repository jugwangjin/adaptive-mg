from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat


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
    mask_root: Optional[Tensor] = None,
    max_level: Optional[int] = None,
):
    """Inplace duplicate the Gaussian with the given mask (multigrid version, default.py style).
    
    Same as default.py duplicate but with hierarchical structure updates.
    
    If mask_root is provided, creates linear tree structure (1 child per level) 
    from level 2 to max_level for root nodes in mask_root.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that may contain level_indices, levels, parent_indices.
        mask: A boolean mask to duplicate the Gaussians.
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent).
        level_indices: Dict mapping level -> list of gaussian indices at that level.
        mask_root: Optional boolean mask indicating which gaussians in mask are roots.
                  If provided along with max_level, creates linear tree children for these roots.
        max_level: Maximum level in the hierarchy. Required if mask_root is provided.
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
    
    # If mask_root is provided, create linear tree structure for root nodes
    if mask_root is not None and max_level is not None:
        # Update structure after duplicate
        levels = state["levels"]
        parent_indices = state["parent_indices"]
        level_indices = state["level_indices"]
        N_old = len(levels)
        
        # Get duplicated gaussians that are roots (they are at the end)
        # mask_root indicates which gaussians in mask were roots
        # After duplicate, these are at indices [N_old - len(sel_root), N_old)
        # where sel_root are the root gaussians from the original mask
        sel_root = torch.where(mask_root)[0]  # Original indices of root gaussians in mask
        if len(sel_root) > 0:
            # After duplicate, root gaussians are at the end: [N_old - len(sel_root), N_old)
            duplicated_start_idx = N_old - len(sel_root)
            duplicated_indices = torch.arange(duplicated_start_idx, N_old, device=mask.device)
            
            # Create linear tree structure for duplicated roots
            _create_linear_tree_children(
                params=params,
                optimizers=optimizers,
                state=state,
                root_indices=duplicated_indices,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                max_level=max_level,
            )


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
    mask_root: Optional[Tensor] = None,
    max_level: Optional[int] = None,
):
    """Inplace split the Gaussian with the given mask (multigrid version).
    
    Same as duplicate but with scale reduced by -0.5.
    Opacity is copied as-is (residual parameter, same level so same parent).
    
    If mask_root is provided, creates linear tree structure (1 child per level) 
    from level 2 to max_level for root nodes in mask_root.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that may contain level_indices, levels, parent_indices.
        mask: A boolean mask to split the Gaussians.
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent).
        level_indices: Dict mapping level -> list of gaussian indices at that level.
        revised_opacity: Whether to use revised opacity formulation from arXiv:2404.06109.
        mask_root: Optional boolean mask indicating which gaussians in mask are roots.
                  If provided along with max_level, creates linear tree children for these roots.
        max_level: Maximum level in the hierarchy. Required if mask_root is provided.
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
    ) * 0.01  # [1, N, 3]

    del splats

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "means":
            p_split = (p[sel] + samples[0]).reshape(-1, 3)  # [2N, 3]
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
    
    # If mask_root is provided, create linear tree structure for root nodes
    if mask_root is not None and max_level is not None:
        # Update structure after split
        levels = state["levels"]
        parent_indices = state["parent_indices"]
        level_indices = state["level_indices"]
        N_old = len(levels)
        
        # Get split gaussians that are roots (they are at the end)
        # mask_root indicates which gaussians in mask were roots
        # After split, these are at indices [N_old - len(sel_root), N_old)
        # where sel_root are the root gaussians from the original mask
        sel_root = torch.where(mask_root)[0]  # Original indices of root gaussians in mask
        if len(sel_root) > 0:
            # After split, root gaussians are at the end: [N_old - len(sel_root), N_old)
            split_start_idx = N_old - len(sel_root)
            split_indices = torch.arange(split_start_idx, N_old, device=mask.device)
            
            # Create linear tree structure for split roots
            _create_linear_tree_children(
                params=params,
                optimizers=optimizers,
                state=state,
                root_indices=split_indices,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                max_level=max_level,
            )


@torch.no_grad()
def _create_linear_tree_children(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    root_indices: Tensor,
    levels: Tensor,
    parent_indices: Tensor,
    level_indices: Dict[int, List[int]],
    max_level: int,
):
    """Create linear tree structure (1 child per level) from level 2 to max_level for given root indices.
    
    For each root, creates a chain: root -> L2 -> L3 -> ... -> max_level
    All children are initialized with zero residual parameters.
    
    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        state: State dictionary that may contain level_indices, levels, parent_indices.
        root_indices: Tensor [M,] with indices of root gaussians to create children for.
        levels: Tensor [N,] with level for each gaussian.
        parent_indices: Tensor [N,] with parent index for each gaussian (-1 means no parent).
        level_indices: Dict mapping level -> list of gaussian indices at that level.
        max_level: Maximum level in the hierarchy.
    """
    if len(root_indices) == 0:
        return
    
    device = root_indices.device
    N_old = len(levels)
    
    # For each root, create linear tree from level 2 to max_level
    # Each level gets 1 child, creating a chain: root -> L2 -> L3 -> ... -> max_level
    total_children_to_create = len(root_indices) * (max_level - 1)  # (max_level - 1) children per root
    
    if total_children_to_create == 0:
        return
    
    # Prepare parent indices for children (linear tree structure)
    # For each root, create children at levels 2, 3, ..., max_level
    # Structure: root[i] -> child_L2[i] -> child_L3[i] -> ... -> child_max_level[i]
    children_parent_indices = []
    children_levels = []
    
    for i, root_idx in enumerate(root_indices):
        # First child (level 2) points to root
        children_parent_indices.append(root_idx)
        children_levels.append(2)
        
        # Subsequent children form a chain: each child points to previous child
        for child_level in range(3, max_level + 1):
            # Parent is the previous child in the chain
            # Index calculation: N_old (current size) + i * (max_level - 1) + (child_level - 3)
            # This will be the index of the previous child after all children are added
            prev_child_idx = N_old + i * (max_level - 1) + (child_level - 3)
            children_parent_indices.append(prev_child_idx)
            children_levels.append(child_level)
    
    children_parent_indices = torch.tensor(children_parent_indices, dtype=torch.long, device=device)
    children_levels = torch.tensor(children_levels, dtype=torch.long, device=device)
    
    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "means":
            # Zero residual means
            p_children = torch.zeros((total_children_to_create, 3), device=device, dtype=p.dtype)
        elif name == "scales":
            # Zero residual scales
            p_children = torch.zeros((total_children_to_create, 3), device=device, dtype=p.dtype)
        elif name == "quats":
            # Identity quaternion (zero residual rotation)
            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=p.dtype)
            p_children = identity_quat.unsqueeze(0).repeat(total_children_to_create, 1)
        elif name == "opacities":
            # Zero residual opacities
            p_children = torch.zeros((total_children_to_create,), device=device, dtype=p.dtype)
        else:
            # For other parameters (sh0, shN, etc.), initialize to zero (residual)
            p_children = torch.zeros((total_children_to_create, *p.shape[1:]), device=device, dtype=p.dtype)
        
        p_new = torch.cat([p, p_children])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
    
    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_children = torch.zeros((total_children_to_create, *v.shape[1:]), device=device)
        return torch.cat([v, v_children])
    
    # Update parameters and optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    # Update state (exclude hierarchical structure, handled separately)
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k not in ["levels", "parent_indices", "level_indices"]:
            v_children = torch.zeros((total_children_to_create, *v.shape[1:]), device=device, dtype=v.dtype)
            state[k] = torch.cat([v, v_children])
    
    # Update hierarchical structure
    new_levels = torch.cat([levels, children_levels])
    new_parent_indices = torch.cat([parent_indices, children_parent_indices])
    
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
        multigrid_gaussians.invalidate_splats_cache()


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

