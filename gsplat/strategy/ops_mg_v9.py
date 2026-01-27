from typing import Any, Callable, Dict, List, Optional, Union

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat

SPLIT_SCALE_LOG = math.log(1.6)


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
    
    # Validate and fix parent indices for new gaussians
    # sel_parents are parent indices in the original array (0 ~ N_old-1)
    # Since we keep originals, these indices remain valid in the new array
    # But we need to check for self-reference (parent == self in sel)
    new_sel_parents = sel_parents.clone()
    
    # Check for invalid parent indices (out of bounds)
    valid_parent_mask = (new_sel_parents == -1) | ((new_sel_parents >= 0) & (new_sel_parents < N_old))
    if not valid_parent_mask.all():
        new_sel_parents[~valid_parent_mask] = -1
    
    # Check for self-reference: if a gaussian's parent is itself (in sel), set to -1
    # Vectorized: check if sel_parents[i] == sel[i] for any i
    sel_tensor = torch.tensor(sel, device=device, dtype=torch.long)  # [len(sel),]
    self_ref_mask = (new_sel_parents >= 0) & (new_sel_parents < N_old) & (new_sel_parents == sel_tensor)
    if self_ref_mask.any():
        new_sel_parents[self_ref_mask] = -1  # Self-reference: set to root

    # Append duplicated levels and parent_indices
    new_levels = torch.cat([levels, sel_levels])
    new_parent_indices = torch.cat([parent_indices, new_sel_parents])
    
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
    
    CHANGED: Uses individual parameters (not residual), similar to ops.py split.
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
    rest = torch.where(~mask)[0]
    
    # Get levels and parent indices of selected gaussians
    sel_levels = levels[sel]
    sel_parents = parent_indices[sel]

    # CHANGED: Use individual parameters directly (like ops.py split)
    # Compute split parameters from individual parameters (not residual)
    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )  # [2, N, 3]

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [1] + [1] * (p.dim() - 1)
        if name == "means":
            # CHANGED: Use individual means directly (not residual)
            # samples is [2, N, 3], so samples[0] and samples[1] are [N, 3]
            p_split = (p[sel] + samples[1]).reshape(-1, 3)  # [N, 3]
            p[sel] = p[sel] + samples[0]  # Update original with samples[0]
        elif name == "scales":
            # CHANGED: Use individual scales directly (not residual)
            p_split = torch.log(scales / 1.6).repeat(1, 1)  # [2N, 3]
            p[sel] = torch.log(scales / 1.6)
        elif name == "opacities" and revised_opacity:
            # CHANGED: Use individual opacities directly (not residual)
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
            p[sel] = torch.logit(new_opacities)
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p, p_split])
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        # CHANGED: Split creates 1 new gaussian per selected (original is kept), so need len(sel) zeros
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    # update the extra running state (exclude hierarchical structure, handled separately)
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k not in ["levels", "parent_indices", "level_indices"]:
            # CHANGED: Split creates 1 new gaussian per selected (original is kept), so just add v[sel]
            state[k] = torch.cat((v, v[sel]))
    
    # Update hierarchical structure
    N_old = len(levels)
    N_new = N_old + len(sel)  # CHANGED: Split creates 1 new gaussian per selected (original is kept)
    
    # Validate and fix parent indices for new gaussians
    # sel_parents are parent indices in the original array (0 ~ N_old-1)
    # Since we keep originals, these indices remain valid in the new array
    # But we need to check for self-reference (parent == self in sel)
    new_sel_parents = sel_parents.clone()
    
    # Check for invalid parent indices (out of bounds)
    valid_parent_mask = (new_sel_parents == -1) | ((new_sel_parents >= 0) & (new_sel_parents < N_old))
    if not valid_parent_mask.all():
        new_sel_parents[~valid_parent_mask] = -1
    
    # Check for self-reference: if a gaussian's parent is itself (in sel), set to -1
    # Vectorized: check if sel_parents[i] == sel[i] for any i
    sel_tensor = torch.tensor(sel, device=device, dtype=torch.long)  # [len(sel),]
    self_ref_mask = (new_sel_parents >= 0) & (new_sel_parents < N_old) & (new_sel_parents == sel_tensor)
    if self_ref_mask.any():
        new_sel_parents[self_ref_mask] = -1  # Self-reference: set to root
    
    # CHANGED: Append split levels and parent_indices (original is kept, so just add new ones)
    new_levels = torch.cat([levels, sel_levels])
    new_parent_indices = torch.cat([parent_indices, new_sel_parents])
    
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
        # Validate: parents_to_add should be within bounds
        valid_parents_mask = (parents_to_add >= 0) & (parents_to_add < len(levels))
        if not valid_parents_mask.all():
            # Filter out invalid parents
            parents_to_add = parents_to_add[valid_parents_mask]
            if len(parents_to_add) == 0:
                continue
            n_parents = len(parents_to_add)
            n_children_per_parent = min_children_per_parent
            total_new_children = n_parents * n_children_per_parent
        
        new_children_parent_indices = parents_to_add.repeat_interleave(n_children_per_parent)  # [total_new_children,]
        new_children_levels = torch.full((total_new_children,), child_level, device=device, dtype=torch.long)

        # CHANGED: Initialize children using individual parameters (like ops.py split)
        # Get parent actual parameters to initialize children
        # CHANGED: Use individual parameters directly (no need for get_splats)
        multigrid_gaussians = state.get("multigrid_gaussians", None)

        # Individual parameters are stored directly in self.splats
        splats = multigrid_gaussians.splats
        # Extract parent parameters
        parent_actual_params = {
            "means": splats["means"][parents_to_add],
            "scales": splats["scales"][parents_to_add],
            "quats": splats["quats"][parents_to_add],
            "opacities": splats["opacities"][parents_to_add],
        }
        # Handle colors (sh0, shN) or features
        if "sh0" in splats:
            parent_actual_params["sh0"] = splats["sh0"][parents_to_add]
            parent_actual_params["shN"] = splats["shN"][parents_to_add]
        elif "features" in splats:
            parent_actual_params["features"] = splats["features"][parents_to_add]
            parent_actual_params["colors"] = splats["colors"][parents_to_add]
        
        # CHANGED: Initialize children with individual parameters based on parent (like ops.py split)
        def param_fn(name: str, p: Tensor) -> Tensor:
            parent_param = parent_actual_params[name]  # [n_parents, ...]
            # Repeat parent parameters for each child
            parent_param_repeated = parent_param.repeat_interleave(n_children_per_parent, dim=0)  # [total_new_children, ...]
            
            if name == "means":
                # CHANGED: Initialize means with small random offsets (like ops.py split)
                # Use parent's scale and rotation to generate offsets
                parent_scales = torch.exp(parent_actual_params["scales"])  # [n_parents, 3]
                parent_quats = F.normalize(parent_actual_params["quats"], dim=-1)  # [n_parents, 4]
                parent_rotmats = normalized_quat_to_rotmat(parent_quats)  # [n_parents, 3, 3]
                # Generate small random samples (same pattern as split function, no intermediate tensors)
                samples = torch.einsum(
                    "nij,nj,bnj->bni",
                    parent_rotmats,  # [n_parents, 3, 3]
                    parent_scales,  # [n_parents, 3]
                    torch.randn(n_children_per_parent, len(parents_to_add), 3, device=device),  # [n_children_per_parent, n_parents, 3]
                )  # [n_children_per_parent, n_parents, 3]
                samples = samples.reshape(-1, 3)  # [total_new_children, 3]
                p_children = parent_param_repeated + samples * 0.01  # Small offset
            else:
                # For colors (sh0, shN, features, colors): copy from parent
                p_children = parent_param_repeated

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
    
    # CHANGED: Use individual parameters (like ops.py split)
    # Get parent individual parameters to initialize children
    multigrid_gaussians = state.get("multigrid_gaussians", None)

    # CHANGED: Get parent individual parameters directly (not actual/residual)
    # Since all parameters are now individual, we can use params directly
    parent_scales = params["scales"][sel]  # [len(sel), 3] - individual scales in log space
    parent_quats = F.normalize(params["quats"][sel], dim=-1)  # [len(sel), 4] - individual quaternions
    parent_means = params["means"][sel]  # [len(sel), 3] - individual means
    
    # For means initialization: use parent's scales and quats to generate offsets
    scales_exp = torch.exp(parent_scales)  # [len(sel), 3] - scales in exp space
    rotmats = normalized_quat_to_rotmat(parent_quats)  # [len(sel), 3, 3] - rotation matrices
    
    # CHANGED: Generate small random samples for means (like ops.py split)
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales_exp,
        torch.randn(n_children_per_split, len(scales_exp), 3, device=device),
    )  # [n_children_per_split, N, 3]

    # Get parent levels and compute child levels (after sel validation)
    parent_levels = levels[sel]
    child_levels = parent_levels + 1

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [n_children_per_split] + [1] * (p.dim() - 1)
        if name == "means":
            # CHANGED: Initialize means with parent + small offset (like ops.py split)
            p_split = (parent_means.unsqueeze(0) + samples).reshape(-1, 3)  # [n_children_per_split*N, 3]
        elif name == "scales":
            # CHANGED: Initialize scales as parent / 1.6 (like ops.py split)
            p_split = torch.log(scales_exp / 1.6).repeat(n_children_per_split, 1)  # [n_children_per_split*N, 3]
        elif name == "quats":
            # CHANGED: Initialize quats same as parent (like ops.py split)
            p_split = parent_quats.repeat(n_children_per_split, 1)  # [n_children_per_split*N, 4]
        elif name == "opacities":
            # CHANGED: Initialize opacities using revised formula (like ops.py split with revised_opacity)
            parent_alpha = torch.sigmoid(p[sel])
            # Each child should satisfy: 1 - (1 - a')^K = a (K = n_children_per_split)
            target_alpha = 1.0 - torch.pow(
                (1.0 - parent_alpha).clamp_min(1e-6),
                1.0 / float(n_children_per_split),
            )
            target_alpha = target_alpha.clamp(1e-6, 1.0 - 1e-6)
            p_split = torch.logit(target_alpha).repeat(n_children_per_split)  # [n_children_per_split*N]
        else:
            # For other parameters (sh0, shN, etc.): copy from parent
            p_split = p[sel].repeat(repeats)
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
    
    # CHANGED: No residual adjustment needed - use individual parameters directly
    # Clone parameters as-is (like ops.py duplicate)
    def param_fn(name: str, p: Tensor) -> Tensor:
        # CHANGED: Copy individual parameters directly (no residual computation)
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
    # - If parent is in sel (self-reference), set to -1
    # Vectorized: avoid Python loop
    new_parent_indices_cloned = torch.full_like(sel_parents, -1)
    valid_parent_mask = (sel_parents >= 0) & (sel_parents < N_old)
    if valid_parent_mask.any():
        old_parents = sel_parents[valid_parent_mask]  # [K,]
        # Check which parents were cloned
        cloned_parent_new_indices = old_to_new[old_parents]  # [K,]
        parent_was_cloned = cloned_parent_new_indices >= 0  # [K,]
        
        # Check if parent is in sel (parent also being cloned)
        sel_mask = torch.zeros(N_old, dtype=torch.bool, device=device)
        sel_mask[sel] = True
        parent_in_sel = sel_mask[old_parents]  # [K,]
        
        # Check direct self-reference: sel_parents[i] == sel[i] for valid entries
        valid_indices_in_sel = torch.where(valid_parent_mask)[0]  # [K,] - indices in sel_parents that are valid
        sel_tensor = torch.tensor(sel, device=device, dtype=torch.long)  # [len(sel),]
        direct_self_ref = (sel_parents[valid_indices_in_sel] == sel_tensor[valid_indices_in_sel])  # [K,]
        
        # Use new parent index if cloned, otherwise keep original (unless parent is in sel or self-reference)
        new_parent_indices_cloned[valid_parent_mask] = torch.where(
            parent_in_sel | direct_self_ref,
            torch.tensor(-1, device=device),  # Self-reference or parent in sel: set to -1
            torch.where(
                parent_was_cloned,
                cloned_parent_new_indices,
                old_parents
            )
        )
    
    # Append cloned levels and parent_indices
    new_levels = torch.cat([levels, sel_levels])
    new_parent_indices = torch.cat([parent_indices, new_parent_indices_cloned])
    
    # CHANGED: No residual adjustment needed - individual parameters are already correct
    # Update hierarchical structure
    _update_hierarchical_structure(state, new_levels, new_parent_indices)
    
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
    
    # CHANGED: Use individual parameters directly (not residual)
    # Simply reset opacities to target value for all max_level gaussians
    max_level_indices = torch.where(max_level_mask)[0]
    
    # Target opacity in logit space
    target_opacity = torch.tensor(value, device=params["opacities"].device, dtype=params["opacities"].dtype)
    target_logit = torch.logit(target_opacity)
    
    # CHANGED: All levels use individual parameters, so reset directly to target
    new_opacities = target_logit
    
    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            # CHANGED: Reset individual opacities directly to target
            new_opacities_tensor = p.clone()
            # Only update max_level gaussians
            new_opacities_tensor[max_level_indices] = new_opacities
            return torch.nn.Parameter(new_opacities_tensor, requires_grad=p.requires_grad)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )

