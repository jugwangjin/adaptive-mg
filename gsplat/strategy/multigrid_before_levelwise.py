from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from .base import Strategy
from .ops_mg import create_children_mg, duplicate, remove_mg, reset_opa_mg, split


@dataclass
class MultigridStrategy(Strategy):
    """A multigrid strategy for hierarchical Gaussian Splatting.

    This strategy extends the default 3DGS strategy with hierarchical level awareness:
    - Pruning only affects leaf nodes (nodes without children)
    - Gradient thresholds are scaled by level
    - Duplication creates gaussians at the same level with the same parent
    - Splitting creates children at the next level

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): Base gradient threshold for creating children. Default is 0.0005.
          Higher levels use lower thresholds (scaled by split_grad_scale).
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".
        split_grad_scale (float): Scale factor for gradient threshold when creating children.
          Higher values make it harder to create children. Default is 1.175.
        beta (float): Exponential scaling factor for level-based threshold (Octree-GS style).
          threshold = grow_grad2d * 2^(beta * level). 
          Positive beta makes higher levels harder to create children. Default is 0.2.
        duplicate_grad_scale (float): Scale factor for gradient threshold when duplicating
          at the same level. Should be smaller than split_grad_scale. Default is 1.5.
        n_children_per_split (int): Number of children to create per split. Default is 4.
        max_children_per_parent (int): Maximum number of children a parent can have.
          A parent can create children only if its current num_child <= (max_children_per_parent - n_children_per_split).
          This ensures that creating n_children_per_split children won't exceed max_children_per_parent.
          Default is 8 (allowing 4+4 children).
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_color: float = 0.00005
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    use_opacity_reset: bool = True
    revised_opacity: bool = True
    refine_every: int = 100
    pause_refine_after_reset: int = 100
    absgrad: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    split_grad_scale: float = 1.175  # Scale for creating children (higher = harder) - kept for backward compatibility
    beta: float = 0.1  # Exponential scaling factor for level-based threshold (Octree-GS style)
    duplicate_grad_scale: float = 1.5  # Scale for same-level duplication (lower = easier)
    n_children_per_split: int = 2  # Number of children to create per split
    max_children_per_parent: int = 5  # Maximum number of children a parent can have

    def initialize_state(self, scene_scale: float = 1.0, levels: Tensor = None, parent_indices: Tensor = None, level_indices: Dict[int, List[int]] = None, max_level: Optional[int] = None, multigrid_gaussians: Optional[Any] = None) -> Dict[str, Any]:
        """Initialize and return thㅁe running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        
        Args:
            scene_scale: Scale of the scene. Default is 1.0.
            levels: Tensor [N,] with level for each gaussian. Required for multigrid strategy.
            parent_indices: Tensor [N,] with parent index for each gaussian. Required for multigrid strategy.
            level_indices: Dict mapping level -> list of gaussian indices. Required for multigrid strategy.
        
        Raises:
            ValueError: If levels, parent_indices, or level_indices are None.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        # - levels: hierarchical level for each gaussian
        # - parent_indices: parent index for each gaussian
        # - level_indices: mapping from level to list of gaussian indices
        
        # Require hierarchical structure for multigrid strategy
        if levels is None:
            raise ValueError("levels is required for MultigridStrategy.initialize_state()")
        if parent_indices is None:
            raise ValueError("parent_indices is required for MultigridStrategy.initialize_state()")
        if level_indices is None:
            raise ValueError("level_indices is required for MultigridStrategy.initialize_state()")
        
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        
        # Store hierarchical structure
        state["levels"] = levels
        state["parent_indices"] = parent_indices
        state["level_indices"] = level_indices
        state["max_level"] = max_level  # Store max_level for limiting hierarchy growth
        state["multigrid_gaussians"] = multigrid_gaussians  # Store MultigridGaussians object for get_splats()
        
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
        visible_mask: Optional[Tensor] = None,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        # Get visible_mask from info if not provided directly
        if visible_mask is None:
            visible_mask = info.get("visible_mask", None)
        
        self._update_state(params, state, info, packed=packed, visible_mask=visible_mask)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # Get actual splats once (shared by _grow_gs and _prune_gs)
            # This avoids calling get_splats() twice, which is expensive
            multigrid_gaussians = state.get("multigrid_gaussians", None)
            actual_splats = None
            if multigrid_gaussians is not None:
                actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False)
            
            # grow GSs
            n_dupli, n_split, n_create_children = self._grow_gs(params, optimizers, state, step, actual_splats=actual_splats)
            if self.verbose:
                # Count children for statistics
                parent_indices = state["parent_indices"]
                N = len(parent_indices)
                num_child = torch.zeros(N, dtype=torch.long, device=parent_indices.device)
                valid_parent_mask = (parent_indices != -1)
                if valid_parent_mask.any():
                    valid_parent_indices = parent_indices[valid_parent_mask]
                    valid_bounds = (valid_parent_indices >= 0) & (valid_parent_indices < N)
                    if valid_bounds.any():
                        valid_parent_indices_in_bounds = valid_parent_indices[valid_bounds]
                        num_child.scatter_add_(0, valid_parent_indices_in_bounds, torch.ones_like(valid_parent_indices_in_bounds))
                
                n_parents = (num_child > 0).sum().item()
                max_children = num_child.max().item() if num_child.max().item() > 0 else 0
                avg_children = num_child[num_child > 0].float().mean().item() if n_parents > 0 else 0.0
                
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split, {n_create_children} GSs created children. "
                    f"Now having {len(params['means'])} GSs. "
                    f"Parents: {n_parents}, Max children: {max_children}, Avg children: {avg_children:.2f}"
                )

            # prune GSs
            n_prune, n_is_too_big = self._prune_gs(params, optimizers, state, step, actual_splats=actual_splats)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned, {n_is_too_big} GSs too big. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if (step % self.reset_every == 0 and step > 0) and self.use_opacity_reset and False:  # Disabled for V-cycle training
            reset_opa_mg(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
        visible_mask: Optional[Tensor] = None,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."
        
        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)
        
        # Verify hierarchical structure is present and consistent
        if "levels" not in state or state["levels"] is None:
            raise ValueError("levels must be provided in state for MultigridStrategy")
        if "parent_indices" not in state or state["parent_indices"] is None:
            raise ValueError("parent_indices must be provided in state for MultigridStrategy")
        if "level_indices" not in state or state["level_indices"] is None:
            raise ValueError("level_indices must be provided in state for MultigridStrategy")
        
        # Verify length consistency (operations should update this, but we check for safety)
        if len(state["levels"]) != n_gaussian:
            raise ValueError(
                f"Length mismatch: state['levels'] has {len(state['levels'])} elements, "
                f"but params have {n_gaussian} gaussians. This should not happen if operations "
                f"are called correctly."
            )
        if len(state["parent_indices"]) != n_gaussian:
            raise ValueError(
                f"Length mismatch: state['parent_indices'] has {len(state['parent_indices'])} elements, "
                f"but params have {n_gaussian} gaussians. This should not happen if operations "
                f"are called correctly."
            )
        
        # Verify level_indices is consistent with current n_gaussian
        total_indices = sum(len(indices) for indices in state["level_indices"].values())
        if total_indices != n_gaussian:
            # Rebuild level_indices from levels (safety check)
            state["level_indices"] = {}
            for level_val in state["levels"].unique():
                level_val_int = level_val.item()
                mask_level = (state["levels"] == level_val_int)
                state["level_indices"][level_val_int] = torch.where(mask_level)[0].tolist()

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] - indices of visible gaussians in the rendered subset
            radii = info["radii"].max(dim=-1).values  # [nnz]
            
            # If visible_mask is provided, filter gs_ids to only include visible gaussians
            # Note: gs_ids are already indices into the visible subset, so we need to map them
            # back to full gaussian indices if visible_mask is provided
            if visible_mask is not None:
                # Get the visible indices that were actually rendered
                visible_indices = torch.where(visible_mask)[0]  # [N_visible]
                # Map gs_ids (indices into visible subset) to full gaussian indices
                if len(visible_indices) > 0:
                    # gs_ids are indices into the visible subset, map to full indices
                    full_gs_ids = visible_indices[gs_ids]  # [nnz]
                else:
                    full_gs_ids = torch.empty((0,), dtype=torch.long, device=gs_ids.device)
            else:
                full_gs_ids = gs_ids
        else:
            # grads is [C, N, 2] where N is the number of visible gaussians
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz] - indices into visible subset
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]
            
            # If visible_mask is provided, map gs_ids to full gaussian indices
            if visible_mask is not None:
                # Get the visible indices that were actually rendered
                visible_indices = torch.where(visible_mask)[0]  # [N_visible]
                # Map gs_ids (indices into visible subset) to full gaussian indices
                if len(visible_indices) > 0:
                    # gs_ids are indices into the visible subset, map to full indices
                    full_gs_ids = visible_indices[gs_ids]  # [nnz]
                else:
                    full_gs_ids = torch.empty((0,), dtype=torch.long, device=gs_ids.device)
            else:
                full_gs_ids = gs_ids
        
        # Only accumulate gradients for visible gaussians
        if len(full_gs_ids) > 0:
            state["grad2d"].index_add_(0, full_gs_ids, grads.norm(dim=-1))
            state["count"].index_add_(
                0, full_gs_ids, torch.ones_like(full_gs_ids, dtype=torch.float32)
            )
            
            if self.refine_scale2d_stop_iter > 0:
                # Should be ideally using scatter max
                state["radii"][full_gs_ids] = torch.maximum(
                    state["radii"][full_gs_ids],
                    # normalize radii to [0, 1] screen space
                    radii / float(max(info["width"], info["height"])),
                )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        actual_splats: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[int, int]:
        """Grow gaussians based on grad2d (default.py style with hierarchical constraints).
        
        - Each gaussian is evaluated individually by grad2d threshold (default.py style)
        - can_duplicate flag: parent != -1 and parent.num_child < max_children_per_parent
        - if can_duplicate: duplicate/split (same level, same parent)
        - if not can_duplicate: create children (next level)
        - Root nodes (parent == -1) cannot duplicate, always create children
        - At max_level: always duplicate (cannot create children)
        """
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)  # Average grad2d gradient
        device = grads.device
        
        # Get levels for all gaussians
        levels = state["levels"]  # [N,]
        parent_indices = state["parent_indices"]  # [N,]
        level_indices = state["level_indices"]  # Dict[int, List[int]]
        max_level = state.get("max_level", None)
        
        # Count number of direct children for each gaussian - OPTIMIZED
        # num_child[i] = number of gaussians that have parent_indices[j] == i
        N = len(levels)
        valid_parent_mask = (parent_indices != -1) & (parent_indices >= 0) & (parent_indices < N)
        if valid_parent_mask.any():
            valid_parent_indices = parent_indices[valid_parent_mask]
            num_child = torch.bincount(valid_parent_indices, minlength=N)
        else:
            num_child = torch.zeros(N, dtype=torch.long, device=device)
        
        # Compute can_duplicate flag for each gaussian
        # can_duplicate = (parent != -1) and (parent.num_child < max_children_per_parent)
        # Root nodes (parent == -1) cannot duplicate (NEVER duplicate/split root nodes)
        has_parent = (parent_indices != -1) & (parent_indices >= 0) & (parent_indices < N)
        can_duplicate = torch.zeros(N, dtype=torch.bool, device=device)
        if has_parent.any():
            parent_ids = parent_indices[has_parent]
            parent_num_children = num_child[parent_ids]
            can_duplicate[has_parent] = parent_num_children < self.max_children_per_parent
        
        # Check max_level constraint
        # If at max_level, force duplication (cannot create children)
        # BUT: root nodes (parent == -1) are NEVER allowed to duplicate/split
        if max_level is not None:
            at_max_level = (levels >= max_level)
            # Only allow duplication at max_level if NOT a root node
            can_duplicate = can_duplicate | (at_max_level & has_parent)  # Force duplicate at max_level, but exclude root nodes
        
        # Evaluate gradient threshold (default.py style, no level-based scaling)
        is_grad_high = grads > self.grow_grad2d
        
        # Get actual scales from hierarchical structure
        # params["scales"] are residuals, so we need actual scales with hierarchical structure applied
        # If actual_splats is provided (from step_post_backward), reuse it to avoid duplicate get_splats() calls
        if actual_splats is not None:
            actual_scales = actual_splats["scales"]  # [N, 3] - actual scales with hierarchy
        else:
            # Fallback: if not provided, call get_splats() (for backward compatibility or direct calls)
            multigrid_gaussians = state.get("multigrid_gaussians", None)
            if multigrid_gaussians is not None:
                actual_splats_fallback = multigrid_gaussians.get_splats(level=None, detach_parents=False)
                actual_scales = actual_splats_fallback["scales"]  # [N, 3] - actual scales with hierarchy
            else:
                # Final fallback: use params directly (should not happen in normal usage)
                actual_scales = params["scales"]  # [N, 3] - residual scales (fallback)
        
        # Evaluate scale conditions (default.py style, using actual scales)
        is_small = (
            torch.exp(actual_scales).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_large = ~is_small
        
        # Determine duplicate vs split vs create_children
        # can_duplicate == True: duplicate (if small) or split (if large)
        # can_duplicate == False: create children (if not at max_level)
        # Root nodes (parent == -1) are NEVER allowed to duplicate/split
        is_dupli = is_grad_high & is_small & can_duplicate & has_parent  # Explicitly exclude root nodes
        is_split = is_grad_high & is_large & can_duplicate & has_parent  # Explicitly exclude root nodes
        if step < self.refine_scale2d_stop_iter:
            is_split = is_split | (state["radii"] > self.grow_scale2d) & can_duplicate & has_parent  # Explicitly exclude root nodes
        
        # Create children: gradient high, cannot duplicate, not at max_level
        is_create_children = is_grad_high & (~can_duplicate)
        if max_level is not None:
            # Exclude gaussians at max_level from creating children
            is_create_children = is_create_children & (~at_max_level)
        
        n_dupli = is_dupli.sum().item()
        n_split = is_split.sum().item()
        n_create_children = is_create_children.sum().item()
        
        # Statistics
        if step % self.refine_every == 0 and self.verbose:
            valid_grads = grads[count > 0]
            if len(valid_grads) > 0:
                percentiles = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], device=device)
                grad2d_percentiles = torch.quantile(valid_grads, percentiles / 100.0)
                print(f"Step {step} - Grad2d gradient percentiles:")
                for p, v in zip(percentiles, grad2d_percentiles):
                    print(f"  {p.item():3.0f}%: {v.item():.6f}")
                print(f"  Threshold: {self.grow_grad2d:.6f}")
                print(f"  Can duplicate: {can_duplicate.sum().item()}/{N}")
                print(f"  Will duplicate: {n_dupli}, split: {n_split}, create_children: {n_create_children}")
        
        # Store original N for mask handling
        N_original = N
        
        # First duplicate (same level, same parent)
        if n_dupli > 0:
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_dupli,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
            )
            # Update levels and parent_indices after duplicate (they may have changed)
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
        
        # Then split (same level, same parent)
        if n_split > 0:
            # is_split mask is still valid for original N gaussians
            # New gaussians from duplicate are not in is_split mask (which is correct)
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                revised_opacity=self.revised_opacity,
            )
            # Update levels and parent_indices after split
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
        
        # Finally create children (next level)
        if n_create_children > 0:
            # is_create_children mask is still valid for original N gaussians
            # New gaussians from duplicate/split are not in is_create_children mask (which is correct)
            # We only want to create children for the original gaussians that met the criteria
            create_children_mg(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_create_children,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                n_children=self.n_children_per_split,
            )
        
        # Return (n_dupli, n_split, n_create_children) for compatibility
        return n_dupli, n_split, n_create_children

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        actual_splats: Optional[Dict[str, Tensor]] = None,
    ) -> int:
        """Prune gaussians, but only leaf nodes (nodes without children)."""
        device = params["opacities"].device
        N = len(params["opacities"])
        parent_indices = state["parent_indices"]  # [N,]

        # Count number of direct children for each gaussian - OPTIMIZED
        # num_child[i] = number of gaussians that have parent_indices[j] == i
        # Use bincount for efficient counting (faster than scatter_add for this use case)
        valid_parent_mask = (parent_indices != -1) & (parent_indices >= 0) & (parent_indices < N)
        if valid_parent_mask.any():
            valid_parent_indices = parent_indices[valid_parent_mask]
            num_child = torch.bincount(valid_parent_indices, minlength=N)
        else:
            num_child = torch.zeros(N, dtype=torch.long, device=device)
        
        # Fix out-of-bounds parent indices: set them to -1 (make them root nodes)
        out_of_bounds_mask = (parent_indices != -1) & ((parent_indices < 0) | (parent_indices >= N))
        if out_of_bounds_mask.any():
            parent_indices = parent_indices.clone()  # Avoid in-place modification if it's a view
            parent_indices[out_of_bounds_mask] = -1
            # Update state
            state["parent_indices"] = parent_indices
            # Also update level_indices for affected gaussians (they become level 1)
            if "level_indices" in state:
                # Rebuild level_indices - OPTIMIZED: only rebuild if needed
                levels = state["levels"]
                unique_levels = levels.unique()
                level_indices_new = {}
                for level_val in unique_levels:
                    level_val_int = level_val.item()
                    level_indices_new[level_val_int] = torch.where(levels == level_val_int)[0].tolist()
                state["level_indices"] = level_indices_new
            
            if self.verbose:
                n_fixed = out_of_bounds_mask.sum().item()
                print(f"Warning: Fixed {n_fixed} out-of-bounds parent indices by setting them to -1 (root nodes)")

        # Get actual scales and opacities from hierarchical structure
        # params["scales"] and params["opacities"] are residuals, so we need actual values with hierarchical structure applied
        # If actual_splats is provided (from step_post_backward), reuse it to avoid duplicate get_splats() calls
        # However, if _grow_gs was called before _prune_gs, the number of gaussians may have changed,
        # so we need to check if actual_splats size matches current N
        if actual_splats is not None:
            actual_splats_N = len(actual_splats["opacities"])
            if actual_splats_N == N:
                # Size matches, can use actual_splats
                actual_scales = actual_splats["scales"]  # [N, 3] - actual scales with hierarchy
                actual_opacities = actual_splats["opacities"]  # [N,] - actual opacities with hierarchy
            else:
                # Size mismatch (likely due to _grow_gs adding gaussians), need to recompute
                if self.verbose:
                    print(f"Warning: actual_splats size ({actual_splats_N}) != current N ({N}), recomputing...")
                multigrid_gaussians = state.get("multigrid_gaussians", None)
                if multigrid_gaussians is not None:
                    actual_splats_fallback = multigrid_gaussians.get_splats(level=None, detach_parents=False)
                    actual_scales = actual_splats_fallback["scales"]  # [N, 3] - actual scales with hierarchy
                    actual_opacities = actual_splats_fallback["opacities"]  # [N,] - actual opacities with hierarchy
                else:
                    # Final fallback: use params directly (should not happen in normal usage)
                    actual_scales = torch.exp(params["scales"])  # Convert from log space
                    actual_opacities = torch.sigmoid(params["opacities"])  # Convert from logit space
        else:
            # Fallback: if not provided, call get_splats() (for backward compatibility or direct calls)
            multigrid_gaussians = state.get("multigrid_gaussians", None)
            if multigrid_gaussians is not None:
                actual_splats_fallback = multigrid_gaussians.get_splats(level=None, detach_parents=False)
                actual_scales = actual_splats_fallback["scales"]  # [N, 3] - actual scales with hierarchy
                actual_opacities = actual_splats_fallback["opacities"]  # [N,] - actual opacities with hierarchy
            else:
                # Final fallback: use params directly (should not happen in normal usage)
                actual_scales = torch.exp(params["scales"])  # Convert from log space
                actual_opacities = torch.sigmoid(params["opacities"])  # Convert from logit space
        
        # Pruning condition: low opacity (use actual opacity, not residual)
        is_prune = torch.sigmoid(actual_opacities.flatten()) < self.prune_opa
        
        if step > self.reset_every:
            # Use actual scales (hierarchical structure applied), not residual scales
            actual_scales_exp = torch.exp(actual_scales)  # [N, 3]
            is_too_big = (
                actual_scales_exp.max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big
            n_is_too_big = is_too_big.sum().item()
        else:
            n_is_too_big = 0

        # Adaptive pruning: only prune leaf nodes (num_child == 0)
        # This ensures that parent nodes with children are not pruned, preventing
        # the loss of entire subtrees in the hierarchical structure
        is_leaf = (num_child == 0)
        final_prune_mask = is_prune & is_leaf

        n_prune = final_prune_mask.sum().item()
        if n_prune > 0:
            remove_mg(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=final_prune_mask,
                levels=state["levels"],
                parent_indices=state["parent_indices"],
                level_indices=state["level_indices"],
            )

        return n_prune, n_is_too_big


