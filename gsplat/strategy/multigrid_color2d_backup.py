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
        grow_grad2d (float): Base gradient threshold for GS splitting/duplication. Default is 0.0002.
        grow_color (float): Color gradient threshold for creating children. Default is 0.0002.
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
          Higher values make it harder to create children. Default is 2.0.
        duplicate_grad_scale (float): Scale factor for gradient threshold when duplicating
          at the same level. Should be smaller than split_grad_scale. Default is 1.0.
        n_children_per_split (int): Number of children to create per split. Default is 4.
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0005
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
    split_grad_scale: float = 1.0  # Scale for creating children (higher = harder)
    duplicate_grad_scale: float = 1.5  # Scale for same-level duplication (lower = easier)
    n_children_per_split: int = 4

    def initialize_state(self, scene_scale: float = 1.0, levels: Tensor = None, parent_indices: Tensor = None, level_indices: Dict[int, List[int]] = None, max_level: Optional[int] = None) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

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
        # - grad_color: running accum of the norm of the color gradients for each GS.
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
        
        state = {"grad2d": None, "grad_color": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        
        # Store hierarchical structure
        state["levels"] = levels
        state["parent_indices"] = parent_indices
        state["level_indices"] = level_indices
        state["max_level"] = max_level  # Store max_level for limiting hierarchy growth
        
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
        
        # Also retain gradient for colors if available
        if "colors" in info:
            info["colors"].retain_grad()

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
            # grow GSs
            n_dupli, n_split, n_create_children = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split, {n_create_children} GSs created children. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            if state["grad_color"] is not None:
                state["grad_color"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if (step % self.reset_every == 0 and step > 0) and self.use_opacity_reset:
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
        
        # colors is optional (only if SH is used)
        has_colors_grad = "colors" in info and info["colors"].grad is not None

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
        if state["grad_color"] is None:
            state["grad_color"] = torch.zeros(n_gaussian, device=grads.device)
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
            
            # Extract color gradients if available (packed mode)
            if has_colors_grad:
                # colors is [nnz, 3] after SH computation in packed mode
                color_grads = info["colors"].grad  # [nnz, 3]
            else:
                color_grads = None
            
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
            
            # Extract color gradients if available (non-packed mode)
            if has_colors_grad:
                # colors is [C, N, 3] after SH computation in non-packed mode
                color_grads = info["colors"].grad[sel]  # [nnz, 3]
            else:
                color_grads = None
            
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
            
            # Accumulate color gradients if available
            if color_grads is not None:
                # Compute norm of color gradients
                color_grad_norms = color_grads.norm(dim=-1)  # [nnz]
                state["grad_color"].index_add_(0, full_gs_ids, color_grad_norms)
            
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
    ) -> Tuple[int, int]:
        """Grow gaussians based on grad2d and color gradients.
        
        - grad2d gradient high: use default.py style duplication/split (based on scale)
        - color gradient high: split
        """
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)  # Average grad2d gradient
        device = grads.device
        
        # Get color gradients (always available)
        color_grads = state["grad_color"] / count.clamp_min(1)  # Average color gradient
        
        # Get levels for all gaussians
        levels = state["levels"]  # [N,]
        parent_indices = state["parent_indices"]  # [N,]
        level_indices = state["level_indices"]  # Dict[int, List[int]]

        # Check scale conditions (for default.py style duplication/split)
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_large = ~is_small
        
        # Check if max_level is set and if any gaussians are at max_level
        max_level = state.get("max_level", None)
        at_max_level = None
        if max_level is not None:
            at_max_level = (levels >= max_level)  # [N,] mask for gaussians at or above max_level
        
        # Compute statistics for grad2d and color gradients
        # Print percentiles every refine_every steps
        if step % self.refine_every == 0 and self.verbose:
            # Grad2d statistics
            valid_grads = grads[count > 0]  # Only consider gaussians that were visible
            if len(valid_grads) > 0:
                percentiles = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], device=device)
                grad2d_percentiles = torch.quantile(valid_grads, percentiles / 100.0)
                print(f"Step {step} - Grad2d gradient percentiles:")
                for p, v in zip(percentiles, grad2d_percentiles):
                    print(f"  {p.item():3.0f}%: {v.item():.6f}")
                
                # Check where threshold falls
                threshold_percentile = (valid_grads <= self.grow_grad2d).float().mean() * 100
                print(f"  Threshold {self.grow_grad2d:.6f} is at {threshold_percentile.item():.2f} percentile")
            else:
                print(f"Step {step} - Grad2d: No valid gradients (all count=0)")
            
            # Color gradient statistics
            valid_color_grads = color_grads[count > 0]
            if len(valid_color_grads) > 0:
                color_percentiles = torch.quantile(valid_color_grads, percentiles / 100.0)
                print(f"Step {step} - Color gradient percentiles:")
                for p, v in zip(percentiles, color_percentiles):
                    print(f"  {p.item():3.0f}%: {v.item():.6f}")
            else:
                print(f"Step {step} - Color gradient: No valid gradients (all count=0)")
        
        # grad2d-based duplication/split (default.py style)
        is_grad2d_high = grads > self.grow_grad2d
        is_dupli_from_grad2d = is_grad2d_high & is_small
        is_split_from_grad2d = is_grad2d_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split_from_grad2d |= state["radii"] > self.grow_scale2d
        
        # color gradient-based create_children
        # Use grow_color threshold (percentile is only for statistics)
        valid_color_grads = color_grads[count > 0]
        if len(valid_color_grads) > 0:
            # Use grow_color threshold
            is_create_children = color_grads > self.grow_color
            
            # Print color threshold statistics if verbose
            if step % self.refine_every == 0 and self.verbose:
                color_threshold_percentile = (valid_color_grads <= self.grow_color).float().mean() * 100
                print(f"  Color threshold {self.grow_color:.6f} is at {color_threshold_percentile.item():.2f} percentile")
        else:
            is_create_children = torch.zeros_like(grads, dtype=torch.bool)
        
        # Exclude gaussians that will be split/duplicated from create_children
        is_create_children = is_create_children & (~is_dupli_from_grad2d) & (~is_split_from_grad2d)
        
        # If max_level is set, gaussians at max_level should only duplicate (not split or create_children)
        if max_level is not None and at_max_level is not None:
            # Gaussians at max_level that would split should instead duplicate
            split_at_max_level = is_split_from_grad2d & at_max_level
            is_dupli_from_grad2d = is_dupli_from_grad2d | split_at_max_level  # Add to duplication
            is_split_from_grad2d = is_split_from_grad2d & (~at_max_level)  # Remove from split
            # Gaussians at max_level should not create children
            is_create_children = is_create_children & (~at_max_level)
        
        n_dupli = is_dupli_from_grad2d.sum().item()
        n_split = is_split_from_grad2d.sum().item()
        n_create_children = is_create_children.sum().item()

        # first duplicate (default.py style)
        if n_dupli > 0:
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_dupli_from_grad2d,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
            )
            # Update levels and parent_indices after duplication
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
            
            # new GSs added by duplication will not be split
            # Update split mask to exclude newly duplicated gaussians
            N_old = len(is_split_from_grad2d)
            N_new = len(levels)
            is_split_from_grad2d = torch.cat(
                [
                    is_split_from_grad2d,
                    torch.zeros(N_new - N_old, dtype=torch.bool, device=device),
                ]
            )
            # Also update create_children mask
            is_create_children = torch.cat(
                [
                    is_create_children,
                    torch.zeros(N_new - N_old, dtype=torch.bool, device=device),
                ]
            )

        # then split (default.py style)
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split_from_grad2d,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                revised_opacity=self.revised_opacity,
            )
            # Update levels and parent_indices after split
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
            
            # Update create_children mask to exclude newly split gaussians
            N_old = len(is_create_children)
            N_new = len(levels)
            is_create_children = torch.cat(
                [
                    is_create_children,
                    torch.zeros(N_new - N_old, dtype=torch.bool, device=device),
                ]
            )

        # then create_children (for color gradient)
        if n_create_children > 0:
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
        
        return n_dupli, n_split, n_create_children

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """Prune gaussians, but only leaf nodes (nodes without children)."""
        device = params["opacities"].device
        N = len(params["opacities"])
        parent_indices = state["parent_indices"]  # [N,]

        # Find which gaussians are parents (have children)
        # is_parent[i] = True if gaussian i is a parent (has children)
        # A gaussian is a parent if any other gaussian has it as a parent
        is_parent = torch.zeros(N, dtype=torch.bool, device=device)
        valid_parent_mask = (parent_indices != -1)
        if valid_parent_mask.any():
            # Get the actual parent indices (values in parent_indices that are not -1)
            valid_parent_indices = parent_indices[valid_parent_mask]
            # Only mark indices that are within bounds as parents
            # (parent indices should be in [0, N), but check for safety)
            valid_bounds = (valid_parent_indices >= 0) & (valid_parent_indices < N)
            if valid_bounds.any():
                valid_parent_indices_in_bounds = valid_parent_indices[valid_bounds]
                # Mark those indices as parents (they have children pointing to them)
                is_parent[valid_parent_indices_in_bounds] = True
            
            # Fix out-of-bounds parent indices: set them to -1 (make them root nodes)
            if not valid_bounds.all():
                out_of_bounds_mask = valid_parent_mask.clone()
                out_of_bounds_mask[valid_parent_mask] = ~valid_bounds
                # Set out-of-bounds parent indices to -1
                parent_indices[out_of_bounds_mask] = -1
                # Update state
                state["parent_indices"] = parent_indices
                # Also update level_indices for affected gaussians (they become level 1)
                if "level_indices" in state:
                    # Rebuild level_indices
                    level_indices_new = {}
                    for level_val in state["levels"].unique():
                        level_val_int = level_val.item()
                        mask_level = (state["levels"] == level_val_int)
                        level_indices_new[level_val_int] = torch.where(mask_level)[0].tolist()
                    state["level_indices"] = level_indices_new
                
                if self.verbose:
                    n_fixed = (~valid_bounds).sum().item()
                    print(f"Warning: Fixed {n_fixed} out-of-bounds parent indices by setting them to -1 (root nodes)")

        # Pruning condition: low opacity
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        # Final prune mask: prune condition AND not a parent (leaf only)
        final_prune_mask = is_prune & (~is_parent)

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

        return n_prune

