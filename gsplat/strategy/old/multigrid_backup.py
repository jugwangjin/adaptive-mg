from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from .base import Strategy
from .ops_mg import duplicate_mg, remove_mg, reset_opa_mg, split_mg


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
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    use_opacity_reset: bool = True
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
            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
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
    ) -> Tuple[int, int]:
        """Grow gaussians with level-aware gradient thresholds."""
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device
        
        # Get levels for all gaussians
        levels = state["levels"]  # [N,]
        parent_indices = state["parent_indices"]  # [N,]
        level_indices = state["level_indices"]  # Dict[int, List[int]]

        # Compute level-scaled gradient thresholds
        # Higher level = easier threshold (multiply by scale^level, where scale > 1.0)
        # For duplicate (same level): use duplicate_grad_scale (higher threshold, harder to trigger)
        # For split (creating children): use split_grad_scale (lower threshold, easier to trigger)
        # Note: duplicate_grad_scale > split_grad_scale means duplication requires higher gradient
        #       This is intentional: we want to create children (split) more easily than duplicate
        duplicate_thresholds = self.grow_grad2d * (self.duplicate_grad_scale ** levels.float())
        split_thresholds = self.grow_grad2d * (self.split_grad_scale ** levels.float())

        # Check if gradient is very high (for duplication at same level)
        # Duplication requires higher gradient threshold than splitting
        is_grad_very_high = grads > duplicate_thresholds
        
        # Check if gradient is high but not very high (for splitting/child creation)
        # Splitting requires lower gradient threshold than duplication
        is_grad_high = (grads > split_thresholds) & (~is_grad_very_high)

        # Check scale conditions
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        
        # Check if max_level is set and if any gaussians are at max_level
        max_level = state.get("max_level", None)
        at_max_level = None
        if max_level is not None:
            at_max_level = (levels >= max_level)  # [N,] mask for gaussians at or above max_level
        
        # For duplication: gradient is high (but not very high) and scale is small
        is_dupli = is_grad_very_high
        # is_dupli = is_grad_very_high & is_small
        
        # For split: gradient is very high (or large scale)
        # is_large = ~is_small
        is_split = is_grad_high
        # is_split = is_grad_high | (is_grad_high & is_large)
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        
        # If max_level is set, gaussians at max_level should only duplicate (not split)
        # Move split candidates at max_level to duplication
        if max_level is not None and at_max_level is not None:
            # Gaussians at max_level that would split should instead duplicate
            split_at_max_level = is_split & at_max_level
            is_dupli = is_dupli | split_at_max_level  # Add to duplication
            is_split = is_split & (~at_max_level)  # Remove from split
        
        n_dupli = is_dupli.sum().item()
        n_split = is_split.sum().item()

        # first duplicate
        if n_dupli > 0:
            duplicate_mg(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_dupli,
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
            N_old = len(is_split)
            N_new = len(levels)
            is_split = torch.cat(
                [
                    is_split,
                    torch.zeros(N_new - N_old, dtype=torch.bool, device=device),
                ]
            )

        # then split
        if n_split > 0:
            split_mg(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                n_children=self.n_children_per_split,
            )
        return n_dupli, n_split

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
            # Mark those indices as parents (they have children pointing to them)
            is_parent[valid_parent_indices] = True

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
            )

        return n_prune

