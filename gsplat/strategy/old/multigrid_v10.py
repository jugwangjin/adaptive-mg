from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import torch
from torch import Tensor
from typing_extensions import Literal

from .base import Strategy
from .ops_mg_v4 import clone_hierarchy_block, duplicate, remove_mg, reset_opa_mg, split, _add_zero_parameter_children


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
        grow_grad2d (float): Base gradient threshold for creating children. Default is 0.0003 (level 1).
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
        reset_every (int): Reset opacities every this V-cycles. Default is 10.
        refine_every (int): Refine GSs every this V-cycles. Default is 3.
        pause_refine_after_reset (int): Pause refining GSs until this number of V-cycles after
          reset. Default is 2.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".
        n_children_per_split (int): Number of children to create per split. Default is 4.
        max_children_per_parent (int): Maximum number of children a parent can have.
          A parent can create children only if its current num_child <= (max_children_per_parent - n_children_per_split).
          This ensures that creating n_children_per_split children won't exceed max_children_per_parent.
          Default is 8 (allowing 4+4 children).
    """

    prune_opa: float = 0.005
    coarsest_grow_grad2d: float = 0.0004 # Base gradient threshold (level 1)
    finest_grow_grad2d: float = 0.0002
    grow_color: float = 9999.0
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 6_000
    reset_every: int = 60 # Reset opacities every N V-cycles
    use_opacity_reset: bool = True
    revised_opacity: bool = True
    refine_every: int = 2 # Refine GSs every N V-cycles
    pause_refine_after_reset: int = 2  # Pause refining for N V-cycles after reset
    verbose: bool = False
    absgrad: bool = True  # Use absolute gradients for GS splitting
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    n_children_per_split: int = 4  # Number of children to create per split
    max_children_per_parent: int = 5  # Maximum number of children a parent can have
    max_level: int = 5  # Maximum level in the hierarchy
    use_gradient_inheritance: bool = False  # Enable gradient inheritance from parent to children

    def initialize_state(self, scene_scale: float = 1.0, levels: Tensor = None, parent_indices: Tensor = None, level_indices: Dict[int, List[int]] = None, max_level: Optional[int] = None, multigrid_gaussians: Optional[Any] = None) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        
        Args:
            scene_scale: Scale of the scene. Default is 1.0.
            levels: Tensor [N,] with level for each gaussian. Required for multigrid strategy.
            parent_indices: Tensor [N,] with parent index for each gaussian. Required for multigrid strategy.
            level_indices: Dict mapping level -> list of gaussian indices. Required for multigrid strategy.
        
        Raises:
            ValueError: If levels, parent_indices, or level_indices are not provided.
        """
        if levels is None or parent_indices is None or level_indices is None:
            raise ValueError("MultigridStrategy requires levels, parent_indices, and level_indices to be provided.")
        
        # Initialize grad2d and count as dicts for render_level-based accumulation
        N = len(levels)
        device = levels.device
        state = {
            "scene_scale": scene_scale,
            "levels": levels,
            "parent_indices": parent_indices,
            "level_indices": level_indices,
            "max_level": max_level,
            "multigrid_gaussians": multigrid_gaussians,
            "grad2d": {},  # Dict[int, Tensor] - render_level -> accumulated gradients
            "count": {},   # Dict[int, Tensor] - render_level -> accumulated counts
        }
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Check if the parameters and optimizers are compatible with this strategy."""
        required_params = ["means", "scales", "quats", "opacities"]
        for name in required_params:
            if name not in params:
                raise ValueError(f"Parameter {name} is required but not found.")
        
        # Check if optimizers exist for all parameters
        for name in required_params:
            if name not in optimizers:
                raise ValueError(f"Optimizer for {name} is required but not found.")

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Called before backward pass. Retains gradients for densification.
        
        This is called before loss.backward() to ensure that gradients for
        means2d (or gradient_2dgs) are retained for later use in state updates
        and densification decisions.
        
        Similar to DefaultStrategy.step_pre_backward().
        """
        assert (
            self.key_for_gradient in info
        ), f"The {self.key_for_gradient} is required but missing."
        info[self.key_for_gradient].retain_grad()
        
        # Color gradients are not used for densification

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Called after backward pass. Updates state by accumulating gradients.
        
        This is called after loss.backward() to accumulate gradients and update
        state (grad2d, count, radii) for densification decisions.
        
        For V-cycle training, densification is NOT performed here. Instead,
        densification is performed in post_cycle() after V-cycle completion
        to maintain hierarchical structure consistency during V-cycle iterations.
        
        Similar to DefaultStrategy.step_post_backward() but without densification.
        
        Note: visible_mask is obtained from info["visible_mask"] if available,
        so it doesn't need to be passed as a separate argument.
        """
        # Update state with rendering information (accumulate gradients)
        self._update_state(params, state, info, packed=packed)

    def post_cycle(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        cycle: int,
        step: int,
    ):
        """Called after V-cycle completion. Performs densification (grow/prune) operations.
        
        This is called once per V-cycle, allowing densification to be synchronized with
        the V-cycle structure. Densification is performed based on cycle count rather than
        step count to maintain consistency with the multigrid training approach.
        
        Args:
            params: Gaussian parameters
            optimizers: Parameter optimizers
            state: Strategy state
            cycle: Current V-cycle index (0-based)
            step: Current training step (for reference)
        """
        # Update hierarchical structure from state (in case it changed)
        if "multigrid_gaussians" in state and state["multigrid_gaussians"] is not None:
            multigrid_gaussians = state["multigrid_gaussians"]
            state["levels"] = multigrid_gaussians.levels
            state["parent_indices"] = multigrid_gaussians.parent_indices
            state["level_indices"] = multigrid_gaussians.level_indices
        
        # Check if we should refine (based on step, not cycle)
        if step < self.refine_start_iter or step >= self.refine_stop_iter:
            return
        
        # Check if we should refine based on cycle count
        if cycle % self.refine_every != 0:
            return
        
        # Check pause after reset (cycle-based)
        if self.use_opacity_reset and self.pause_refine_after_reset > 0:
            if "last_reset_cycle" in state:
                if cycle - state["last_reset_cycle"] < self.pause_refine_after_reset:
                    return
        
        # Get coarsest and finest levels for grow_gs iteration
        level_indices = state["level_indices"]
        if len(level_indices) == 0:
            return
        
        coarsest_level = min(level_indices.keys())
        finest_level = max(level_indices.keys())
        max_level = state.get("max_level", self.max_level)
        
        # Grow gaussians once using aggregated max gradients across levels
        total_n_dupli_split = 0
        total_n_create_children = 0
        any_densification = False
        
        # Initialize already_densified at post_cycle level
        already_densified = torch.zeros_like(multigrid_gaussians.levels, dtype=torch.bool, device=multigrid_gaussians.levels.device)
        
        # Aggregate gradients: use max of level-wise average gradients
        levels = state["levels"]
        N = len(levels)
        device = levels.device
        agg_grad2d = torch.zeros(N, device=device)
        agg_has_grad = torch.zeros(N, dtype=torch.bool, device=device)
        
        if "grad2d" in state and "count" in state:
            for render_level, grad2d in state["grad2d"].items():
                if render_level not in state["count"]:
                    continue
                count = state["count"][render_level]
                grad2d = self._ensure_tensor_size(grad2d, N, device, torch.float32)
                count = self._ensure_tensor_size(count, N, device, torch.float32)
                avg_grad2d = grad2d / count.clamp_min(1.0)
                agg_grad2d = torch.maximum(agg_grad2d, avg_grad2d)
                agg_has_grad |= (count > 0)
                
        
        agg_count = agg_has_grad.to(dtype=torch.float32)
        
        n_dupli_split, n_create_children, already_densified = self._grow_gs(
            params,
            optimizers,
            state,
            step,
            target_level=None,
            already_densified=already_densified,
            agg_grad2d=agg_grad2d,
            agg_count=agg_count,
        )
        
        # Free aggregated gradient buffers
        del agg_grad2d, agg_has_grad, agg_count
        
        total_n_dupli_split += n_dupli_split
        total_n_create_children += n_create_children
        
        if n_dupli_split > 0 or n_create_children > 0:
            any_densification = True
            # Update hierarchical structure after grow (structure may have changed)
            if "multigrid_gaussians" in state and state["multigrid_gaussians"] is not None:
                multigrid_gaussians = state["multigrid_gaussians"]
                old_N = len(state["levels"])
                state["levels"] = multigrid_gaussians.levels
                state["parent_indices"] = multigrid_gaussians.parent_indices
                state["level_indices"] = multigrid_gaussians.level_indices
                new_N = len(state["levels"])
                
                # Update level_indices after grow
                level_indices = state["level_indices"]
                
                # Expand grad2d and count tensors if gaussian count increased
                # Optimization: Use torch.cat with zeros instead of creating new tensor and copying
                if new_N > old_N:
                    device = state["levels"].device
                    num_new = new_N - old_N
                    # Expand all render_level tensors efficiently
                    if "grad2d" in state:
                        for render_level in state["grad2d"]:
                            old_tensor = state["grad2d"][render_level]
                            # Use cat instead of zeros + copy for better performance
                            state["grad2d"][render_level] = torch.cat([
                                old_tensor,
                                torch.zeros(num_new, device=device, dtype=old_tensor.dtype)
                            ])
                    if "count" in state:
                        for render_level in state["count"]:
                            old_tensor = state["count"][render_level]
                            state["count"][render_level] = torch.cat([
                                old_tensor,
                                torch.zeros(num_new, device=device, dtype=old_tensor.dtype)
                            ])
                    if "radii" in state:
                        old_radii = state["radii"]
                        state["radii"] = torch.cat([
                            old_radii,
                            torch.zeros(num_new, device=device, dtype=old_radii.dtype)
                        ])
        
        # Update hierarchical structure after all grow operations
        if "multigrid_gaussians" in state and state["multigrid_gaussians"] is not None:
            multigrid_gaussians = state["multigrid_gaussians"]
            state["levels"] = multigrid_gaussians.levels
            state["parent_indices"] = multigrid_gaussians.parent_indices
            state["level_indices"] = multigrid_gaussians.level_indices
        
        # Prune gaussians
        n_pruned = self._prune_gs(params, optimizers, state, step)
        
        # Update hierarchical structure after prune
        if "multigrid_gaussians" in state and state["multigrid_gaussians"] is not None:
            multigrid_gaussians = state["multigrid_gaussians"]
            state["levels"] = multigrid_gaussians.levels
            state["parent_indices"] = multigrid_gaussians.parent_indices
            state["level_indices"] = multigrid_gaussians.level_indices
        
        # After all grow and prune operations, ensure all parents have children
        # This is done once after all operations to avoid redundant processing
        if max_level is not None and max_level > 1:
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
            N_final = len(levels)
            
            if N_final > 0:
                # Add zero-parameter children to all parents that need them
                # Process level by level from coarse to fine (like initialization)
                _add_zero_parameter_children(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    levels=levels,
                    parent_indices_tensor=parent_indices,
                    level_indices=level_indices,
                    max_level=max_level,
                )
                
                # Update structure after adding children
                if "multigrid_gaussians" in state and state["multigrid_gaussians"] is not None:
                    multigrid_gaussians = state["multigrid_gaussians"]
                    state["levels"] = multigrid_gaussians.levels
                    state["parent_indices"] = multigrid_gaussians.parent_indices
                    state["level_indices"] = multigrid_gaussians.level_indices
        
        # Reset opacities (cycle-based)
        if self.use_opacity_reset and cycle % self.reset_every == 0 and cycle > 0:
            reset_opa_mg(
                params=params,
                optimizers=optimizers,
                state=state,
                value=0.01,
            )
            state["last_reset_cycle"] = cycle
        
        # Reset running stats after densification (reset all render_levels)
        # If any densification occurred, reset all grad/count to zero

        if "grad2d" in state:
            for render_level in state["grad2d"]:
                state["grad2d"][render_level].zero_()
        if "count" in state:
            for render_level in state["count"]:
                state["count"][render_level].zero_()
        if "radii" in state and self.refine_scale2d_stop_iter > 0:
            state["radii"].zero_()
        torch.cuda.empty_cache()
        
        if self.verbose:
            # Calculate level-wise statistics
            levels = state["levels"]  # [N,]
            parent_indices = state["parent_indices"]  # [N,]
            level_indices = state["level_indices"]  # Dict[int, List[int]]
            N = len(levels)
            device = levels.device
            
            # Count number of children for each gaussian
            valid_parent_mask = (parent_indices != -1) & (parent_indices >= 0) & (parent_indices < N)
            if valid_parent_mask.any():
                valid_parent_indices = parent_indices[valid_parent_mask]
                n_child = torch.bincount(valid_parent_indices, minlength=N)
            else:
                n_child = torch.zeros(N, dtype=torch.long, device=device)
            
            # Build level-wise statistics
            level_stats = []
            for level in sorted(level_indices.keys()):
                level_gaussians = level_indices[level]
                n_gaussians = len(level_gaussians)
                
                if n_gaussians > 0:
                    level_n_child = n_child[level_gaussians]
                    min_n_child = level_n_child.min().item()
                    avg_n_child = level_n_child.float().mean().item()
                    max_n_child = level_n_child.max().item()
                    level_stats.append(f"L{level}: {n_gaussians} gaussians, n_children: [{min_n_child}, {avg_n_child:.2f}, {max_n_child}]")
            
            level_stats_str = ", ".join(level_stats)
            print(f"Cycle {cycle} (step {step}): Duplicated/Split: {total_n_dupli_split}, Created children: {total_n_create_children}, Pruned: {n_pruned}")
            print(f"  Level-wise: {level_stats_str}")

    @torch.no_grad()
    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Update state with rendering information using visible_mask.
        
        For multigrid, only visible gaussians are rendered, so we use visible_mask
        to correctly accumulate gradients and counts for all gaussians.
        """
        # Check required keys
        for key in ["width", "height", "n_cameras", "radii", self.key_for_gradient]:
            assert key in info, f"{key} is required but missing."
        
        # Initialize state on the first run
        N = len(params["means"])
        device = params["means"].device
        
        # Get render_level from info (default to 1 if not provided)
        render_level = info.get("render_level", 1)
        
        # Initialize grad2d and count dicts if not present
        if "grad2d" not in state:
            state["grad2d"] = {}
        if "count" not in state:
            state["count"] = {}
        
        # Initialize or resize tensors for this render_level
        # If tensor exists but size doesn't match, resize it
        if render_level not in state["grad2d"]:
            state["grad2d"][render_level] = torch.zeros(N, device=device)
        elif len(state["grad2d"][render_level]) != N:
            # Resize if gaussian count changed (e.g., after densification)
            old_tensor = state["grad2d"][render_level]
            new_tensor = torch.zeros(N, device=device, dtype=old_tensor.dtype)
            old_N = len(old_tensor)
            if old_N > 0:
                new_tensor[:min(old_N, N)] = old_tensor[:min(old_N, N)]
            state["grad2d"][render_level] = new_tensor
        
        if render_level not in state["count"]:
            state["count"][render_level] = torch.zeros(N, dtype=torch.float32, device=device)
        elif len(state["count"][render_level]) != N:
            # Resize if gaussian count changed (e.g., after densification)
            old_tensor = state["count"][render_level]
            new_tensor = torch.zeros(N, dtype=torch.float32, device=device)
            old_N = len(old_tensor)
            if old_N > 0:
                new_tensor[:min(old_N, N)] = old_tensor[:min(old_N, N)]
            state["count"][render_level] = new_tensor
        
        # Get gradients from info (similar to default.py)
        # Use absolute gradients if absgrad is enabled
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        
        # Normalize grads to [-1, 1] screen space (same as default.py)
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        
        # Ensure visible_mask is defined (set by multigrid_gaussians.rasterize_splats)
        # visible_mask is [N_total] indicating which gaussians were rendered
        if "visible_mask" not in info:
            visible_mask = torch.ones_like(params["means"][:, 0], dtype=torch.bool)
            warnings.warn("visible_mask not found in info. Assuming all gaussians are visible.", UserWarning)
        else:
            visible_mask = info["visible_mask"]  # [N_total,]
        
        visible_indices = torch.where(visible_mask)[0]  # [M,] indices of visible gaussians
        
        # Get gradient and count for visible gaussians
        if packed:
            # In packed mode, info["gaussian_ids"] maps rendered gaussians to visible_indices
            # But we need to map to full gaussian indices
            gaussian_ids = info["gaussian_ids"]  # [nnz,] indices in visible_indices
            # Map to full indices: visible_indices[gaussian_ids]
            full_gs_ids = visible_indices[gaussian_ids]  # [nnz,]
            # grads is [nnz, 2] for packed mode
            visible_grads = grads  # [nnz, 2]
            grad2d = visible_grads.norm(dim=-1)  # [nnz,]
            count = torch.ones_like(full_gs_ids, dtype=torch.float32)  # [nnz,]
            
            # Accumulate to the specific render_level
            state["grad2d"][render_level].index_add_(0, full_gs_ids, grad2d)
            state["count"][render_level].index_add_(0, full_gs_ids, count)
        else:
            # Not packed: grads is [C, M, 2] where C is number of cameras, M is visible gaussians
            # Note: multigrid_gaussians.rasterize_splats only passes visible_indices to rasterization,
            # so grads and radii are for visible gaussians only, not the full array.
            # Select visible gaussians that actually contributed to rendering (radii > 0)
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, M] where M is number of visible gaussians
            # gs_ids_local are indices within the visible gaussians array (0 to M-1)
            gs_ids_local = torch.where(sel)[1]  # [nnz] - indices in visible gaussians array
            visible_grads = grads[sel]  # [nnz, 2]
            
            # Map local indices to full gaussian indices using visible_indices
            # gs_ids_local are indices into visible_indices, so we need visible_indices[gs_ids_local]
            full_gs_ids = visible_indices[gs_ids_local]  # [nnz] - indices in full array
            
            # Compute norm
            grad2d = visible_grads.norm(dim=-1)  # [nnz,]
            count = torch.ones_like(full_gs_ids, dtype=torch.float32)  # [nnz,]
            
            # Aggregate across cameras (same gaussian might appear multiple times across cameras)
            # Accumulate to the specific render_level
            state["grad2d"][render_level].index_add_(0, full_gs_ids, grad2d)
            state["count"][render_level].index_add_(0, full_gs_ids, count)
            
        # Update radii for scale2d-based refinement
        if "radii" not in state:
            N = len(params["means"])
            device = params["means"].device
            state["radii"] = torch.zeros(
                N, dtype=torch.float32, device=device
            )
        
        radii = info["radii"]  # [B, N, 2] or [N, 2] or [M, 2] for visible only
        if radii.dim() == 3:
            radii = radii[0]  # [N, 2] or [M, 2]
        
        # Radii are for visible gaussians only
        if packed:
            # In packed mode, radii correspond to gaussian_ids
            gaussian_ids = info["gaussian_ids"]  # [nnz,] indices in visible_indices
            full_gs_ids = visible_indices[gaussian_ids]  # [nnz,]
            visible_radii = radii.max(dim=-1).values  # [nnz,]
            
            state["radii"][full_gs_ids] = torch.maximum(
                state["radii"][full_gs_ids],
                visible_radii,
            )
        else:
            # Radii are for visible gaussians: [M, 2]
            visible_radii = radii.max(dim=-1).values  # [M,]
            
            state["radii"][visible_indices] = torch.maximum(
                state["radii"][visible_indices],
                visible_radii,
            )
        
        if self.refine_scale2d_stop_iter > 0:
            # Normalize radii to [0, 1] screen space
            if packed:
                gaussian_ids = info["gaussian_ids"]
                full_gs_ids = visible_indices[gaussian_ids]
                normalized_radii = visible_radii / float(max(info["width"], info["height"]))
                
                state["radii"][full_gs_ids] = torch.maximum(
                    state["radii"][full_gs_ids],
                    normalized_radii,
                )
            else:
                normalized_radii = visible_radii / float(max(info["width"], info["height"]))
                
                state["radii"][visible_indices] = torch.maximum(
                    state["radii"][visible_indices],
                    normalized_radii,
                )
        
        # Note: Gradient inheritance is now done in _grow_gs after normalization

    @torch.no_grad()
    def _ensure_tensor_size(self, tensor: Tensor, target_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Ensure tensor matches target size, resizing if necessary."""
        if len(tensor) == target_size:
            return tensor
        new_tensor = torch.zeros(target_size, device=device, dtype=dtype)
        old_size = len(tensor)
        if old_size > 0:
            new_tensor[:min(old_size, target_size)] = tensor[:min(old_size, target_size)]
        return new_tensor

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        target_level: Optional[int] = None,
        actual_splats: Optional[Dict[str, Tensor]] = None,
        already_densified: Optional[Tensor] = None,
        agg_grad2d: Optional[Tensor] = None,
        agg_count: Optional[Tensor] = None,
    ) -> Tuple[int, int, Tensor]:
        """Grow gaussians using hierarchical densification algorithm.
        
        Algorithm Overview:
        1. Validate and prepare gradients/counts (target_level or aggregated)
        2. Pre-compute all flags (node_full, parent_full, grad_high, etc.)
        3. Apply gradient inheritance from parents to children (if enabled)
        4. Process densification level-by-level (coarsest to finest):
           - Root nodes: always densify (duplicate/split)
           - Non-root nodes: densify if parent not full, else signal parent
        5. Recursively propagate parent signals upward until root
        6. Execute duplicate/split operations
        
        Note: Children are pre-created during initialization, so no child creation here.
        
        Args:
            target_level: Render level to use for gradient/count accumulation.
                If None, uses aggregated gradients provided via agg_*.
            agg_grad2d: Optional aggregated gradient (max over levels).
            agg_count: Optional count mask for aggregated gradients (1 if any grad seen).
            already_densified: Boolean mask tracking already processed gaussians.
            
        Returns:
            Tuple of (n_duplicate + n_split, n_create_children, updated_already_densified)
        """
        # print("already_densified.sum().item():", already_densified.sum().item(), "target_level:", target_level)
        # Get hierarchical structure
        levels = state["levels"]  # [N,]
        parent_indices = state["parent_indices"]  # [N,]
        level_indices = state["level_indices"]  # Dict[int, List[int]]
        max_level = state.get("max_level", self.max_level)
        N = len(levels)
        device = levels.device
        
        use_aggregated = agg_grad2d is not None
        if not use_aggregated:
            # Early exit if no gradients accumulated for target_level
            if target_level is None or target_level not in state.get("count", {}) or target_level not in state.get("grad2d", {}):
                if already_densified is None:
                    already_densified = torch.zeros(len(levels), dtype=torch.bool, device=levels.device)
                return 0, 0, already_densified
            
            # Ensure all tensors match current gaussian count
            count = self._ensure_tensor_size(
                state["count"][target_level], N, device, torch.float32
            )
            state["count"][target_level] = count
            
            grad2d = self._ensure_tensor_size(
                state["grad2d"][target_level], N, device, torch.float32
            )
            state["grad2d"][target_level] = grad2d
            
        else:
            count_source = agg_count if agg_count is not None else torch.zeros_like(agg_grad2d)
            count = self._ensure_tensor_size(count_source, N, device, torch.float32)
            grad2d = self._ensure_tensor_size(agg_grad2d, N, device, torch.float32)
        
        # Ensure already_densified matches size
        if already_densified is None or len(already_densified) != N:
            new_already_densified = torch.zeros(N, dtype=torch.bool, device=device)
            if already_densified is not None and len(already_densified) > 0:
                old_size = len(already_densified)
                new_already_densified[:min(old_size, N)] = already_densified[:min(old_size, N)]
            already_densified = new_already_densified

        # ========== Step 1: Pre-compute hierarchical flags ==========
        
        # Count children per gaussian
        valid_parent_mask = (parent_indices >= 0) & (parent_indices < N)
        n_child = torch.bincount(
            parent_indices[valid_parent_mask], minlength=N
        ) if valid_parent_mask.any() else torch.zeros(N, dtype=torch.long, device=device)
        
        # Normalize gradients by count (average gradients)
        grads = grad2d / count.clamp_min(1.0)  # [N,]
        color_grads = torch.zeros_like(grads)
        
        # ========== Step 2: Apply gradient inheritance (if enabled) ==========
        # Propagate parent gradients to children: child_grad += (parent_grad / parent_radii * child_radii) / num_children
        # Larger children receive more gradient proportionally to their size
        # Must process level-by-level from coarsest to finest to ensure parent gradients are computed before children
        if self.use_gradient_inheritance and len(level_indices) > 0:
            all_radii = state.get("radii", torch.zeros(N, device=device))
            coarsest_level = min(level_indices.keys())
            finest_level = max(level_indices.keys())
            
            for level in range(coarsest_level, finest_level + 1):
                if level == 1 or level not in level_indices:
                    continue  # Skip level 1 (no parents) or missing level
                
                # Pre-allocate tensor once instead of creating new tensor each time
                level_indices_list = level_indices[level]
                if len(level_indices_list) == 0:
                    continue
                level_gaussians = torch.tensor(level_indices_list, dtype=torch.long, device=device)
                level_gaussians = level_gaussians[(level_gaussians >= 0) & (level_gaussians < N)]
                if len(level_gaussians) == 0:
                    continue
                
                # Get valid parent-child pairs
                level_parents = parent_indices[level_gaussians]
                valid_pair_mask = (level_parents >= 0) & (level_parents < N)
                if not valid_pair_mask.any():
                    continue
                
                children = level_gaussians[valid_pair_mask]  # [K,]
                parents = level_parents[valid_pair_mask]  # [K,]
                
                # Check visibility (radii > 0)
                parent_radii = all_radii[parents]
                child_radii = all_radii[children]
                visible_mask = (parent_radii >= 1e-8) & (child_radii >= 1e-8)
                
                if visible_mask.any():
                    # Compute inherited gradients for visible pairs
                    p_radii = parent_radii[visible_mask].clamp_min(1e-6)
                    c_radii = child_radii[visible_mask].clamp_min(1e-6)
                    p_grads = grads[parents[visible_mask]]
                    p_n_child = n_child[parents[visible_mask]].float().clamp_min(1.0)
                    
                    # Inherited gradient formula: (parent_grad / parent_radii * child_radii) / num_children
                    inherited_grad = ((p_grads / p_radii) * c_radii) / p_n_child
                    
                    # Add to children
                    grads[children[visible_mask]] += inherited_grad
        
        # ========== Step 3: Compute densification flags ==========
        
        # Compute level-dependent gradient thresholds (linear interpolation)
        if max_level > 1:
            level_weights = (levels.float() - 1.0) / max(float(max_level) - 1.0, 1.0)  # [N,] in [0, 1]
            level_thresholds = (
                self.coarsest_grow_grad2d
                - (self.coarsest_grow_grad2d - self.finest_grow_grad2d) * level_weights
            )  # [N,]
        else:
            level_thresholds = torch.full((N,), self.coarsest_grow_grad2d, device=device)
        
        # Check if gradients exceed thresholds (grad2d)
        is_grad2d_high = grads > level_thresholds  # [N,]
        is_color_grad_high = torch.zeros_like(is_grad2d_high)
        is_grad_high = is_grad2d_high  # [N,]
        
        # Debug: Check if gradients are accumulated for target_level (using actual thresholds)
        # Compute hierarchical structure flags
        has_parent = (parent_indices >= 0) & (parent_indices < N)
        is_root = ~has_parent  # [N,]
        is_node_full = n_child >= self.max_children_per_parent  # [N,]
        
        # Check if parent is full
        is_parent_full = torch.zeros(N, dtype=torch.bool, device=device)
        if has_parent.any():
            parent_ids = parent_indices[has_parent]
            is_parent_full[has_parent] = n_child[parent_ids] >= self.max_children_per_parent
        
        # Get scales for duplicate vs split decision
        multigrid_gaussians = state.get("multigrid_gaussians")
        with torch.no_grad():
            actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False, current_splats=None)
            actual_scales = torch.exp(actual_splats["scales"]).max(dim=-1).values  # [N,]
            scale_threshold = self.grow_scale3d * state["scene_scale"]
            is_small = actual_scales <= scale_threshold  # [N,]
            is_large = ~is_small  # [N,]
            del actual_splats
        
        # ========== Step 4: Determine densification candidates and hierarchy clone targets ==========
        
        should_densify = torch.zeros(N, dtype=torch.bool, device=device)
        # Parents that need to be cloned (with their signal children)
        clone_parents = torch.zeros(N, dtype=torch.bool, device=device)
        # Children that need to be cloned (original signal senders)
        clone_children_mask = torch.zeros(N, dtype=torch.bool, device=device)
        
        if len(level_indices) == 0:
            coarsest_level = finest_level = 1
        else:
            coarsest_level = min(level_indices.keys())
            finest_level = max(level_indices.keys())
        
        # Process each level from finest to coarsest
        # This order ensures that children are processed before parents, which is important
        # for hierarchy clone signal propagation (children signal parents upward)
        for level in range(finest_level, coarsest_level - 1, -1):
            if level not in level_indices:
                continue
            
            # Pre-allocate tensor once instead of creating new tensor each time
            level_indices_list = level_indices[level]
            if len(level_indices_list) == 0:
                continue
            level_gaussians = torch.tensor(level_indices_list, dtype=torch.long, device=device)
            level_gaussians = level_gaussians[(level_gaussians >= 0) & (level_gaussians < N)]
            if len(level_gaussians) == 0:
                continue
            
            # Filter to only high-gradient gaussians
            level_grad_high = is_grad_high[level_gaussians]
            if not level_grad_high.any():
                continue
            
            candidates = level_gaussians[level_grad_high]  # [M,]
            candidates_is_root = is_root[candidates]  # [M,]
            candidates_is_parent_full = is_parent_full[candidates]  # [M,]
            candidates_parents = parent_indices[candidates]  # [M,]
            
            # Root nodes: always densify
            root_candidates = candidates[candidates_is_root]
            should_densify[root_candidates] = True
            
            # Non-root nodes: densify if parent not full, else send hierarchy clone signal
            non_root_candidates = candidates[~candidates_is_root]
            non_root_parents_full = candidates_is_parent_full[~candidates_is_root]
            non_root_parents = candidates_parents[~candidates_is_root]
            
            # Can densify directly
            can_densify = ~non_root_parents_full
            should_densify[non_root_candidates[can_densify]] = True
            
            # Cannot densify: need hierarchy clone
            need_clone = non_root_parents_full
            if need_clone.any():
                clone_children = non_root_candidates[need_clone]
                clone_parents_candidates = non_root_parents[need_clone]
                # Filter valid parents and not yet densified
                valid_mask = (clone_parents_candidates >= 0) & (clone_parents_candidates < N) & ~already_densified[clone_parents_candidates]
                if valid_mask.any():
                    clone_children_mask[clone_children[valid_mask]] = True
                    clone_parents[clone_parents_candidates[valid_mask]] = True
        
        # Step 4b: Recursively propagate clone signals upward
        changed = True
        while changed:
            changed = False
            clone_parents_candidates = torch.where(clone_parents & ~already_densified)[0]
            if len(clone_parents_candidates) == 0:
                break
            
            clone_parents_is_root = is_root[clone_parents_candidates]
            clone_parents_parents = parent_indices[clone_parents_candidates]
            clone_parents_parents_full = is_parent_full[clone_parents_candidates]
            
            # Root nodes: can densify, remove from clone_parents
            root_clone_parents = clone_parents_candidates[clone_parents_is_root]
            if len(root_clone_parents) > 0:
                should_densify[root_clone_parents] = True
                clone_parents[root_clone_parents] = False
                changed = True
            
            # Non-root nodes that can densify: remove from clone_parents
            non_root_clone_parents = clone_parents_candidates[~clone_parents_is_root]
            if len(non_root_clone_parents) > 0:
                non_root_parents_full = clone_parents_parents_full[~clone_parents_is_root]
                can_densify = ~non_root_parents_full
                if can_densify.any():
                    should_densify[non_root_clone_parents[can_densify]] = True
                    clone_parents[non_root_clone_parents[can_densify]] = False
                    changed = True
                
                # Must propagate further upward
                must_propagate = non_root_parents_full
                if must_propagate.any():
                    propagate_children = non_root_clone_parents[must_propagate]
                    propagate_parents = clone_parents_parents[~clone_parents_is_root][must_propagate]
                    # Filter valid parents and not yet densified
                    valid_mask = (propagate_parents >= 0) & (propagate_parents < N) & ~already_densified[propagate_parents]
                    if valid_mask.any():
                        clone_children_mask[propagate_children[valid_mask]] = True
                        clone_parents[propagate_parents[valid_mask]] = True
                        clone_parents[propagate_children[valid_mask]] = False
                        changed = True

        # Prevent child densification if its parent is scheduled to densify
        # densified_parents = should_densify
        # if densified_parents.any():
        #     child_mask = (
        #         (parent_indices >= 0)
        #         & (parent_indices < N)
        #         & densified_parents[parent_indices]
        #     )
        #     if child_mask.any():
        #         should_densify[child_mask] = False

        # Prevent parent densification if any of its children is scheduled to densify
        densified_children = should_densify & ~is_root
        if densified_children.any():
            parent_ids = parent_indices[densified_children]
            valid_parent_ids = (parent_ids >= 0) & (parent_ids < N)
            if valid_parent_ids.any():
                should_densify[parent_ids[valid_parent_ids]] = False
        
        # ========== Step 5: Clone hierarchy blocks ==========
        # Track initial N to identify newly created gaussians
        N_before_all_operations = N
        
        n_hierarchy_clones = 0
        clone_parents_indices = torch.where(clone_parents & ~already_densified)[0]
        cloned_indices_for_densify = None  # Track cloned gaussians for duplicate/split
        
        if len(clone_parents_indices) > 0:
            # Mark clone children as already_densified (they will be cloned)
            clone_children_indices = torch.where(clone_children_mask & ~already_densified)[0]
            if len(clone_children_indices) > 0:
                already_densified[clone_children_indices] = True
            
            # Reuse is_small from Step 3 (line 731) - no need to recompute get_splats
            # is_small is already computed for all gaussians at line 734
            
            # Clone hierarchy blocks with scale adjustment
            cloned_indices_for_densify = clone_hierarchy_block(
                params=params,
                optimizers=optimizers,
                state=state,
                parent_indices_to_clone=clone_parents_indices,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                signal_indices=clone_children_mask,
                is_small_mask=is_small,  # Pass scale flags for adjustment
            )
            
            n_hierarchy_clones = len(clone_parents_indices)
            
            # Update structure
            old_N = len(levels)
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
            new_N = len(levels)
            N = new_N
            
            # Expand already_densified and should_densify to match new structure size
            if new_N > len(already_densified):
                new_already_densified = torch.zeros(new_N, dtype=torch.bool, device=device)
                old_size = len(already_densified)
                if old_size > 0:
                    new_already_densified[:old_size] = already_densified
                already_densified = new_already_densified
            
            # Expand should_densify to match new structure size
            if new_N > len(should_densify):
                new_should_densify = torch.zeros(new_N, dtype=torch.bool, device=device)
                old_size = len(should_densify)
                if old_size > 0:
                    new_should_densify[:old_size] = should_densify[:old_size]
                should_densify = new_should_densify
            
            # Recompute scales for all gaussians (including cloned ones)
            # Note: get_splats computes hierarchical parameters, so we need full recomputation
            # However, this is only called after clone, so it's necessary
            with torch.no_grad():
                actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False, current_splats=None)
                actual_scales = torch.exp(actual_splats["scales"]).max(dim=-1).values  # [new_N,]
                scale_threshold = self.grow_scale3d * state["scene_scale"]
                is_small = actual_scales <= scale_threshold  # [new_N,]
                is_large = ~is_small  # [new_N,]
                del actual_splats
            
            # Recompute is_root for new structure
            has_parent = (parent_indices >= 0) & (parent_indices < new_N)
            is_root = ~has_parent  # [new_N,]
            
            # Mark all cloned gaussians as already_densified
            # Clone된 parents와 children 모두 이미 clone되었으므로 추가 densification 불필요
            if cloned_indices_for_densify is not None and len(cloned_indices_for_densify) > 0:
                already_densified[cloned_indices_for_densify] = True
                
                # Reset gradients for all cloned gaussians
                for render_level in state.get("grad2d", {}):
                    if len(state["grad2d"][render_level]) == new_N:
                        state["grad2d"][render_level][cloned_indices_for_densify] = 0
        
        # ========== Step 7: Determine duplicate vs split and execute ==========
        
        # Ensure already_densified matches current structure size (may have changed after clone)
        current_N = len(levels)
        if len(already_densified) != current_N:
            new_already_densified = torch.zeros(current_N, dtype=torch.bool, device=device)
            old_size = len(already_densified)
            if old_size > 0:
                new_already_densified[:min(old_size, current_N)] = already_densified[:min(old_size, current_N)]
            already_densified = new_already_densified
        
        # Ensure should_densify matches current structure size
        if len(should_densify) != current_N:
            new_should_densify = torch.zeros(current_N, dtype=torch.bool, device=device)
            old_size = len(should_densify)
            if old_size > 0:
                new_should_densify[:min(old_size, current_N)] = should_densify[:min(old_size, current_N)]
            should_densify = new_should_densify
        
        # Recompute is_root for current structure (in case structure changed after clone)
        has_parent = (parent_indices >= 0) & (parent_indices < current_N)
        is_root = ~has_parent  # [current_N,]
        
        # Filter out already densified gaussians
        candidates = should_densify & ~already_densified
        
        # Determine action: duplicate (small scale) or split (large scale)
        is_duplicate = candidates & is_small
        is_split = candidates & is_large
        
        # Apply scale2d condition for split (if enabled)
        if step < self.refine_scale2d_stop_iter and "radii" in state:
            radii = state["radii"]
            if len(radii) == current_N:
                is_split = is_split | (candidates & (radii > self.grow_scale2d))
        
        # Separate root and non-root
        is_duplicate_root = is_duplicate & is_root
        is_duplicate_non_root = is_duplicate & ~is_root
        is_split_root = is_split & is_root
        is_split_non_root = is_split & ~is_root
        
        n_dupli_root = is_duplicate_root.sum().item()
        n_dupli_non_root = is_duplicate_non_root.sum().item()
        n_split_root = is_split_root.sum().item()
        n_split_non_root = is_split_non_root.sum().item()
        n_create_children = 0  # Children pre-created during initialization
        
        if self.verbose:
            print(f"  Densification: duplication(root={n_dupli_root}, non-root={n_dupli_non_root}), "
                  f"split(root={n_split_root}, non-root={n_split_non_root}), "
                  f"hierarchy_clones={n_hierarchy_clones}, create_children={n_create_children})")
        
        # Reset gradients for gaussians that will be densified (prevent excessive densification)
        if n_dupli_root > 0 or n_dupli_non_root > 0 or n_split_root > 0 or n_split_non_root > 0:
            is_densified = is_duplicate | is_split
            densified_indices = torch.where(is_densified)[0]
            
            # Reset gradients at all render levels
            for render_level in state.get("grad2d", {}):
                if len(state["grad2d"][render_level]) == N:
                    state["grad2d"][render_level][densified_indices] = 0
            
            already_densified[densified_indices] = True
        
        # Execute duplicate operations (combine root and non-root)
        if n_dupli_non_root > 0 or n_dupli_root > 0:
            # Combine masks
            is_duplicate_combined = is_duplicate_non_root | is_duplicate_root
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_duplicate_combined,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                max_level=max_level,
            )
            # Update structure after duplicate
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
        
        # Execute split operations (combine root and non-root)
        if n_split_non_root > 0 or n_split_root > 0:
            # Combine masks
            is_split_combined = is_split_non_root | is_split_root
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split_combined,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                revised_opacity=self.revised_opacity,
                max_level=max_level,
            )
            # Update structure after split
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
        
        # Ensure already_densified matches new structure size
        new_N = len(levels)
        if new_N != len(already_densified):
            new_already_densified = torch.zeros(new_N, dtype=torch.bool, device=device)
            old_size = len(already_densified)
            if old_size > 0:
                new_already_densified[:min(old_size, new_N)] = already_densified[:min(old_size, new_N)]
            already_densified = new_already_densified
        
        total_dupli = n_dupli_root + n_dupli_non_root
        total_split = n_split_root + n_split_non_root
        
        # Note: Zero-parameter children are added after all grow_gs and prune_gs operations
        # in post_cycle to ensure all parents have children even after pruning
        
        # Count gaussians added by hierarchy cloning (approximate: each clone adds parent + children)
        # This is approximate because we don't track exact count, but it's for reporting only
        return total_dupli + total_split + n_hierarchy_clones, n_create_children, already_densified

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """Prune gaussians based on opacity and scale (bottom-up: finest to coarsest).
        
        Prunes leaf nodes from finest level to coarsest level in a single pass.
        After pruning at each level, the structure is updated and leaf nodes are recalculated.
        """
        # Get hierarchical structure
        levels = state["levels"]  # [N,]
        parent_indices = state["parent_indices"]  # [N,]
        level_indices = state["level_indices"]  # Dict[int, List[int]]
        
        if len(levels) == 0:
            return 0
        
        # Get coarsest and finest levels
        if len(level_indices) == 0:
            return 0
        
        coarsest_level = min(level_indices.keys())
        finest_level = max(level_indices.keys())
        
        total_pruned = 0
        total_pruned_opacity = 0
        total_pruned_too_big = 0
        multigrid_gaussians = state.get("multigrid_gaussians", None)
        
        # Prune from finest to coarsest level
        for level in range(finest_level, coarsest_level - 1, -1):
            if level not in level_indices:
                continue
            
            # Get current structure (updated after each level's pruning)
            levels = state["levels"]  # [N,]
            parent_indices = state["parent_indices"]  # [N,]
            level_indices = state["level_indices"]  # Dict[int, List[int]]
            N = len(levels)
            
            if N == 0:
                break
            
            level_indices_tensor = torch.tensor(level_indices[level], dtype=torch.long, device=levels.device)
            if len(level_indices_tensor) == 0:
                continue
            
            # Filter out invalid indices (out of bounds)
            valid_indices_mask = (level_indices_tensor >= 0) & (level_indices_tensor < N)
            if not valid_indices_mask.any():
                del valid_indices_mask, level_indices_tensor
                continue
            level_indices_tensor = level_indices_tensor[valid_indices_mask]
            del valid_indices_mask
            
            # Count number of children for each gaussian (updated after each level's pruning)
            valid_parent_mask = (parent_indices != -1) & (parent_indices >= 0) & (parent_indices < N)
            if valid_parent_mask.any():
                valid_parent_indices = parent_indices[valid_parent_mask]
                num_child = torch.bincount(valid_parent_indices, minlength=N)
                del valid_parent_indices
            else:
                num_child = torch.zeros(N, dtype=torch.long, device=levels.device)
            del valid_parent_mask
            
            # Only prune leaf nodes (nodes with no children)
            is_leaf = (num_child == 0)
            
            if not is_leaf.any():
                del num_child, is_leaf, level_indices_tensor
                continue
            
            # Get actual opacities and scales (hierarchical) - recalculate after each pruning
            # No gradients needed for pruning decisions
            with torch.no_grad():
                # Always use current splats (no caching during pruning)
                actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False, current_splats=None)
                actual_opacities = torch.sigmoid(actual_splats["opacities"])
                actual_scales = torch.exp(actual_splats["scales"])  # Use residual scales for pruning
                # Free actual_splats immediately after extracting needed values
                del actual_splats
                
                # Prune conditions - recalculate after each pruning
                is_opacity_low = actual_opacities < self.prune_opa
                
                # Level-dependent scale threshold: coarser levels (lower level) need larger threshold
                # Image size is downsampled by 1/2 per level, so same 3D scale appears 2x larger in 2D
                # Therefore, threshold should increase by 2^(max_level - level)
                # Example: max_level=4, level=1 -> threshold_multiplier = 2^(4-1) = 8
                #          max_level=4, level=4 -> threshold_multiplier = 2^(4-4) = 1
                max_level = state.get("max_level", None)
                if max_level is None:
                    max_level = self.max_level
                
                # Calculate level-dependent threshold multiplier
                # For each gaussian at this level, apply the same multiplier
                threshold_multiplier = 2.0 ** (max_level - level)  # float
                level_threshold = self.prune_scale3d * state["scene_scale"] * threshold_multiplier
                
                is_too_big = actual_scales.max(dim=-1).values > level_threshold
                # Free intermediate tensors
                del actual_opacities, actual_scales
            
            # Combine conditions: prune if opacity is low OR scale is too big
            to_prune = is_opacity_low | is_too_big
            
            # Get leaf nodes at this level that should be pruned
            level_is_leaf = is_leaf[level_indices_tensor]
            level_to_prune = to_prune[level_indices_tensor]
            level_prune_mask = level_is_leaf & level_to_prune
            
            # Separate opacity and too_big conditions for leaf nodes
            level_is_opacity_low = is_opacity_low[level_indices_tensor]
            level_is_too_big = is_too_big[level_indices_tensor]
            level_opacity_prune_mask = level_is_leaf & level_is_opacity_low & ~level_is_too_big  # Only opacity, not too_big
            level_too_big_prune_mask = level_is_leaf & level_is_too_big  # too_big (may also have opacity_low)
            
            # Free intermediate tensors
            del level_is_leaf, level_to_prune
            
            if level_prune_mask.any():
                level_prune_indices = level_indices_tensor[level_prune_mask]
                prune_mask = torch.zeros(N, dtype=torch.bool, device=levels.device)
                prune_mask[level_prune_indices] = True
                
                n_pruned_level = prune_mask.sum().item()
                if n_pruned_level > 0:
                    # Count opacity-only and too_big pruning
                    n_pruned_opacity_level = level_opacity_prune_mask.sum().item()
                    n_pruned_too_big_level = level_too_big_prune_mask.sum().item()
                    
                    remove_mg(
                        params=params,
                        optimizers=optimizers,
                        state=state,
                        mask=prune_mask,
                    )
                    total_pruned += n_pruned_level
                    total_pruned_opacity += n_pruned_opacity_level
                    total_pruned_too_big += n_pruned_too_big_level
                    
                    # Structure is automatically updated by remove_mg
                    # Continue to next level with updated structure
                
                # Free pruning-related tensors after use
                del prune_mask, level_prune_indices, level_prune_mask, level_opacity_prune_mask, level_too_big_prune_mask
            
            # Free level-specific tensors after processing this level
            del to_prune, level_indices_tensor, num_child, is_leaf, is_opacity_low, is_too_big
            torch.cuda.empty_cache()
        
        if total_pruned > 0 and self.verbose:
            # Print level-wise counts after pruning
            levels_after = state["levels"]
            level_indices_after = state["level_indices"]
            print(f"After pruning (step {step}, {total_pruned} total): opacity={total_pruned_opacity}, too_big={total_pruned_too_big}")
            for level in sorted(level_indices_after.keys()):
                count = len(level_indices_after[level])
                print(f"  Level {level}: {count} gaussians")
        
        return total_pruned
