from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import torch
from torch import Tensor
from typing_extensions import Literal

from .base import Strategy
from .ops_mg_v2 import create_children_mg, duplicate, remove_mg, reset_opa_mg, split


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
    coarsest_grow_grad2d: float = 0.0002 # Base gradient threshold (level 1)
    finest_grow_grad2d: float = 0.0002
    grow_color: float = 9999.
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
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
        
        # Initialize grad2d, grad_color, and count as dicts for render_level-based accumulation
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
            "grad_color": {},  # Dict[int, Tensor] - render_level -> accumulated color gradients
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
        
        # Grow gaussians: iterate from coarsest to finest level
        total_n_dupli_split = 0
        total_n_create_children = 0
        any_densification = False
        
        # Initialize already_densified at post_cycle level to share across all target_levels
        already_densified = torch.zeros_like(multigrid_gaussians.levels, dtype=torch.bool, device=multigrid_gaussians.levels.device)
        
        for target_level in range(coarsest_level, finest_level + 1):
            if target_level not in level_indices:
                continue

            n_dupli_split, n_create_children, already_densified = self._grow_gs(
                params, optimizers, state, step, target_level=target_level, already_densified=already_densified
            )
            
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
                    
                    # Update level_indices for next iteration
                    level_indices = state["level_indices"]
                    
                    # Expand grad2d and count tensors if gaussian count increased
                    if new_N > old_N:
                        device = state["levels"].device
                        # Expand all render_level tensors
                        if "grad2d" in state:
                            for render_level in state["grad2d"]:
                                old_tensor = state["grad2d"][render_level]
                                new_tensor = torch.zeros(new_N, device=device, dtype=old_tensor.dtype)
                                new_tensor[:old_N] = old_tensor
                                state["grad2d"][render_level] = new_tensor
                        if "grad_color" in state:
                            for render_level in state["grad_color"]:
                                old_tensor = state["grad_color"][render_level]
                                new_tensor = torch.zeros(new_N, device=device, dtype=old_tensor.dtype)
                                new_tensor[:old_N] = old_tensor
                                state["grad_color"][render_level] = new_tensor
                        if "count" in state:
                            for render_level in state["count"]:
                                old_tensor = state["count"][render_level]
                                new_tensor = torch.zeros(new_N, device=device, dtype=old_tensor.dtype)
                                new_tensor[:old_N] = old_tensor
                                state["count"][render_level] = new_tensor
                        if "radii" in state:
                            old_radii = state["radii"]
                            new_radii = torch.zeros(new_N, device=device, dtype=old_radii.dtype)
                            new_radii[:old_N] = old_radii
                            state["radii"] = new_radii
        
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
        if any_densification:
            if "grad2d" in state:
                for render_level in state["grad2d"]:
                    state["grad2d"][render_level].zero_()
            if "grad_color" in state:
                for render_level in state["grad_color"]:
                    state["grad_color"][render_level].zero_()
            if "count" in state:
                for render_level in state["count"]:
                    state["count"][render_level].zero_()
        else:
            # Normal reset after densification cycle (even if no densification occurred)
            if "grad2d" in state:
                for render_level in state["grad2d"]:
                    state["grad2d"][render_level].zero_()
            if "grad_color" in state:
                for render_level in state["grad_color"]:
                    state["grad_color"][render_level].zero_()
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
        
        # Initialize grad2d, grad_color, and count dicts if not present
        if "grad2d" not in state:
            state["grad2d"] = {}
        if "grad_color" not in state:
            state["grad_color"] = {}
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
        
        if render_level not in state["grad_color"]:
            state["grad_color"][render_level] = torch.zeros(N, device=device)
        elif len(state["grad_color"][render_level]) != N:
            # Resize if gaussian count changed (e.g., after densification)
            old_tensor = state["grad_color"][render_level]
            new_tensor = torch.zeros(N, device=device, dtype=old_tensor.dtype)
            old_N = len(old_tensor)
            if old_N > 0:
                new_tensor[:min(old_N, N)] = old_tensor[:min(old_N, N)]
            state["grad_color"][render_level] = new_tensor
        
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
        
        # Extract color gradients if available
        has_colors_grad = "colors" in info and info["colors"].grad is not None
        
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
            visible_grads = grads[gaussian_ids]  # [nnz, 2]
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
            
            # Accumulate color gradients if available
            if has_colors_grad:
                # colors is [C, N, 3] after SH computation in non-packed mode
                color_grads = info["colors"].grad[sel]  # [nnz, 3]
                color_grad_norms = color_grads.norm(dim=-1)  # [nnz,]
                state["grad_color"][render_level].index_add_(0, full_gs_ids, color_grad_norms)
        
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
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        target_level: int,
        actual_splats: Optional[Dict[str, Tensor]] = None,
        already_densified: Optional[Tensor] = None,
    ) -> Tuple[int, int]:
        """Grow gaussians with new hierarchical algorithm: coarse-to-fine with gradient inheritance.
        
        Algorithm:
        1. Pre-compute flags: is_node_full, is_parent_full, n_child, is_grad_high, is_max_level
        2. Process from coarsest level to finest level:
           - is_grad_high일 경우:
             * root이고 child가 없으면: child 생성
             * root이고 child가 있고 full이면: duplication/split
             * root이고 child가 있지만 full이 아니면: child 생성 (grad2d 기반)
             * root가 아니고 parent가 full이 아니면: duplication/split
             * root가 아니고 parent가 full이고 child가 없으면: child 생성
             * root가 아니고 parent가 full이고 child가 있지만 full이 아니면: do_nothing
             * root가 아니고 parent가 full이고 child가 full이면: parent에 signal
             * max level이면 child 생성 불가능
        3. After each level: inherit parent gradient to children (if enabled)
        
        Args:
            target_level: The render level to use for gradient/count accumulation.
                          Only gradients/counts from this render_level will be used.
        """
        # Use gradients and counts from target_level only
        if "count" not in state or target_level not in state["count"]:
            # No gradients accumulated for this render_level yet, skip densification
            return 0, 0
        
        if "grad2d" not in state or target_level not in state["grad2d"]:
            # No gradients accumulated for this render_level yet, skip densification
            return 0, 0, already_densified
        
        # Use counts and gradients from target_level only
        count = state["count"][target_level]  # Use target_level count
        grad2d = state["grad2d"][target_level]  # Use target_level gradients
        device = grad2d.device
        
        # Get color gradients if available
        if "grad_color" in state and target_level in state["grad_color"]:
            grad_color = state["grad_color"][target_level]  # Use target_level color gradients
        else:
            grad_color = torch.zeros_like(grad2d)  # No color gradients available
        
        # Get hierarchical structure
        levels = state["levels"]  # [N,]
        parent_indices = state["parent_indices"]  # [N,]
        level_indices = state["level_indices"]  # Dict[int, List[int]]
        # max_level should always come from state (which is set from cfg.max_level)
        # If not in state, use strategy's max_level attribute
        max_level = state.get("max_level", None)
        
        if max_level is None:
            # Fallback to strategy's max_level (which should be set from cfg.max_level)
            max_level = self.max_level
        
        N = len(levels)
        
        # Ensure already_densified matches the current gaussian count
        if already_densified is not None:
            if len(already_densified) != N:
                # Resize already_densified to match current N
                old_already_densified = already_densified
                already_densified = torch.zeros(N, dtype=torch.bool, device=levels.device)
                old_N = len(old_already_densified)
                if old_N > 0:
                    already_densified[:min(old_N, N)] = old_already_densified[:min(old_N, N)]
        else:
            already_densified = torch.zeros(N, dtype=torch.bool, device=levels.device)
        
        # Ensure grad2d and count match the current gaussian count
        # This can happen if densification occurred but tensors weren't resized yet
        if len(grad2d) != N:
            # Resize grad2d to match current gaussian count
            old_grad2d = grad2d
            new_grad2d = torch.zeros(N, device=device, dtype=old_grad2d.dtype)
            old_N = len(old_grad2d)
            if old_N > 0:
                new_grad2d[:min(old_N, N)] = old_grad2d[:min(old_N, N)]
            state["grad2d"][target_level] = new_grad2d
            grad2d = new_grad2d
        
        if len(count) != N:
            # Resize count to match current gaussian count
            old_count = count
            new_count = torch.zeros(N, device=device, dtype=old_count.dtype)
            old_N = len(old_count)
            if old_N > 0:
                new_count[:min(old_N, N)] = old_count[:min(old_N, N)]
            state["count"][target_level] = new_count
            count = new_count
        
        if len(grad_color) != N:
            # Resize grad_color to match current gaussian count
            old_grad_color = grad_color
            new_grad_color = torch.zeros(N, device=device, dtype=old_grad_color.dtype)
            old_N = len(old_grad_color)
            if old_N > 0:
                new_grad_color[:min(old_N, N)] = old_grad_color[:min(old_N, N)]
            state["grad_color"][target_level] = new_grad_color
            grad_color = new_grad_color

        # ========== Step 1: Pre-compute all flags ==========
        
        # Count number of children for each gaussian
        valid_parent_mask = (parent_indices != -1) & (parent_indices >= 0) & (parent_indices < N)
        if valid_parent_mask.any():
            valid_parent_indices = parent_indices[valid_parent_mask]
            n_child = torch.bincount(valid_parent_indices, minlength=N)
        else:
            n_child = torch.zeros(N, dtype=torch.long, device=device)
        
        # Compute average gradients (normalized by count)
        grads = grad2d / count.clamp_min(1)  # [N,]
        color_grads = grad_color / count.clamp_min(1)  # [N,] - average color gradients
        
        # ========== Parent->Children Gradient Inheritance ==========
        # Propagate parent gradients to children using: child_grad += (parent_grad / radii[parent] * radii[child]) / num_child[parent]
        # This gives larger children more gradient (proportional to their size)
        if self.use_gradient_inheritance:
            # Get radii for all gaussians (use state["radii"] which accumulates max radii)
            all_radii = state["radii"]  # [N,] - max radii for all gaussians
            
            # Process from coarsest to finest level
            if len(level_indices) > 0:
                coarsest_level = min(level_indices.keys())
                finest_level = max(level_indices.keys())
                
                for level in range(coarsest_level, finest_level + 1):
                    if level not in level_indices or level == 1:
                        continue  # Skip level 1 (no parents)
                    
                    # Get gaussians at this level
                    level_indices_list = level_indices[level]
                    if len(level_indices_list) == 0:
                        continue
                    
                    level_indices_tensor = torch.tensor(level_indices_list, dtype=torch.long, device=device)
                    
                    # Filter out invalid indices
                    valid_indices_mask = (level_indices_tensor >= 0) & (level_indices_tensor < N)
                    if not valid_indices_mask.any():
                        continue
                    level_indices_tensor = level_indices_tensor[valid_indices_mask]
                    
                    # Get parent indices for children at this level
                    level_parent_indices = parent_indices[level_indices_tensor]  # [M,]
                    valid_parent_mask = (level_parent_indices != -1) & (level_parent_indices >= 0) & (level_parent_indices < N)
                    
                    if not valid_parent_mask.any():
                        continue
                    
                    valid_children = level_indices_tensor[valid_parent_mask]  # [K,]
                    valid_parents = level_parent_indices[valid_parent_mask]  # [K,]
                    
                    # Get parent gradients (already normalized) and radii
                    parent_grads = grads[valid_parents]  # [K,] - parent gradients (normalized)
                    parent_radii = all_radii[valid_parents]  # [K,] - parent radii
                    child_radii = all_radii[valid_children]  # [K,] - child radii
                    
                    # Only inherit gradient if both parent and child are visible (radii > 0)
                    # In rendering.py, only radii > 0 are considered visible
                    # If radii < 1e-8, set inherit grad to 0
                    visible_mask = (parent_radii >= 1e-8) & (child_radii >= 1e-8)  # [K,]
                    
                    if visible_mask.any():
                        # Compute inherited gradient only for visible pairs
                        # Use clamp_min to avoid division by zero for visible pairs
                        parent_radii_visible = parent_radii[visible_mask].clamp_min(1e-6)  # [K_visible,]
                        child_radii_visible = child_radii[visible_mask].clamp_min(1e-6)  # [K_visible,]
                        parent_grads_visible = parent_grads[visible_mask]  # [K_visible,]
                        parent_n_child_visible = n_child[valid_parents[visible_mask]].float().clamp_min(1.0)  # [K_visible,] - number of children per parent
                        
                        # Compute inherited gradient: (parent_grad / parent_radii * child_radii) / num_child[parent]
                        # This gives larger children more gradient proportional to their size
                        # Normalize by num_child to distribute parent gradient evenly among children
                        inherited_grad_visible = ((parent_grads_visible / parent_radii_visible) * child_radii_visible) / parent_n_child_visible  # [K_visible,]
                        
                        # Initialize inherited_grad with zeros, then fill in visible ones
                        inherited_grad = torch.zeros_like(parent_grads)  # [K,]
                        inherited_grad[visible_mask] = inherited_grad_visible
                        
                        # Add inherited gradient to children
                        grads[valid_children] += inherited_grad
        
        # Compute target_level-based gradient thresholds using linear interpolation
        # Base threshold from target_level: Higher target_level (finest) -> lower base threshold
        # Use linear interpolation between coarsest and finest thresholds based on each gaussian's level
        if max_level > 1:
            # For each gaussian, compute threshold based on its level using linear interpolation
            # level=1 (coarsest): threshold = coarsest_grow_grad2d
            # level=max_level (finest): threshold = finest_grow_grad2d
            max_level_float = float(max_level)
            level_weights = (levels.float() - 1.0) / max(max_level_float - 1.0, 1.0)  # [N,] in [0, 1]
            level_thresholds = self.coarsest_grow_grad2d - (self.coarsest_grow_grad2d - self.finest_grow_grad2d) * level_weights  # [N,]
        else:
            level_thresholds = torch.full((N,), 0.0002, device=device)
        
        # ========== Step 2: Compute is_grad_high (integrated: grad2d or color) ==========
        # Check grad2d and color gradients separately with their respective thresholds
        is_grad2d_high = grads > level_thresholds  # [N,] - grad2d threshold (level-dependent)
        is_color_grad_high = color_grads > self.grow_color  # [N,] - color gradient threshold
        # Integrate: either grad2d or color gradient high triggers densification
        is_grad_high = is_grad2d_high | is_color_grad_high  # [N,]
        is_max_level = (levels >= max_level) if max_level is not None else torch.zeros(N, dtype=torch.bool, device=device)
        is_node_full = n_child >= self.max_children_per_parent  # [N,]
        
        # Check if parent is full
        has_parent = (parent_indices != -1) & (parent_indices >= 0) & (parent_indices < N)
        is_parent_full = torch.zeros(N, dtype=torch.bool, device=device)
        if has_parent.any():
            parent_ids = parent_indices[has_parent]
            parent_n_child = n_child[parent_ids]
            is_parent_full[has_parent] = parent_n_child >= self.max_children_per_parent
        
        is_root = ~has_parent  # [N,]
        has_child = (n_child > 0)  # [N,]
        
        # Get actual scales for scale-based decisions
        # No gradients needed for densification decisions
        multigrid_gaussians = state.get("multigrid_gaussians", None)
        with torch.no_grad():
            # Always use current splats (no caching during densification)
            actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False, current_splats=None)
            actual_scales = actual_splats["scales"]
            # Apply level-dependent threshold multiplier (similar to too_big)
            # Level-dependent scale threshold: coarser levels (lower level) need larger threshold
            # Image size is downsampled by 1/2 per level, so same 3D scale appears 2x larger in 2D
            # Therefore, threshold should increase by 2^(max_level - level)
            # Example: max_level=4, level=1 -> threshold_multiplier = 2^(4-1) = 8
            #          max_level=4, level=4 -> threshold_multiplier = 2^(4-4) = 1
            max_level_for_scale = state.get("max_level", None)
            if max_level_for_scale is None:
                max_level_for_scale = self.max_level if hasattr(self, 'max_level') else max_level
            
            # Calculate level-dependent threshold multiplier for each gaussian
            scale_threshold_multipliers = 2.0 ** (max_level_for_scale - target_level)  # [1]
            # scale_threshold_multipliers = 2.0 ** (max_level_for_scale - levels.float())  # [N,]
            scale_thresholds = self.grow_scale3d * state["scene_scale"] * scale_threshold_multipliers  # [N,]
            is_small = torch.exp(actual_scales).max(dim=-1).values <= scale_thresholds
            is_large = ~is_small
            # Free actual_splats immediately after extracting scales
            del actual_splats, actual_scales
        
        # ========== Step 3: Process densification from coarsest to finest level ==========
        
        # Track actions to take
        should_densify = torch.zeros(N, dtype=torch.bool, device=device)  # Flag for densification (duplicate/split will be determined later)
        is_create_children = torch.zeros(N, dtype=torch.bool, device=device)  # From integrated gradient (grad2d or color)
        parent_duplicate_split = torch.zeros(N, dtype=torch.bool, device=device)  # Signal parent to densify
        
        # Get coarsest and finest levels
        if len(level_indices) > 0:
            coarsest_level = min(level_indices.keys())
            finest_level = max(level_indices.keys())
        else:
            coarsest_level = 1
            finest_level = 1
        
        # Process from coarsest to finest
        for level in range(coarsest_level, finest_level + 1):
            if level not in level_indices:
                continue
            
            level_indices_tensor = torch.tensor(level_indices[level], dtype=torch.long, device=device)
            if len(level_indices_tensor) == 0:
                continue
            
            # Filter out invalid indices (out of bounds)
            valid_indices_mask = (level_indices_tensor >= 0) & (level_indices_tensor < N)
            if not valid_indices_mask.any():
                continue
            level_indices_tensor = level_indices_tensor[valid_indices_mask]
            
            # Note: Gradient inheritance is now done in _update_state using radii-based formula
            # No need to inherit here as it's already done during gradient accumulation
            
            # Get flags for this level
            level_is_grad_high = is_grad_high[level_indices_tensor]  # [M,] - integrated gradient (grad2d or color)
            level_is_root = is_root[level_indices_tensor]
            level_has_child = has_child[level_indices_tensor]
            level_is_node_full = is_node_full[level_indices_tensor]
            level_is_parent_full = is_parent_full[level_indices_tensor]
            level_is_max_level = is_max_level[level_indices_tensor]
            level_is_small = is_small[level_indices_tensor]
            level_is_large = is_large[level_indices_tensor]
            level_parent_indices = parent_indices[level_indices_tensor]
            
            # Filter to only process gaussians with high gradient (integrated: grad2d or color)
            grad_high_mask = level_is_grad_high
            if not grad_high_mask.any():
                continue
            
            # Get indices of gaussians to process
            process_indices = level_indices_tensor[grad_high_mask]  # [M,]
            process_is_root = level_is_root[grad_high_mask]  # [M,]
            process_has_child = level_has_child[grad_high_mask]  # [M,]
            process_is_node_full = level_is_node_full[grad_high_mask]  # [M,]
            process_is_parent_full = level_is_parent_full[grad_high_mask]  # [M,]
            process_is_max_level = level_is_max_level[grad_high_mask]  # [M,]
            process_is_small = is_small[process_indices]  # [M,] - get from original arrays
            process_is_large = is_large[process_indices]  # [M,]
            process_parent_indices = level_parent_indices[grad_high_mask]  # [M,]
            
            # ========== Root node logic (vectorized) ==========
            root_mask = process_is_root  # [M,]
            
            # Root with no child: create children (if not max level)
            root_no_child = root_mask & ~process_has_child & ~process_is_max_level  # [M,]
            is_create_children[process_indices[root_no_child]] = True
            
            # Root with child: check if child count exceeds max_children_per_parent
            root_with_child = root_mask & process_has_child  # [M,]
            # If child count >= max_children_per_parent, use duplicate/split instead of creating more children
            root_with_child_full = root_with_child & process_is_node_full  # [M,]
            root_with_child_not_full = root_with_child & ~process_is_node_full  # [M,]
            
            # Root with child and full: densify (duplicate/split will be determined later)
            should_densify[process_indices[root_with_child_full]] = True
            
            # Root with child but not full: create children (if not max level) - for integrated gradient
            root_with_child_not_full_and_not_max = root_with_child_not_full & ~process_is_max_level  # [M,]
            is_create_children[process_indices[root_with_child_not_full_and_not_max]] = True
            
            # ========== Non-root node logic (vectorized) ==========
            non_root_mask = ~root_mask  # [M,]
            
            # Try densification first (if parent is not full)
            can_densify = non_root_mask & ~process_is_parent_full  # [M,]
            should_densify[process_indices[can_densify]] = True
            
            # Cannot densify (parent is full), check child status
            cannot_densify = non_root_mask & process_is_parent_full  # [M,]
            
            # Case 1: No child and not max level -> create children (for integrated gradient)
            no_child = cannot_densify & ~process_has_child & ~process_is_max_level  # [M,]
            is_create_children[process_indices[no_child]] = True
            
            # Case 2: Has child but not full -> do_nothing (no action)
            # This is handled implicitly by not adding to any action mask
            
            # Case 3: Has child and full -> signal parent
            has_child_and_full = cannot_densify & process_has_child & process_is_node_full  # [M,]
            if has_child_and_full.any():
                signal_parent_indices = process_parent_indices[has_child_and_full]  # [K,]
                valid_parent_mask = (signal_parent_indices >= 0) & (signal_parent_indices < N)  # [K,]
                
                if valid_parent_mask.any():
                    valid_signal_parents = signal_parent_indices[valid_parent_mask]  # [L,]
                    # Only signal parent if it's not already densified
                    parent_not_densified = ~already_densified[valid_signal_parents]  # [L,]
                    if parent_not_densified.any():
                        parent_duplicate_split[valid_signal_parents[parent_not_densified]] = True
            
            # Case 4: Cannot densify and max level -> signal parent
            cannot_densify_max_level = cannot_densify & process_is_max_level  # [M,]
            if cannot_densify_max_level.any():
                signal_parent_indices = process_parent_indices[cannot_densify_max_level]  # [K,]
                valid_parent_mask = (signal_parent_indices >= 0) & (signal_parent_indices < N)  # [K,]
                
                if valid_parent_mask.any():
                    valid_signal_parents = signal_parent_indices[valid_parent_mask]  # [L,]
                    # Only signal parent if it's not already densified
                    parent_not_densified = ~already_densified[valid_signal_parents]  # [L,]
                    if parent_not_densified.any():
                        parent_duplicate_split[valid_signal_parents[parent_not_densified]] = True
        
        # ========== Recursively propagate parent signals to root ==========
        # Process from finest to coarsest level: if a parent receives a signal but cannot
        # densify/create children, signal its parent. This continues until root node is reached.
        # Process from finest to coarsest level (reverse order)
        for level in range(finest_level, coarsest_level - 1, -1):
            if level not in level_indices:
                continue
            
            # Get gaussians at this level that received signals
            level_indices_tensor = torch.tensor(level_indices[level], dtype=torch.long, device=device)
            if len(level_indices_tensor) == 0:
                continue
            
            # Filter out invalid indices (out of bounds)
            valid_indices_mask = (level_indices_tensor >= 0) & (level_indices_tensor < N)
            if not valid_indices_mask.any():
                continue
            level_indices_tensor = level_indices_tensor[valid_indices_mask]
            
            # Get signaled gaussians at this level
            level_signaled_mask = parent_duplicate_split[level_indices_tensor]  # [M,]
            if not level_signaled_mask.any():
                continue
            
            signaled_gaussians = level_indices_tensor[level_signaled_mask]  # [K,]
            
            # Check if these signaled gaussians can densify or create children
            signaled_is_root = is_root[signaled_gaussians]  # [K,]
            signaled_parent_indices = parent_indices[signaled_gaussians]  # [K,]
            signaled_is_node_full = is_node_full[signaled_gaussians]  # [K,]
            signaled_is_max_level = is_max_level[signaled_gaussians]  # [K,]
            signaled_is_parent_full = is_parent_full[signaled_gaussians]  # [K,]
            signaled_has_child = has_child[signaled_gaussians]  # [K,]
            
            # Root nodes: can always densify (duplicate/split), so no need to signal further
            root_signaled = signaled_is_root  # [K,]
            if root_signaled.any():
                root_signaled_indices = signaled_gaussians[root_signaled]  # [L,]
                should_densify[root_signaled_indices] = True
                parent_duplicate_split[root_signaled_indices] = False
            
            # Non-root nodes: check if they can densify or create children
            non_root_signaled = ~signaled_is_root  # [K,]
            if not non_root_signaled.any():
                continue  # All handled, move to next level
            
            non_root_signaled_indices = signaled_gaussians[non_root_signaled]  # [L,]
            non_root_signaled_parent_full = signaled_is_parent_full[non_root_signaled]  # [L,]
            non_root_signaled_node_full = signaled_is_node_full[non_root_signaled]  # [L,]
            non_root_signaled_max_level = signaled_is_max_level[non_root_signaled]  # [L,]
            non_root_signaled_parent_indices = signaled_parent_indices[non_root_signaled]  # [L,]
            non_root_signaled_has_child = signaled_has_child[non_root_signaled]  # [L,]
            
            # Can densify if parent is not full
            can_densify_signaled = ~non_root_signaled_parent_full  # [L,]
            should_densify[non_root_signaled_indices[can_densify_signaled]] = True
            parent_duplicate_split[non_root_signaled_indices[can_densify_signaled]] = False
            
            # Cannot densify (parent is full), check child status
            cannot_densify_signaled = non_root_signaled_parent_full  # [L,]
            
            # Case 1: No child and not max level -> create children
            no_child_signaled = cannot_densify_signaled & ~non_root_signaled_has_child & ~non_root_signaled_max_level  # [L,]
            is_create_children[non_root_signaled_indices[no_child_signaled]] = True
            parent_duplicate_split[non_root_signaled_indices[no_child_signaled]] = False
            
            # Case 2: Has child but not full -> do_nothing (no action, remove signal)
            has_child_not_full_signaled = cannot_densify_signaled & non_root_signaled_has_child & ~non_root_signaled_node_full  # [L,]
            parent_duplicate_split[non_root_signaled_indices[has_child_not_full_signaled]] = False
            
            # Case 3: Has child and full -> signal parent
            has_child_and_full_signaled = cannot_densify_signaled & non_root_signaled_has_child & non_root_signaled_node_full  # [L,]
            if has_child_and_full_signaled.any():
                signal_parent_indices = non_root_signaled_parent_indices[has_child_and_full_signaled]  # [M,]
                valid_parent_mask = (signal_parent_indices >= 0) & (signal_parent_indices < N)  # [M,]
                
                if valid_parent_mask.any():
                    valid_parents = signal_parent_indices[valid_parent_mask]  # [P,]
                    parent_not_densified = ~already_densified[valid_parents]  # [P,]
                    if parent_not_densified.any():
                        parent_duplicate_split[valid_parents[parent_not_densified]] = True
                
                # Remove current signal (will be replaced by parent signal)
                parent_duplicate_split[non_root_signaled_indices[has_child_and_full_signaled]] = False
            
            # Case 4: Cannot densify and max level -> signal parent
            cannot_densify_max_level_signaled = cannot_densify_signaled & non_root_signaled_max_level  # [L,]
            if cannot_densify_max_level_signaled.any():
                signal_parent_indices = non_root_signaled_parent_indices[cannot_densify_max_level_signaled]  # [M,]
                valid_parent_mask = (signal_parent_indices >= 0) & (signal_parent_indices < N)  # [M,]
                
                if valid_parent_mask.any():
                    valid_parents = signal_parent_indices[valid_parent_mask]  # [P,]
                    parent_not_densified = ~already_densified[valid_parents]  # [P,]
                    if parent_not_densified.any():
                        parent_duplicate_split[valid_parents[parent_not_densified]] = True
                
                # Remove current signal (will be replaced by parent signal)
                parent_duplicate_split[non_root_signaled_indices[cannot_densify_max_level_signaled]] = False
        
        # Handle parent duplicate/split signals (now propagated to root if needed)
        should_densify = should_densify | parent_duplicate_split
        
        # Determine duplicate vs split based on is_small / is_large
        is_duplicate = should_densify & is_small & ~already_densified
        is_split_action = should_densify & is_large & ~already_densified
        
        # Apply scale2d condition for split
        if step < self.refine_scale2d_stop_iter:
            is_split_action = is_split_action | (should_densify & (state["radii"] > self.grow_scale2d))
        
        n_dupli = is_duplicate.sum().item()
        n_split = is_split_action.sum().item()
        # Exclude already densified
        is_create_children = is_create_children & ~already_densified
        n_create_children = is_create_children.sum().item()
        
        print("duplication: ", n_dupli, "split: ", n_split, "create_children: ", n_create_children)

        # Reset grad2d and grad_color for all densified gaussians BEFORE densification to prevent excessive densification
        # This is simpler than tracking indices after densification
        if (n_dupli > 0 or n_split > 0 or n_create_children > 0):
            # Combine all densification actions
            is_densified = is_duplicate | is_split_action | is_create_children  # [N,]
            densified_indices = torch.where(is_densified)[0]  # Indices of gaussians that will be densified
            
            # Reset grad2d for all densified gaussians at all render levels
            if "grad2d" in state:
                for render_level in state["grad2d"]:
                    grad2d_tensor = state["grad2d"][render_level]
                    if len(grad2d_tensor) == N:
                        # Reset grad2d for all densified gaussians
                        grad2d_tensor[densified_indices] = 0
            
            # Reset grad_color for all densified gaussians at all render levels
            if "grad_color" in state:
                for render_level in state["grad_color"]:
                    grad_color_tensor = state["grad_color"][render_level]
                    if len(grad_color_tensor) == N:
                        # Reset grad_color for all densified gaussians
                        grad_color_tensor[densified_indices] = 0
            
            already_densified[densified_indices] = True
        
        # Execute actions
        if n_dupli > 0:
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_duplicate,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
            )
            # Update after duplicate
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
        
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split_action,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                revised_opacity=self.revised_opacity,
            )
            # Update after split
            levels = state["levels"]
            parent_indices = state["parent_indices"]
            level_indices = state["level_indices"]
        
        if n_create_children > 0:
            create_children_mg(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_create_children,
                levels=levels,
                parent_indices=parent_indices,
                level_indices=level_indices,
                n_children_per_split=self.n_children_per_split,
            )
        

        already_densified_new = torch.zeros_like(levels, dtype=torch.bool, device=levels.device)
        already_densified_new[:len(already_densified)] = already_densified

        return n_dupli + n_split, n_create_children, already_densified_new

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
