from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

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
    finest_grow_grad2d: float = 0.00015
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
    n_children_per_split: int = 10  # Number of children to create per split
    max_children_per_parent: int = 5  # Maximum number of children a parent can have
    max_level: int = 5  # Maximum level in the hierarchy

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
        
        for target_level in range(coarsest_level, finest_level + 1):
            if target_level not in level_indices:
                continue
            
            n_dupli_split, n_create_children = self._grow_gs(
                params, optimizers, state, step, target_level=target_level
            )
            
            total_n_dupli_split += n_dupli_split
            total_n_create_children += n_create_children
            
            if n_dupli_split > 0 or n_create_children > 0:
                any_densification = True
                # Update hierarchical structure after grow (structure may have changed)
                if "multigrid_gaussians" in state and state["multigrid_gaussians"] is not None:
                    multigrid_gaussians = state["multigrid_gaussians"]
                    state["levels"] = multigrid_gaussians.levels
                    state["parent_indices"] = multigrid_gaussians.parent_indices
                    state["level_indices"] = multigrid_gaussians.level_indices
                    # Update level_indices for next iteration
                    level_indices = state["level_indices"]
        
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
            if "count" in state:
                for render_level in state["count"]:
                    state["count"][render_level].zero_()
        else:
            # Normal reset after densification cycle (even if no densification occurred)
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
        
        # Initialize tensors for this render_level if not present
        if render_level not in state["grad2d"]:
            state["grad2d"][render_level] = torch.zeros(N, device=device)
        if render_level not in state["count"]:
            state["count"][render_level] = torch.zeros(N, dtype=torch.float32, device=device)
        
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

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        target_level: int,
        actual_splats: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[int, int]:
        """Grow gaussians with new hierarchical algorithm: coarse-to-fine with gradient inheritance.
        
        Algorithm:
        1. Pre-compute flags: is_node_full, is_parent_full, n_child, is_grad_high, is_max_level
        2. Process from coarsest level to finest level:
           - is_grad_high일 경우:
             * root이고 child가 없으면: child 생성
             * root이고 child가 있으면: duplication/split
             * root가 아니고 가능하면: duplication/split
             * root가 아니고 duplication/split 불가능하면: child 생성
             * child 생성도 불가능하면: parent의 duplication/split 호출
             * max level이면 child 생성 불가능
        3. After each level: inherit parent gradient to children
        
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
            return 0, 0
        
        # Use counts and gradients from target_level only
        count = state["count"][target_level]  # Use target_level count
        grad2d = state["grad2d"][target_level]  # Use target_level gradients
        device = grad2d.device
        
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
        
        # ========== Step 1: Pre-compute all flags ==========
        
        # Count number of children for each gaussian
        valid_parent_mask = (parent_indices != -1) & (parent_indices >= 0) & (parent_indices < N)
        if valid_parent_mask.any():
            valid_parent_indices = parent_indices[valid_parent_mask]
            n_child = torch.bincount(valid_parent_indices, minlength=N)
        else:
            n_child = torch.zeros(N, dtype=torch.long, device=device)
        
        # Compute average gradients (will be updated with inheritance)
        grads = grad2d / count.clamp_min(1)  # [N,]
        
        # Compute level-dependent gradient thresholds
        # Base: self.coarsest_grow_grad2d, Max level: self.finest_grow_grad2d
        # Linear interpolation: threshold = self.coarsest_grow_grad2d - (self.coarsest_grow_grad2d - self.finest_grow_grad2d) * (level - 1) / (max_level - 1)
        if max_level > 1:
            level_weights = (levels.float() - 1.0) / (max_level - 1.0)  # [N,] in [0, 1]
            level_thresholds = self.coarsest_grow_grad2d - (self.coarsest_grow_grad2d - self.finest_grow_grad2d) * level_weights  # [N,]
        else:
            level_thresholds = torch.full((N,), 0.0003, device=device)
        
        # ========== Step 2: Compute is_grad_high (will be updated after inheritance) ==========
        is_grad_high = grads > level_thresholds  # [N,]
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
            actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False)
            actual_scales = actual_splats["scales"]
            is_small = torch.exp(actual_scales).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]
            is_large = ~is_small
            # Free actual_splats immediately after extracting scales
            del actual_splats, actual_scales
        
        # ========== Step 3: Process densification from coarsest to finest level ==========
        
        # Track actions to take
        should_densify = torch.zeros(N, dtype=torch.bool, device=device)  # Flag for densification (duplicate/split will be determined later)
        is_create_children = torch.zeros(N, dtype=torch.bool, device=device)
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
            
            # ========== Inherit parent gradients for this level (before processing densification) ==========
            if level > 1:  # Only inherit for levels > 1 (level 1 has no parents)
                # Get children at this level and their parents
                level_children_parents = parent_indices[level_indices_tensor]  # [M,]
                valid_parent_mask_level = (level_children_parents != -1) & (level_children_parents >= 0) & (level_children_parents < N)
                
                if valid_parent_mask_level.any():
                    valid_children = level_indices_tensor[valid_parent_mask_level]
                    valid_parents = level_children_parents[valid_parent_mask_level]
                    
                    # Inherit: grad[children] += grad[parents] / n_child[parents]
                    # This propagates coarse level gradients to fine level children
                    parent_grads = grads[valid_parents]  # [M,] - average gradients
                    parent_n_child = n_child[valid_parents].float().clamp_min(1)  # [M,]
                    inherited_grad = parent_grads / parent_n_child  # [M,] - per-child gradient
                    
                    # Add to children's average gradients
                    grads[valid_children] += inherited_grad
                    
                    # Re-evaluate is_grad_high for children after inheritance
                    level_thresholds_children = level_thresholds[valid_children]
                    is_grad_high[valid_children] = grads[valid_children] > level_thresholds_children
            
            # Get flags for this level
            level_is_grad_high = is_grad_high[level_indices_tensor]
            level_is_root = is_root[level_indices_tensor]
            level_has_child = has_child[level_indices_tensor]
            level_is_node_full = is_node_full[level_indices_tensor]
            level_is_parent_full = is_parent_full[level_indices_tensor]
            level_is_max_level = is_max_level[level_indices_tensor]
            level_is_small = is_small[level_indices_tensor]
            level_is_large = is_large[level_indices_tensor]
            level_parent_indices = parent_indices[level_indices_tensor]
            
            # Filter to only process gaussians with high gradient
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
            
            # Root with child but not full: create children (if not max level)
            root_with_child_not_full_and_not_max = root_with_child_not_full & ~process_is_max_level  # [M,]
            is_create_children[process_indices[root_with_child_not_full_and_not_max]] = True
            
            # ========== Non-root node logic (vectorized) ==========
            non_root_mask = ~root_mask  # [M,]
            
            # Try densification first (if parent is not full)
            can_densify = non_root_mask & ~process_is_parent_full  # [M,]
            should_densify[process_indices[can_densify]] = True
            
            # Cannot densify, try create children
            cannot_densify = non_root_mask & process_is_parent_full  # [M,]
            can_create_children = cannot_densify & ~process_is_max_level & ~process_is_node_full  # [M,]
            is_create_children[process_indices[can_create_children]] = True
            
            # Cannot create children either, signal parent
            cannot_create_children = cannot_densify & (process_is_max_level | process_is_node_full)  # [M,]
            if cannot_create_children.any():
                # Get valid parent indices (within bounds)
                signal_parent_indices = process_parent_indices[cannot_create_children]  # [K,]
                valid_parent_mask = (signal_parent_indices >= 0) & (signal_parent_indices < N)  # [K,]
                
                if valid_parent_mask.any():
                    valid_signal_parents = signal_parent_indices[valid_parent_mask]  # [L,]
                    # Signal parent to densify
                    parent_duplicate_split[valid_signal_parents] = True
        
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
            
            # Can densify if parent is not full
            can_densify_signaled = ~non_root_signaled_parent_full  # [L,]
            # Can create children if not max level and not node full
            can_create_children_signaled = ~non_root_signaled_max_level & ~non_root_signaled_node_full  # [L,]
            
            # If can densify, add to should_densify and remove from parent_duplicate_split
            can_handle_densify = can_densify_signaled  # [L,]
            should_densify[non_root_signaled_indices[can_handle_densify]] = True
            parent_duplicate_split[non_root_signaled_indices[can_handle_densify]] = False
            
            # If can create children, add to is_create_children and remove from parent_duplicate_split
            can_handle_create = can_create_children_signaled & ~can_handle_densify  # [L,] - only if cannot densify
            is_create_children[non_root_signaled_indices[can_handle_create]] = True
            parent_duplicate_split[non_root_signaled_indices[can_handle_create]] = False
            
            # Combined: can handle if can densify or create children
            can_handle = can_handle_densify | can_handle_create  # [L,]
            
            # If cannot handle (cannot densify and cannot create children), signal parent
            cannot_handle = ~can_handle  # [L,]
            if cannot_handle.any():
                cannot_handle_indices = non_root_signaled_indices[cannot_handle]  # [M,]
                cannot_handle_parent_indices = non_root_signaled_parent_indices[cannot_handle]  # [M,]
                
                # Remove current signal (will be replaced by parent signal)
                parent_duplicate_split[cannot_handle_indices] = False
                
                # Signal parent (will be processed in next iteration at coarser level)
                valid_parent_mask = (cannot_handle_parent_indices >= 0) & (cannot_handle_parent_indices < N)  # [M,]
                if valid_parent_mask.any():
                    valid_parents = cannot_handle_parent_indices[valid_parent_mask]  # [P,]
                    parent_duplicate_split[valid_parents] = True
        
        # Handle parent duplicate/split signals (now propagated to root if needed)
        should_densify = should_densify | parent_duplicate_split
        
        # Determine duplicate vs split based on is_small / is_large
        is_duplicate = should_densify & is_small
        is_split_action = should_densify & is_large
        
        # Apply scale2d condition for split
        if step < self.refine_scale2d_stop_iter:
            is_split_action = is_split_action | (should_densify & (state["radii"] > self.grow_scale2d))
        
        n_dupli = is_duplicate.sum().item()
        n_split = is_split_action.sum().item()
        n_create_children = is_create_children.sum().item()
        
        print("duplication: ", n_dupli, "split: ", n_split, "create_children: ", n_create_children)

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
        
        return n_dupli + n_split, n_create_children

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
                actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False)
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
