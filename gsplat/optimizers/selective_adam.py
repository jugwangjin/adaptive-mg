import torch

from ..cuda._wrapper import adam


class SelectiveAdam(torch.optim.Adam):
    """
    A custom optimizer that extends the standard Adam optimizer by
    incorporating selective updates.

    This class is useful for situations where only a subset of parameters
    should be updated at each step, such as in sparse models or in cases where
    parameter visibility is controlled by an external mask.

    Additionally, the operations are fused into a single kernel. This optimizer
    leverages the `adam` function from a CUDA backend for
    optimized sparse updates.

    This is one of the two optimizers mentioned in the Taming3DGS paper.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).

    Examples:

        >>> N = 100
        >>> param = torch.randn(N, requires_grad=True)
        >>> optimizer = SelectiveAdam([param], eps=1e-8, betas=(0.9, 0.999))
        >>> visibility_mask = torch.cat([torch.ones(50), torch.zeros(50)])  # Visible first half, hidden second half

        >>> # Forward pass
        >>> loss = torch.sum(param ** 2)

        >>> # Backward pass
        >>> loss.backward()

        >>> # Optimization step with selective updates
        >>> optimizer.step(visibility=visibility_mask)

    """

    def __init__(self, params, eps, betas, **kwargs):
        # Ignore unsupported Adam kwargs (e.g., fused) to match torch.optim.Adam signature usage
        super().__init__(params=params, eps=eps, betas=betas)

    @torch.no_grad()
    def step(self, visibility):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue
            if visibility.device != param.device:
                visibility = visibility.to(param.device)
            visibility = visibility.view(-1)
            N = visibility.numel()

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N

            if param.is_cuda:
                adam(
                    param,
                    param.grad,
                    exp_avg,
                    exp_avg_sq,
                    visibility,
                    lr,
                    beta1,
                    beta2,
                    eps,
                )
            else:
                if visibility.dtype != torch.bool:
                    visible_mask = visibility != 0
                else:
                    visible_mask = visibility
                if not visible_mask.any():
                    continue
                visible_indices = torch.where(visible_mask)[0]
                param_view = param.view(N, -1)
                grad_view = param.grad.view(N, -1)
                exp_avg_view = exp_avg.view(N, -1)
                exp_avg_sq_view = exp_avg_sq.view(N, -1)
                grad_sel = grad_view[visible_indices]
                exp_avg_sel = exp_avg_view[visible_indices]
                exp_avg_sq_sel = exp_avg_sq_view[visible_indices]
                exp_avg_sel = beta1 * exp_avg_sel + (1.0 - beta1) * grad_sel
                exp_avg_sq_sel = beta2 * exp_avg_sq_sel + (1.0 - beta2) * grad_sel * grad_sel
                denom = exp_avg_sq_sel.sqrt().add_(eps)
                param_view[visible_indices] = param_view[visible_indices] - lr * (
                    exp_avg_sel / denom
                )
                exp_avg_view[visible_indices] = exp_avg_sel
                exp_avg_sq_view[visible_indices] = exp_avg_sq_sel
