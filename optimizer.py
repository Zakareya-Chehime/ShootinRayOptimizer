import torch
import numpy as np
import logging
from typing import Callable, List, Tuple, Optional

__all__ = ["RayShootingOptimizer"]

logger = logging.getLogger(__name__)


class RayShootingOptimizer:
    """
    Combines gradient ascent with a geometric ray-shooting search to escape
    local maxima on multimodal loss surfaces.

    After climbing to a local maximum, rays are fired outward in evenly-spaced
    directions. Each ray walks until it finds a region meaningfully higher than
    the current position. Gradient ascent then runs from the top-k landing
    points by value. This repeats until no improvement is found or
    max_iterations is reached.

    Ray shooting is vectorized: all ray positions at each walk step are
    evaluated in a single batched call to f_batch. f_batch must accept a
    (num_rays, d) tensor and return a (num_rays,) tensor. A scalar wrapper
    f is still used for gradient ascent.

    Args:
        lr:                  Learning rate for gradient ascent.
        ascent_steps:        Max gradient ascent steps per climb.
        num_rays:            Number of rays fired per iteration.
        ray_step_size:       Step size along each ray direction.
        ray_max_steps:       Max walk steps per ray.
        top_k_landings:      Only run gradient ascent from the top-k landing
                             points by value. Reduces ascent calls significantly.
        distance_threshold:  Min movement to continue iterating.
        epsilon:             Min value improvement to continue iterating.
        max_iterations:      Max ray-shooting iterations.
        use_momentum:        Use Adam-style momentum in gradient ascent.
        beta1:               Adam momentum decay (first moment).
        beta2:               Adam variance decay (second moment).
        lr_decay:            Multiplicative LR decay factor.
        lr_patience:         Steps without improvement before decaying LR.
        tol:                 Objective change below this triggers early stop.
        max_param_step_norm: Clips parameter update norm each step.
        max_grad_norm:       Clips gradient norm before update.
        early_stop_patience: Steps below tol before early stopping ascent.
    """

    def __init__(self,
                 lr: float = 0.01,
                 ascent_steps: int = 200,
                 num_rays: int = 38,
                 ray_step_size: float = 0.05,
                 ray_max_steps: int = 200,
                 top_k_landings: int = 5,
                 distance_threshold: float = 0.2,
                 epsilon: float = 1e-4,
                 max_iterations: int = 6,
                 use_momentum: bool = True,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 lr_decay: float = 0.99,
                 lr_patience: int = 3,
                 tol: float = 1e-3,
                 max_param_step_norm: float = 0.001,
                 max_grad_norm: float = 1.0,
                 early_stop_patience: int = 10):
        self.lr = lr
        self.ascent_steps = ascent_steps
        self.num_rays = num_rays
        self.ray_step_size = ray_step_size
        self.ray_max_steps = ray_max_steps
        self.top_k_landings = top_k_landings
        self.distance_threshold = distance_threshold
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.use_momentum = use_momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.tol = tol
        self.max_param_step_norm = max_param_step_norm
        self.max_grad_norm = max_grad_norm
        self.early_stop_patience = early_stop_patience

    def _gradient_ascent(self, f: Callable[[torch.Tensor], torch.Tensor], start: torch.Tensor) -> torch.Tensor:
        x = start.clone().detach().requires_grad_(True)
        current_lr = self.lr
        velocity = torch.zeros_like(x)
        variance = torch.zeros_like(x)
        prev_obj = None
        no_improve = 0
        no_improve_early_stop = 0
        best_obj = float('-inf')

        for step in range(self.ascent_steps):
            if self.use_momentum:
                x_lookahead = (x + self.beta1 * velocity).detach().requires_grad_(True)
                obj = f(x_lookahead)
                obj.backward()
                g = x_lookahead.grad.detach().clone()
            else:
                if x.grad is not None:
                    x.grad.zero_()
                obj = f(x)
                obj.backward()
                g = x.grad.detach().clone()

            grad_norm = torch.norm(g)
            if grad_norm > self.max_grad_norm:
                g = g * (self.max_grad_norm / (grad_norm + 1e-8))

            with torch.no_grad():
                if self.use_momentum:
                    velocity = self.beta1 * velocity + (1 - self.beta1) * g
                    variance = self.beta2 * variance + (1 - self.beta2) * (g ** 2)
                    m = velocity / (1 - self.beta1 ** (step + 1))
                    v = variance / (1 - self.beta2 ** (step + 1))
                    delta = current_lr * (m / (torch.sqrt(v) + self.epsilon))
                else:
                    delta = current_lr * g

                step_norm = torch.norm(delta)
                if step_norm > self.max_param_step_norm:
                    delta = delta * (self.max_param_step_norm / (step_norm + 1e-8))

                x = (x + delta).detach().requires_grad_(True)

            current_obj = obj.item()

            if not torch.isfinite(obj):
                logger.warning(f"Non-finite objective at step {step}, terminating ascent.")
                break

            if prev_obj is not None and abs(current_obj - prev_obj) < self.tol:
                no_improve_early_stop += 1
            else:
                no_improve_early_stop = 0

            if no_improve_early_stop >= self.early_stop_patience:
                logger.info(f"Early stopping at step {step}.")
                break

            if current_obj > best_obj:
                best_obj = current_obj
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.lr_patience:
                    current_lr *= self.lr_decay
                    no_improve = 0

            prev_obj = current_obj

        return x.detach()

    def _shoot_rays(
        self,
        x_max: torch.Tensor,
        f_batch: Callable[[torch.Tensor], torch.Tensor],
        f_start: float,
    ) -> List[torch.Tensor]:
        """
        Vectorized ray shooting. Evaluates all active rays in a single
        batched call per walk step instead of one call per ray per step.

        f_batch: accepts (N, d) tensor, returns (N,) tensor.
        """
        d = x_max.shape[0]
        n = self.num_rays
        thetas = torch.linspace(0, 2 * np.pi * (1 - 1 / n), n)
        # directions: (n, d) — works for any d, not just 2D
        directions = torch.zeros(n, d)
        directions[:, 0] = torch.cos(thetas)
        if d > 1:
            directions[:, 1] = torch.sin(thetas)

        # positions: (n, d) — all rays start at x_max
        positions = x_max.unsqueeze(0).expand(n, -1).clone()
        # active[i] = True means ray i has not yet landed or failed
        active = torch.ones(n, dtype=torch.bool)
        landings: List[torch.Tensor] = []
        landing_vals: List[float] = []

        for step in range(self.ray_max_steps):
            positions[active] = positions[active] + self.ray_step_size * directions[active]

            if step < 3:
                continue

            # evaluate all active rays in one batch call
            active_idx = active.nonzero(as_tuple=True)[0]
            if active_idx.numel() == 0:
                break

            vals = f_batch(positions[active_idx])   # (num_active,)

            for local_i, global_i in enumerate(active_idx.tolist()):
                v = vals[local_i].item()
                if not np.isfinite(v):
                    active[global_i] = False
                elif v > f_start * 1.05:
                    landings.append(positions[global_i].clone())
                    landing_vals.append(v)
                    active[global_i] = False

        if not landings:
            return []

        # return only top-k landings by value to cap gradient ascent calls
        k = min(self.top_k_landings, len(landings))
        top_idx = sorted(range(len(landing_vals)), key=lambda i: landing_vals[i], reverse=True)[:k]
        return [landings[i] for i in top_idx]

    def optimize(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        start: torch.Tensor,
        f_batch: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Find the maximum of f starting from start.

        Args:
            f:       Scalar objective. Takes a (d,) tensor, returns a scalar tensor.
                     Used for gradient ascent.
            start:   Starting point as a 1D tensor.
            f_batch: Batched objective. Takes a (N, d) tensor, returns a (N,) tensor.
                     Used for vectorized ray evaluation. If None, falls back to a
                     loop over f (slower, but maintains compatibility).

        Returns:
            Tuple of (best_point, best_value).
        """
        # build a batch wrapper if none supplied
        if f_batch is None:
            def f_batch(xs: torch.Tensor) -> torch.Tensor:
                return torch.stack([f(xs[i]) for i in range(xs.shape[0])])

        current_max = self._gradient_ascent(f, start)
        current_value = f(current_max).item()

        for _ in range(self.max_iterations):
            landings = self._shoot_rays(current_max, f_batch, current_value)
            if not landings:
                break

            best_iter_max, best_iter_value = current_max.clone(), current_value

            for landing in landings:
                candidate = self._gradient_ascent(f, landing)
                candidate_value = f(candidate).item()
                if candidate_value > best_iter_value:
                    best_iter_value = candidate_value
                    best_iter_max = candidate.clone()

            if best_iter_value - current_value < self.epsilon:
                break

            current_max, current_value = best_iter_max, best_iter_value

        return current_max, current_value
