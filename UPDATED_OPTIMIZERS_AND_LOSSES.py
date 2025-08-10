# optimizers_and_losses.py
import torch
import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any

torch.set_default_dtype(torch.float32)

def himmelblau(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    return -((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

def rastrigin_2d(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    A = 10
    n = 2
    return -(A * n + (x**2 - A * torch.cos(2 * np.pi * x)) + (y**2 - A * torch.cos(2 * np.pi * y)))

def six_hump_camel(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    return -((4 - 2.1*x**2 + x**4/3)*x**2 + x*y + (-4 + 4*y**2)*y**2)

def ackley(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    return -(-20*torch.exp(-0.2*torch.sqrt(0.5*(x**2 + y**2))) - torch.exp(0.5*(torch.cos(2*np.pi*x) + torch.cos(2*np.pi*y))) + torch.e + 20)

def sphere(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    return -(x**2 + y**2)

def rosenbrock(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    return -(100*(y - x**2)**2 + (1 - x)**2)

def goldstein_price(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    term1 = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    term2 = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return -(term1 * term2)

def three_minima(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    return (4.0 * torch.exp(-0.5 * ((x - 2.0)**2/5 + (y - 2.0)**2/1)) +
            2.0 * torch.exp(-0.5 * ((x + 2.0)**2/1 + (y + 2.0)**2/2)) +
            1.0 * torch.exp(-0.5 * ((x - 4.0)**2/5 + (y + 4.0)**2/1)))

LOSS_FUNCTIONS = {
    'himmelblau': himmelblau,
    'rastrigin_2d': rastrigin_2d,
    'six_hump_camel': six_hump_camel,
    'ackley': ackley,
    'sphere': sphere,
    'rosenbrock': rosenbrock,
    'goldstein_price': goldstein_price,
    'three_minima': three_minima
}

class RayShootingOptimizer:
    def __init__(self,
                 lr: float = 0.01,
                 ascent_steps: int = 200,
                 num_rays: int = 16,
                 ray_step_size: float = 0.05,
                 distance_threshold: float = 0.2,
                 epsilon: float = 1e-4,
                 max_iterations: int = 6,
                 # adaptive
                 use_momentum: bool = True,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 use_adaptive_lr: bool = False,
                 lr_decay: float = 0.99,
                 lr_patience: int = 10,
                 tol: float = 1e-3, # Apply grid search to optimize this
                 max_param_step_norm: float = 0.1, # Apply grid search to optimize this
                 max_grad_norm: float = 1.0, # Apply grid search to optimize this
                 early_stop_patience: int = 10
                 ):
        # core
        self.lr = lr
        self.ascent_steps = ascent_steps
        self.num_rays = num_rays
        self.ray_step_size = ray_step_size
        self.distance_threshold = distance_threshold
        self.epsilon = epsilon
        self.max_iterations = max_iterations

        # adaptive
        self.use_momentum = use_momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.use_adaptive_lr = use_adaptive_lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.tol = tol
        self.max_param_step_norm = max_param_step_norm
        self.max_grad_norm = max_grad_norm
        self.early_stop_patience = early_stop_patience

        # history structure
        self.history: Dict[str, Any] = {
#           'all_paths': [],          # list of np arrays (paths)
            'main_ascent_paths': [],
            'ray_paths' : [],
            'iteration_data': [],     # per-iteration dicts
            'best_point': None,
            'best_value': None
        }

    def gradient_ascent(self, f: Callable[[torch.Tensor], torch.Tensor], start: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray]]:
        x = start.clone().detach().requires_grad_(True)
        path = [x.detach().cpu().numpy().copy()]    
        current_lr = self.lr
        velocity = torch.zeros_like(x)
        variance = torch.zeros_like(x)
        max_grad_norm = self.max_grad_norm
        max_param_step_norm = self.max_param_step_norm
        epsilon = 1e-8
        tol = self.tol
        prev_obj = None
        no_improve = 0  # Counter for LR decay patience
        no_improve_early_stop = 0
        best_obj = float('-inf')


        for step in range(self.ascent_steps):
            
            if self.use_momentum:
                # Nesterov lookahead
                x_lookahead = (x + self.beta1 * velocity).detach().requires_grad_(True)
                obj = f(x_lookahead)
                obj.backward()
                g = x_lookahead.grad.detach().clone()

            else:
                # Regular Gradient Ascent
                x.grad = None
                obj = f(x)
                obj.backward()
                g = x.grad.detach().clone()

            # Gradient Clipping
            grad_norm = torch.norm(g)
            if grad_norm > max_grad_norm:
                g = g * (max_grad_norm / (grad_norm + 1e-8))

            with torch.no_grad():
                delta = torch.zeros_like(x)

                # Nesterov Momentum with RMSProp Adaptive LR Update
                if self.use_momentum:
                    
                    # First and second moment calculation
                    velocity = self.beta1 * velocity + (1 - self.beta1) * g
                    variance = self.beta2 * variance + (1 - self.beta2) * (g ** 2)
                    
                    # Bias correction
                    m = velocity / (1 - self.beta1**(step + 1))
                    v = variance / (1 - self.beta2**(step + 1))
                    
                    # Delta used for updating and Parameter Clipping
                    delta = current_lr * (m / (torch.sqrt(v) + epsilon))
                else:
                    delta = current_lr * g

                # Parameter Step Clipping
                step_norm = torch.norm(delta)
                if step_norm > max_param_step_norm:
                    delta = delta * (max_param_step_norm / (step_norm + 1e-8))

                x = x + delta
                
            path.append(x.detach().cpu().numpy().copy())

            current_obj = obj.item()

            # If the objective value improves by less than 0.001 stop and consider it converged
            if prev_obj is not None and abs(current_obj - prev_obj) < tol:
                no_improve_early_stop += 1
            else:
                no_improve_early_stop = 0

            # Early stopping condition
            if no_improve_early_stop >= self.early_stop_patience:
                print(f"Early stopping at step {step} due to no improvement in objective for {self.early_stop_patience} steps.")
                break
            
            # Adaptive Learning Rate Decay
            if current_obj > best_obj:
                best_obj = current_obj
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.lr_patience:
                    old_lr = current_lr
                    current_lr *= self.lr_decay
                    print(f"[Step {step}] Learning rate decayed from {old_lr:.6f} to {current_lr:.6f} after {no_improve} stagnant steps")
                    no_improve = 0

            prev_obj = current_obj

            # If the objective is infinite break
            if not torch.isfinite(obj):
                print(f"Terminating early at step {step} due to non-finite objective.")
                break

            # Diagnostic logging every 10 steps
            if step % 10 == 0 or step == self.ascent_steps - 1:
                print(f"[Step {step}] Obj: {current_obj:.6f}, Grad Norm: {grad_norm:.6f}, Step Norm: {step_norm:.6f}, LR: {current_lr:.6f}")

        return x.detach(), path


    def find_surface_intersection(self, x_start: torch.Tensor, direction: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[Optional[torch.Tensor], List[np.ndarray]]:
        x = x_start.clone().detach()
        f_start = f(x_start).item()
        path = [x.detach().cpu().numpy().copy()]

        step_size = self.ray_step_size
        max_steps = 200
        min_steps = 3

        for _ in range(max_steps):
            x = x + step_size * direction
            path.append(x.detach().cpu().numpy().copy())

            if _ < min_steps:
                continue

            current_value = f(x).item()

            #  find region that is meaningfully higher than the original
            if current_value > f_start * 1.05:
                return x, path

        return None, path

    def shoot_rays(self, x_max: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[List[torch.Tensor], List[np.ndarray], List[str]]:
        ray_landings: List[torch.Tensor] = []
        ray_paths: List[np.ndarray] = []
        ray_colors: List[str] = []
        roygbiv = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]

        for i in range(self.num_rays):
            theta = 2 * np.pi * i / self.num_rays
            direction = torch.tensor([np.cos(theta), np.sin(theta)], dtype=torch.float32)
            landing, path = self.find_surface_intersection(x_max, direction, f)
            if landing is not None:
                ray_landings.append(landing)
                ray_paths.append(np.array(path))
                ray_colors.append(roygbiv[i % len(roygbiv)])

        return ray_landings, ray_paths, ray_colors

    def optimize(self, f: Callable[[torch.Tensor], torch.Tensor], start: torch.Tensor) -> torch.Tensor:
        # reset history
        self.history = {'all_paths': [], 'iteration_data': [], 'best_point': None, 'best_value': None}

        current_max, main_path = self.gradient_ascent(f, start)
        current_value = f(current_max).item()

        self.history['all_paths'] = [np.array(main_path)]
        self.history['best_point'] = current_max.clone()
        self.history['best_value'] = current_value

        for iteration in range(self.max_iterations):
            ray_landings, ray_paths, ray_colors = self.shoot_rays(current_max, f)
            if not ray_landings:
                # no promising landings — stop
                break

            ascent_paths = []
            found_maxima = []
            best_iter_max = current_max.clone()
            best_iter_value = current_value

            # try gradient ascent from each landing
            for k, landing in enumerate(ray_landings):
                # landing may be a point away; ensure grad enabled
                new_max, asc_path = self.gradient_ascent(f, landing)
                ascent_paths.append(np.array(asc_path))
                new_val = f(new_max).item()
                found_maxima.append((new_max.clone(), new_val))
                if new_val > best_iter_value:
                    best_iter_value = new_val
                    best_iter_max = new_max.clone()

            distances = [torch.norm(m - current_max).item() for m, _ in found_maxima] if found_maxima else [0.0]
            max_distance = float(max(distances)) if distances else 0.0

            iter_record = {
                'iteration': iteration + 1,
                'starting_max': current_max.clone(),
                'starting_value': current_value,
                'ray_paths': ray_paths,       # list of np arrays
                'ray_colors': ray_colors,     # matched list of color strings
                'ascent_paths': ascent_paths, # list of np arrays
                'best_max': best_iter_max.clone(),
                'best_value': best_iter_value,
                'max_distance': max_distance
            }

            self.history['iteration_data'].append(iter_record)
            self.history['all_paths'].extend(ascent_paths)

            improvement = best_iter_value - current_value
            if improvement < self.epsilon or max_distance < self.distance_threshold:
                # not enough improvement or rays too close — stop
                self.history['best_point'] = best_iter_max.clone()
                self.history['best_value'] = best_iter_value
                break

            # otherwise update and continue
            current_max = best_iter_max.clone()
            current_value = best_iter_value
            self.history['best_point'] = current_max.clone()
            self.history['best_value'] = current_value

        return self.history['best_point']


# ----------------------
# SGDOptimizer (from scratch) - gradient ascent
# ----------------------
class SGDOptimizer:
    """
    Simple SGD implemented from scratch that logs the path.
    Performs gradient-ascent: x <- x + lr * grad
    """

    def __init__(self, lr: float = 0.01, steps: int = 500, momentum: float = 0.0):
        self.lr = lr
        self.steps = steps
        self.momentum = momentum
        self.history: Dict[str, Any] = {}

    def optimize(self, f: Callable[[torch.Tensor], torch.Tensor], start: torch.Tensor) -> torch.Tensor:
        x = start.clone().detach().requires_grad_(True)
        v = torch.zeros_like(x)
        path = [x.detach().cpu().numpy().copy()]

        for t in range(self.steps):
            obj = f(x)
            obj.backward()
            with torch.no_grad():
                g = x.grad
                v = self.momentum * v + self.lr * g
                x += v
            x.grad.zero_()
            path.append(x.detach().cpu().numpy().copy())

        self.history = {'all_paths': [np.array(path)], 'best_point': x.detach().clone(), 'best_value': f(x).item()}
        return x.detach()

# ----------------------
# AdamOptimizer (from scratch) - gradient ascent
# ----------------------
class AdamOptimizer:
    """
    Adam implemented from scratch for gradient-ascent (i.e., step in +grad).
    """

    def __init__(self, lr: float = 0.01, steps: int = 500, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.steps = steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.history: Dict[str, Any] = {}

    def optimize(self, f: Callable[[torch.Tensor], torch.Tensor], start: torch.Tensor) -> torch.Tensor:
        x = start.clone().detach().requires_grad_(True)
        m = torch.zeros_like(x)
        v = torch.zeros_like(x)
        path = [x.detach().cpu().numpy().copy()]

        for t in range(1, self.steps + 1):
            obj = f(x)
            obj.backward()
            with torch.no_grad():
                g = x.grad
                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * (g * g)
                m_hat = m / (1 - self.beta1 ** t)
                v_hat = v / (1 - self.beta2 ** t)
                x += self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            x.grad.zero_()
            path.append(x.detach().cpu().numpy().copy())

        self.history = {'all_paths': [np.array(path)], 'best_point': x.detach().clone(), 'best_value': f(x).item()}
        return x.detach()
