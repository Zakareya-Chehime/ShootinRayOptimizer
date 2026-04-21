# RayShootingOptimizer (RSO)

A gradient-based global optimizer that escapes local maxima on multimodal loss surfaces using a geometric ray-shooting search.

---

## The Problem

Adam and SGD are local optimizers. They follow the gradient from wherever they start and converge to the nearest basin. On multimodal surfaces with many peaks, they get stuck regardless of how well their hyperparameters are tuned. This is a structural limitation, not a tuning problem.

## The Idea

After climbing to a local maximum via gradient ascent, RSO fires rays outward in evenly-spaced directions. Each ray walks forward until it finds a region meaningfully higher than the current position. Gradient ascent then runs from the top-k landing points by value. This cycle repeats until no improvement is found or a maximum iteration count is reached.

```
Climb → Scout → Climb from best landings → repeat
```

Ray shooting is vectorized: all active ray positions at each walk step are evaluated in a single batched tensor call, keeping runtime competitive with standard baselines.

---

## Results

Monte Carlo benchmark: 30 random starts uniformly distributed across each surface's domain. **Mean gap** is the primary metric (distance from the global optimum — lower is better).

### Ackley
| Method | Mean Gap | Std | Mean Time |
|--------|----------|-----|-----------|
| **RSO** | **0.2717** | 0.6289 | 0.118s |
| Adam | 6.4687 | 2.3485 | 0.111s |
| SGD | 5.5230 | 3.1018 | 0.100s |

### Rastrigin
| Method | Mean Gap | Std | Mean Time |
|--------|----------|-----|-----------|
| **RSO** | **1.6597** | 1.2048 | 0.044s |
| Adam | 10.6460 | 6.7488 | 0.081s |
| SGD | 9.5746 | 6.2908 | 0.069s |

### Levy
| Method | Mean Gap | Std | Mean Time |
|--------|----------|-----|-----------|
| **RSO** | **0.0096** | 0.0157 | 0.246s |
| Adam | 2.9292 | 2.1482 | 0.169s |
| SGD | 2.6869 | 2.2405 | 0.159s |

### Griewank
| Method | Mean Gap | Std | Mean Time |
|--------|----------|-----|-----------|
| **RSO** | **0.0110** | 0.0097 | 0.094s |
| Adam | 0.2588 | 0.1679 | 0.093s |
| SGD | 0.2588 | 0.1679 | 0.082s |

### Bukin N.6 (known hard case)
| Method | Mean Gap | Std | Mean Time |
|--------|----------|-----|-----------|
| Adam | **1.3515** | 0.6674 | 0.098s |
| SGD | 10.2321 | 4.0925 | 0.087s |
| RSO | 5.4662 | 2.3663 | 1.150s |

RSO dominates on 4 of 5 surfaces, often by a factor of 5–10x in mean gap, while matching baselines in runtime. Bukin N.6 is a known failure case explained below.

---

## How It Works

### Phase 1 — Gradient Ascent

RSO uses a hand-implemented Adam-style ascent with Nesterov lookahead momentum, gradient norm clipping, parameter step norm clipping, and patience-based learning rate decay. This climbs reliably from any starting point to the nearest local maximum.

### Phase 2 — Ray Shooting

From the local maximum, `num_rays` rays are fired outward in evenly-spaced angular directions. All ray positions are advanced together each step and evaluated in a single batched call to `f_batch`. A ray "lands" when it finds a position whose value exceeds the current maximum by a meaningful margin. Only the top-k landings by value are kept to cap the cost of subsequent ascent calls.

### Phase 3 — Iterate

Gradient ascent runs from each landing point. If any candidate beats the current best, RSO moves there and repeats the scout phase. The loop exits when no ray finds higher ground or the improvement falls below `epsilon`.

---

## Known Limitation: Bukin N.6

Bukin N.6 has a narrow parabolic ridge. RSO's rays travel in straight lines, so they cross the curved ridge briefly and continue off it. Landing points end up on the slope rather than the ridge crest, and subsequent ascent climbs to mediocre local points. Straight-line ray shooting is structurally ill-suited to curved manifold optima. The domain is also highly asymmetric (`x1 ∈ [-15, -5]`, `x2 ∈ [-3, 3]`), so most uniform rays immediately walk into out-of-domain regions.

---

## Installation

No package installation required beyond standard scientific Python:

```bash
pip install torch numpy matplotlib
```

Clone the repo and import directly:

```python
from optimizer import RayShootingOptimizer
```

---

## Usage

```python
import torch
from optimizer import RayShootingOptimizer

# Define a scalar objective (maximization)
def f(x: torch.Tensor) -> torch.Tensor:
    return -torch.sum(x ** 2)  # maximum at origin

# Define a batched version for ray evaluation
def f_batch(xs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(xs ** 2, dim=1)

start = torch.tensor([3.0, -2.5])

rso = RayShootingOptimizer(
    lr=0.05,
    ascent_steps=300,
    num_rays=24,
    ray_step_size=0.1,
    max_iterations=8,
    top_k_landings=5,
)

best_point, best_value = rso.optimize(f, start, f_batch=f_batch)
print(f"Best point: {best_point}")
print(f"Best value: {best_value:.5f}")
```

If `f_batch` is not provided, RSO falls back to a sequential loop over `f` — correct but slower.

---

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate for gradient ascent | `0.01` |
| `ascent_steps` | Max gradient ascent steps per climb | `200` |
| `num_rays` | Number of rays fired per iteration | `38` |
| `ray_step_size` | Step size along each ray direction | `0.05` |
| `ray_max_steps` | Max walk steps per ray | `200` |
| `top_k_landings` | Only climb from the top-k landing points | `5` |
| `max_iterations` | Max climb → scout → climb cycles | `6` |
| `epsilon` | Min value improvement to keep iterating | `1e-4` |
| `max_param_step_norm` | Clips parameter update norm each step | `0.001` |
| `max_grad_norm` | Clips gradient norm before update | `1.0` |
| `early_stop_patience` | Steps below `tol` before early stopping | `10` |

The most impactful parameters for a new surface are `ray_step_size` (should be on the order of the spacing between local optima), `num_rays` (more rays = more coverage, higher cost), and `top_k_landings` (controls the accuracy vs. speed tradeoff).

---

## Files

```
├── optimizer.py          # RayShootingOptimizer implementation
├── demo.ipynb            # Benchmarking notebook (single-run + Monte Carlo)
└── assets/
    ├── single_run_comparison.png
    └── monte_carlo_comparison.png
```

---

## Relation to Existing Work

RSO is structurally related to basin-hopping (Wales & Doye, 1997), which combines local optimization with structured perturbations to escape local minima. The ray-shooting phase implicitly probes the geometry of the loss surface via linear interpolation, connecting to loss landscape characterization work (Li et al., 2018; Goodfellow et al., 2015). The key distinction is that RSO uses directional geometric search rather than random perturbations, and integrates vectorized batched evaluation for competitive runtime.

---

## Limitations and Future Work

- Ray directions are fixed to the first two dimensions of the parameter space. For high-dimensional problems (d >> 2), directions should be sampled randomly in the full space rather than projected onto a 2D plane.
- The landing threshold assumes the surface is not heavily negative-valued near the start. The threshold `f_start + |f_start| * 0.05 + 1e-3` handles this more robustly than a simple percentage.
- A natural extension is Hessian-informed ray directions — concentrating exploration where curvature suggests the surface is most likely to rise, rather than uniform angular spacing.

---

*Undergraduate research project. Benchmarks run on standard 2D test functions; scaling behavior in higher dimensions is an open question.*
