# RayShootingOptimizer (RSO)

A gradient-based global optimizer that escapes local maxima on multimodal loss surfaces using a geometric ray-shooting search.

---

## The Problem

Adam and SGD are local optimizers. They follow the gradient from wherever they start and converge to the nearest basin. On multimodal surfaces with many peaks, they get stuck regardless of how well their hyperparameters are tuned.

## The Idea

Imagine you're a climber trying to reach the highest peak in a mountain range. One way you could try and find the highest peak is as follows: Start at the base of a random mountain and climb it. Once you're at the top look around, do you see taller mountains? If yes make a note for each. Then, if those mountains are at least 5% higher than the one you were standing on, climb up them and whichever is tallest is the one you'll pick. Granted this is a lot of work for you as a climber but given enough time you will eventually find the highest mountain. This is the idea of this optimizer, although of course the analogy could use some work in certain areas. A quick side note, the optimizer conducts gradient **ascent** on the **negative** version of a loss surface. This is done purely so I can keep the analogy of the mountain climber looking around and no other reason, since solving for the maximum of a negative loss surface is equivalent to minimizing a normal loss surface. The optimizer starts in a random area and, after climbing to a local maximum via gradient **ascent**, RSO fires rays outward in evenly spaced directions. Each ray walks forward until it finds a region meaningfully higher than the current position. Gradient ascent then runs from the top-k landing points by value. This cycle repeats until no meaningful improvement is found or a maximum iteration count is reached.

```
Climb → Scout → Climb from best landings → repeat
```

Ray shooting is vectorized: all active ray positions at each walk step are evaluated in a single batched tensor call, keeping runtime competitive with standard baselines.

---

## Results

For a single run through the different benchmarks:
![Single run comparison](assets/single_run_comparison.png)


Then, the Monte Carlo benchmark: 30 random starts uniformly distributed across each surface's domain. **Mean gap** is the primary metric (distance from the global optimum — lower is better).

![Monte Carlo comparison](assets/monte_carlo_comparison.png)

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

Bukin N.6 has a narrow parabolic ridge. RSO's rays travel in straight lines, so they cross the curved ridge briefly and continue off it. Landing points end up on the slope rather than the ridge crest, and subsequent ascent climbs to mediocre local points. Straight-line ray shooting is structurally ill-suited for this. The domain is also highly asymmetric (`x1 ∈ [-15, -5]`, `x2 ∈ [-3, 3]`), so most uniform rays immediately walk into out of domain regions.

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

If `f_batch` is not provided, RSO falls back to a sequential loop over `f` — it works but is slower.

---

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate for gradient ascent | `0.01` |
| `ascent_steps` | Gradient ascent steps per climb | `200` |
| `num_rays` | Number of rays fired per iteration | `38` |
| `ray_step_size` | Step size along each ray direction | `0.05` |
| `ray_max_steps` | Max walk steps per ray | `200` |
| `top_k_landings` | Only climb from the top-k highest landing points | `5` |
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
└── images/
    ├── single_run_comparison.png
    └── monte_carlo_comparison.png
```

---


## Limitations and Future Work

- Ray directions are fixed to the first two dimensions of the parameter space. For high-dimensional problems (d >> 2), directions should be sampled randomly in the full space rather than projected onto a 2D plane.
- A good extension is to use Hessian-informed ray directions, concentrating exploration where curvature suggests the surface is most likely to rise, rather than uniform angular spacing.
