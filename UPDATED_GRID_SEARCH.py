from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import math
import torch
import numpy as np
from UPDATED_OPTIMIZERS_AND_LOSSES import (
    LOSS_FUNCTIONS,
    RayShootingOptimizer,
    SGDOptimizer,
    AdamOptimizer
)

# for reproducibility
torch.manual_seed(0)
np.random.seed(0)

START_POINT = torch.tensor([-4.43, -6.64], dtype=torch.float32)

GRID = {
    'RayShootingOptimizer': {
        'lr': [0.001, 0.01],
        'ascent_steps': [200, 500, 1000],
        'num_rays': [16, 32],
        'ray_step_size': [0.05, 0.5],
        'distance_threshold': [0.2, 0.5],
        'use_momentum': [False, True],
        'max_param_step_norm': [0.1, 0.5],
        'max_grad_norm': [1.0, 3.0],
    },
    'SGDOptimizer': {
        'lr': [0.005, 0.01, 0.02, 0.05],
        'steps': [200, 400],
        'momentum': [0.0, 0.8]
    },
    'AdamOptimizer': {
        'lr': [0.005, 0.01, 0.02],
        'steps': [200, 400],
        'beta1': [0.9, 0.8]
    }
}

TIME_LIMIT_PER_TRIAL = 17.0

def evaluate_optimizer_on_loss(opt_name: str, loss_name: str, params: dict) -> tuple:
    """
    Runs a single optimizer-loss trial and returns (opt_name, loss_name, params, result_dict)
    This wrapper is needed so ProcessPoolExecutor can pickle and run it in a separate process.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    loss_fn = LOSS_FUNCTIONS[loss_name]
    start = START_POINT.clone()
    t0 = time.time()

    def timed_loss_fn(x):
        if time.time() - t0 > TIME_LIMIT_PER_TRIAL:
            raise TimeoutError("Trial exceeded time limit")
        return loss_fn(x)

    try:
        if opt_name == 'RayShootingOptimizer':
            opt = RayShootingOptimizer(**params)
        elif opt_name == 'SGDOptimizer':
            opt = SGDOptimizer(**params)
        elif opt_name == 'AdamOptimizer':
            opt = AdamOptimizer(**params)
        else:
            raise ValueError("Unknown optimizer")

        best = opt.optimize(timed_loss_fn, start)
        best_val = loss_fn(best).item()
        hist = opt.history

    except TimeoutError:
        return (opt_name, loss_name, params, {
            'final_value': -1e9,
            'history': None,
            'duration': time.time() - t0
        })
    except Exception:
        return (opt_name, loss_name, params, {
            'final_value': -1e9,
            'history': None,
            'duration': time.time() - t0
        })

    duration = time.time() - t0
    return (opt_name, loss_name, params, {
        'final_value': best_val,
        'history': hist,
        'duration': duration
    })


def iterate_grid_for_optimizer(grid: dict):
    from itertools import product
    keys = sorted(grid.keys())
    lists = [grid[k] for k in keys]
    for vals in product(*lists):
        yield dict(zip(keys, vals))


def run_grid_search():
    best_params = {}
    overall_start = time.time()
    max_workers = min(8, torch.get_num_threads())  # adjust based on CPU cores

    for loss_name in LOSS_FUNCTIONS.keys():
        print(f"\n=== Grid search for loss: {loss_name} ===")
        best_params[loss_name] = {}

        for opt_name, grid in [
            ('RayShootingOptimizer', GRID['RayShootingOptimizer']),
            ('SGDOptimizer', GRID['SGDOptimizer']),
            ('AdamOptimizer', GRID['AdamOptimizer'])
        ]:
            print(f" optimizer: {opt_name}")
            candidates = list(iterate_grid_for_optimizer(grid))
            if not candidates:
                candidates = [{}]  # default trial

            futures = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for params in candidates:
                    futures.append(
                        executor.submit(evaluate_optimizer_on_loss, opt_name, loss_name, params)
                    )

                best_val = float('-inf')
                best_conf = None
                trials = 0
                nan_trials = []

                for future in as_completed(futures):
                    opt_name_r, loss_name_r, params_r, trial_info = future.result()
                    trials += 1
                    val = trial_info['final_value']
                    dur = trial_info.get('duration', 0.0)

                    # NaN check
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        print(f"  [NaN] trial {trials} params={params_r} produced NaN value")
                        nan_trials.append(params_r)
                        continue

                    print(f"  trial {trials}: val={val:.4f} dur={dur:.2f}s params={params_r}")

                    # Track best
                    if val > best_val:
                        best_val = val
                        best_conf = {
                            'params': params_r,
                            'final_value': float(val),
                            'duration': float(dur)
                        }

            # Store results for THIS optimizer-loss pair
            if nan_trials:
                print(f"  [WARNING] {len(nan_trials)} trials produced NaN values for optimizer {opt_name} on loss {loss_name}")
                for idx, bad_params in enumerate(nan_trials, 1):
                    print(f"    NaN trial {idx}: params={bad_params}")

            if best_conf is None:
                print(f"  -> No valid trial for {opt_name}")
                best_params[loss_name][opt_name] = None
            else:
                print(f"  -> best for {opt_name}: val={best_conf['final_value']} params={best_conf['params']}")
                best_params[loss_name][opt_name] = best_conf

    overall_dur = time.time() - overall_start
    print(f"\nGrid search complete in {overall_dur:.1f}s")
    with open('BEST_PARAMS.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("Saved BEST_PARAMS.json")


if __name__ == "__main__":
    run_grid_search()
