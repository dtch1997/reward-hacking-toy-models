"""Quick test with smaller parameter grid."""

import numpy as np
import pickle
from pathlib import Path
from track1_simulation import run_parameter_sweep

# Small test
pi_proper_values = [0.3, 0.5, 0.7]
pi_hack_values = [0.7, 0.5, 0.3]
reward_ratios = [0.5, 0.9]
alphas = [0.1]
n_trajectories = 10  # Much smaller for quick test
n_steps = 1000

print("Running quick test...")
results = run_parameter_sweep(
    pi_proper_values=pi_proper_values,
    pi_hack_values=pi_hack_values,
    reward_ratios=reward_ratios,
    alphas=alphas,
    n_trajectories=n_trajectories,
    n_steps=n_steps
)

# Save
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "simulation_results_test.pkl"

with open(output_file, "wb") as f:
    pickle.dump(results, f)

print(f"\n✓ Saved {len(results)} results to {output_file}")
