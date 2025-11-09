"""
Run the full parameter sweep for Track 1 simulations.
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from track1_simulation import run_parameter_sweep

# Create output directory
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# Define parameter grid (from plan.md)
pi_proper_values = [0.1, 0.3, 0.5, 0.7, 0.9]
pi_hack_values = [0.9, 0.7, 0.5, 0.3, 0.1]  # Complementary to pi_proper

reward_ratios = [0.5, 0.7, 0.9, 1.0]
alphas = [0.01, 0.1, 0.5]

# Simulation parameters
n_trajectories = 1000  # As specified in plan
n_steps = 10000

print("=" * 80)
print("Track 1: Mathematical Simulation of Reward Hacking")
print("=" * 80)
print(f"\nParameter grid:")
print(f"  Initial probabilities: {len(pi_proper_values)} pairs")
print(f"  Reward ratios: {reward_ratios}")
print(f"  Learning rates (α): {alphas}")
print(f"  Trajectories per config: {n_trajectories}")
print(f"  Steps per trajectory: {n_steps}")
print()

# Run parameter sweep
results = run_parameter_sweep(
    pi_proper_values=pi_proper_values,
    pi_hack_values=pi_hack_values,
    reward_ratios=reward_ratios,
    alphas=alphas,
    n_trajectories=n_trajectories,
    n_steps=n_steps
)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"simulation_results_{timestamp}.pkl"

print(f"\nSaving results to {output_file}")
with open(output_file, "wb") as f:
    pickle.dump(results, f)

print(f"\n✓ Saved {len(results)} simulation results")
print(f"  File size: {output_file.stat().st_size / 1e6:.1f} MB")
