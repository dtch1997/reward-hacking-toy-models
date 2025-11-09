"""
Track 1: Mathematical simulation of reward hacking dynamics.

Two competing behaviors (proper solution vs. hack) with logit-based probability updates.
Update rule: logit_k(t+1) = logit_k(t) + α × R_k (with R_baseline = 0)
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    # Initial propensities (probabilities)
    pi_proper_init: float  # Initial probability of proper solution
    pi_hack_init: float    # Initial probability of hack

    # Rewards
    r_proper: float = 1.0
    r_hack: float = 0.5

    # Learning parameters
    alpha: float = 0.1  # Learning rate

    # Simulation parameters
    n_steps: int = 10000

    def __post_init__(self):
        """Validate configuration."""
        assert 0 <= self.pi_proper_init <= 1, "Initial probabilities must be in [0, 1]"
        assert 0 <= self.pi_hack_init <= 1, "Initial probabilities must be in [0, 1]"
        # Note: pi_proper_init + pi_hack_init should equal 1 for a two-choice system


@dataclass
class SimulationResult:
    """Results from a single simulation trajectory."""
    config: SimulationConfig

    # Trajectories over time
    logit_proper: np.ndarray  # Shape: (n_steps,)
    logit_hack: np.ndarray    # Shape: (n_steps,)
    prob_proper: np.ndarray   # Shape: (n_steps,)
    prob_hack: np.ndarray     # Shape: (n_steps,)

    # Summary statistics
    final_prob_proper: float
    final_prob_hack: float
    converged_to_hack: bool  # True if model prefers hack at end
    convergence_step: int    # Step when probability crossed 0.5


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits."""
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / exp_logits.sum()


def simulate_trajectory(config: SimulationConfig, seed: int = None) -> SimulationResult:
    """
    Simulate a single training trajectory.

    At each step:
    1. Sample an action according to current probabilities
    2. Receive reward for that action
    3. Update the logit for the chosen action
    4. Recompute probabilities via softmax

    Args:
        config: Simulation configuration
        seed: Random seed for reproducibility

    Returns:
        SimulationResult with full trajectory and summary stats
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize logits from initial probabilities
    # If pi_proper_init and pi_hack_init are given, convert to logits
    # For simplicity, we'll use: logit_k = log(pi_k)
    # Then normalize via softmax to ensure they sum to 1
    init_logits = np.array([
        np.log(config.pi_proper_init + 1e-10),  # Avoid log(0)
        np.log(config.pi_hack_init + 1e-10)
    ])

    # Storage for trajectories
    logit_proper_traj = np.zeros(config.n_steps)
    logit_hack_traj = np.zeros(config.n_steps)
    prob_proper_traj = np.zeros(config.n_steps)
    prob_hack_traj = np.zeros(config.n_steps)

    # Current state
    logits = init_logits.copy()
    rewards = np.array([config.r_proper, config.r_hack])

    convergence_step = -1

    for t in range(config.n_steps):
        # Compute current probabilities
        probs = softmax(logits)

        # Record state
        logit_proper_traj[t] = logits[0]
        logit_hack_traj[t] = logits[1]
        prob_proper_traj[t] = probs[0]
        prob_hack_traj[t] = probs[1]

        # Check convergence to hack (first time prob_hack > 0.5)
        if convergence_step == -1 and probs[1] > 0.5:
            convergence_step = t

        # Sample action according to current probabilities
        action = np.random.choice(2, p=probs)  # 0=proper, 1=hack

        # Get reward for chosen action
        reward = rewards[action]

        # Update logit for chosen action
        # logit_k(t+1) = logit_k(t) + α × (R_k - R_baseline)
        # With R_baseline = 0: logit_k(t+1) = logit_k(t) + α × R_k
        logits[action] += config.alpha * reward

    return SimulationResult(
        config=config,
        logit_proper=logit_proper_traj,
        logit_hack=logit_hack_traj,
        prob_proper=prob_proper_traj,
        prob_hack=prob_hack_traj,
        final_prob_proper=prob_proper_traj[-1],
        final_prob_hack=prob_hack_traj[-1],
        converged_to_hack=(prob_hack_traj[-1] > 0.5),
        convergence_step=convergence_step
    )


def run_parameter_sweep(
    pi_proper_values: List[float],
    pi_hack_values: List[float],
    reward_ratios: List[float],
    alphas: List[float],
    n_trajectories: int = 1000,
    n_steps: int = 10000
) -> List[SimulationResult]:
    """
    Run simulations across a parameter grid.

    Args:
        pi_proper_values: List of initial proper probabilities to try
        pi_hack_values: List of initial hack probabilities to try
        reward_ratios: List of R_hack/R_proper ratios
        alphas: List of learning rates
        n_trajectories: Number of trajectories per configuration
        n_steps: Steps per trajectory

    Returns:
        List of all simulation results
    """
    results = []
    total_configs = (len(pi_proper_values) * len(pi_hack_values) *
                    len(reward_ratios) * len(alphas))

    print(f"Running {total_configs} configurations × {n_trajectories} trajectories = "
          f"{total_configs * n_trajectories} total simulations")

    config_idx = 0
    for pi_proper in pi_proper_values:
        for pi_hack in pi_hack_values:
            # Ensure probabilities sum to 1
            if not np.isclose(pi_proper + pi_hack, 1.0):
                continue

            for reward_ratio in reward_ratios:
                r_proper = 1.0
                r_hack = reward_ratio * r_proper

                for alpha in alphas:
                    config_idx += 1
                    print(f"Config {config_idx}/{total_configs}: "
                          f"π_proper={pi_proper:.2f}, π_hack={pi_hack:.2f}, "
                          f"R_ratio={reward_ratio:.2f}, α={alpha:.3f}")

                    # Run multiple trajectories for this configuration
                    for traj_idx in range(n_trajectories):
                        config = SimulationConfig(
                            pi_proper_init=pi_proper,
                            pi_hack_init=pi_hack,
                            r_proper=r_proper,
                            r_hack=r_hack,
                            alpha=alpha,
                            n_steps=n_steps
                        )

                        result = simulate_trajectory(config, seed=traj_idx)
                        results.append(result)

    print(f"\nCompleted {len(results)} simulations")
    return results


if __name__ == "__main__":
    # Quick test
    config = SimulationConfig(
        pi_proper_init=0.5,
        pi_hack_init=0.5,
        r_proper=1.0,
        r_hack=0.7,
        alpha=0.1,
        n_steps=1000
    )

    result = simulate_trajectory(config, seed=42)
    print(f"Final probabilities: proper={result.final_prob_proper:.3f}, "
          f"hack={result.final_prob_hack:.3f}")
    print(f"Converged to hack: {result.converged_to_hack}")
    if result.convergence_step > 0:
        print(f"Convergence step: {result.convergence_step}")
