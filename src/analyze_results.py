"""
Analyze and visualize results from Track 1 simulations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from typing import List
from track1_simulation import SimulationResult

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(results_file: Path) -> List[SimulationResult]:
    """Load simulation results from pickle file."""
    with open(results_file, "rb") as f:
        return pickle.load(f)


def create_summary_dataframe(results: List[SimulationResult]) -> pd.DataFrame:
    """Convert simulation results to a pandas DataFrame for analysis."""
    data = []
    for result in results:
        data.append({
            'pi_proper_init': result.config.pi_proper_init,
            'pi_hack_init': result.config.pi_hack_init,
            'r_proper': result.config.r_proper,
            'r_hack': result.config.r_hack,
            'reward_ratio': result.config.r_hack / result.config.r_proper,
            'alpha': result.config.alpha,
            'n_steps': result.config.n_steps,
            'final_prob_proper': result.final_prob_proper,
            'final_prob_hack': result.final_prob_hack,
            'converged_to_hack': result.converged_to_hack,
            'convergence_step': result.convergence_step
        })
    return pd.DataFrame(data)


def analyze_convergence(df: pd.DataFrame):
    """Analyze convergence patterns across configurations."""
    print("=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    # Group by configuration (average over trajectories)
    grouped = df.groupby(['pi_proper_init', 'reward_ratio', 'alpha']).agg({
        'converged_to_hack': 'mean',  # Fraction that converged to hack
        'final_prob_hack': 'mean',
        'convergence_step': lambda x: x[x > 0].mean() if (x > 0).any() else -1
    }).reset_index()

    print("\nFraction converging to hack by configuration:")
    print(grouped.to_string())
    print()

    return grouped


def plot_phase_diagram(df: pd.DataFrame, output_dir: Path):
    """
    Create phase diagrams showing when reward hacking emerges.

    For each alpha, plot a 2D heatmap with:
    - X-axis: initial proper probability
    - Y-axis: reward ratio
    - Color: fraction of trajectories converging to hack
    """
    alphas = sorted(df['alpha'].unique())

    fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 5))
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        # Aggregate data
        pivot_data = df[df['alpha'] == alpha].groupby(
            ['pi_proper_init', 'reward_ratio']
        )['converged_to_hack'].mean().reset_index()

        pivot_table = pivot_data.pivot(
            index='reward_ratio',
            columns='pi_proper_init',
            values='converged_to_hack'
        )

        # Plot heatmap
        sns.heatmap(
            pivot_table,
            ax=ax,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Fraction converged to hack'},
            annot=True,
            fmt='.2f'
        )

        ax.set_title(f'α = {alpha}')
        ax.set_xlabel('Initial P(proper)')
        ax.set_ylabel('Reward ratio (R_hack/R_proper)')

    plt.tight_layout()
    output_file = output_dir / 'phase_diagram.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved phase diagram to {output_file}")
    plt.close()


def plot_convergence_curves(results: List[SimulationResult], output_dir: Path, n_examples: int = 5):
    """
    Plot example trajectories showing probability evolution over time.

    Show a few different configurations to illustrate dynamics.
    """
    # Select diverse configurations
    configs_of_interest = [
        {'pi_proper_init': 0.5, 'reward_ratio': 0.5, 'alpha': 0.1},
        {'pi_proper_init': 0.5, 'reward_ratio': 0.9, 'alpha': 0.1},
        {'pi_proper_init': 0.1, 'reward_ratio': 0.7, 'alpha': 0.1},
        {'pi_proper_init': 0.9, 'reward_ratio': 0.7, 'alpha': 0.1},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, config_params in zip(axes, configs_of_interest):
        # Find matching results
        matching = [
            r for r in results
            if (abs(r.config.pi_proper_init - config_params['pi_proper_init']) < 0.01 and
                abs(r.config.r_hack / r.config.r_proper - config_params['reward_ratio']) < 0.01 and
                abs(r.config.alpha - config_params['alpha']) < 0.001)
        ]

        if not matching:
            continue

        # Plot first few trajectories
        for result in matching[:n_examples]:
            ax.plot(result.prob_hack, alpha=0.6, linewidth=0.8)

        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Training step')
        ax.set_ylabel('P(hack)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"π_proper={config_params['pi_proper_init']}, "
            f"R_ratio={config_params['reward_ratio']}, "
            f"α={config_params['alpha']}"
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'convergence_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved convergence curves to {output_file}")
    plt.close()


def plot_threshold_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Analyze the threshold where systems switch from proper → hack.

    For each alpha, plot P(converged to hack) vs. reward ratio.
    """
    alphas = sorted(df['alpha'].unique())
    pi_proper_values = sorted(df['pi_proper_init'].unique())

    fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 5))
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        for pi_proper in pi_proper_values:
            subset = df[(df['alpha'] == alpha) & (df['pi_proper_init'] == pi_proper)]

            agg = subset.groupby('reward_ratio')['converged_to_hack'].mean()

            ax.plot(agg.index, agg.values, marker='o', label=f'π_proper={pi_proper:.1f}')

        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Reward ratio (R_hack/R_proper)')
        ax.set_ylabel('Fraction converged to hack')
        ax.set_title(f'α = {alpha}')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'threshold_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved threshold analysis to {output_file}")
    plt.close()


def extract_scaling_laws(df: pd.DataFrame):
    """
    Attempt to derive simple scaling laws or predictive equations.

    Questions to answer:
    1. At what reward ratio does P(hack) = 0.5?
    2. How does this threshold depend on initial bias and alpha?
    3. Can we predict final behavior from initial conditions?
    """
    print("=" * 80)
    print("SCALING LAWS & PREDICTIVE EQUATIONS")
    print("=" * 80)

    # For each (pi_proper_init, alpha), find the critical reward ratio
    grouped = df.groupby(['pi_proper_init', 'alpha', 'reward_ratio'])['converged_to_hack'].mean().reset_index()

    print("\nCritical reward ratios (where P(converged to hack) ≈ 0.5):")

    for pi_proper in sorted(df['pi_proper_init'].unique()):
        for alpha in sorted(df['alpha'].unique()):
            subset = grouped[(grouped['pi_proper_init'] == pi_proper) & (grouped['alpha'] == alpha)]

            # Find where it crosses 0.5
            below = subset[subset['converged_to_hack'] < 0.5]
            above = subset[subset['converged_to_hack'] >= 0.5]

            if not below.empty and not above.empty:
                r_below = below['reward_ratio'].max()
                r_above = above['reward_ratio'].min()
                r_critical = (r_below + r_above) / 2
                print(f"  π_proper={pi_proper:.1f}, α={alpha:.2f}: R_critical ≈ {r_critical:.2f}")

    print()


def main():
    """Run full analysis pipeline."""
    # Find most recent results file
    results_dir = Path("results")
    results_files = sorted(results_dir.glob("simulation_results_*.pkl"))

    if not results_files:
        print("No results files found. Run run_experiments.py first.")
        return

    latest_file = results_files[-1]
    print(f"Loading results from: {latest_file}\n")

    # Load and analyze
    results = load_results(latest_file)
    df = create_summary_dataframe(results)

    print(f"Loaded {len(results)} simulation results")
    print(f"Unique configurations: {len(df.groupby(['pi_proper_init', 'reward_ratio', 'alpha']))}\n")

    # Analysis
    analyze_convergence(df)
    extract_scaling_laws(df)

    # Visualization
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")
    plot_phase_diagram(df, output_dir)
    plot_convergence_curves(results, output_dir)
    plot_threshold_analysis(df, output_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
