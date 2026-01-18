"""
Visualization utilities for the Adaptive Periodization Agent.

This module provides plotting functions for training curves, policy analysis,
and baseline comparisons.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Check matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, plotting disabled")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def set_style() -> None:
    """Set plotting style."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.style.use("seaborn-v0_8-whitegrid")
    
    if SEABORN_AVAILABLE:
        sns.set_palette("husl")


def plot_training_curves(
    metrics_history: List[Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot training curves (reward, loss, entropy).
    
    Args:
        metrics_history: List of training metrics per episode.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
        
    Returns:
        Figure object or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available")
        return None
    
    set_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    episodes = range(len(metrics_history))
    
    # Extract metrics
    rewards = [m.get("episode_reward", 0) for m in metrics_history]
    mean_rewards = [m.get("mean_reward", 0) for m in metrics_history]
    
    # Plot 1: Episode Reward
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, color="blue", label="Episode")
    ax1.plot(episodes, mean_rewards, color="blue", linewidth=2, label="Rolling Mean")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation metrics if available
    ax2 = axes[0, 1]
    eval_rewards = [m.get("eval_mean_reward") for m in metrics_history if m.get("eval_mean_reward")]
    eval_episodes = [i for i, m in enumerate(metrics_history) if m.get("eval_mean_reward")]
    
    if eval_rewards:
        ax2.plot(eval_episodes, eval_rewards, "o-", color="green", linewidth=2)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Evaluation Reward")
        ax2.set_title("Evaluation Performance")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No evaluation data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Evaluation Performance")
    
    # Plot 3: Training steps
    ax3 = axes[1, 0]
    total_steps = np.cumsum([m.get("episode_steps", 90) for m in metrics_history])
    ax3.plot(episodes, total_steps, color="purple")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Total Steps")
    ax3.set_title("Training Progress")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Entropy (if available)
    ax4 = axes[1, 1]
    entropies = [m.get("entropy") for m in metrics_history if m.get("entropy")]
    entropy_eps = [i for i, m in enumerate(metrics_history) if m.get("entropy")]
    
    if entropies:
        ax4.plot(entropy_eps, entropies, color="orange")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Policy Entropy")
        ax4.set_title("Action Diversity")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No entropy data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Action Diversity")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training curves to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_policy_analysis(
    action_distribution: Dict[int, float],
    action_names: Optional[List[str]] = None,
    title: str = "Action Distribution",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot action distribution for policy analysis.
    
    Args:
        action_distribution: Dictionary of action -> probability.
        action_names: Optional names for actions.
        title: Plot title.
        save_path: Optional save path.
        show: Whether to display.
        
    Returns:
        Figure or None.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    set_style()
    
    if action_names is None:
        action_names = ["Rest", "Active Recovery", "Aerobic", "Tempo", "HIIT", "Strength"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    actions = list(range(len(action_names)))
    probs = [action_distribution.get(i, 0) for i in actions]
    
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e74c3c", "#1abc9c"]
    
    bars = ax.bar(actions, probs, color=colors, edgecolor="white", linewidth=2)
    
    ax.set_xticks(actions)
    ax.set_xticklabels(action_names, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.set_ylim(0, max(probs) * 1.2 if probs else 1)
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prob:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def plot_baseline_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = "mean_reward",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot comparison of agent with baselines.
    
    Args:
        results: Evaluation results dictionary.
        metric: Metric to compare.
        save_path: Optional save path.
        show: Whether to display.
        
    Returns:
        Figure or None.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    set_style()
    
    # Extract policy names and metrics
    policies = []
    values = []
    errors = []
    
    for name, data in results.items():
        if isinstance(data, dict) and metric in data:
            policies.append(name.replace("_", " ").title())
            values.append(data[metric])
            errors.append(data.get(f"std_{metric.replace('mean_', '')}", 0))
    
    if not policies:
        logger.warning("No data to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(policies))
    
    # Color agent differently
    colors = ["#e74c3c" if "Agent" in p else "#3498db" for p in policies]
    
    bars = ax.bar(x, values, yerr=errors if any(errors) else None, 
                  color=colors, edgecolor="white", linewidth=2,
                  capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Policy Comparison: {metric.replace('_', ' ').title()}")
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    # Legend
    agent_patch = mpatches.Patch(color="#e74c3c", label="Trained Agent")
    baseline_patch = mpatches.Patch(color="#3498db", label="Baseline")
    ax.legend(handles=[agent_patch, baseline_patch], loc="upper right")
    
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def plot_episode_trajectory(
    episode_data: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot a single episode trajectory showing actions and metrics.
    
    Args:
        episode_data: List of step data dictionaries.
        save_path: Optional save path.
        show: Whether to display.
        
    Returns:
        Figure or None.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    set_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    days = range(len(episode_data))
    
    # Extract data
    actions = [d.get("action", 0) for d in episode_data]
    recoveries = [d.get("recovery", 50) for d in episode_data]
    
    # Plot 1: Actions
    ax1 = axes[0]
    action_names = ["Rest", "AR", "Aerobic", "Tempo", "HIIT", "Str"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e74c3c", "#1abc9c"]
    action_colors = [colors[a] for a in actions]
    
    ax1.bar(days, [1] * len(days), color=action_colors, width=1)
    ax1.set_ylabel("Training Type")
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks([])
    ax1.set_title("Training Prescription Timeline")
    
    # Legend
    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(colors, action_names)]
    ax1.legend(handles=patches, loc="upper right", ncol=6)
    
    # Plot 2: Recovery
    ax2 = axes[1]
    ax2.plot(days, recoveries, color="#2ecc71", linewidth=2)
    ax2.axhline(y=60, color="orange", linestyle="--", label="Target (60%)")
    ax2.axhline(y=30, color="red", linestyle="--", label="Critical (30%)")
    ax2.fill_between(days, 0, recoveries, alpha=0.3, color="#2ecc71")
    ax2.set_ylabel("Recovery Score")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right")
    ax2.set_title("Recovery Over Time")
    
    # Plot 3: Action intensity
    ax3 = axes[2]
    intensities = [a for a in actions]  # Simple mapping
    ax3.plot(days, intensities, "o-", color="#3498db", markersize=3)
    ax3.set_ylabel("Action Intensity")
    ax3.set_xlabel("Day")
    ax3.set_ylim(-0.5, 5.5)
    ax3.set_yticks(range(6))
    ax3.set_yticklabels(action_names)
    ax3.set_title("Action Sequence")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def create_report_figures(
    training_metrics: List[Dict[str, float]],
    evaluation_results: Dict[str, Any],
    output_dir: Union[str, Path],
) -> List[Path]:
    """
    Create all figures for a training report.
    
    Args:
        training_metrics: Training history.
        evaluation_results: Evaluation results.
        output_dir: Output directory.
        
    Returns:
        List of saved figure paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # Training curves
    path = output_dir / "training_curves.png"
    plot_training_curves(training_metrics, save_path=path, show=False)
    saved_paths.append(path)
    
    # Action distribution (if available)
    if "agent" in evaluation_results:
        agent_dist = evaluation_results["agent"].get("action_distribution", {})
        if agent_dist:
            path = output_dir / "action_distribution.png"
            plot_policy_analysis(agent_dist, save_path=path, show=False)
            saved_paths.append(path)
    
    # Baseline comparison
    path = output_dir / "baseline_comparison.png"
    plot_baseline_comparison(evaluation_results, save_path=path, show=False)
    saved_paths.append(path)
    
    logger.info(f"Created {len(saved_paths)} report figures in {output_dir}")
    
    return saved_paths
