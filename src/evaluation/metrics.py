"""
Evaluation metrics for the Adaptive Periodization Agent.

This module provides domain-specific metrics for evaluating agent performance
including fitness improvement, recovery management, and safety compliance.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation results."""
    
    # Reward metrics
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    
    # Fitness metrics
    ctl_improvement: float
    fitness_trend: float
    
    # Recovery metrics
    mean_recovery: float
    days_above_60_pct: float
    recovery_volatility: float
    
    # Safety metrics
    constraint_violations: int
    violation_rate: float
    overtraining_score: float
    
    # Action distribution
    action_distribution: Dict[int, float]
    action_entropy: float
    
    # Additional stats
    episode_length_mean: float
    total_steps: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "ctl_improvement": self.ctl_improvement,
            "fitness_trend": self.fitness_trend,
            "mean_recovery": self.mean_recovery,
            "days_above_60_pct": self.days_above_60_pct,
            "recovery_volatility": self.recovery_volatility,
            "constraint_violations": self.constraint_violations,
            "violation_rate": self.violation_rate,
            "overtraining_score": self.overtraining_score,
            "action_distribution": self.action_distribution,
            "action_entropy": self.action_entropy,
            "episode_length_mean": self.episode_length_mean,
            "total_steps": self.total_steps,
        }


def compute_fitness_metrics(
    episode_data: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute fitness-related metrics from episode data.
    
    Args:
        episode_data: List of episode info dictionaries.
        
    Returns:
        Dictionary of fitness metrics.
    """
    if not episode_data:
        return {
            "ctl_improvement": 0.0,
            "fitness_trend": 0.0,
            "ctl_start": 0.0,
            "ctl_end": 0.0,
        }
    
    # Extract CTL values
    ctl_values = []
    for ep in episode_data:
        if "ctl" in ep:
            ctl_values.append(ep["ctl"])
    
    if len(ctl_values) < 2:
        return {
            "ctl_improvement": 0.0,
            "fitness_trend": 0.0,
            "ctl_start": ctl_values[0] if ctl_values else 0.0,
            "ctl_end": ctl_values[-1] if ctl_values else 0.0,
        }
    
    # CTL improvement
    ctl_start = np.mean(ctl_values[:7])  # First week average
    ctl_end = np.mean(ctl_values[-7:])   # Last week average
    ctl_improvement = ctl_end - ctl_start
    
    # Fitness trend (linear slope)
    x = np.arange(len(ctl_values))
    slope, _, r_value, _, _ = stats.linregress(x, ctl_values)
    fitness_trend = slope * 100  # Per 100 days
    
    return {
        "ctl_improvement": ctl_improvement,
        "fitness_trend": fitness_trend,
        "ctl_start": ctl_start,
        "ctl_end": ctl_end,
        "ctl_r_squared": r_value ** 2,
    }


def compute_recovery_metrics(
    episode_data: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute recovery management metrics from episode data.
    
    Args:
        episode_data: List of episode info dictionaries.
        
    Returns:
        Dictionary of recovery metrics.
    """
    if not episode_data:
        return {
            "mean_recovery": 0.0,
            "days_above_60_pct": 0.0,
            "recovery_volatility": 0.0,
            "min_recovery": 0.0,
            "max_recovery": 0.0,
        }
    
    # Extract recovery values
    recovery_values = []
    for ep in episode_data:
        if "recovery" in ep:
            recovery_values.append(ep["recovery"])
    
    if not recovery_values:
        return {
            "mean_recovery": 0.0,
            "days_above_60_pct": 0.0,
            "recovery_volatility": 0.0,
            "min_recovery": 0.0,
            "max_recovery": 0.0,
        }
    
    recovery_array = np.array(recovery_values)
    
    return {
        "mean_recovery": float(np.mean(recovery_array)),
        "days_above_60_pct": float(np.mean(recovery_array >= 60)) * 100,
        "recovery_volatility": float(np.std(recovery_array)),
        "min_recovery": float(np.min(recovery_array)),
        "max_recovery": float(np.max(recovery_array)),
    }


def compute_safety_metrics(
    actions: List[int],
    action_masks: List[np.ndarray],
    state_history: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute safety and constraint metrics.
    
    Args:
        actions: List of actions taken.
        action_masks: List of action masks at each step.
        state_history: List of state dictionaries.
        
    Returns:
        Dictionary of safety metrics.
    """
    if not actions:
        return {
            "constraint_violations": 0,
            "violation_rate": 0.0,
            "overtraining_score": 0.0,
            "consecutive_high_max": 0,
        }
    
    violations = 0
    consecutive_high = 0
    max_consecutive_high = 0
    
    for i, (action, mask) in enumerate(zip(actions, action_masks)):
        # Count violations
        if not mask[action]:
            violations += 1
        
        # Track consecutive high intensity
        if action >= 3:  # Tempo, HIIT, Strength
            consecutive_high += 1
            max_consecutive_high = max(max_consecutive_high, consecutive_high)
        else:
            consecutive_high = 0
    
    # Calculate overtraining score based on state history
    overtraining_indicators = 0
    for state in state_history:
        recovery = state.get("recovery", 50)
        if recovery < 40:
            overtraining_indicators += 1
    
    overtraining_score = overtraining_indicators / len(state_history) if state_history else 0
    
    return {
        "constraint_violations": violations,
        "violation_rate": violations / len(actions) * 100,
        "overtraining_score": overtraining_score,
        "consecutive_high_max": max_consecutive_high,
    }


def compute_action_metrics(
    actions: List[int],
    n_actions: int = 6,
) -> Dict[str, Any]:
    """
    Compute action distribution metrics.
    
    Args:
        actions: List of actions taken.
        n_actions: Number of action types.
        
    Returns:
        Dictionary with action distribution and entropy.
    """
    if not actions:
        return {
            "action_distribution": {i: 0.0 for i in range(n_actions)},
            "action_entropy": 0.0,
        }
    
    action_counts = np.bincount(actions, minlength=n_actions)
    action_probs = action_counts / len(actions)
    
    # Distribution as dict
    distribution = {i: float(p) for i, p in enumerate(action_probs)}
    
    # Entropy (higher = more diverse actions)
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
    
    return {
        "action_distribution": distribution,
        "action_entropy": float(entropy),
    }


def aggregate_episode_metrics(
    episode_results: List[Dict[str, Any]],
) -> EvaluationMetrics:
    """
    Aggregate metrics across multiple episodes.
    
    Args:
        episode_results: List of per-episode metrics.
        
    Returns:
        Aggregated EvaluationMetrics.
    """
    if not episode_results:
        raise ValueError("No episode results to aggregate")
    
    # Reward statistics
    rewards = [ep.get("reward", 0) for ep in episode_results]
    
    # Aggregate fitness
    ctl_improvements = [ep.get("ctl_improvement", 0) for ep in episode_results]
    
    # Aggregate recovery
    mean_recoveries = [ep.get("mean_recovery", 50) for ep in episode_results]
    pct_above_60 = [ep.get("days_above_60_pct", 50) for ep in episode_results]
    
    # Aggregate safety
    violations = sum(ep.get("constraint_violations", 0) for ep in episode_results)
    total_steps = sum(ep.get("steps", 90) for ep in episode_results)
    
    # Aggregate actions
    all_actions = []
    for ep in episode_results:
        all_actions.extend(ep.get("actions", []))
    action_metrics = compute_action_metrics(all_actions)
    
    return EvaluationMetrics(
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
        ctl_improvement=float(np.mean(ctl_improvements)),
        fitness_trend=float(np.mean([ep.get("fitness_trend", 0) for ep in episode_results])),
        mean_recovery=float(np.mean(mean_recoveries)),
        days_above_60_pct=float(np.mean(pct_above_60)),
        recovery_volatility=float(np.std(mean_recoveries)),
        constraint_violations=violations,
        violation_rate=violations / total_steps * 100 if total_steps > 0 else 0.0,
        overtraining_score=float(np.mean([ep.get("overtraining_score", 0) for ep in episode_results])),
        action_distribution=action_metrics["action_distribution"],
        action_entropy=action_metrics["action_entropy"],
        episode_length_mean=float(np.mean([ep.get("steps", 90) for ep in episode_results])),
        total_steps=total_steps,
    )


def statistical_comparison(
    agent_rewards: List[float],
    baseline_rewards: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform statistical comparison between agent and baseline.
    
    Args:
        agent_rewards: Agent episode rewards.
        baseline_rewards: Baseline episode rewards.
        alpha: Significance level.
        
    Returns:
        Statistical comparison results.
    """
    # Wilcoxon signed-rank test (non-parametric)
    n_samples = min(len(agent_rewards), len(baseline_rewards))
    
    if n_samples < 5:
        return {
            "test": "insufficient_samples",
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
        }
    
    agent_sample = agent_rewards[:n_samples]
    baseline_sample = baseline_rewards[:n_samples]
    
    try:
        stat, p_value = stats.wilcoxon(agent_sample, baseline_sample)
        
        # Effect size (rank-biserial correlation)
        diff = np.array(agent_sample) - np.array(baseline_sample)
        effect_size = np.mean(diff) / (np.std(diff) + 1e-8)  # Cohen's d approximation
        
        return {
            "test": "wilcoxon",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "effect_size": float(effect_size),
            "agent_mean": float(np.mean(agent_sample)),
            "baseline_mean": float(np.mean(baseline_sample)),
            "improvement": float(np.mean(agent_sample) - np.mean(baseline_sample)),
        }
    except Exception as e:
        logger.warning(f"Statistical test failed: {e}")
        return {
            "test": "failed",
            "error": str(e),
            "p_value": 1.0,
            "significant": False,
        }
