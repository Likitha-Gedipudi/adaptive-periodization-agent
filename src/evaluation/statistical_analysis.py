"""
Statistical analysis utilities for comparing policies.

Provides rigorous statistical testing to prove significance
of improvements over baselines.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import scipy for statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, statistical tests disabled")


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two methods."""
    method_a: str
    method_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    difference: float
    relative_improvement: float
    p_value: float
    significant: bool
    test_used: str
    effect_size: float
    confidence_interval: Tuple[float, float]


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """
    Wilcoxon signed-rank test for paired samples.
    
    Args:
        scores_a: Scores from method A.
        scores_b: Scores from method B.
        alpha: Significance level.
        
    Returns:
        Tuple of (p_value, is_significant).
    """
    if not SCIPY_AVAILABLE:
        return 1.0, False
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")
    
    if len(scores_a) < 5:
        logger.warning("Fewer than 5 samples, results may not be reliable")
    
    try:
        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
        return float(p_value), p_value < alpha
    except ValueError as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        return 1.0, False


def mann_whitney_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """
    Mann-Whitney U test for independent samples.
    
    Args:
        scores_a: Scores from method A.
        scores_b: Scores from method B.
        alpha: Significance level.
        
    Returns:
        Tuple of (p_value, is_significant).
    """
    if not SCIPY_AVAILABLE:
        return 1.0, False
    
    statistic, p_value = stats.mannwhitneyu(
        scores_a, scores_b, alternative="two-sided"
    )
    return float(p_value), p_value < alpha


def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.
    
    Args:
        scores: Sample scores.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    rng = np.random.default_rng(42)
    scores = np.array(scores)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return float(lower), float(upper)


def cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        scores_a: Scores from method A.
        scores_b: Scores from method B.
        
    Returns:
        Cohen's d value.
    """
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    
    # Pooled standard deviation
    n_a = len(scores_a)
    n_b = len(scores_b)
    var_a = np.var(scores_a, ddof=1)
    var_b = np.var(scores_b, ddof=1)
    
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((mean_a - mean_b) / pooled_std)


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value.
        
    Returns:
        Interpretation string.
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compare_methods(
    method_a_name: str,
    method_b_name: str,
    scores_a: List[float],
    scores_b: List[float],
    paired: bool = True,
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    Compare two methods with statistical testing.
    
    Args:
        method_a_name: Name of method A.
        method_b_name: Name of method B.
        scores_a: Scores from method A.
        scores_b: Scores from method B.
        paired: Whether samples are paired.
        alpha: Significance level.
        
    Returns:
        ComparisonResult with all statistics.
    """
    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))
    std_a = float(np.std(scores_a))
    std_b = float(np.std(scores_b))
    
    difference = mean_a - mean_b
    
    if mean_b != 0:
        relative_improvement = (mean_a - mean_b) / abs(mean_b) * 100
    else:
        relative_improvement = float("inf") if mean_a > 0 else 0
    
    # Statistical test
    if paired:
        p_value, significant = wilcoxon_test(scores_a, scores_b, alpha)
        test_used = "Wilcoxon signed-rank"
    else:
        p_value, significant = mann_whitney_test(scores_a, scores_b, alpha)
        test_used = "Mann-Whitney U"
    
    # Effect size
    effect_size = cohens_d(scores_a, scores_b)
    
    # Confidence interval for the difference
    diffs = np.array(scores_a) - np.array(scores_b)
    ci = bootstrap_confidence_interval(diffs.tolist())
    
    return ComparisonResult(
        method_a=method_a_name,
        method_b=method_b_name,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        difference=difference,
        relative_improvement=relative_improvement,
        p_value=p_value,
        significant=significant,
        test_used=test_used,
        effect_size=effect_size,
        confidence_interval=ci,
    )


def print_comparison_table(results: List[ComparisonResult]) -> None:
    """
    Print formatted comparison table.
    
    Args:
        results: List of comparison results.
    """
    print("\n" + "=" * 100)
    print("STATISTICAL COMPARISON RESULTS")
    print("=" * 100)
    
    print(f"{'Method A':<15} {'Method B':<15} {'Diff':<10} {'%Improve':<10} "
          f"{'p-value':<10} {'Sig.':<6} {'Effect':<10}")
    print("-" * 100)
    
    for r in results:
        sig_str = "✓" if r.significant else "✗"
        effect_str = f"{r.effect_size:.2f} ({interpret_effect_size(r.effect_size)})"
        
        print(f"{r.method_a:<15} {r.method_b:<15} {r.difference:>+8.2f}  "
              f"{r.relative_improvement:>+7.1f}%   {r.p_value:>8.4f}  {sig_str:<6} {effect_str}")
    
    print("=" * 100)
    print(f"Significance level: α = 0.05")
    print(f"Effect size interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large")


def run_full_comparison(
    agent_scores: Dict[str, List[float]],
    baseline_scores: Dict[str, List[float]],
) -> List[ComparisonResult]:
    """
    Run full comparison of agent against all baselines.
    
    Args:
        agent_scores: Dictionary with agent name -> list of scores.
        baseline_scores: Dictionary with baseline name -> list of scores.
        
    Returns:
        List of comparison results.
    """
    results = []
    
    for agent_name, agent_vals in agent_scores.items():
        for baseline_name, baseline_vals in baseline_scores.items():
            result = compare_methods(
                agent_name,
                baseline_name,
                agent_vals,
                baseline_vals,
                paired=len(agent_vals) == len(baseline_vals),
            )
            results.append(result)
    
    return results
