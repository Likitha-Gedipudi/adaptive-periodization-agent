"""
Cross-validation framework for user-level evaluation.

Provides k-fold cross-validation to verify generalization
to unseen users.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result from a single CV fold."""
    fold: int
    train_users: List[int]
    test_users: List[int]
    train_reward: float
    test_reward: float
    train_std: float
    test_std: float
    violations: int


@dataclass
class CVResult:
    """Complete cross-validation result."""
    n_folds: int
    train_mean: float
    train_std: float
    test_mean: float
    test_std: float
    generalization_gap: float
    fold_results: List[FoldResult]


def user_kfold_split(
    data: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Create k-fold splits at the user level.
    
    Args:
        data: DataFrame with user_id column.
        n_folds: Number of folds.
        seed: Random seed.
        
    Returns:
        List of (train_users, test_users) tuples.
    """
    users = data["user_id"].unique()
    n_users = len(users)
    
    rng = np.random.default_rng(seed)
    shuffled_users = rng.permutation(users)
    
    fold_size = n_users // n_folds
    folds = []
    
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_users
        
        test_users = shuffled_users[start:end].tolist()
        train_users = [u for u in shuffled_users if u not in test_users]
        
        folds.append((train_users, test_users))
    
    return folds


def run_cross_validation(
    data: pd.DataFrame,
    n_folds: int = 5,
    n_episodes: int = 100,
    eval_episodes: int = 20,
    seed: int = 42,
    device: str = "cpu",
) -> CVResult:
    """
    Run k-fold cross-validation.
    
    Args:
        data: Preprocessed data with all features.
        n_folds: Number of CV folds.
        n_episodes: Training episodes per fold.
        eval_episodes: Evaluation episodes.
        seed: Random seed.
        device: Device for training.
        
    Returns:
        CVResult with all metrics.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.environment.periodization_env import PeriodizationEnv
    from src.models.sac_agent import SACAgent
    
    folds = user_kfold_split(data, n_folds, seed)
    fold_results = []
    
    for fold_idx, (train_users, test_users) in enumerate(folds):
        logger.info(f"Running fold {fold_idx + 1}/{n_folds}")
        logger.info(f"  Train users: {len(train_users)}, Test users: {len(test_users)}")
        
        # Split data
        train_data = data[data["user_id"].isin(train_users)]
        test_data = data[data["user_id"].isin(test_users)]
        
        # Create environments
        train_env = PeriodizationEnv(
            data=train_data,
            episode_length=90,
            seed=seed + fold_idx,
        )
        
        test_env = PeriodizationEnv(
            data=test_data,
            episode_length=90,
            seed=seed + fold_idx + 1000,
        )
        
        # Create agent
        state_dim = train_env.observation_space.shape[0]
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=6,
            device=device,
        )
        
        # Training
        train_rewards = []
        for ep in range(n_episodes):
            obs, _ = train_env.reset(seed=seed + fold_idx * 1000 + ep)
            episode_reward = 0
            
            done = False
            while not done:
                mask = train_env.get_action_mask()
                action = agent.select_action(obs, mask, deterministic=False)
                
                next_obs, reward, terminated, truncated, _ = train_env.step(action)
                agent.store_transition(obs, action, reward, next_obs, terminated)
                
                if len(agent.buffer) >= agent.batch_size:
                    agent.train_step()
                
                episode_reward += reward
                obs = next_obs
                done = terminated or truncated
            
            train_rewards.append(episode_reward)
        
        # Evaluate on test set
        test_rewards = []
        violations = 0
        
        for ep in range(eval_episodes):
            obs, _ = test_env.reset(seed=seed + fold_idx * 10000 + ep)
            episode_reward = 0
            
            done = False
            while not done:
                mask = test_env.get_action_mask()
                action = agent.select_action(obs, mask, deterministic=True)
                
                next_obs, reward, terminated, truncated, info = test_env.step(action)
                
                if info.get("recovery", 50) < 30 and action >= 3:
                    violations += 1
                
                episode_reward += reward
                obs = next_obs
                done = terminated or truncated
            
            test_rewards.append(episode_reward)
        
        fold_result = FoldResult(
            fold=fold_idx,
            train_users=train_users,
            test_users=test_users,
            train_reward=float(np.mean(train_rewards[-20:])),
            test_reward=float(np.mean(test_rewards)),
            train_std=float(np.std(train_rewards[-20:])),
            test_std=float(np.std(test_rewards)),
            violations=violations,
        )
        
        fold_results.append(fold_result)
        logger.info(f"  Train reward: {fold_result.train_reward:.2f}")
        logger.info(f"  Test reward:  {fold_result.test_reward:.2f}")
    
    # Aggregate results
    train_means = [f.train_reward for f in fold_results]
    test_means = [f.test_reward for f in fold_results]
    
    cv_result = CVResult(
        n_folds=n_folds,
        train_mean=float(np.mean(train_means)),
        train_std=float(np.std(train_means)),
        test_mean=float(np.mean(test_means)),
        test_std=float(np.std(test_means)),
        generalization_gap=float(np.mean(train_means) - np.mean(test_means)),
        fold_results=fold_results,
    )
    
    return cv_result


def print_cv_results(result: CVResult) -> None:
    """Print formatted cross-validation results."""
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"\nNumber of folds: {result.n_folds}")
    print(f"\nTrain Reward:  {result.train_mean:.2f} ± {result.train_std:.2f}")
    print(f"Test Reward:   {result.test_mean:.2f} ± {result.test_std:.2f}")
    print(f"Gap:           {result.generalization_gap:.2f} ({result.generalization_gap / result.train_mean * 100:.1f}%)")
    
    print("\nPer-Fold Results:")
    print("-" * 60)
    print(f"{'Fold':<6} {'Train':<15} {'Test':<15} {'Violations':<12}")
    print("-" * 60)
    
    for f in result.fold_results:
        print(f"{f.fold + 1:<6} {f.train_reward:>10.2f} ± {f.train_std:>4.1f}  "
              f"{f.test_reward:>10.2f} ± {f.test_std:>4.1f}  {f.violations}")
    
    print("=" * 60)
    
    # Interpretation
    gap_percent = abs(result.generalization_gap / result.train_mean * 100)
    if gap_percent < 10:
        print("✓ Excellent generalization (gap < 10%)")
    elif gap_percent < 20:
        print("○ Good generalization (gap 10-20%)")
    else:
        print("✗ Potential overfitting (gap > 20%)")
