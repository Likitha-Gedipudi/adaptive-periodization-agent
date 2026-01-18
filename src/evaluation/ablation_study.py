"""
Ablation study framework for the Adaptive Periodization Agent.

This module allows systematic evaluation of different configuration variants
to understand which components are critical for performance.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_data import generate_synthetic_dataset
from src.data.preprocess import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer, compute_reward_components
from src.environment.periodization_env import PeriodizationEnv
from src.models.sac_agent import SACAgent

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""
    name: str
    base_reward: float = 6.0
    penalty_weight: float = 0.3
    apply_masking: bool = True
    short_weight: float = 0.5
    medium_weight: float = 0.3
    long_weight: float = 0.2
    learning_rate: float = 3e-4
    hidden_dims: List[int] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


# Define ablation experiments
ABLATION_EXPERIMENTS = {
    "baseline": AblationConfig(
        name="baseline",
        base_reward=6.0,
        penalty_weight=0.3,
        apply_masking=True,
    ),
    "no_base_reward": AblationConfig(
        name="no_base_reward",
        base_reward=0.0,
        penalty_weight=0.3,
        apply_masking=True,
    ),
    "high_penalty": AblationConfig(
        name="high_penalty",
        base_reward=6.0,
        penalty_weight=1.0,
        apply_masking=True,
    ),
    "no_masking": AblationConfig(
        name="no_masking",
        base_reward=6.0,
        penalty_weight=0.3,
        apply_masking=False,
    ),
    "short_term_only": AblationConfig(
        name="short_term_only",
        base_reward=6.0,
        penalty_weight=0.3,
        apply_masking=True,
        short_weight=1.0,
        medium_weight=0.0,
        long_weight=0.0,
    ),
    "equal_weights": AblationConfig(
        name="equal_weights",
        base_reward=6.0,
        penalty_weight=0.3,
        apply_masking=True,
        short_weight=0.33,
        medium_weight=0.33,
        long_weight=0.34,
    ),
    "small_network": AblationConfig(
        name="small_network",
        base_reward=6.0,
        penalty_weight=0.3,
        apply_masking=True,
        hidden_dims=[64, 64],
    ),
}


@dataclass
class AblationResult:
    """Results from an ablation experiment."""
    config_name: str
    seed: int
    mean_reward: float
    std_reward: float
    best_reward: float
    violations: int
    episodes: int
    training_time: float


def run_single_ablation(
    config: AblationConfig,
    n_episodes: int = 100,
    seed: int = 42,
    n_users: int = 100,
    device: str = "cpu",
) -> AblationResult:
    """
    Run a single ablation experiment.
    
    Args:
        config: Ablation configuration.
        n_episodes: Number of training episodes.
        seed: Random seed.
        n_users: Number of synthetic users.
        device: Device for training.
        
    Returns:
        AblationResult with metrics.
    """
    import time
    start_time = time.time()
    
    # Set seeds
    np.random.seed(seed)
    
    # Generate data
    data = generate_synthetic_dataset(n_users=n_users, n_days=180, seed=seed)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess(data, is_training=True)
    
    # Feature engineering
    engineer = FeatureEngineer()
    data = engineer.engineer_features(data)
    data = compute_reward_components(data)
    
    # Create environment with config
    reward_weights = {
        "short": config.short_weight,
        "medium": config.medium_weight,
        "long": config.long_weight,
        "penalty": config.penalty_weight,
    }
    
    env = PeriodizationEnv(
        data=data,
        episode_length=90,
        reward_weights=reward_weights,
        apply_constraints=config.apply_masking,
        seed=seed,
    )
    
    # Patch base reward if needed (temporary modification)
    original_calc = env._calculate_reward
    def patched_reward(current_idx, action):
        reward = original_calc(current_idx, action)
        # Adjust base reward
        default_base = 6.0
        reward = reward - default_base + config.base_reward
        return reward
    env._calculate_reward = patched_reward
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=6,
        hidden_dims=config.hidden_dims,
        lr_actor=config.learning_rate,
        lr_critic=config.learning_rate,
        device=device,
    )
    
    # Training
    episode_rewards = []
    total_violations = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0
        
        done = False
        while not done:
            mask = env.get_action_mask() if config.apply_masking else None
            action = agent.select_action(obs, mask, deterministic=False)
            
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Count violations (high intensity with low recovery)
            if step_info.get("recovery", 50) < 30 and action >= 3:
                total_violations += 1
            
            agent.store_transition(obs, action, reward, next_obs, terminated)
            
            if len(agent.buffer) >= agent.batch_size:
                agent.train_step()
            
            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
    
    training_time = time.time() - start_time
    
    return AblationResult(
        config_name=config.name,
        seed=seed,
        mean_reward=float(np.mean(episode_rewards[-20:])),  # Last 20 episodes
        std_reward=float(np.std(episode_rewards[-20:])),
        best_reward=float(np.max(episode_rewards)),
        violations=total_violations,
        episodes=n_episodes,
        training_time=training_time,
    )


def run_ablation_study(
    experiments: Optional[List[str]] = None,
    n_seeds: int = 3,
    n_episodes: int = 100,
    output_dir: str = "experiments/ablations",
) -> Dict[str, List[AblationResult]]:
    """
    Run complete ablation study.
    
    Args:
        experiments: List of experiment names (None = all).
        n_seeds: Number of random seeds per experiment.
        n_episodes: Episodes per run.
        output_dir: Output directory for results.
        
    Returns:
        Dictionary mapping experiment name to list of results.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    if experiments is None:
        experiments = list(ABLATION_EXPERIMENTS.keys())
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for exp_name in experiments:
        config = ABLATION_EXPERIMENTS[exp_name]
        logger.info(f"Running ablation: {exp_name}")
        
        results = []
        for seed in range(n_seeds):
            logger.info(f"  Seed {seed + 1}/{n_seeds}")
            
            result = run_single_ablation(
                config=config,
                n_episodes=n_episodes,
                seed=42 + seed,
            )
            results.append(result)
            logger.info(f"    Mean reward: {result.mean_reward:.2f}")
        
        all_results[exp_name] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"ablation_results_{timestamp}.json"
    
    # Convert to serializable format
    serializable = {
        name: [asdict(r) for r in results]
        for name, results in all_results.items()
    }
    
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Experiment':<20} {'Mean Reward':<15} {'Std':<10} {'Violations':<12} {'Interpretation'}")
    print("-" * 80)
    
    for name, results in all_results.items():
        mean_rewards = [r.mean_reward for r in results]
        violations = sum(r.violations for r in results)
        
        avg = np.mean(mean_rewards)
        std = np.std(mean_rewards)
        
        # Simple interpretation
        if name == "baseline":
            interp = "Full system"
        elif avg < all_results["baseline"][0].mean_reward * 0.5:
            interp = "Critical component!"
        elif violations > 10:
            interp = "Unsafe configuration"
        else:
            interp = "Minor impact"
        
        print(f"{name:<20} {avg:>10.2f} ± {std:>4.2f}    {violations:>5}       {interp}")
    
    print("=" * 80)
    
    return all_results


def main():
    """CLI entry point for ablation studies."""
    parser = argparse.ArgumentParser(description="Run ablation studies")
    
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        default=None,
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        default=3,
        help="Number of random seeds (default: 3)",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=100,
        help="Episodes per run (default: 100)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="experiments/ablations",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    run_ablation_study(
        experiments=args.experiments,
        n_seeds=args.seeds,
        n_episodes=args.episodes,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
