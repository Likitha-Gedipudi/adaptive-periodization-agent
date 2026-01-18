"""
Hyperparameter tuning using Optuna for the Adaptive Periodization Agent.

This module provides automated hyperparameter search for SAC and reward
function parameters.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.synthetic_data import generate_synthetic_dataset
from src.data.preprocess import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer, compute_reward_components
from src.environment.periodization_env import PeriodizationEnv
from src.models.sac_agent import create_sac_agent

logger = logging.getLogger(__name__)


def objective(
    trial: "optuna.Trial",
    base_config: Dict[str, Any],
    data,
    n_eval_episodes: int = 5,
) -> float:
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object.
        base_config: Base configuration.
        data: Preprocessed training data.
        n_eval_episodes: Episodes for evaluation.
        
    Returns:
        Mean evaluation reward.
    """
    # Sample hyperparameters
    sac_config = {
        "learning_rate_actor": trial.suggest_float("lr_actor", 1e-5, 1e-3, log=True),
        "learning_rate_critic": trial.suggest_float("lr_critic", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.01),
        "alpha": trial.suggest_float("alpha", 0.05, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "hidden_dims": [256, 128, 64],  # Keep architecture fixed
        "buffer_size": 100000,
        "auto_alpha": trial.suggest_categorical("auto_alpha", [True, False]),
    }
    
    # Reward weights
    reward_config = {
        "short_weight": trial.suggest_float("short_weight", 0.1, 0.4),
        "medium_weight": trial.suggest_float("medium_weight", 0.2, 0.5),
        "long_weight": trial.suggest_float("long_weight", 0.3, 0.6),
        "penalty_weight": trial.suggest_float("penalty_weight", 0.5, 2.0),
    }
    
    # Normalize reward weights
    total = reward_config["short_weight"] + reward_config["medium_weight"] + reward_config["long_weight"]
    reward_config["short_weight"] /= total
    reward_config["medium_weight"] /= total
    reward_config["long_weight"] /= total
    
    # Create environment
    env = PeriodizationEnv(
        data=data,
        episode_length=base_config["environment"]["episode_length"],
        apply_constraints=True,
        reward_weights=reward_config,
        seed=base_config["experiment"]["seed"],
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    agent = create_sac_agent(
        state_dim=state_dim,
        action_dim=6,
        config=sac_config,
        device="cpu",  # Use CPU for tuning (faster for small batches)
    )
    
    # Training parameters
    n_episodes = 100  # Shorter training for tuning
    warmup_steps = 500
    
    total_steps = 0
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        done = False
        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(obs, action_mask)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, next_obs, done, action_mask)
            
            if total_steps >= warmup_steps:
                agent.train_step()
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
        
        episode_rewards.append(episode_reward)
        
        # Report intermediate value for pruning
        if episode > 0 and episode % 20 == 0:
            intermediate_value = np.mean(episode_rewards[-20:])
            trial.report(intermediate_value, episode)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # Final evaluation
    eval_rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        done = False
        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(obs, action_mask, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)


def run_tuning(
    n_trials: int = 50,
    timeout: Optional[int] = None,
    config_path: Optional[str] = None,
    output_dir: str = "./experiments/tuning",
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning.
    
    Args:
        n_trials: Number of trials to run.
        timeout: Optional timeout in seconds.
        config_path: Path to base config.
        output_dir: Output directory for results.
        
    Returns:
        Best hyperparameters and study results.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")
    
    # Load base config
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    
    # Set up output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data once
    logger.info("Generating synthetic data for tuning...")
    data = generate_synthetic_dataset(
        n_users=50,  # Smaller dataset for faster tuning
        n_days=180,
        seed=base_config["experiment"]["seed"],
    )
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess(data, is_training=True)
    
    engineer = FeatureEngineer()
    data = engineer.engineer_features(data)
    data = compute_reward_components(data)
    
    logger.info(f"Data prepared: {len(data)} rows")
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=base_config["experiment"]["seed"]),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20),
        study_name="adaptive_periodization_tuning",
    )
    
    # Run optimization
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
    
    study.optimize(
        lambda trial: objective(trial, base_config, data),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )
    
    # Get results
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best trial value: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Save results
    results = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": len(study.trials),
        "datetime": datetime.now().isoformat(),
    }
    
    with open(output_dir / "best_params.yaml", "w") as f:
        yaml.dump(results, f)
    
    # Save study for visualization
    try:
        import joblib
        joblib.dump(study, output_dir / "study.pkl")
    except ImportError:
        pass
    
    return results


def main():
    """CLI entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Adaptive Periodization Agent")
    
    parser.add_argument(
        "--n-trials", "-n",
        type=int,
        default=50,
        help="Number of trials",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=None,
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to base config",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./experiments/tuning",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run tuning
    results = run_tuning(
        n_trials=args.n_trials,
        timeout=args.timeout,
        config_path=args.config,
        output_dir=args.output,
    )
    
    print("\n" + "=" * 50)
    print("Hyperparameter Tuning Complete!")
    print("=" * 50)
    print(f"Best Value: {results['best_value']:.4f}")
    print(f"Trials: {results['n_trials']}")
    print("\nBest Parameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
