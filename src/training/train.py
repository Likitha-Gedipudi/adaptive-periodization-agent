"""
Main training script for the Adaptive Periodization Agent.

This module provides the training loop and CLI interface for training
the SAC agent on fitness data.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.synthetic_data import generate_synthetic_dataset
from src.data.preprocess import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer, compute_reward_components
from src.environment.periodization_env import PeriodizationEnv
from src.models.sac_agent import SACAgent, create_sac_agent
from src.training.callbacks import (
    CallbackList,
    EarlyStoppingCallback,
    CheckpointCallback,
    MetricsLoggerCallback,
    ProgressCallback,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for the Adaptive Periodization Agent.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary.
            experiment_dir: Directory for experiment outputs.
        """
        self.config = config
        
        # Set up experiment directory
        if experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = config.get("experiment", {}).get("name") or f"run_{timestamp}"
            experiment_dir = Path(config["experiment"]["output_dir"]) / exp_name
        
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Set random seeds
        self.seed = config["experiment"]["seed"]
        self._set_seeds(self.seed)
        
        # Determine device
        self.device = self._get_device(config["experiment"]["device"])
        logger.info(f"Using device: {self.device}")
        
        # Initialize components (lazy loading)
        self.data = None
        self.env = None
        self.agent: Optional[SACAgent] = None
        self.callbacks = None
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _get_device(self, device_config: str) -> str:
        """Determine device to use."""
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device_config
    
    def setup(self) -> None:
        """Set up training components."""
        logger.info("Setting up training...")
        
        # Load or generate data
        self._setup_data()
        
        # Create environment
        self._setup_environment()
        
        # Create agent
        self._setup_agent()
        
        # Set up callbacks
        self._setup_callbacks()
        
        logger.info("Training setup complete")
    
    def _setup_data(self) -> None:
        """Load and preprocess data."""
        data_config = self.config["data"]
        
        if data_config["synthetic"]:
            logger.info("Generating synthetic data...")
            self.data = generate_synthetic_dataset(
                n_users=data_config["n_users"],
                n_days=data_config["n_days"],
                seed=self.seed,
            )
        else:
            raise NotImplementedError("Real data loading not yet implemented")
        
        # Preprocess
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        self.data = preprocessor.preprocess(self.data, is_training=True)
        
        # Feature engineering
        logger.info("Engineering features...")
        engineer = FeatureEngineer()
        self.data = engineer.engineer_features(self.data)
        
        # Compute reward components
        self.data = compute_reward_components(self.data)
        
        logger.info(f"Data prepared: {len(self.data)} rows, {len(self.data.columns)} features")
    
    def _setup_environment(self) -> None:
        """Create training environment."""
        env_config = self.config["environment"]
        reward_config = self.config["reward"]
        
        self.env = PeriodizationEnv(
            data=self.data,
            episode_length=env_config["episode_length"],
            apply_constraints=env_config["apply_constraints"],
            reward_weights={
                "short": reward_config["short_weight"],
                "medium": reward_config["medium_weight"],
                "long": reward_config["long_weight"],
                "penalty": reward_config["penalty_weight"],
            },
            seed=self.seed,
        )
        
        logger.info(f"Environment created: state_dim={self.env.observation_space.shape[0]}")
    
    def _setup_agent(self) -> None:
        """Create SAC agent."""
        sac_config = self.config["sac"]
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.agent = create_sac_agent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=sac_config,
            device=self.device,
        )
        
        logger.info(f"Agent created: SAC with state_dim={state_dim}, action_dim={action_dim}")
    
    def _setup_callbacks(self) -> None:
        """Set up training callbacks."""
        train_config = self.config["training"]
        exp_config = self.config["experiment"]
        es_config = self.config.get("early_stopping", {})
        
        callbacks = []
        
        # Progress bar
        callbacks.append(ProgressCallback(train_config["episodes"]))
        
        # Metrics logger
        callbacks.append(MetricsLoggerCallback(
            log_frequency=train_config["log_frequency"],
            log_to_tensorboard=exp_config["log_to_tensorboard"],
            log_to_wandb=exp_config["log_to_wandb"],
            tensorboard_dir=str(self.experiment_dir / "tensorboard"),
        ))
        
        # Checkpointing
        callbacks.append(CheckpointCallback(
            save_dir=str(self.experiment_dir / "checkpoints"),
            save_frequency=train_config["checkpoint_frequency"],
            save_best=True,
        ))
        
        # Early stopping
        if es_config.get("enabled", True):
            callbacks.append(EarlyStoppingCallback(
                patience=es_config.get("patience", 50),
                min_delta=es_config.get("min_delta", 0.01),
                monitor=es_config.get("monitor", "mean_reward"),
            ))
        
        self.callbacks = CallbackList(callbacks)
    
    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Returns:
            Training results dictionary.
        """
        train_config = self.config["training"]
        n_episodes = train_config["episodes"]
        warmup_steps = train_config["warmup_steps"]
        eval_frequency = train_config["eval_frequency"]
        
        logger.info(f"Starting training for {n_episodes} episodes...")
        
        # Training state
        total_steps = 0
        episode_rewards = []
        training_metrics = []
        
        self.callbacks.on_training_start(self)
        
        for episode in range(n_episodes):
            self.callbacks.on_episode_start(episode, self)
            
            # Run episode
            obs, info = self.env.reset(seed=self.seed + episode)
            episode_reward = 0
            episode_steps = 0
            
            done = False
            while not done:
                # Get action mask
                action_mask = self.env.get_action_mask()
                
                # Select action
                action = self.agent.select_action(
                    obs,
                    action_mask=action_mask,
                    deterministic=False,
                )
                
                # Step environment
                next_obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(
                    obs, action, reward, next_obs, done, action_mask
                )
                
                # Train agent
                if total_steps >= warmup_steps:
                    step_metrics = self.agent.train_step()
                    if step_metrics:
                        self.callbacks.on_step(total_steps, step_metrics, self)
                
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
            
            episode_rewards.append(episode_reward)
            
            # Episode metrics
            metrics = {
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "total_steps": total_steps,
                "mean_reward": np.mean(episode_rewards[-100:]),
            }
            
            # Periodic evaluation
            if episode > 0 and episode % eval_frequency == 0:
                eval_metrics = self._evaluate()
                metrics.update(eval_metrics)
            
            training_metrics.append(metrics)
            
            # Callbacks
            continue_training = self.callbacks.on_episode_end(episode, metrics, self)
            if not continue_training:
                logger.info(f"Training stopped at episode {episode}")
                break
        
        self.callbacks.on_training_end(self)
        
        # Final save
        self.agent.save(self.experiment_dir / "final_model.pt")
        
        results = {
            "total_episodes": episode + 1,
            "total_steps": total_steps,
            "best_reward": max(episode_rewards),
            "final_reward": np.mean(episode_rewards[-10:]),
            "metrics": training_metrics,
        }
        
        logger.info(
            f"Training complete: {results['total_episodes']} episodes, "
            f"best reward: {results['best_reward']:.2f}"
        )
        
        return results
    
    def _evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            n_episodes: Number of evaluation episodes.
            
        Returns:
            Evaluation metrics.
        """
        eval_rewards = []
        constraint_violations = []
        
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=self.seed + 10000 + ep)
            episode_reward = 0
            violations = 0
            
            done = False
            while not done:
                action_mask = self.env.get_action_mask()
                action = self.agent.select_action(obs, action_mask, deterministic=True)
                
                # Check violation
                if not action_mask[action]:
                    violations += 1
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            eval_rewards.append(episode_reward)
            constraint_violations.append(violations)
        
        return {
            "eval_mean_reward": np.mean(eval_rewards),
            "eval_std_reward": np.std(eval_rewards),
            "eval_violations": np.mean(constraint_violations),
        }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def train(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Train the agent with the given configuration.
    
    Args:
        config: Configuration dictionary (loads default if None).
        
    Returns:
        Training results.
    """
    if config is None:
        config = load_config()
    
    trainer = Trainer(config)
    trainer.setup()
    results = trainer.train()
    
    return results


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train Adaptive Periodization Agent")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=None,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cpu, cuda, mps, auto)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
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
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI arguments
    if args.episodes is not None:
        config["training"]["episodes"] = args.episodes
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.synthetic:
        config["data"]["synthetic"] = True
    if args.device:
        config["experiment"]["device"] = args.device
    if args.output:
        config["experiment"]["output_dir"] = args.output
    
    # Train
    results = train(config)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Episodes: {results['total_episodes']}")
    print(f"Total Steps: {results['total_steps']}")
    print(f"Best Reward: {results['best_reward']:.2f}")
    print(f"Final Reward (avg last 10): {results['final_reward']:.2f}")
    
    return results


if __name__ == "__main__":
    main()
