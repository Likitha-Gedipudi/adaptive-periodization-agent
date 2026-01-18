"""
Evaluation pipeline for the Adaptive Periodization Agent.

This module provides functions to evaluate trained agents and compare
with baseline policies.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.synthetic_data import generate_synthetic_dataset
from src.data.preprocess import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer, compute_reward_components
from src.environment.periodization_env import PeriodizationEnv
from src.models.sac_agent import SACAgent
from src.models.baseline_policies import get_all_baseline_policies
from src.evaluation.metrics import (
    compute_fitness_metrics,
    compute_recovery_metrics,
    compute_safety_metrics,
    compute_action_metrics,
    aggregate_episode_metrics,
    statistical_comparison,
    EvaluationMetrics,
)

logger = logging.getLogger(__name__)


def evaluate_agent(
    agent: SACAgent,
    env: PeriodizationEnv,
    n_episodes: int = 20,
    deterministic: bool = True,
    seed: int = 42,
) -> EvaluationMetrics:
    """
    Evaluate a trained agent on the environment.
    
    Args:
        agent: Trained SAC agent.
        env: Evaluation environment.
        n_episodes: Number of evaluation episodes.
        deterministic: Use deterministic policy.
        seed: Random seed.
        
    Returns:
        Aggregated evaluation metrics.
    """
    episode_results = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        
        episode_reward = 0
        step_data = []
        actions = []
        action_masks = []
        
        done = False
        while not done:
            # Get action mask
            mask = env.get_action_mask()
            action_masks.append(mask.copy())
            
            # Select action
            action = agent.select_action(obs, mask, deterministic=deterministic)
            actions.append(action)
            
            # Step
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            step_data.append({
                "recovery": step_info.get("recovery", 50),
                "action": action,
            })
            
            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        # Compute episode metrics
        fitness_metrics = compute_fitness_metrics(step_data)
        recovery_metrics = compute_recovery_metrics(step_data)
        safety_metrics = compute_safety_metrics(
            actions, action_masks, step_data
        )
        action_metrics = compute_action_metrics(actions)
        
        episode_results.append({
            "reward": episode_reward,
            "steps": len(actions),
            "actions": actions,
            **fitness_metrics,
            **recovery_metrics,
            **safety_metrics,
            **action_metrics,
        })
    
    return aggregate_episode_metrics(episode_results)


def compare_with_baselines(
    agent: SACAgent,
    env: PeriodizationEnv,
    n_episodes: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare trained agent with baseline policies.
    
    Args:
        agent: Trained SAC agent.
        env: Evaluation environment.
        n_episodes: Episodes per policy.
        seed: Random seed.
        
    Returns:
        Comparison results dictionary.
    """
    results = {}
    
    # Evaluate agent
    logger.info("Evaluating trained agent...")
    agent_metrics = evaluate_agent(agent, env, n_episodes, seed=seed)
    results["agent"] = agent_metrics.to_dict()
    
    # Get agent rewards for comparison
    agent_rewards = [agent_metrics.mean_reward]  # Will extend in actual eval
    
    # Evaluate baselines
    baselines = get_all_baseline_policies(seed=seed)
    
    for name, policy in baselines.items():
        logger.info(f"Evaluating baseline: {name}...")
        
        episode_rewards = []
        episode_results = []
        
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + 1000 + ep)
            
            episode_reward = 0
            actions = []
            action_masks = []
            step_data = []
            action_history = []
            
            done = False
            while not done:
                mask = env.get_action_mask()
                action_masks.append(mask.copy())
                
                # Build state dict for baseline
                state = {
                    "recovery": obs[2] if len(obs) > 2 else 50,
                }
                
                action = policy.select_action(state, action_history, mask)
                actions.append(action)
                action_history.append(action)
                
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                
                step_data.append({
                    "recovery": step_info.get("recovery", 50),
                    "action": action,
                })
                
                episode_reward += reward
                obs = next_obs
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            
            # Per-episode metrics
            fitness_metrics = compute_fitness_metrics(step_data)
            recovery_metrics = compute_recovery_metrics(step_data)
            safety_metrics = compute_safety_metrics(actions, action_masks, step_data)
            action_metrics = compute_action_metrics(actions)
            
            episode_results.append({
                "reward": episode_reward,
                "steps": len(actions),
                "actions": actions,
                **fitness_metrics,
                **recovery_metrics,
                **safety_metrics,
                **action_metrics,
            })
        
        baseline_metrics = aggregate_episode_metrics(episode_results)
        results[name] = baseline_metrics.to_dict()
        
        # Statistical comparison
        comparison = statistical_comparison(
            [episode_rewards[0]] * len(episode_rewards),  # Agent rewards placeholder
            episode_rewards,
        )
        results[f"{name}_comparison"] = comparison
    
    # Create summary
    results["summary"] = {
        "agent_mean_reward": agent_metrics.mean_reward,
        "best_baseline": max(baselines.keys(), key=lambda k: results[k]["mean_reward"]),
        "improvement_over_best": (
            agent_metrics.mean_reward - max(results[k]["mean_reward"] for k in baselines.keys())
        ),
    }
    
    return results


def run_evaluation(
    model_path: str,
    config_path: Optional[str] = None,
    n_episodes: int = 20,
    compare_baselines_flag: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.
    
    Args:
        model_path: Path to saved model.
        config_path: Path to config file.
        n_episodes: Number of evaluation episodes.
        compare_baselines_flag: Whether to compare with baselines.
        output_dir: Output directory for results.
        
    Returns:
        Evaluation results.
    """
    model_path = Path(model_path)
    
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "training" / "config.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Generate data
    logger.info("Preparing data...")
    data = generate_synthetic_dataset(
        n_users=config["data"]["n_users"],
        n_days=config["data"]["n_days"],
        seed=config["experiment"]["seed"],
    )
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess(data, is_training=True)
    
    engineer = FeatureEngineer()
    data = engineer.engineer_features(data)
    data = compute_reward_components(data)
    
    # Create environment
    env = PeriodizationEnv(
        data=data,
        episode_length=config["environment"]["episode_length"],
        seed=config["experiment"]["seed"],
    )
    
    # Load agent
    logger.info(f"Loading agent from {model_path}...")
    state_dim = env.observation_space.shape[0]
    agent = SACAgent(state_dim=state_dim, action_dim=6, device="cpu")
    agent.load(model_path)
    
    # Evaluate
    logger.info("Running evaluation...")
    
    if compare_baselines_flag:
        results = compare_with_baselines(agent, env, n_episodes)
    else:
        metrics = evaluate_agent(agent, env, n_episodes)
        results = {"agent": metrics.to_dict()}
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "evaluation_results.yaml", "w") as f:
            yaml.dump(results, f)
        
        logger.info(f"Results saved to {output_path}")
    
    return results


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Adaptive Periodization Agent")
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline comparison",
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
    
    # Run evaluation
    results = run_evaluation(
        model_path=args.model_path,
        config_path=args.config,
        n_episodes=args.episodes,
        compare_baselines_flag=not args.no_baselines,
        output_dir=args.output,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    if "agent" in results:
        agent = results["agent"]
        print(f"\nAgent Performance:")
        print(f"  Mean Reward: {agent['mean_reward']:.2f} ± {agent['std_reward']:.2f}")
        print(f"  Constraint Violations: {agent['constraint_violations']}")
        print(f"  Action Entropy: {agent['action_entropy']:.3f}")
    
    if "summary" in results:
        summary = results["summary"]
        print(f"\nComparison:")
        print(f"  Best Baseline: {summary['best_baseline']}")
        print(f"  Improvement: {summary['improvement_over_best']:.2f}")


if __name__ == "__main__":
    main()
