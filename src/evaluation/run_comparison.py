"""
Comprehensive comparison script for agent vs baselines.

Runs full evaluation comparing SAC agent against all baselines
with statistical testing and visualization.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.synthetic_data import generate_synthetic_dataset
from src.data.preprocess import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer, compute_reward_components
from src.environment.periodization_env import PeriodizationEnv
from src.models.sac_agent import SACAgent
from src.models.baseline_policies import (
    RandomPolicy,
    RuleBasedPolicy,
    FixedPeriodizationPolicy,
    RecoveryBasedPolicy,
)
from src.evaluation.strong_baselines import (
    get_all_strong_baselines,
    evaluate_strong_baseline,
)
from src.evaluation.statistical_analysis import (
    compare_methods,
    print_comparison_table,
)

logger = logging.getLogger(__name__)


def evaluate_agent(
    agent: SACAgent,
    env: PeriodizationEnv,
    n_episodes: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate trained SAC agent."""
    episode_rewards = []
    violations = 0
    all_actions = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = 0
        
        done = False
        while not done:
            mask = env.get_action_mask()
            action = agent.select_action(obs, mask, deterministic=True)
            all_actions.append(action)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if info.get("recovery", 50) < 30 and action >= 3:
                violations += 1
            
            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
    
    action_counts = np.bincount(all_actions, minlength=6)
    action_dist = action_counts / len(all_actions)
    
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "rewards_list": episode_rewards,
        "violations": violations,
        "episodes": n_episodes,
        "action_distribution": action_dist.tolist(),
    }


def run_comprehensive_comparison(
    model_path: Optional[str] = None,
    n_users: int = 100,
    n_episodes_train: int = 100,
    n_episodes_eval: int = 20,
    n_seeds: int = 3,
    output_dir: str = "experiments/comparisons",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run comprehensive comparison of agent vs all baselines.
    
    Args:
        model_path: Path to pre-trained model (None = train new).
        n_users: Number of synthetic users.
        n_episodes_train: Training episodes for new agent.
        n_episodes_eval: Evaluation episodes per method.
        n_seeds: Number of random seeds.
        output_dir: Output directory.
        device: Device for training.
        
    Returns:
        Comprehensive results dictionary.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "metadata": {
            "n_users": n_users,
            "n_episodes_train": n_episodes_train,
            "n_episodes_eval": n_episodes_eval,
            "n_seeds": n_seeds,
            "timestamp": datetime.now().isoformat(),
        },
        "agent_results": [],
        "baseline_results": {},
        "comparisons": [],
    }
    
    # Run for multiple seeds
    for seed in range(n_seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed + 1}/{n_seeds}")
        logger.info(f"{'='*60}")
        
        base_seed = 42 + seed * 1000
        
        # Generate data
        logger.info("Generating data...")
        data = generate_synthetic_dataset(n_users=n_users, n_days=180, seed=base_seed)
        
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess(data, is_training=True)
        
        engineer = FeatureEngineer()
        data = engineer.engineer_features(data)
        data = compute_reward_components(data)
        
        # Create environment
        env = PeriodizationEnv(
            data=data,
            episode_length=90,
            seed=base_seed,
        )
        
        # Train or load agent
        state_dim = env.observation_space.shape[0]
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=6,
            device=device,
        )
        
        if model_path and Path(model_path).exists():
            logger.info(f"Loading model from {model_path}")
            agent.load(model_path)
        else:
            logger.info(f"Training agent for {n_episodes_train} episodes...")
            for ep in range(n_episodes_train):
                obs, _ = env.reset(seed=base_seed + ep)
                episode_reward = 0
                
                done = False
                while not done:
                    mask = env.get_action_mask()
                    action = agent.select_action(obs, mask, deterministic=False)
                    
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    agent.store_transition(obs, action, reward, next_obs, terminated)
                    
                    if len(agent.buffer) >= agent.batch_size:
                        agent.train_step()
                    
                    episode_reward += reward
                    obs = next_obs
                    done = terminated or truncated
                
                if (ep + 1) % 20 == 0:
                    logger.info(f"  Episode {ep + 1}: reward = {episode_reward:.2f}")
        
        # Evaluate agent
        logger.info("Evaluating agent...")
        agent_result = evaluate_agent(agent, env, n_episodes_eval, base_seed + 10000)
        agent_result["seed"] = seed
        all_results["agent_results"].append(agent_result)
        logger.info(f"  Agent: {agent_result['mean_reward']:.2f} ± {agent_result['std_reward']:.2f}")
        
        # Evaluate strong baselines
        logger.info("Evaluating baselines...")
        strong_baselines = get_all_strong_baselines(seed=base_seed)
        
        for name, policy in strong_baselines.items():
            baseline_result = evaluate_strong_baseline(
                policy, data, n_episodes_eval, seed=base_seed + 20000
            )
            baseline_result["seed"] = seed
            
            if name not in all_results["baseline_results"]:
                all_results["baseline_results"][name] = []
            all_results["baseline_results"][name].append(baseline_result)
            
            logger.info(f"  {name}: {baseline_result['mean_reward']:.2f} ± {baseline_result['std_reward']:.2f}")
        
        # Evaluate simple baselines
        simple_baselines = {
            "random": RandomPolicy(seed=base_seed),
            "rule_based": RuleBasedPolicy(),
            "fixed_periodization": FixedPeriodizationPolicy(),
            "recovery_based": RecoveryBasedPolicy(),
        }
        
        for name, policy in simple_baselines.items():
            # Use similar evaluation
            episode_rewards = []
            violations = 0
            
            for ep in range(n_episodes_eval):
                obs, _ = env.reset(seed=base_seed + 30000 + ep)
                episode_reward = 0
                action_history = []
                
                done = False
                while not done:
                    mask = env.get_action_mask()
                    
                    # Get state dict
                    current_idx = env._episode_start_idx + env._current_step
                    row = env._current_user_data.iloc[current_idx]
                    state = {"recovery": row.get("recovery_score", 50)}
                    
                    action = policy.select_action(state, action_history, mask)
                    action_history.append(action)
                    
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    
                    if info.get("recovery", 50) < 30 and action >= 3:
                        violations += 1
                    
                    episode_reward += reward
                    obs = next_obs
                    done = terminated or truncated
                
                episode_rewards.append(episode_reward)
            
            baseline_result = {
                "mean_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "rewards_list": episode_rewards,
                "violations": violations,
                "seed": seed,
            }
            
            if name not in all_results["baseline_results"]:
                all_results["baseline_results"][name] = []
            all_results["baseline_results"][name].append(baseline_result)
            
            logger.info(f"  {name}: {baseline_result['mean_reward']:.2f}")
    
    # Statistical comparisons
    logger.info("\nRunning statistical comparisons...")
    
    agent_rewards = []
    for r in all_results["agent_results"]:
        agent_rewards.extend(r["rewards_list"])
    
    comparisons = []
    for baseline_name, baseline_results in all_results["baseline_results"].items():
        baseline_rewards = []
        for r in baseline_results:
            if "rewards_list" in r:
                baseline_rewards.extend(r["rewards_list"])
            else:
                # Approximate from mean
                baseline_rewards.extend([r["mean_reward"]] * n_episodes_eval)
        
        if len(baseline_rewards) > 0:
            comparison = compare_methods(
                "SAC_Agent",
                baseline_name,
                agent_rewards[:len(baseline_rewards)],
                baseline_rewards,
                paired=True,
            )
            
            comparisons.append({
                "baseline": baseline_name,
                "difference": comparison.difference,
                "relative_improvement": comparison.relative_improvement,
                "p_value": comparison.p_value,
                "significant": comparison.significant,
                "effect_size": comparison.effect_size,
            })
    
    all_results["comparisons"] = comparisons
    
    # Print results table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 80)
    
    # Aggregate across seeds
    agent_mean = np.mean([r["mean_reward"] for r in all_results["agent_results"]])
    agent_std = np.std([r["mean_reward"] for r in all_results["agent_results"]])
    
    print(f"\n{'Method':<25} {'Mean Reward':<15} {'Std':<10} {'Violations':<12}")
    print("-" * 80)
    print(f"{'SAC Agent':<25} {agent_mean:>10.2f} {agent_std:>10.2f}")
    
    for name, results in all_results["baseline_results"].items():
        mean = np.mean([r["mean_reward"] for r in results])
        std = np.std([r["mean_reward"] for r in results])
        violations = sum(r.get("violations", 0) for r in results)
        print(f"{name:<25} {mean:>10.2f} {std:>10.2f}      {violations}")
    
    print("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"comparison_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(results_file, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    
    return all_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive comparison")
    
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to pre-trained model")
    parser.add_argument("--users", "-u", type=int, default=100, help="Number of users")
    parser.add_argument("--train-episodes", "-t", type=int, default=100, help="Training episodes")
    parser.add_argument("--eval-episodes", "-e", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--seeds", "-s", type=int, default=3, help="Number of seeds")
    parser.add_argument("--output", "-o", type=str, default="experiments/comparisons", help="Output dir")
    
    args = parser.parse_args()
    
    run_comprehensive_comparison(
        model_path=args.model,
        n_users=args.users,
        n_episodes_train=args.train_episodes,
        n_episodes_eval=args.eval_episodes,
        n_seeds=args.seeds,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
