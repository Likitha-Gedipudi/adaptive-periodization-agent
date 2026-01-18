"""
Baseline policies for comparison with the trained RL agent.

This module implements various baseline policies including random,
rule-based, and fixed periodization strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaselinePolicy(ABC):
    """Abstract base class for baseline policies."""
    
    @abstractmethod
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current state as a dictionary.
            action_history: List of previous actions.
            action_mask: Optional mask for valid actions.
            
        Returns:
            Selected action index (0-5).
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for logging."""
        pass


class RandomPolicy(BaselinePolicy):
    """
    Random baseline policy.
    
    Selects actions uniformly at random (respecting action mask if provided).
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random policy.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
    
    @property
    def name(self) -> str:
        return "Random"
    
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Select a random valid action."""
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                return 0  # Fallback to rest
            return self.rng.choice(valid_actions)
        
        return self.rng.integers(0, 6)


class RuleBasedPolicy(BaselinePolicy):
    """
    Rule-based heuristic policy.
    
    Uses simple rules based on recovery and recent training to decide actions:
    - High recovery (>70%): Moderate to high intensity
    - Moderate recovery (50-70%): Light to moderate
    - Low recovery (<50%): Rest or active recovery
    """
    
    def __init__(
        self,
        high_recovery_threshold: float = 70,
        low_recovery_threshold: float = 50,
        seed: Optional[int] = None,
    ):
        """
        Initialize rule-based policy.
        
        Args:
            high_recovery_threshold: Threshold for high intensity training.
            low_recovery_threshold: Threshold below which to rest.
            seed: Random seed.
        """
        self.high_threshold = high_recovery_threshold
        self.low_threshold = low_recovery_threshold
        self.rng = np.random.default_rng(seed)
    
    @property
    def name(self) -> str:
        return "RuleBased"
    
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Select action based on recovery-based rules."""
        recovery = state.get("recovery", 50)
        
        # Check how many days since last rest
        days_since_rest = len(action_history)
        if action_history and 0 in action_history:
            for i, a in enumerate(reversed(action_history)):
                if a == 0:
                    days_since_rest = i
                    break
        
        # Force rest every 7 days
        if days_since_rest >= 6:
            action = 0
        elif recovery >= self.high_threshold:
            # High recovery - can do hard training
            # Mix of moderate and high intensity
            action = self.rng.choice([2, 3, 4, 5], p=[0.35, 0.30, 0.20, 0.15])
        elif recovery >= self.low_threshold:
            # Moderate recovery - lighter training
            action = self.rng.choice([1, 2, 5], p=[0.30, 0.50, 0.20])
        else:
            # Low recovery - rest or very light
            action = self.rng.choice([0, 1], p=[0.5, 0.5])
        
        # Apply action mask if provided
        if action_mask is not None and not action_mask[action]:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                # Choose the most conservative valid action
                for preferred in [0, 1, 2, 5, 3, 4]:
                    if preferred in valid_actions:
                        return preferred
            return 0
        
        return action


class FixedPeriodizationPolicy(BaselinePolicy):
    """
    Fixed periodization policy (3 weeks progressive + 1 week deload).
    
    Follows a classic periodization structure:
    - Week 1-3: Progressive overload
    - Week 4: Deload/recovery week
    """
    
    # Weekly pattern templates
    BUILD_WEEK_PATTERN = [
        [2, 3, 1, 4, 2, 5, 0],  # Week 1: Moderate
        [2, 4, 1, 3, 2, 5, 0],  # Week 2: Harder
        [3, 4, 1, 4, 2, 5, 0],  # Week 3: Peak
    ]
    
    DELOAD_PATTERN = [1, 2, 0, 1, 2, 1, 0]  # Week 4: Recovery
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize fixed periodization policy.
        
        Args:
            seed: Random seed for slight variations.
        """
        self.rng = np.random.default_rng(seed)
        self.day_counter = 0
    
    @property
    def name(self) -> str:
        return "FixedPeriodization"
    
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Select action based on periodization schedule."""
        day = len(action_history)
        
        # Determine week and day within week
        week_in_cycle = (day // 7) % 4  # 4-week cycle
        day_in_week = day % 7
        
        # Get planned action
        if week_in_cycle < 3:
            action = self.BUILD_WEEK_PATTERN[week_in_cycle][day_in_week]
        else:
            action = self.DELOAD_PATTERN[day_in_week]
        
        # Add slight randomness (±1 intensity level)
        if self.rng.random() < 0.1:
            action = max(0, min(5, action + self.rng.choice([-1, 1])))
        
        # Override if recovery is critically low
        recovery = state.get("recovery", 50)
        if recovery < 30:
            action = 0
        
        # Apply action mask
        if action_mask is not None and not action_mask[action]:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                # Find closest valid action
                distances = np.abs(valid_actions - action)
                action = valid_actions[np.argmin(distances)]
            else:
                action = 0
        
        return action


class RecoveryBasedPolicy(BaselinePolicy):
    """
    Recovery-based adaptive policy.
    
    Strictly follows recovery score with some randomness.
    More aggressive than rule-based when recovery is high.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize recovery-based policy."""
        self.rng = np.random.default_rng(seed)
    
    @property
    def name(self) -> str:
        return "RecoveryBased"
    
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Select action proportional to recovery score."""
        recovery = state.get("recovery", 50)
        
        # Map recovery to action distribution
        if recovery >= 80:
            probs = [0.05, 0.05, 0.25, 0.30, 0.25, 0.10]  # High intensity
        elif recovery >= 65:
            probs = [0.05, 0.10, 0.40, 0.25, 0.10, 0.10]  # Moderate-high
        elif recovery >= 50:
            probs = [0.10, 0.25, 0.35, 0.15, 0.05, 0.10]  # Moderate
        elif recovery >= 35:
            probs = [0.20, 0.40, 0.25, 0.10, 0.00, 0.05]  # Low-moderate
        else:
            probs = [0.50, 0.40, 0.10, 0.00, 0.00, 0.00]  # Low
        
        # Apply action mask
        if action_mask is not None:
            probs = np.array(probs) * action_mask.astype(float)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.zeros(6)
                probs[0] = 1.0
        
        return self.rng.choice(6, p=probs)


def get_all_baseline_policies(seed: int = 42) -> Dict[str, BaselinePolicy]:
    """
    Get all baseline policies for comparison.
    
    Args:
        seed: Random seed.
        
    Returns:
        Dictionary of policy name to policy instance.
    """
    return {
        "random": RandomPolicy(seed=seed),
        "rule_based": RuleBasedPolicy(seed=seed),
        "fixed_periodization": FixedPeriodizationPolicy(seed=seed),
        "recovery_based": RecoveryBasedPolicy(seed=seed),
    }


def evaluate_policy(
    policy: BaselinePolicy,
    env,
    n_episodes: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate a policy on the environment.
    
    Args:
        policy: Policy to evaluate.
        env: Environment instance.
        n_episodes: Number of evaluation episodes.
        seed: Random seed.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    from src.environment.constraints import SafetyConstraints
    
    np.random.seed(seed)
    constraints = SafetyConstraints()
    
    episode_rewards = []
    constraint_violations = []
    action_distributions = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0
        violations = 0
        actions_taken = []
        action_history = []
        
        done = False
        while not done:
            # Build state dictionary
            state = {
                "recovery": obs[2] if len(obs) > 2 else 50,  # recovery_score
                "hrv": obs[0] if len(obs) > 0 else 60,
            }
            
            # Get action mask
            mask = env.get_action_mask() if hasattr(env, "get_action_mask") else None
            
            # Select action
            action = policy.select_action(state, action_history, mask)
            
            # Check for constraint violation
            if mask is not None and not mask[action]:
                violations += 1
            
            actions_taken.append(action)
            action_history.append(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        constraint_violations.append(violations)
        action_distributions.append(actions_taken)
    
    # Compute metrics
    all_actions = [a for ep_actions in action_distributions for a in ep_actions]
    action_counts = np.bincount(all_actions, minlength=6)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_violations": np.mean(constraint_violations),
        "action_distribution": action_counts / len(all_actions),
    }
