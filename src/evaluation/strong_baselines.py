"""
Strong baseline policies for comparison with the SAC agent.

These implement domain-expert heuristics, imitation learning,
and fixed periodization schedules.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class StrongBaseline(ABC):
    """Abstract base class for strong baseline policies."""
    
    @abstractmethod
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Select an action given the current state."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name."""
        pass


class ExpertCoachBaseline(StrongBaseline):
    """
    Rule-based expert coach that mimics professional training recommendations.
    
    Implements evidence-based periodization principles:
    - Force rest when recovery is critically low
    - Active recovery when fatigued (negative TSB)
    - Build fitness when fresh
    - Taper before peak performance windows
    """
    
    @property
    def name(self) -> str:
        return "ExpertCoach"
    
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        recovery = state.get("recovery", 50)
        tsb = state.get("tsb", 0)
        hrv = state.get("hrv", 60)
        hrv_baseline = state.get("hrv_baseline", 60)
        ctl = state.get("ctl", 50)
        
        # Count days since last rest
        days_since_rest = len(action_history)
        if 0 in action_history:
            for i, a in enumerate(reversed(action_history)):
                if a == 0:
                    days_since_rest = i
                    break
        
        # Count consecutive high intensity days
        consecutive_high = 0
        for a in reversed(action_history):
            if a >= 3:
                consecutive_high += 1
            else:
                break
        
        # Rule 1: Force rest if critically low recovery
        if recovery < 30:
            return 0  # REST
        
        # Rule 2: Force rest after 7+ days without rest
        if days_since_rest >= 7:
            return 0  # REST
        
        # Rule 3: Rest if HRV crashed (>15% below baseline)
        if hrv < hrv_baseline * 0.85:
            return 0  # REST
        
        # Rule 4: Active recovery if very fatigued (TSB < -15)
        if tsb < -15:
            return 1  # ACTIVE_RECOVERY
        
        # Rule 5: Active recovery after 3+ consecutive hard days
        if consecutive_high >= 3:
            return 1  # ACTIVE_RECOVERY
        
        # Rule 6: Light day if moderately fatigued
        if tsb < -5 or recovery < 50:
            return 2  # AEROBIC_BASE
        
        # Rule 7: Quality session if fresh and recovered
        if recovery >= 70 and tsb > 0:
            # Alternate between tempo and HIIT
            if len(action_history) % 3 == 0:
                return 4  # HIIT
            else:
                return 3  # TEMPO
        
        # Rule 8: Strength training if very recovered and CTL is building
        if recovery >= 80 and ctl > 50 and days_since_rest >= 2:
            if len(action_history) % 4 == 0:
                return 5  # STRENGTH
        
        # Default: steady state aerobic
        return 2  # AEROBIC_BASE


class FixedPeriodizationBaseline(StrongBaseline):
    """
    Classic 3-week build, 1-week recovery periodization.
    
    Follows traditional training blocks:
    - Week 1-3: Progressive overload with varied intensities
    - Week 4: Recovery/deload week
    """
    
    @property
    def name(self) -> str:
        return "FixedPeriodization"
    
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        day = len(action_history)
        week_in_cycle = (day // 7) % 4  # 0-3
        day_in_week = day % 7  # 0-6 (Mon-Sun)
        
        # Week 4: Recovery week
        if week_in_cycle == 3:
            if day_in_week in [0, 3, 6]:  # Mon, Thu, Sun
                return 0  # REST
            else:
                return 1  # ACTIVE_RECOVERY
        
        # Build weeks (1-3)
        # Sunday and Monday: Rest or easy
        if day_in_week == 0:  # Monday
            return 1  # ACTIVE_RECOVERY
        if day_in_week == 6:  # Sunday
            return 0  # REST
        
        # Tuesday: Quality session 1
        if day_in_week == 1:
            if week_in_cycle == 0:
                return 3  # TEMPO
            elif week_in_cycle == 1:
                return 4  # HIIT
            else:
                return 3  # TEMPO
        
        # Wednesday: Easy/recovery
        if day_in_week == 2:
            return 2  # AEROBIC_BASE
        
        # Thursday: Quality session 2
        if day_in_week == 3:
            if week_in_cycle == 0:
                return 5  # STRENGTH
            elif week_in_cycle == 1:
                return 3  # TEMPO
            else:
                return 4  # HIIT
        
        # Friday: Easy
        if day_in_week == 4:
            return 2  # AEROBIC_BASE
        
        # Saturday: Long session
        if day_in_week == 5:
            return 2  # AEROBIC_BASE (long)
        
        return 2  # Default


class AdaptiveThresholdBaseline(StrongBaseline):
    """
    Simple adaptive policy that uses recovery thresholds.
    
    Maps recovery score directly to training intensity:
    - Very low recovery → Rest
    - Low recovery → Active recovery
    - Moderate recovery → Aerobic
    - High recovery → Tempo/HIIT
    - Very high recovery → Strength
    """
    
    @property
    def name(self) -> str:
        return "AdaptiveThreshold"
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        randomness: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize threshold-based policy.
        
        Args:
            thresholds: Recovery thresholds for each action.
            randomness: Probability of random action.
            seed: Random seed.
        """
        self.thresholds = thresholds or {
            "rest": 25,
            "active_recovery": 40,
            "aerobic": 55,
            "tempo": 70,
            "hiit": 80,
        }
        self.randomness = randomness
        self.rng = np.random.default_rng(seed)
    
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        # Occasionally take random action (exploration)
        if self.rng.random() < self.randomness:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                return int(self.rng.choice(valid_actions))
            return int(self.rng.integers(0, 6))
        
        recovery = state.get("recovery", 50)
        
        if recovery < self.thresholds["rest"]:
            return 0  # REST
        elif recovery < self.thresholds["active_recovery"]:
            return 1  # ACTIVE_RECOVERY
        elif recovery < self.thresholds["aerobic"]:
            return 2  # AEROBIC_BASE
        elif recovery < self.thresholds["tempo"]:
            return 3  # TEMPO
        elif recovery < self.thresholds["hiit"]:
            return 4  # HIIT
        else:
            return 5  # STRENGTH


class TSBOptimizedBaseline(StrongBaseline):
    """
    Policy that optimizes for Training Stress Balance (TSB).
    
    Tries to maintain TSB in the optimal range (-10 to +15)
    for peak performance while preventing overtraining.
    """
    
    @property
    def name(self) -> str:
        return "TSBOptimized"
    
    def __init__(self, target_tsb_range: tuple = (-5, 10)):
        """
        Initialize TSB-optimized policy.
        
        Args:
            target_tsb_range: Target TSB range (min, max).
        """
        self.target_min, self.target_max = target_tsb_range
    
    def select_action(
        self,
        state: Dict[str, Any],
        action_history: List[int],
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        tsb = state.get("tsb", 0)
        recovery = state.get("recovery", 50)
        
        # Safety first: rest if low recovery
        if recovery < 30:
            return 0
        
        # If TSB too low (overtrained), recover
        if tsb < self.target_min - 10:
            return 0  # REST
        elif tsb < self.target_min:
            return 1  # ACTIVE_RECOVERY
        
        # If TSB too high (undertrained), increase load
        elif tsb > self.target_max + 10:
            return 4  # HIIT
        elif tsb > self.target_max:
            return 3  # TEMPO
        
        # In optimal range: maintain with moderate training
        else:
            # Vary training to build different systems
            day = len(action_history)
            if day % 4 == 0:
                return 3  # TEMPO
            elif day % 4 == 1:
                return 2  # AEROBIC_BASE
            elif day % 4 == 2:
                return 4 if recovery >= 60 else 2  # HIIT or AEROBIC
            else:
                return 1  # ACTIVE_RECOVERY


def get_all_strong_baselines(seed: int = 42) -> Dict[str, StrongBaseline]:
    """
    Get all strong baseline policies.
    
    Args:
        seed: Random seed.
        
    Returns:
        Dictionary of baseline name -> policy.
    """
    return {
        "expert_coach": ExpertCoachBaseline(),
        "fixed_periodization": FixedPeriodizationBaseline(),
        "adaptive_threshold": AdaptiveThresholdBaseline(seed=seed),
        "tsb_optimized": TSBOptimizedBaseline(),
    }


def evaluate_strong_baseline(
    policy: StrongBaseline,
    data,
    n_episodes: int = 20,
    episode_length: int = 90,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a strong baseline policy.
    
    Args:
        policy: Baseline policy to evaluate.
        data: Preprocessed data.
        n_episodes: Number of evaluation episodes.
        episode_length: Steps per episode.
        seed: Random seed.
        
    Returns:
        Evaluation metrics.
    """
    from src.environment.periodization_env import PeriodizationEnv
    
    env = PeriodizationEnv(
        data=data,
        episode_length=episode_length,
        seed=seed,
    )
    
    episode_rewards = []
    total_violations = 0
    all_actions = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        
        episode_reward = 0
        actions = []
        action_history = []
        
        done = False
        while not done:
            # Build state dict
            current_idx = env._episode_start_idx + env._current_step
            row = env._current_user_data.iloc[current_idx]
            
            state = {
                "recovery": row.get("recovery_score", 50),
                "tsb": row.get("tsb", 0),
                "hrv": row.get("hrv_rmssd", 60),
                "hrv_baseline": row.get("hrv_rmssd_mean_28d", 60),
                "ctl": row.get("ctl", 50),
            }
            
            mask = env.get_action_mask()
            action = policy.select_action(state, action_history, mask)
            
            # Apply mask if action is blocked
            if not mask[action]:
                valid_actions = np.where(mask)[0]
                action = int(valid_actions[0]) if len(valid_actions) > 0 else 0
            
            actions.append(action)
            action_history.append(action)
            
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Count violations
            if step_info.get("recovery", 50) < 30 and action >= 3:
                total_violations += 1
            
            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        all_actions.extend(actions)
    
    # Compute action distribution
    action_counts = np.bincount(all_actions, minlength=6)
    action_dist = action_counts / len(all_actions)
    
    return {
        "policy_name": policy.name,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "violations": total_violations,
        "episodes": n_episodes,
        "action_distribution": {
            "rest": float(action_dist[0]),
            "active_recovery": float(action_dist[1]),
            "aerobic_base": float(action_dist[2]),
            "tempo": float(action_dist[3]),
            "hiit": float(action_dist[4]),
            "strength": float(action_dist[5]),
        },
    }
