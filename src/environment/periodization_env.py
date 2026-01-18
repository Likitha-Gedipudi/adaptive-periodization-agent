"""
Custom Gymnasium environment for the Adaptive Periodization Agent.

This module implements the training recommendation environment where the agent
observes physiological state and prescribes training actions.
"""

import logging
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Training action types."""
    REST = 0
    ACTIVE_RECOVERY = 1
    AEROBIC_BASE = 2
    TEMPO = 3
    HIIT = 4
    STRENGTH = 5


# Action descriptions for interpretability
ACTION_DESCRIPTIONS = {
    ActionType.REST: "Complete rest day",
    ActionType.ACTIVE_RECOVERY: "Active recovery (Zone 1, <60% max HR, 20-40 min)",
    ActionType.AEROBIC_BASE: "Aerobic base (Zone 2, 60-70% max HR, 45-90 min)",
    ActionType.TEMPO: "Tempo/Threshold (Zone 3-4, 70-85% max HR, 30-60 min)",
    ActionType.HIIT: "High intensity intervals (Zone 5, >85% max HR, 20-40 min)",
    ActionType.STRENGTH: "Strength training (resistance-based)",
}


# State features used for observation
STATE_FEATURES = [
    # Core physiological
    "hrv_rmssd_scaled",
    "resting_hr_scaled",
    "recovery_score",
    "sleep_duration_scaled",
    "sleep_efficiency_scaled",
    
    # Training load
    "atl",
    "ctl",
    "tsb",
    
    # Rolling statistics
    "hrv_rmssd_zscore_7d",
    "hrv_trend_7d",
    "recovery_trend_7d",
    
    # Temporal
    "day_sin",
    "day_cos",
    "days_since_high_recovery",
    "consecutive_high_strain",
    
    # Derived
    "recovery_fatigue_interaction",
    "tsb_recovery_interaction",
]


class PeriodizationEnv(gym.Env):
    """
    Gymnasium environment for adaptive training periodization.
    
    The agent observes the user's physiological state and prescribes
    a training action (one of 6 types). The reward is based on short,
    medium, and long-term fitness outcomes.
    
    Attributes:
        action_space: Discrete(6) - 6 training types
        observation_space: Box with state features
        episode_length: Number of days per episode (default 90)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data: pd.DataFrame,
        user_id: Optional[int] = None,
        episode_length: int = 90,
        state_features: Optional[List[str]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        apply_constraints: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize the environment.
        
        Args:
            data: DataFrame with preprocessed and feature-engineered data.
            user_id: Optional specific user to train on (None = sample randomly).
            episode_length: Number of steps (days) per episode.
            state_features: List of feature columns for state observation.
            reward_weights: Weights for reward components {short, medium, long}.
            apply_constraints: Whether to apply safety constraints (action masking).
            seed: Random seed for reproducibility.
        """
        super().__init__()
        
        self.data = data.copy()
        self.user_id = user_id
        self.episode_length = episode_length
        self.state_features = state_features or STATE_FEATURES
        self.apply_constraints = apply_constraints
        
        # Validate state features exist
        self._validate_features()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(ActionType))
        
        state_dim = len(self.state_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        
        # Reward configuration
        self.reward_weights = reward_weights or {
            "short": 0.2,
            "medium": 0.3,
            "long": 0.5,
            "penalty": 1.0,
        }
        
        # Import reward and constraint modules
        from src.environment.reward_functions import CompositeReward
        from src.environment.constraints import SafetyConstraints
        
        self.reward_fn = CompositeReward(
            short_weight=self.reward_weights["short"],
            medium_weight=self.reward_weights["medium"],
            long_weight=self.reward_weights["long"],
        )
        self.constraints = SafetyConstraints()
        
        # Episode state
        self._current_user_data: Optional[pd.DataFrame] = None
        self._current_step: int = 0
        self._episode_start_idx: int = 0
        self._action_history: List[int] = []
        
        # Set random seed
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()
        
        logger.debug(
            f"PeriodizationEnv initialized: state_dim={state_dim}, "
            f"episode_length={episode_length}"
        )
    
    def _validate_features(self) -> None:
        """Validate that required state features exist in the data."""
        available = set(self.data.columns)
        required = set(self.state_features)
        missing = required - available
        
        if missing:
            # Try to use fallback features
            fallback_map = {
                "hrv_rmssd_scaled": "hrv_rmssd",
                "resting_hr_scaled": "resting_hr",
                "sleep_duration_scaled": "sleep_duration",
                "sleep_efficiency_scaled": "sleep_efficiency",
            }
            
            for feat in list(missing):
                if feat in fallback_map and fallback_map[feat] in available:
                    # Replace with unscaled version
                    idx = self.state_features.index(feat)
                    self.state_features[idx] = fallback_map[feat]
                    missing.remove(feat)
                    logger.warning(f"Using fallback feature: {fallback_map[feat]} for {feat}")
            
            # Remove still-missing features
            if missing:
                logger.warning(f"Removing missing features from state: {missing}")
                self.state_features = [f for f in self.state_features if f in available]
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Optional random seed.
            options: Optional reset options (e.g., {"user_id": 5}).
            
        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        
        # Select user for this episode
        user_id = self.user_id
        if options and "user_id" in options:
            user_id = options["user_id"]
        
        if user_id is None:
            # Sample a random user
            users = self.data["user_id"].unique()
            user_id = self._np_random.choice(users)
        
        # Get user data
        user_data = self.data[self.data["user_id"] == user_id].sort_values("date")
        user_data = user_data.reset_index(drop=True)
        
        # Ensure enough data for episode
        if len(user_data) < self.episode_length + 30:  # Need buffer for rewards
            # Try another user
            logger.warning(f"User {user_id} has insufficient data, sampling another")
            users = self.data["user_id"].unique()
            for u in self._np_random.permutation(users):
                user_data = self.data[self.data["user_id"] == u].sort_values("date")
                if len(user_data) >= self.episode_length + 30:
                    user_id = u
                    user_data = user_data.reset_index(drop=True)
                    break
        
        self._current_user_data = user_data
        
        # Random start point (with buffer for future rewards)
        max_start = len(user_data) - self.episode_length - 30
        max_start = max(0, max_start)
        self._episode_start_idx = self._np_random.integers(0, max_start + 1)
        
        self._current_step = 0
        self._action_history = []
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            "user_id": user_id,
            "episode_start_date": str(user_data.iloc[self._episode_start_idx]["date"]),
            "step": self._current_step,
        }
        
        return obs, info
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action index (0-5).
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self._current_user_data is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Apply action masking if enabled
        if self.apply_constraints:
            current_state = self._get_current_state_dict()
            action = self.constraints.apply_mask(
                action, current_state, self._action_history
            )
        
        # Record action
        self._action_history.append(action)
        
        # Current and next indices
        current_idx = self._episode_start_idx + self._current_step
        
        # Get current state data
        current_row = self._current_user_data.iloc[current_idx]
        
        # Calculate reward (using pre-computed reward components if available)
        reward = self._calculate_reward(current_idx, action)
        
        # Apply constraint penalties
        if self.apply_constraints:
            penalty = self.constraints.calculate_penalty(
                action, self._get_current_state_dict(), self._action_history
            )
            reward -= self.reward_weights["penalty"] * penalty
        
        # Move to next step
        self._current_step += 1
        
        # Check termination
        terminated = self._current_step >= self.episode_length
        truncated = False
        
        # Get next observation (or zeros if terminated)
        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(len(self.state_features), dtype=np.float32)
        
        info = {
            "step": self._current_step,
            "action": action,
            "action_name": ActionType(action).name,
            "reward": reward,
            "date": str(current_row["date"]),
            "recovery": current_row.get("recovery_score", 0),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        current_idx = self._episode_start_idx + self._current_step
        row = self._current_user_data.iloc[current_idx]
        
        obs = []
        for feat in self.state_features:
            val = row.get(feat, 0.0)
            if pd.isna(val):
                val = 0.0
            obs.append(float(val))
        
        return np.array(obs, dtype=np.float32)
    
    def _get_current_state_dict(self) -> Dict[str, Any]:
        """Get current state as a dictionary for constraints."""
        current_idx = self._episode_start_idx + self._current_step
        row = self._current_user_data.iloc[current_idx]
        
        return {
            "recovery": row.get("recovery_score", 50),
            "hrv": row.get("hrv_rmssd", 60),
            "hrv_baseline": row.get("hrv_rmssd_mean_28d", row.get("hrv_rmssd", 60)),
            "hrv_std": row.get("hrv_rmssd_std_28d", 10),
            "days_since_rest": len(self._action_history) - (
                self._action_history[::-1].index(0) 
                if 0 in self._action_history else len(self._action_history)
            ),
            "consecutive_high_intensity": self._count_consecutive_high(),
        }
    
    def _count_consecutive_high(self) -> int:
        """Count consecutive high-intensity days."""
        count = 0
        for action in reversed(self._action_history):
            if action >= 3:  # Tempo, HIIT, Strength
                count += 1
            else:
                break
        return count
    
    def _calculate_reward(self, current_idx: int, action: int) -> float:
        """
        Calculate reward based on action and outcomes.
        
        Uses pre-computed reward components if available.
        Scaled to achieve ~500+ reward per 90-step episode.
        """
        row = self._current_user_data.iloc[current_idx]
        
        # BASE REWARD: +6 per step (gives 540 base for 90 steps)
        base_reward = 6.0
        
        # Short-term: HRV change next day (scaled up)
        short_reward = 0.0
        if "hrv_change_short" in row:
            short_reward = row["hrv_change_short"]  # Remove normalization
        elif current_idx + 1 < len(self._current_user_data):
            next_row = self._current_user_data.iloc[current_idx + 1]
            hrv_change = next_row.get("hrv_rmssd", 0) - row.get("hrv_rmssd", 0)
            short_reward = hrv_change
        
        # Medium-term: CTL growth (scaled up)
        medium_reward = 0.0
        if "ctl_growth_medium" in row:
            medium_reward = row["ctl_growth_medium"]  # Remove normalization
        elif current_idx + 14 < len(self._current_user_data):
            future_row = self._current_user_data.iloc[current_idx + 14]
            ctl_growth = future_row.get("ctl", 0) - row.get("ctl", 0)
            medium_reward = ctl_growth
        
        # Long-term: Fitness improvement (scaled up)
        long_reward = 0.0
        if "fitness_improvement_long" in row:
            long_reward = row["fitness_improvement_long"]  # Remove normalization
        elif current_idx + 30 < len(self._current_user_data):
            future_row = self._current_user_data.iloc[current_idx + 30]
            fitness_gain = future_row.get("ctl", 0) - row.get("ctl", 0)
            long_reward = fitness_gain
        
        # Recovery bonus: reward maintaining high recovery
        recovery = row.get("recovery_score", 50)
        recovery_bonus = 0.0
        if recovery >= 70:
            recovery_bonus = 2.0
        elif recovery >= 50:
            recovery_bonus = 1.0
        elif recovery < 30:
            recovery_bonus = -1.0
        
        # Combine rewards with weights
        total = (
            base_reward
            + self.reward_weights["short"] * short_reward
            + self.reward_weights["medium"] * medium_reward
            + self.reward_weights["long"] * long_reward
            + recovery_bonus
        )
        
        return total
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of valid actions for current state.
        
        Returns:
            Boolean array where True = action allowed.
        """
        if not self.apply_constraints or self._current_user_data is None:
            return np.ones(len(ActionType), dtype=bool)
        
        state = self._get_current_state_dict()
        return self.constraints.get_action_mask(state, self._action_history)
    
    def render(self) -> None:
        """Render the environment (human-readable)."""
        if self._current_user_data is None:
            print("Environment not initialized")
            return
        
        current_idx = self._episode_start_idx + self._current_step
        row = self._current_user_data.iloc[current_idx]
        
        print(f"\n{'='*50}")
        print(f"Day {self._current_step + 1} / {self.episode_length}")
        print(f"Date: {row['date']}")
        print(f"Recovery: {row.get('recovery_score', 'N/A'):.1f}%")
        print(f"HRV: {row.get('hrv_rmssd', 'N/A'):.1f} ms")
        print(f"RHR: {row.get('resting_hr', 'N/A'):.1f} bpm")
        
        if self._action_history:
            last_action = ActionType(self._action_history[-1])
            print(f"Last Action: {last_action.name}")
        
        mask = self.get_action_mask()
        print(f"Available Actions: {[ActionType(i).name for i, m in enumerate(mask) if m]}")
        print(f"{'='*50}")
    
    def close(self) -> None:
        """Clean up resources."""
        pass


def make_env(
    data: pd.DataFrame,
    episode_length: int = 90,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> PeriodizationEnv:
    """
    Factory function to create a PeriodizationEnv.
    
    Args:
        data: Preprocessed DataFrame.
        episode_length: Days per episode.
        seed: Random seed.
        **kwargs: Additional arguments for PeriodizationEnv.
        
    Returns:
        Configured environment instance.
    """
    return PeriodizationEnv(
        data=data,
        episode_length=episode_length,
        seed=seed,
        **kwargs,
    )
