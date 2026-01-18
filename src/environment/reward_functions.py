"""
Reward functions for the Adaptive Periodization Agent.

This module implements modular reward components for short-term recovery,
medium-term adaptation, and long-term fitness optimization.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def calculate(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        """
        Calculate reward for a state-action-next_state transition.
        
        Args:
            state: Current state dictionary.
            action: Action taken.
            next_state: Resulting state dictionary.
            info: Additional info (e.g., action history).
            
        Returns:
            Reward value.
        """
        pass


class RecoveryReward(RewardFunction):
    """
    Short-term reward based on recovery metrics.
    
    Rewards improvements in HRV and maintains good recovery scores.
    """
    
    def __init__(
        self,
        hrv_weight: float = 0.6,
        recovery_weight: float = 0.4,
        target_recovery: float = 70.0,
    ):
        """
        Initialize recovery reward.
        
        Args:
            hrv_weight: Weight for HRV improvement component.
            recovery_weight: Weight for recovery score component.
            target_recovery: Target recovery score (penalty below this).
        """
        self.hrv_weight = hrv_weight
        self.recovery_weight = recovery_weight
        self.target_recovery = target_recovery
    
    def calculate(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        """Calculate short-term recovery reward."""
        reward = 0.0
        
        # HRV improvement
        hrv_current = state.get("hrv", 0)
        hrv_next = next_state.get("hrv", hrv_current)
        hrv_change = (hrv_next - hrv_current) / max(hrv_current, 1)  # Relative change
        reward += self.hrv_weight * hrv_change * 10  # Scale to reasonable range
        
        # Recovery maintenance
        recovery_next = next_state.get("recovery", 50)
        if recovery_next >= self.target_recovery:
            reward += self.recovery_weight * 1.0
        else:
            # Penalty proportional to deficit
            deficit = (self.target_recovery - recovery_next) / self.target_recovery
            reward -= self.recovery_weight * deficit
        
        return reward


class AdaptationReward(RewardFunction):
    """
    Medium-term reward based on training adaptation.
    
    Rewards positive CTL (Chronic Training Load) growth while
    avoiding excessive fatigue accumulation.
    """
    
    def __init__(
        self,
        ctl_weight: float = 0.7,
        tsb_weight: float = 0.3,
        optimal_tsb_range: tuple = (-10, 25),
    ):
        """
        Initialize adaptation reward.
        
        Args:
            ctl_weight: Weight for CTL growth component.
            tsb_weight: Weight for TSB (form) component.
            optimal_tsb_range: Optimal TSB range (min, max).
        """
        self.ctl_weight = ctl_weight
        self.tsb_weight = tsb_weight
        self.optimal_tsb_range = optimal_tsb_range
    
    def calculate(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        """Calculate medium-term adaptation reward."""
        reward = 0.0
        
        # CTL growth (positive is good)
        ctl_current = state.get("ctl", 0)
        ctl_next = next_state.get("ctl", ctl_current)
        ctl_growth = ctl_next - ctl_current
        reward += self.ctl_weight * ctl_growth * 0.1  # Scale growth
        
        # TSB in optimal range
        tsb = next_state.get("tsb", 0)
        min_tsb, max_tsb = self.optimal_tsb_range
        
        if min_tsb <= tsb <= max_tsb:
            # Optimal range - small bonus
            reward += self.tsb_weight * 0.5
        elif tsb < min_tsb:
            # Too fatigued
            penalty = (min_tsb - tsb) / abs(min_tsb)
            reward -= self.tsb_weight * penalty * 0.5
        else:
            # Too fresh (not training enough)
            penalty = (tsb - max_tsb) / max_tsb
            reward -= self.tsb_weight * penalty * 0.3
        
        return reward


class FitnessReward(RewardFunction):
    """
    Long-term reward based on fitness improvement.
    
    Rewards sustained fitness gains over extended periods.
    """
    
    def __init__(
        self,
        fitness_weight: float = 1.0,
        consistency_bonus: float = 0.2,
    ):
        """
        Initialize fitness reward.
        
        Args:
            fitness_weight: Weight for fitness improvement.
            consistency_bonus: Bonus for consistent training.
        """
        self.fitness_weight = fitness_weight
        self.consistency_bonus = consistency_bonus
    
    def calculate(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        """Calculate long-term fitness reward."""
        reward = 0.0
        
        # Fitness proxy improvement (CTL over longer term)
        fitness_start = info.get("episode_start_ctl", state.get("ctl", 0))
        fitness_current = next_state.get("ctl", fitness_start)
        fitness_gain = fitness_current - fitness_start
        
        reward += self.fitness_weight * fitness_gain * 0.05
        
        # Consistency bonus (not too many rest days)
        action_history = info.get("action_history", [])
        if len(action_history) >= 7:
            recent_rest_days = sum(1 for a in action_history[-7:] if a == 0)
            if 1 <= recent_rest_days <= 2:
                reward += self.consistency_bonus
        
        return reward


class OvertrainingPenalty(RewardFunction):
    """
    Penalty for overtraining indicators.
    
    Penalizes states indicating excessive fatigue or injury risk.
    """
    
    def __init__(
        self,
        hrv_crash_penalty: float = 5.0,
        low_recovery_penalty: float = 3.0,
        consecutive_intensity_penalty: float = 2.0,
    ):
        """
        Initialize overtraining penalty.
        
        Args:
            hrv_crash_penalty: Penalty for HRV drops > 2 SD.
            low_recovery_penalty: Penalty for sustained low recovery.
            consecutive_intensity_penalty: Penalty for too many hard days.
        """
        self.hrv_crash_penalty = hrv_crash_penalty
        self.low_recovery_penalty = low_recovery_penalty
        self.consecutive_intensity_penalty = consecutive_intensity_penalty
    
    def calculate(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        """Calculate overtraining penalty (returns negative value)."""
        penalty = 0.0
        
        # HRV crash detection
        hrv = next_state.get("hrv", 60)
        hrv_baseline = state.get("hrv_baseline", hrv)
        hrv_std = state.get("hrv_std", 10)
        
        if hrv < hrv_baseline - 2 * hrv_std:
            penalty += self.hrv_crash_penalty
        
        # Sustained low recovery
        recovery = next_state.get("recovery", 50)
        if recovery < 33:
            penalty += self.low_recovery_penalty * (33 - recovery) / 33
        
        # Too many consecutive hard days
        action_history = info.get("action_history", [])
        consecutive_hard = 0
        for a in reversed(action_history):
            if a >= 3:  # Tempo, HIIT, Strength
                consecutive_hard += 1
            else:
                break
        
        if consecutive_hard > 2:
            penalty += self.consecutive_intensity_penalty * (consecutive_hard - 2)
        
        return -penalty  # Return as negative reward


class CompositeReward(RewardFunction):
    """
    Composite reward combining multiple reward components.
    
    Combines short, medium, and long-term objectives with
    safety penalties.
    """
    
    def __init__(
        self,
        short_weight: float = 0.2,
        medium_weight: float = 0.3,
        long_weight: float = 0.5,
        penalty_weight: float = 1.0,
    ):
        """
        Initialize composite reward.
        
        Args:
            short_weight: Weight for short-term (recovery) reward.
            medium_weight: Weight for medium-term (adaptation) reward.
            long_weight: Weight for long-term (fitness) reward.
            penalty_weight: Weight for overtraining penalty.
        """
        self.short_weight = short_weight
        self.medium_weight = medium_weight
        self.long_weight = long_weight
        self.penalty_weight = penalty_weight
        
        # Component rewards
        self.recovery_reward = RecoveryReward()
        self.adaptation_reward = AdaptationReward()
        self.fitness_reward = FitnessReward()
        self.overtraining_penalty = OvertrainingPenalty()
    
    def calculate(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        """Calculate composite reward from all components."""
        # Calculate individual components
        r_short = self.recovery_reward.calculate(state, action, next_state, info)
        r_medium = self.adaptation_reward.calculate(state, action, next_state, info)
        r_long = self.fitness_reward.calculate(state, action, next_state, info)
        r_penalty = self.overtraining_penalty.calculate(state, action, next_state, info)
        
        # Combine with weights
        total = (
            self.short_weight * r_short
            + self.medium_weight * r_medium
            + self.long_weight * r_long
            + self.penalty_weight * r_penalty
        )
        
        return total
    
    def calculate_components(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Calculate and return all reward components separately.
        
        Useful for logging and analysis.
        """
        return {
            "short_term": self.recovery_reward.calculate(state, action, next_state, info),
            "medium_term": self.adaptation_reward.calculate(state, action, next_state, info),
            "long_term": self.fitness_reward.calculate(state, action, next_state, info),
            "penalty": self.overtraining_penalty.calculate(state, action, next_state, info),
        }


def create_reward_function(
    config: Optional[Dict[str, Any]] = None,
) -> CompositeReward:
    """
    Factory function to create a configured reward function.
    
    Args:
        config: Optional configuration dictionary.
        
    Returns:
        Configured CompositeReward instance.
    """
    if config is None:
        config = {}
    
    return CompositeReward(
        short_weight=config.get("short_weight", 0.2),
        medium_weight=config.get("medium_weight", 0.3),
        long_weight=config.get("long_weight", 0.5),
        penalty_weight=config.get("penalty_weight", 1.0),
    )
