"""
Safety constraints for the Adaptive Periodization Agent.

This module implements hard and soft constraints to ensure safe training
recommendations, including action masking and penalty calculation.
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class SafetyConstraints:
    """
    Safety constraint system for training recommendations.
    
    Implements:
    - Hard constraints: Block unsafe actions (action masking)
    - Soft constraints: Penalize suboptimal but not dangerous actions
    """
    
    # Thresholds
    LOW_RECOVERY_THRESHOLD = 30  # Force rest
    MODERATE_RECOVERY_THRESHOLD = 50  # Limit intensity
    HRV_CRASH_SD = 2.0  # Standard deviations below baseline
    MAX_CONSECUTIVE_HIGH_INTENSITY = 3
    MIN_REST_DAYS_PER_WEEK = 1
    
    def __init__(
        self,
        low_recovery_threshold: float = 30,
        moderate_recovery_threshold: float = 50,
        max_consecutive_high: int = 3,
        min_rest_per_week: int = 1,
    ):
        """
        Initialize safety constraints.
        
        Args:
            low_recovery_threshold: Recovery below this forces rest.
            moderate_recovery_threshold: Recovery below this limits intensity.
            max_consecutive_high: Maximum consecutive high-intensity days.
            min_rest_per_week: Minimum rest days per 7-day window.
        """
        self.low_recovery_threshold = low_recovery_threshold
        self.moderate_recovery_threshold = moderate_recovery_threshold
        self.max_consecutive_high = max_consecutive_high
        self.min_rest_per_week = min_rest_per_week
    
    def get_action_mask(
        self,
        state: Dict[str, Any],
        action_history: List[int],
    ) -> np.ndarray:
        """
        Generate action mask based on current state.
        
        Args:
            state: Current physiological state.
            action_history: List of previous actions.
            
        Returns:
            Boolean mask array where True = action allowed.
            Shape: (6,) for 6 action types.
        """
        # Start with all actions allowed
        mask = np.ones(6, dtype=bool)
        
        recovery = state.get("recovery", 50)
        hrv = state.get("hrv", 60)
        hrv_baseline = state.get("hrv_baseline", hrv)
        hrv_std = state.get("hrv_std", 10)
        
        # Rule 1: Force rest if recovery < 30% for 2+ consecutive days
        if self._check_sustained_low_recovery(recovery, action_history):
            mask[:] = False
            mask[0] = True  # Only rest allowed
            return mask
        
        # Rule 2: Block Zone 4-5 if HRV crashed
        if hrv < hrv_baseline - self.HRV_CRASH_SD * hrv_std:
            mask[3] = False  # Block Tempo
            mask[4] = False  # Block HIIT
        
        # Rule 3: Max consecutive high-intensity days
        consecutive_high = self._count_consecutive_high_intensity(action_history)
        if consecutive_high >= self.max_consecutive_high:
            mask[3] = False  # Block Tempo
            mask[4] = False  # Block HIIT
            mask[5] = False  # Block Strength
        
        # Rule 4: Minimum rest days per week
        if len(action_history) >= 6:
            rest_count = sum(1 for a in action_history[-6:] if a == 0)
            if rest_count < self.min_rest_per_week:
                # Not quite a hard mask, but strongly encourage rest
                # For safety, if 0 rest days in last 6, only allow easy options
                if rest_count == 0:
                    mask[3] = False
                    mask[4] = False
        
        # Rule 5: Limit intensity at moderate recovery
        if recovery < self.moderate_recovery_threshold:
            mask[4] = False  # Block HIIT
        
        return mask
    
    def _check_sustained_low_recovery(
        self,
        current_recovery: float,
        action_history: List[int],
    ) -> bool:
        """Check if recovery has been low for multiple days."""
        if current_recovery >= self.low_recovery_threshold:
            return False
        
        # Check if this pattern has persisted
        # (Simplified: if current is low and we've been active recently)
        if len(action_history) >= 1:
            recent_actions = action_history[-2:] if len(action_history) >= 2 else action_history
            # If last actions were not rest and recovery is low, force rest
            if sum(1 for a in recent_actions if a == 0) == 0:
                return True
        
        return False
    
    def _count_consecutive_high_intensity(self, action_history: List[int]) -> int:
        """Count consecutive high-intensity days from recent history."""
        count = 0
        for action in reversed(action_history):
            if action >= 3:  # Tempo (3), HIIT (4), Strength (5)
                count += 1
            else:
                break
        return count
    
    def apply_mask(
        self,
        action: int,
        state: Dict[str, Any],
        action_history: List[int],
    ) -> int:
        """
        Apply action masking, returning a valid action.
        
        If the proposed action is masked, returns the safest alternative.
        
        Args:
            action: Proposed action.
            state: Current state.
            action_history: Action history.
            
        Returns:
            Valid action (may be different from input).
        """
        mask = self.get_action_mask(state, action_history)
        
        if mask[action]:
            return action
        
        # Proposed action is blocked - find safest alternative
        # Priority: Rest > Active Recovery > Aerobic > Tempo > Strength > HIIT
        priority = [0, 1, 2, 5, 3, 4]
        
        for alt_action in priority:
            if mask[alt_action]:
                logger.debug(
                    f"Action {action} blocked, using {alt_action} instead"
                )
                return alt_action
        
        # Fallback to rest (should always be allowed)
        return 0
    
    def calculate_penalty(
        self,
        action: int,
        state: Dict[str, Any],
        action_history: List[int],
    ) -> float:
        """
        Calculate soft penalty for suboptimal actions.
        
        Unlike hard constraints (masking), soft constraints add penalties
        to the reward function to discourage but not prevent actions.
        
        Args:
            action: Action taken.
            state: Current state.
            action_history: Action history.
            
        Returns:
            Penalty value (non-negative, higher = worse).
        """
        penalty = 0.0
        
        recovery = state.get("recovery", 50)
        
        # Penalty 1: Low recovery + any training
        if recovery < self.low_recovery_threshold and action > 0:
            penalty += 10 * (self.low_recovery_threshold - recovery) / self.low_recovery_threshold
        
        # Penalty 2: Moderate recovery + high intensity
        if recovery < self.moderate_recovery_threshold and action >= 3:
            penalty += 5 * (self.moderate_recovery_threshold - recovery) / self.moderate_recovery_threshold
        
        # Penalty 3: Too many consecutive hard days
        consecutive_hard = self._count_consecutive_high_intensity(action_history)
        if consecutive_hard >= 2 and action >= 3:
            penalty += 3 * (consecutive_hard - 1)
        
        # Penalty 4: No rest days in extended period
        if len(action_history) >= 7:
            rest_count = sum(1 for a in action_history[-7:] if a == 0)
            if rest_count == 0:
                penalty += 5  # Missing rest day penalty
        
        # Penalty 5: HRV below normal + training
        hrv = state.get("hrv", 60)
        hrv_baseline = state.get("hrv_baseline", hrv)
        if hrv < hrv_baseline * 0.85 and action >= 2:
            penalty += 3 * (1 - hrv / hrv_baseline)
        
        return penalty
    
    def is_action_safe(
        self,
        action: int,
        state: Dict[str, Any],
        action_history: List[int],
    ) -> bool:
        """
        Check if an action is safe (passes hard constraints).
        
        Args:
            action: Action to check.
            state: Current state.
            action_history: Action history.
            
        Returns:
            True if action is allowed, False otherwise.
        """
        mask = self.get_action_mask(state, action_history)
        return mask[action]
    
    def get_overtraining_score(
        self,
        state: Dict[str, Any],
        action_history: List[int],
    ) -> float:
        """
        Calculate an overtraining risk score.
        
        Args:
            state: Current state.
            action_history: Action history.
            
        Returns:
            Score from 0 (no risk) to 1 (high risk).
        """
        score = 0.0
        factors = 0
        
        # Factor 1: Low recovery
        recovery = state.get("recovery", 50)
        if recovery < 40:
            score += (40 - recovery) / 40
            factors += 1
        
        # Factor 2: HRV below baseline
        hrv = state.get("hrv", 60)
        hrv_baseline = state.get("hrv_baseline", hrv)
        if hrv < hrv_baseline * 0.9:
            score += (hrv_baseline - hrv) / hrv_baseline
            factors += 1
        
        # Factor 3: No recent rest
        if len(action_history) >= 7:
            rest_count = sum(1 for a in action_history[-7:] if a == 0)
            if rest_count == 0:
                score += 0.5
                factors += 1
        
        # Factor 4: Consecutive hard days
        consecutive_hard = self._count_consecutive_high_intensity(action_history)
        if consecutive_hard >= 3:
            score += 0.3 * (consecutive_hard / 5)
            factors += 1
        
        # Normalize
        if factors > 0:
            score = min(score / factors, 1.0)
        
        return score


def apply_action_mask(
    action_probs: np.ndarray,
    mask: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Apply action mask to probability distribution.
    
    Zeros out masked actions and renormalizes probabilities.
    
    Args:
        action_probs: Original action probabilities (shape: (6,)).
        mask: Boolean mask (True = allowed).
        temperature: Temperature for probability scaling.
        
    Returns:
        Masked and normalized probabilities.
    """
    # Apply mask
    masked_probs = action_probs * mask.astype(float)
    
    # Handle case where all actions are masked
    if masked_probs.sum() == 0:
        # Force rest as only option
        masked_probs = np.zeros_like(action_probs)
        masked_probs[0] = 1.0
        return masked_probs
    
    # Apply temperature and renormalize
    if temperature != 1.0:
        masked_probs = masked_probs ** (1 / temperature)
    
    return masked_probs / masked_probs.sum()
