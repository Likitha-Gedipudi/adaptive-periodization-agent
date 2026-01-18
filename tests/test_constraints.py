"""
Unit tests for safety constraints.
"""

import numpy as np
import pytest

from src.environment.constraints import SafetyConstraints, apply_action_mask


class TestSafetyConstraints:
    """Tests for safety constraint system."""
    
    @pytest.fixture
    def constraints(self):
        """Create default constraints instance."""
        return SafetyConstraints()
    
    def test_action_mask_shape(self, constraints, sample_state, sample_action_history):
        """Test that action mask has correct shape."""
        mask = constraints.get_action_mask(sample_state, sample_action_history)
        
        assert mask.shape == (6,)
        assert mask.dtype == bool
    
    def test_rest_always_allowed(self, constraints):
        """Test that rest is always allowed."""
        # Even in worst conditions
        bad_state = {
            "recovery": 10,
            "hrv": 30,
            "hrv_baseline": 60,
            "hrv_std": 10,
            "days_since_rest": 10,
            "consecutive_high_intensity": 5,
        }
        
        mask = constraints.get_action_mask(bad_state, [3, 4, 5, 4, 3])
        
        assert mask[0] == True  # Rest is always allowed
    
    def test_force_rest_on_low_recovery(self, constraints):
        """Test that very low recovery forces rest."""
        low_recovery_state = {
            "recovery": 25,  # Below threshold of 30
            "hrv": 50,
            "hrv_baseline": 60,
            "hrv_std": 10,
        }
        
        # Recent active days
        active_history = [2, 3, 4]  # No rest
        
        mask = constraints.get_action_mask(low_recovery_state, active_history)
        
        # Only rest should be allowed
        assert mask[0] == True
        assert mask[1:].sum() == 0  # All other actions blocked
    
    def test_hrv_crash_blocks_high_intensity(self, constraints):
        """Test that HRV crash blocks Zone 4-5."""
        crashed_state = {
            "recovery": 50,
            "hrv": 35,  # Below baseline - 2*std = 60 - 20 = 40
            "hrv_baseline": 60,
            "hrv_std": 10,
        }
        
        mask = constraints.get_action_mask(crashed_state, [])
        
        # Tempo (3) and HIIT (4) should be blocked
        assert mask[3] == False  # Tempo
        assert mask[4] == False  # HIIT
    
    def test_max_consecutive_high_intensity(self, constraints):
        """Test that consecutive high intensity limit is enforced."""
        ok_state = {
            "recovery": 70,
            "hrv": 60,
            "hrv_baseline": 60,
            "hrv_std": 10,
        }
        
        # Three consecutive high intensity days (max is 3)
        consecutive_high = [3, 4, 5]  # Tempo, HIIT, Strength
        
        mask = constraints.get_action_mask(ok_state, consecutive_high)
        
        # High intensity should be blocked after 3 consecutive
        assert mask[3] == False  # Tempo
        assert mask[4] == False  # HIIT
        assert mask[5] == False  # Strength
    
    def test_apply_mask_returns_valid_action(self, constraints, sample_state):
        """Test that apply_mask always returns a valid action."""
        # Try to apply an invalid action
        action_history = [3, 4, 5]  # 3 consecutive high
        
        # Agent wants HIIT but it's blocked
        masked_action = constraints.apply_mask(4, sample_state, action_history)
        
        # Should return a valid action
        mask = constraints.get_action_mask(sample_state, action_history)
        assert mask[masked_action] == True
    
    def test_calculate_penalty_low_recovery(self, constraints):
        """Test that low recovery with training gives penalty."""
        low_recovery_state = {
            "recovery": 25,
            "hrv": 50,
            "hrv_baseline": 60,
            "hrv_std": 10,
        }
        
        # Penalty for training with low recovery
        penalty_training = constraints.calculate_penalty(3, low_recovery_state, [])
        # No penalty for rest
        penalty_rest = constraints.calculate_penalty(0, low_recovery_state, [])
        
        assert penalty_training > penalty_rest
        assert penalty_rest == 0  # Rest should not be penalized
    
    def test_calculate_penalty_consecutive_hard(self, constraints):
        """Test penalty for consecutive hard days."""
        state = {"recovery": 70, "hrv": 60, "hrv_baseline": 60, "hrv_std": 10}
        
        # Many consecutive hard days
        hard_history = [3, 4, 5, 4]
        
        penalty = constraints.calculate_penalty(4, state, hard_history)
        
        assert penalty > 0
    
    def test_is_action_safe(self, constraints, sample_state, sample_action_history):
        """Test is_action_safe returns boolean."""
        result = constraints.is_action_safe(0, sample_state, sample_action_history)
        
        assert isinstance(result, bool)
        assert result == True  # Rest is always safe
    
    def test_overtraining_score_range(self, constraints, sample_state, sample_action_history):
        """Test overtraining score is in valid range."""
        score = constraints.get_overtraining_score(sample_state, sample_action_history)
        
        assert 0 <= score <= 1


class TestApplyActionMask:
    """Tests for action mask application function."""
    
    def test_mask_zeros_invalid_actions(self):
        """Test that mask properly zeros invalid actions."""
        probs = np.array([0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        mask = np.array([True, True, True, False, False, True])
        
        result = apply_action_mask(probs, mask)
        
        assert result[3] == 0
        assert result[4] == 0
        assert result.sum() == pytest.approx(1.0)
    
    def test_mask_renormalizes(self):
        """Test that probabilities are renormalized after masking."""
        probs = np.array([0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        mask = np.array([True, True, True, False, False, True])
        
        result = apply_action_mask(probs, mask)
        
        assert result.sum() == pytest.approx(1.0)
    
    def test_all_masked_returns_rest(self):
        """Test that if all actions masked, rest (0) is returned."""
        probs = np.array([0.0, 0.2, 0.3, 0.2, 0.15, 0.15])
        mask = np.array([False, False, False, False, False, False])  # All masked
        
        result = apply_action_mask(probs, mask)
        
        # Should default to action 0 (rest)
        assert result[0] == 1.0
        assert result[1:].sum() == 0
    
    def test_temperature_scaling(self):
        """Test that temperature affects distribution."""
        probs = np.array([0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        mask = np.ones(6, dtype=bool)
        
        result_default = apply_action_mask(probs, mask, temperature=1.0)
        result_high_temp = apply_action_mask(probs, mask, temperature=2.0)
        
        # Higher temperature should make distribution more uniform
        # (higher entropy)
        entropy_default = -np.sum(result_default * np.log(result_default + 1e-8))
        entropy_high = -np.sum(result_high_temp * np.log(result_high_temp + 1e-8))
        
        assert entropy_high > entropy_default


class TestConstraintEdgeCases:
    """Test edge cases for constraints."""
    
    def test_empty_action_history(self):
        """Test constraints work with empty action history."""
        constraints = SafetyConstraints()
        state = {"recovery": 70, "hrv": 60, "hrv_baseline": 60, "hrv_std": 10}
        
        mask = constraints.get_action_mask(state, [])
        
        # All actions should be available with high recovery
        assert mask.all()
    
    def test_missing_state_keys(self):
        """Test constraints handle missing state keys gracefully."""
        constraints = SafetyConstraints()
        partial_state = {"recovery": 60}  # Missing hrv, baseline, etc.
        
        # Should not raise, use defaults
        mask = constraints.get_action_mask(partial_state, [])
        
        assert mask.shape == (6,)
        assert mask.any()
