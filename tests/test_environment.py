"""
Unit tests for the PeriodizationEnv environment.
"""

import numpy as np
import pytest

from src.environment.periodization_env import PeriodizationEnv, ActionType


class TestPeriodizationEnv:
    """Tests for the Gymnasium environment."""
    
    def test_environment_creation(self, preprocessed_data, env_config, seed):
        """Test that environment can be created."""
        env = PeriodizationEnv(
            data=preprocessed_data,
            episode_length=env_config["episode_length"],
            seed=seed,
        )
        
        assert env is not None
        assert env.action_space.n == 6
        assert env.observation_space.shape[0] > 0
    
    def test_observation_space_shape(self, preprocessed_data, seed):
        """Test observation space dimensions."""
        env = PeriodizationEnv(data=preprocessed_data, seed=seed)
        
        obs, info = env.reset()
        
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
    
    def test_reset_returns_valid_state(self, preprocessed_data, seed):
        """Test that reset returns valid initial state."""
        env = PeriodizationEnv(data=preprocessed_data, seed=seed)
        
        obs, info = env.reset()
        
        assert obs is not None
        assert "user_id" in info
        assert "step" in info
        assert info["step"] == 0
    
    def test_step_returns_correct_format(self, preprocessed_data, seed):
        """Test that step returns correct tuple format."""
        env = PeriodizationEnv(data=preprocessed_data, seed=seed)
        env.reset()
        
        action = 2  # Aerobic base
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_episode_terminates_at_length(self, preprocessed_data, seed):
        """Test that episode terminates after episode_length steps."""
        episode_length = 30
        env = PeriodizationEnv(
            data=preprocessed_data,
            episode_length=episode_length,
            seed=seed,
        )
        env.reset()
        
        for i in range(episode_length):
            _, _, terminated, truncated, _ = env.step(0)
            
            if i < episode_length - 1:
                assert not terminated
            else:
                assert terminated
    
    def test_all_actions_valid(self, preprocessed_data, seed):
        """Test that all action types can be executed."""
        env = PeriodizationEnv(
            data=preprocessed_data,
            apply_constraints=False,  # Disable constraints for this test
            seed=seed,
        )
        
        for action in range(6):
            env.reset()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert info["action"] == action
            assert info["action_name"] == ActionType(action).name
    
    def test_action_mask_shape(self, preprocessed_data, seed):
        """Test that action mask has correct shape."""
        env = PeriodizationEnv(
            data=preprocessed_data,
            apply_constraints=True,
            seed=seed,
        )
        env.reset()
        
        mask = env.get_action_mask()
        
        assert mask.shape == (6,)
        assert mask.dtype == bool
        assert mask.any()  # At least one action should be allowed
    
    def test_reproducibility_with_seed(self, preprocessed_data):
        """Test that same seed produces same episode."""
        seed = 123
        
        env1 = PeriodizationEnv(data=preprocessed_data, seed=seed)
        obs1, _ = env1.reset()
        
        env2 = PeriodizationEnv(data=preprocessed_data, seed=seed)
        obs2, _ = env2.reset()
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_info_contains_required_keys(self, preprocessed_data, seed):
        """Test that step info contains required debugging information."""
        env = PeriodizationEnv(data=preprocessed_data, seed=seed)
        env.reset()
        
        _, _, _, _, info = env.step(0)
        
        assert "step" in info
        assert "action" in info
        assert "action_name" in info
        assert "reward" in info


class TestActionTypes:
    """Tests for action type enumeration."""
    
    def test_action_type_values(self):
        """Test that action types have correct integer values."""
        assert ActionType.REST == 0
        assert ActionType.ACTIVE_RECOVERY == 1
        assert ActionType.AEROBIC_BASE == 2
        assert ActionType.TEMPO == 3
        assert ActionType.HIIT == 4
        assert ActionType.STRENGTH == 5
    
    def test_action_type_count(self):
        """Test that there are exactly 6 action types."""
        assert len(ActionType) == 6


class TestEnvironmentEdgeCases:
    """Test edge cases and error handling."""
    
    def test_step_without_reset_raises(self, preprocessed_data, seed):
        """Test that stepping without reset raises error."""
        env = PeriodizationEnv(data=preprocessed_data, seed=seed)
        
        with pytest.raises(RuntimeError):
            env.step(0)
    
    def test_invalid_action_handled(self, preprocessed_data, seed):
        """Test that invalid action raises appropriate error."""
        env = PeriodizationEnv(data=preprocessed_data, seed=seed)
        env.reset()
        
        # Action outside valid range
        with pytest.raises(Exception):  # Could be IndexError or ValueError
            env.step(100)
    
    def test_observation_no_nans(self, preprocessed_data, seed):
        """Test that observations don't contain NaN values."""
        env = PeriodizationEnv(data=preprocessed_data, seed=seed)
        
        obs, _ = env.reset()
        assert not np.isnan(obs).any()
        
        for _ in range(10):
            obs, _, terminated, _, _ = env.step(np.random.randint(0, 6))
            if not terminated:
                assert not np.isnan(obs).any()
