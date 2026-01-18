"""
Pytest configuration and fixtures for the Adaptive Periodization Agent tests.
"""

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture
def sample_user_data(seed):
    """Generate sample user data for testing."""
    np.random.seed(seed)
    
    n_days = 100
    
    data = {
        "user_id": [1] * n_days,
        "date": pd.date_range("2025-01-01", periods=n_days, freq="D"),
        "hrv_rmssd": np.random.normal(60, 10, n_days),
        "resting_hr": np.random.normal(55, 5, n_days),
        "sleep_duration": np.random.uniform(6, 9, n_days),
        "sleep_efficiency": np.random.uniform(75, 95, n_days),
        "recovery_score": np.random.uniform(40, 90, n_days),
        "strain_score": np.random.uniform(5, 15, n_days),
        "action": np.random.randint(0, 6, n_days),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_multi_user_data(seed):
    """Generate sample data for multiple users."""
    np.random.seed(seed)
    
    dfs = []
    for user_id in range(5):
        n_days = 90
        
        data = {
            "user_id": [user_id] * n_days,
            "date": pd.date_range("2025-01-01", periods=n_days, freq="D"),
            "hrv_rmssd": np.random.normal(60 + user_id * 5, 10, n_days),
            "resting_hr": np.random.normal(55 - user_id * 2, 5, n_days),
            "sleep_duration": np.random.uniform(6, 9, n_days),
            "sleep_efficiency": np.random.uniform(75, 95, n_days),
            "recovery_score": np.random.uniform(40, 90, n_days),
            "strain_score": np.random.uniform(5, 15, n_days),
            "action": np.random.randint(0, 6, n_days),
        }
        
        dfs.append(pd.DataFrame(data))
    
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def preprocessed_data(sample_multi_user_data):
    """Preprocessed data with engineered features."""
    from src.data.preprocess import DataPreprocessor
    from src.data.feature_engineering import FeatureEngineer, compute_reward_components
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess(sample_multi_user_data, is_training=True)
    
    engineer = FeatureEngineer()
    data = engineer.engineer_features(data)
    data = compute_reward_components(data)
    
    return data


@pytest.fixture
def sample_state():
    """Sample state dictionary for testing constraints."""
    return {
        "recovery": 65.0,
        "hrv": 55.0,
        "hrv_baseline": 60.0,
        "hrv_std": 10.0,
        "days_since_rest": 3,
        "consecutive_high_intensity": 1,
    }


@pytest.fixture
def sample_action_history():
    """Sample action history."""
    return [2, 3, 1, 4, 2, 5, 0, 2, 3, 1]  # Mix of actions


@pytest.fixture
def sample_observation(seed):
    """Sample observation vector."""
    np.random.seed(seed)
    state_dim = 17  # Number of state features
    return np.random.randn(state_dim).astype(np.float32)


@pytest.fixture
def device():
    """Device for model testing."""
    return "cpu"


@pytest.fixture
def env_config():
    """Environment configuration."""
    return {
        "episode_length": 90,
        "apply_constraints": True,
        "reward_weights": {
            "short": 0.2,
            "medium": 0.3,
            "long": 0.5,
            "penalty": 1.0,
        },
    }


@pytest.fixture
def sac_config():
    """SAC agent configuration."""
    return {
        "hidden_dims": [64, 32],  # Smaller for faster tests
        "learning_rate_actor": 3e-4,
        "learning_rate_critic": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "auto_alpha": True,
        "batch_size": 32,
        "buffer_size": 1000,
    }


# Markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
