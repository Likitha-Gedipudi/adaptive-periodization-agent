"""
Data loading utilities for the Adaptive Periodization Agent.

This module provides functions to load, validate, and split time-series
fitness data for training the RL agent.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Required columns for the dataset
REQUIRED_COLUMNS = [
    "user_id",
    "date",
    "hrv_rmssd",
    "resting_hr",
    "sleep_duration",
    "sleep_efficiency",
    "recovery_score",
    "strain_score",
]

# Optional physiological columns
OPTIONAL_PHYSIO_COLUMNS = [
    "rem_sleep_pct",
    "deep_sleep_pct",
    "respiratory_rate",
    "skin_temp_deviation",
    "spo2",
]

# Training load columns (computed during feature engineering)
TRAINING_LOAD_COLUMNS = [
    "atl",
    "ctl",
    "tsb",
]


class DataLoader:
    """
    Data loading and validation for fitness time-series data.
    
    Handles loading from various formats (CSV, Parquet) and ensures
    data integrity for the RL training pipeline.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing the data files.
        """
        self.data_dir = Path(data_dir)
        self._validate_directory()
    
    def _validate_directory(self) -> None:
        """Ensure the data directory exists."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def load(
        self,
        filename: str,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            filename: Name of the file to load.
            validate: Whether to validate required columns.
            
        Returns:
            DataFrame containing the loaded data.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If required columns are missing.
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load based on file extension
        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath, parse_dates=["date"])
        elif filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded {len(df)} rows from {filename}")
        
        if validate:
            self._validate_columns(df)
        
        return df
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns are present.
        
        Args:
            df: DataFrame to validate.
            
        Raises:
            ValueError: If required columns are missing.
        """
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def load_all_users(
        self,
        pattern: str = "*.csv",
    ) -> pd.DataFrame:
        """
        Load and concatenate data from all matching files.
        
        Args:
            pattern: Glob pattern for files to load.
            
        Returns:
            Combined DataFrame from all matching files.
        """
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files matching pattern: {pattern}")
        
        dfs = []
        for f in files:
            try:
                df = self.load(f.name, validate=True)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {f.name}: {e}")
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} total rows from {len(dfs)} files")
        
        return combined


def load_user_data(
    filepath: Union[str, Path],
    user_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience function to load user data from a file.
    
    Args:
        filepath: Path to the data file.
        user_id: Optional user ID to filter for.
        
    Returns:
        DataFrame with user data.
    """
    filepath = Path(filepath)
    
    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath, parse_dates=["date"])
    elif filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")
    
    if user_id is not None:
        df = df[df["user_id"] == user_id].copy()
    
    # Sort by user and date
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    
    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    by_user: bool = True,
    temporal: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Supports both temporal and user-level splitting strategies:
    - Temporal: First 70% of each user's timeline for training
    - By user: 80% of users for training, 20% for testing
    
    Args:
        df: DataFrame to split.
        train_ratio: Proportion for training (default 0.7).
        val_ratio: Proportion for validation (default 0.15).
        by_user: If True, split by unique users rather than rows.
        temporal: If True, split temporally within each user.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    
    if by_user:
        # Split by unique users
        users = df["user_id"].unique()
        np.random.shuffle(users)
        
        n_train = int(len(users) * train_ratio)
        n_val = int(len(users) * val_ratio)
        
        train_users = users[:n_train]
        val_users = users[n_train : n_train + n_val]
        test_users = users[n_train + n_val :]
        
        train_df = df[df["user_id"].isin(train_users)].copy()
        val_df = df[df["user_id"].isin(val_users)].copy()
        test_df = df[df["user_id"].isin(test_users)].copy()
        
    elif temporal:
        # Split temporally within each user
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for user_id, user_df in df.groupby("user_id"):
            user_df = user_df.sort_values("date")
            n = len(user_df)
            
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_dfs.append(user_df.iloc[:train_end])
            val_dfs.append(user_df.iloc[train_end:val_end])
            test_dfs.append(user_df.iloc[val_end:])
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
    else:
        # Simple random split (not recommended for time-series)
        n = len(df)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[indices[:train_end]].copy()
        val_df = df.iloc[indices[train_end:val_end]].copy()
        test_df = df.iloc[indices[val_end:]].copy()
    
    logger.info(
        f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
    return train_df, val_df, test_df


def get_user_trajectories(
    df: pd.DataFrame,
    min_days: int = 30,
) -> Dict[int, pd.DataFrame]:
    """
    Extract individual user trajectories from the dataset.
    
    Args:
        df: DataFrame with all user data.
        min_days: Minimum number of days required per user.
        
    Returns:
        Dictionary mapping user_id to their trajectory DataFrame.
    """
    trajectories = {}
    
    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("date").reset_index(drop=True)
        
        if len(user_df) >= min_days:
            trajectories[user_id] = user_df
        else:
            logger.debug(
                f"Skipping user {user_id}: only {len(user_df)} days (min: {min_days})"
            )
    
    logger.info(f"Extracted {len(trajectories)} user trajectories (min {min_days} days)")
    
    return trajectories
