"""
Feature engineering for the Adaptive Periodization Agent.

This module computes derived features including training load metrics (ATL, CTL, TSB),
temporal features, rolling statistics, and cyclic encodings.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for fitness time-series data.
    
    Computes:
    - Training load metrics (ATL, CTL, TSB)
    - Rolling statistics (7d, 14d, 28d windows)
    - Temporal features (day of week encoding, trends)
    - Interaction features
    """
    
    # Rolling window sizes
    WINDOWS = [7, 14, 28]
    
    def __init__(
        self,
        compute_training_load: bool = True,
        compute_rolling_stats: bool = True,
        compute_temporal: bool = True,
        compute_interactions: bool = True,
    ):
        """
        Initialize the feature engineer.
        
        Args:
            compute_training_load: Whether to compute ATL/CTL/TSB.
            compute_rolling_stats: Whether to compute rolling statistics.
            compute_temporal: Whether to compute temporal features.
            compute_interactions: Whether to compute interaction terms.
        """
        self.compute_training_load = compute_training_load
        self.compute_rolling_stats = compute_rolling_stats
        self.compute_temporal = compute_temporal
        self.compute_interactions = compute_interactions
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input DataFrame with raw features.
            
        Returns:
            DataFrame with engineered features added.
        """
        df = df.copy()
        
        # Ensure data is sorted
        df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
        
        # Apply feature engineering per user
        processed_dfs = []
        
        for user_id, user_df in df.groupby("user_id"):
            user_df = user_df.sort_values("date").copy()
            
            if self.compute_training_load:
                user_df = self._add_training_load(user_df)
            
            if self.compute_rolling_stats:
                user_df = self._add_rolling_stats(user_df)
            
            if self.compute_temporal:
                user_df = self._add_temporal_features(user_df)
            
            if self.compute_interactions:
                user_df = self._add_interaction_features(user_df)
            
            processed_dfs.append(user_df)
        
        result = pd.concat(processed_dfs, ignore_index=True)
        logger.info(f"Engineered features: {result.shape[1]} columns")
        
        return result
    
    def _add_training_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Acute Training Load (ATL), Chronic Training Load (CTL), and TSB.
        
        ATL: 7-day exponentially weighted moving average of strain
        CTL: 42-day EWMA of strain
        TSB: CTL - ATL (Training Stress Balance)
        
        Args:
            df: User DataFrame sorted by date.
            
        Returns:
            DataFrame with ATL, CTL, TSB columns added.
        """
        strain_col = "strain_score"
        
        if strain_col not in df.columns:
            logger.warning(f"Column {strain_col} not found, using recovery_score")
            strain_col = "recovery_score"
            # Invert recovery as proxy for strain
            df["_strain_proxy"] = 100 - df[strain_col]
            strain_col = "_strain_proxy"
        
        # Compute EWMA with appropriate spans
        # ATL: 7-day acute load (faster response)
        df["atl"] = df[strain_col].ewm(span=7, adjust=False).mean()
        
        # CTL: 42-day chronic load (slower response)
        df["ctl"] = df[strain_col].ewm(span=42, adjust=False).mean()
        
        # TSB: Training Stress Balance (positive = fresh, negative = fatigued)
        df["tsb"] = df["ctl"] - df["atl"]
        
        # Clean up proxy column if created
        if "_strain_proxy" in df.columns:
            df = df.drop(columns=["_strain_proxy"])
        
        return df
    
    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling statistics for key physiological metrics.
        
        Args:
            df: User DataFrame sorted by date.
            
        Returns:
            DataFrame with rolling statistics added.
        """
        # Key metrics to compute rolling stats for
        metrics = ["hrv_rmssd", "resting_hr", "sleep_duration", "recovery_score"]
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            for window in self.WINDOWS:
                # Rolling mean
                df[f"{metric}_mean_{window}d"] = (
                    df[metric].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std (for detecting variability)
                df[f"{metric}_std_{window}d"] = (
                    df[metric].rolling(window=window, min_periods=1).std().fillna(0)
                )
            
            # Rolling z-score (current value vs recent mean)
            df[f"{metric}_zscore_7d"] = (
                (df[metric] - df[f"{metric}_mean_7d"])
                / (df[f"{metric}_std_7d"] + 1e-8)
            )
        
        # Trend features (slope of recent window)
        if "hrv_rmssd" in df.columns:
            df["hrv_trend_7d"] = self._compute_trend(df["hrv_rmssd"], window=7)
        
        if "recovery_score" in df.columns:
            df["recovery_trend_7d"] = self._compute_trend(df["recovery_score"], window=7)
        
        return df
    
    def _compute_trend(
        self,
        series: pd.Series,
        window: int = 7,
    ) -> pd.Series:
        """
        Compute rolling linear trend (slope) for a series.
        
        Args:
            series: Input time series.
            window: Window size for trend calculation.
            
        Returns:
            Series with trend values (positive = improving).
        """
        def linear_slope(values: np.ndarray) -> float:
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            # Linear regression slope
            slope = np.polyfit(x, values, 1)[0]
            return slope
        
        return series.rolling(window=window, min_periods=2).apply(
            linear_slope, raw=True
        ).fillna(0)
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features including cyclic encodings.
        
        Args:
            df: User DataFrame with 'date' column.
            
        Returns:
            DataFrame with temporal features added.
        """
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        
        # Day of week (0-6)
        day_of_week = df["date"].dt.dayofweek
        
        # Cyclic encoding for day of week
        df["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        df["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Month for seasonality
        month = df["date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        
        # Days since last rest day (requires action history or recovery-based proxy)
        if "recovery_score" in df.columns:
            # Proxy: count days since recovery > 80%
            df["days_since_high_recovery"] = self._days_since_condition(
                df["recovery_score"] > 80
            )
        
        # Consecutive training days (based on strain)
        if "strain_score" in df.columns:
            df["consecutive_high_strain"] = self._consecutive_count(
                df["strain_score"] > 10  # High strain threshold
            )
        
        return df
    
    def _days_since_condition(self, condition: pd.Series) -> pd.Series:
        """
        Count days since condition was last True.
        
        Args:
            condition: Boolean series.
            
        Returns:
            Series with day counts.
        """
        # Create groups that increment when condition is True
        groups = (~condition).cumsum()
        
        # Count within each group
        counts = condition.groupby(groups).cumcount() + 1
        
        # Reset count when condition is True
        counts = counts.where(~condition, 0)
        
        return counts
    
    def _consecutive_count(self, condition: pd.Series) -> pd.Series:
        """
        Count consecutive True values.
        
        Args:
            condition: Boolean series.
            
        Returns:
            Series with consecutive counts.
        """
        # Create groups that increment when condition changes
        groups = (condition != condition.shift()).cumsum()
        
        # Count within each group, but only for True values
        counts = condition.groupby(groups).cumcount() + 1
        
        # Set to 0 when condition is False
        counts = counts.where(condition, 0)
        
        return counts
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between key metrics.
        
        Args:
            df: User DataFrame.
            
        Returns:
            DataFrame with interaction features added.
        """
        # Recovery × Days since rest (fatigue accumulation signal)
        if "recovery_score" in df.columns and "days_since_high_recovery" in df.columns:
            df["recovery_fatigue_interaction"] = (
                (100 - df["recovery_score"]) * df["days_since_high_recovery"] / 10
            )
        
        # HRV × Sleep quality (combined readiness)
        if "hrv_rmssd" in df.columns and "sleep_efficiency" in df.columns:
            df["hrv_sleep_interaction"] = (
                df["hrv_rmssd"] * df["sleep_efficiency"] / 100
            )
        
        # TSB × Recovery (overall freshness indicator)
        if "tsb" in df.columns and "recovery_score" in df.columns:
            df["tsb_recovery_interaction"] = df["tsb"] * df["recovery_score"] / 100
        
        # Resting HR deviation (lower than baseline is good)
        if "resting_hr" in df.columns and "resting_hr_mean_28d" in df.columns:
            df["rhr_deviation"] = df["resting_hr"] - df["resting_hr_mean_28d"]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all possible feature names that can be generated.
        
        Returns:
            List of feature column names.
        """
        features = []
        
        # Training load
        features.extend(["atl", "ctl", "tsb"])
        
        # Rolling stats (for key metrics)
        metrics = ["hrv_rmssd", "resting_hr", "sleep_duration", "recovery_score"]
        for metric in metrics:
            for window in self.WINDOWS:
                features.append(f"{metric}_mean_{window}d")
                features.append(f"{metric}_std_{window}d")
            features.append(f"{metric}_zscore_7d")
        
        # Trends
        features.extend(["hrv_trend_7d", "recovery_trend_7d"])
        
        # Temporal
        features.extend([
            "day_sin", "day_cos",
            "month_sin", "month_cos",
            "days_since_high_recovery",
            "consecutive_high_strain",
        ])
        
        # Interactions
        features.extend([
            "recovery_fatigue_interaction",
            "hrv_sleep_interaction",
            "tsb_recovery_interaction",
            "rhr_deviation",
        ])
        
        return features


def compute_reward_components(
    df: pd.DataFrame,
    short_term_horizon: int = 1,
    medium_term_horizon: int = 14,
    long_term_horizon: int = 30,
) -> pd.DataFrame:
    """
    Pre-compute reward components from future data (for offline RL).
    
    This requires access to future data points to calculate delayed rewards.
    Must be run on complete user trajectories.
    
    Args:
        df: User trajectory DataFrame sorted by date.
        short_term_horizon: Days for short-term reward (default 1).
        medium_term_horizon: Days for medium-term reward (default 14).
        long_term_horizon: Days for long-term reward (default 30).
        
    Returns:
        DataFrame with reward component columns added.
    """
    df = df.copy()
    
    # Short-term: HRV change next day
    if "hrv_rmssd" in df.columns:
        df["hrv_next_day"] = df["hrv_rmssd"].shift(-short_term_horizon)
        df["hrv_change_short"] = df["hrv_next_day"] - df["hrv_rmssd"]
    
    # Medium-term: CTL growth over horizon
    if "ctl" in df.columns:
        df["ctl_future"] = df["ctl"].shift(-medium_term_horizon)
        df["ctl_growth_medium"] = df["ctl_future"] - df["ctl"]
    
    # Long-term: Overall fitness proxy improvement
    if "ctl" in df.columns:
        df["ctl_long_future"] = df["ctl"].shift(-long_term_horizon)
        df["fitness_improvement_long"] = df["ctl_long_future"] - df["ctl"]
    
    # Overtraining indicators
    if "recovery_score" in df.columns:
        # Look ahead for recovery crashes
        df["recovery_min_next_7d"] = (
            df["recovery_score"]
            .rolling(window=7, min_periods=1)
            .min()
            .shift(-7)
        )
    
    return df
