"""
Data preprocessing pipeline for the Adaptive Periodization Agent.

This module handles normalization, missing data imputation, and data cleaning
for physiological and training load metrics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessing pipeline for fitness time-series data.
    
    Handles:
    - Normalization (RobustScaler for physiological, StandardScaler for load)
    - Missing data imputation (forward-fill, interpolation)
    - Outlier detection and capping
    """
    
    # Feature groups for different scaling strategies
    PHYSIO_FEATURES = [
        "hrv_rmssd",
        "resting_hr",
        "respiratory_rate",
        "skin_temp_deviation",
        "spo2",
    ]
    
    SLEEP_FEATURES = [
        "sleep_duration",
        "sleep_efficiency",
        "rem_sleep_pct",
        "deep_sleep_pct",
    ]
    
    LOAD_FEATURES = [
        "strain_score",
        "atl",
        "ctl",
        "tsb",
    ]
    
    def __init__(
        self,
        clip_outliers: bool = True,
        outlier_std: float = 3.0,
        max_missing_pct: float = 0.2,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            clip_outliers: Whether to clip outlier values.
            outlier_std: Number of standard deviations for outlier detection.
            max_missing_pct: Maximum allowed missing data percentage per user.
        """
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        self.max_missing_pct = max_missing_pct
        
        # Scalers (fitted during preprocessing)
        self.physio_scaler: Optional[RobustScaler] = None
        self.sleep_scaler: Optional[StandardScaler] = None
        self.load_scaler: Optional[StandardScaler] = None
        
        # Fitted feature lists
        self.fitted_physio_features: List[str] = []
        self.fitted_sleep_features: List[str] = []
        self.fitted_load_features: List[str] = []
        
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the scalers on training data.
        
        Args:
            df: Training DataFrame.
            
        Returns:
            Self for method chaining.
        """
        # Identify available features
        self.fitted_physio_features = [
            f for f in self.PHYSIO_FEATURES if f in df.columns
        ]
        self.fitted_sleep_features = [
            f for f in self.SLEEP_FEATURES if f in df.columns
        ]
        self.fitted_load_features = [
            f for f in self.LOAD_FEATURES if f in df.columns
        ]
        
        # Fit scalers
        if self.fitted_physio_features:
            self.physio_scaler = RobustScaler()
            self.physio_scaler.fit(df[self.fitted_physio_features].values)
        
        if self.fitted_sleep_features:
            self.sleep_scaler = StandardScaler()
            self.sleep_scaler.fit(df[self.fitted_sleep_features].values)
        
        if self.fitted_load_features:
            self.load_scaler = StandardScaler()
            self.load_scaler.fit(df[self.fitted_load_features].values)
        
        self._is_fitted = True
        logger.info("Preprocessor fitted on training data")
        
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Transform data using fitted scalers.
        
        Args:
            df: DataFrame to transform.
            inplace: Whether to modify the DataFrame in place.
            
        Returns:
            Transformed DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        if not inplace:
            df = df.copy()
        
        # Apply scaling
        if self.physio_scaler and self.fitted_physio_features:
            scaled = self.physio_scaler.transform(df[self.fitted_physio_features].values)
            for i, feat in enumerate(self.fitted_physio_features):
                df[f"{feat}_scaled"] = scaled[:, i]
        
        if self.sleep_scaler and self.fitted_sleep_features:
            scaled = self.sleep_scaler.transform(df[self.fitted_sleep_features].values)
            for i, feat in enumerate(self.fitted_sleep_features):
                df[f"{feat}_scaled"] = scaled[:, i]
        
        if self.load_scaler and self.fitted_load_features:
            scaled = self.load_scaler.transform(df[self.fitted_load_features].values)
            for i, feat in enumerate(self.fitted_load_features):
                df[f"{feat}_scaled"] = scaled[:, i]
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform.
            
        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)
    
    def handle_missing(
        self,
        df: pd.DataFrame,
        strategy: str = "forward_fill",
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            df: DataFrame with potential missing values.
            strategy: Imputation strategy ('forward_fill', 'interpolate', 'median').
            
        Returns:
            DataFrame with imputed values.
        """
        df = df.copy()
        
        # Process each user separately
        processed_dfs = []
        
        for user_id, user_df in df.groupby("user_id"):
            user_df = user_df.sort_values("date").copy()
            
            # Check missing percentage
            missing_pct = user_df.isna().mean().mean()
            if missing_pct > self.max_missing_pct:
                logger.warning(
                    f"User {user_id} has {missing_pct:.1%} missing data, skipping"
                )
                continue
            
            # Apply imputation strategy
            numeric_cols = user_df.select_dtypes(include=[np.number]).columns
            
            if strategy == "forward_fill":
                user_df[numeric_cols] = user_df[numeric_cols].fillna(method="ffill")
                # Backward fill for remaining NaNs at the start
                user_df[numeric_cols] = user_df[numeric_cols].fillna(method="bfill")
                
            elif strategy == "interpolate":
                user_df[numeric_cols] = user_df[numeric_cols].interpolate(
                    method="linear", limit_direction="both"
                )
                
            elif strategy == "median":
                for col in numeric_cols:
                    median_val = user_df[col].median()
                    user_df[col] = user_df[col].fillna(median_val)
            
            processed_dfs.append(user_df)
        
        result = pd.concat(processed_dfs, ignore_index=True)
        logger.info(f"Handled missing values for {len(processed_dfs)} users")
        
        return result
    
    def clip_outliers_in_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip outlier values to reasonable bounds.
        
        Args:
            df: DataFrame with potential outliers.
            
        Returns:
            DataFrame with clipped values.
        """
        if not self.clip_outliers:
            return df
        
        df = df.copy()
        
        # Physiological bounds (based on domain knowledge)
        bounds = {
            "hrv_rmssd": (5, 200),        # ms
            "resting_hr": (30, 120),       # bpm
            "sleep_duration": (0, 14),     # hours
            "sleep_efficiency": (0, 100),  # percent
            "recovery_score": (0, 100),    # percent
            "strain_score": (0, 21),       # Whoop scale
            "spo2": (85, 100),             # percent
            "respiratory_rate": (8, 25),   # breaths/min
        }
        
        for col, (lower, upper) in bounds.items():
            if col in df.columns:
                original_outliers = (
                    (df[col] < lower) | (df[col] > upper)
                ).sum()
                df[col] = df[col].clip(lower, upper)
                if original_outliers > 0:
                    logger.debug(f"Clipped {original_outliers} outliers in {col}")
        
        return df
    
    def preprocess(
        self,
        df: pd.DataFrame,
        is_training: bool = True,
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Args:
            df: Raw DataFrame.
            is_training: If True, fit scalers; otherwise just transform.
            
        Returns:
            Preprocessed DataFrame.
        """
        # Step 1: Clip outliers
        df = self.clip_outliers_in_df(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing(df, strategy="forward_fill")
        
        # Step 3: Fit/transform scalers
        if is_training:
            df = self.fit_transform(df)
        else:
            df = self.transform(df)
        
        return df
    
    def get_scaled_features(self) -> List[str]:
        """
        Get list of all scaled feature names.
        
        Returns:
            List of scaled feature column names.
        """
        scaled = []
        scaled.extend([f"{f}_scaled" for f in self.fitted_physio_features])
        scaled.extend([f"{f}_scaled" for f in self.fitted_sleep_features])
        scaled.extend([f"{f}_scaled" for f in self.fitted_load_features])
        return scaled
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor.
        """
        import pickle
        
        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved preprocessor to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DataPreprocessor":
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath: Path to the saved preprocessor.
            
        Returns:
            Loaded DataPreprocessor instance.
        """
        import pickle
        
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info(f"Loaded preprocessor from {filepath}")
        return preprocessor


def normalize_per_user(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Normalize specified columns per user (z-score within each user).
    
    Args:
        df: DataFrame to normalize.
        columns: Columns to normalize.
        
    Returns:
        DataFrame with normalized columns added.
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # Calculate per-user z-score
        grouped = df.groupby("user_id")[col]
        df[f"{col}_user_norm"] = grouped.transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    
    return df
