"""
Synthetic data generation for the Adaptive Periodization Agent.

This module generates realistic synthetic fitness data for development and testing,
simulating physiological patterns, training responses, and recovery dynamics.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# User archetype configurations
ARCHETYPES = {
    "beginner": {
        "baseline_hrv": 55,
        "baseline_rhr": 65,
        "hrv_variability": 10,
        "recovery_rate": 0.8,
        "fitness_responsiveness": 0.7,
    },
    "intermediate": {
        "baseline_hrv": 70,
        "baseline_rhr": 55,
        "hrv_variability": 12,
        "recovery_rate": 1.0,
        "fitness_responsiveness": 1.0,
    },
    "advanced": {
        "baseline_hrv": 85,
        "baseline_rhr": 48,
        "hrv_variability": 15,
        "recovery_rate": 1.2,
        "fitness_responsiveness": 1.3,
    },
    "elite": {
        "baseline_hrv": 100,
        "baseline_rhr": 42,
        "hrv_variability": 18,
        "recovery_rate": 1.5,
        "fitness_responsiveness": 1.5,
    },
}

# Training action types with their characteristics
TRAINING_TYPES = {
    0: {"name": "rest", "strain": 0, "duration": 0, "recovery_impact": 0.15},
    1: {"name": "active_recovery", "strain": 3, "duration": 30, "recovery_impact": 0.10},
    2: {"name": "aerobic_base", "strain": 8, "duration": 60, "recovery_impact": -0.05},
    3: {"name": "tempo", "strain": 12, "duration": 45, "recovery_impact": -0.15},
    4: {"name": "hiit", "strain": 16, "duration": 30, "recovery_impact": -0.25},
    5: {"name": "strength", "strain": 10, "duration": 50, "recovery_impact": -0.10},
}


class SyntheticDataGenerator:
    """
    Generate realistic synthetic fitness data for training and testing.
    
    Simulates:
    - Daily physiological metrics (HRV, RHR, sleep)
    - Training load and recovery dynamics
    - Seasonal and weekly patterns
    - User-specific characteristics
    """
    
    def __init__(
        self,
        seed: Optional[int] = 42,
    ):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def generate_dataset(
        self,
        n_users: int = 100,
        n_days: int = 180,
        start_date: str = "2025-01-01",
    ) -> pd.DataFrame:
        """
        Generate a complete synthetic dataset.
        
        Args:
            n_users: Number of users to simulate.
            n_days: Number of days per user.
            start_date: Starting date for the dataset.
            
        Returns:
            DataFrame with all user trajectories.
        """
        all_data = []
        
        for user_id in range(n_users):
            # Assign archetype
            archetype = self.rng.choice(list(ARCHETYPES.keys()))
            user_config = ARCHETYPES[archetype]
            
            # Generate user demographics
            age = self.rng.integers(18, 60)
            sex = self.rng.choice(["male", "female"])
            
            # Generate trajectory
            user_df = self._generate_user_trajectory(
                user_id=user_id,
                n_days=n_days,
                start_date=start_date,
                archetype=archetype,
                age=age,
                sex=sex,
                **user_config,
            )
            
            all_data.append(user_df)
        
        dataset = pd.concat(all_data, ignore_index=True)
        logger.info(
            f"Generated synthetic dataset: {n_users} users, {n_days} days each, "
            f"{len(dataset)} total rows"
        )
        
        return dataset
    
    def _generate_user_trajectory(
        self,
        user_id: int,
        n_days: int,
        start_date: str,
        archetype: str,
        age: int,
        sex: str,
        baseline_hrv: float,
        baseline_rhr: float,
        hrv_variability: float,
        recovery_rate: float,
        fitness_responsiveness: float,
    ) -> pd.DataFrame:
        """
        Generate a single user's trajectory.
        
        Args:
            user_id: Unique user identifier.
            n_days: Number of days to simulate.
            start_date: Starting date.
            archetype: User archetype name.
            age: User age.
            sex: User sex.
            baseline_hrv: Baseline HRV value.
            baseline_rhr: Baseline resting heart rate.
            hrv_variability: Natural HRV variability.
            recovery_rate: Recovery speed multiplier.
            fitness_responsiveness: Fitness adaptation rate.
            
        Returns:
            DataFrame with user's daily data.
        """
        dates = pd.date_range(start=start_date, periods=n_days, freq="D")
        
        # Initialize state
        state = {
            "hrv": baseline_hrv,
            "rhr": baseline_rhr,
            "recovery": 75.0,
            "fitness": 50.0,  # CTL proxy
            "fatigue": 0.0,   # ATL proxy
        }
        
        records = []
        
        for i, date in enumerate(dates):
            # Simulate training decision (realistic pattern)
            action = self._simulate_training_decision(
                day_of_week=date.dayofweek,
                recovery=state["recovery"],
                consecutive_hard_days=self._count_recent_hard_days(records),
            )
            
            training_info = TRAINING_TYPES[action]
            
            # Update physiological state
            state = self._update_state(
                state=state,
                action=action,
                recovery_rate=recovery_rate,
                fitness_responsiveness=fitness_responsiveness,
                baseline_hrv=baseline_hrv,
                baseline_rhr=baseline_rhr,
                hrv_variability=hrv_variability,
            )
            
            # Generate daily record with noise
            record = self._generate_daily_record(
                user_id=user_id,
                date=date,
                state=state,
                action=action,
                training_info=training_info,
                archetype=archetype,
                age=age,
                sex=sex,
                baseline_hrv=baseline_hrv,
                baseline_rhr=baseline_rhr,
                hrv_variability=hrv_variability,
            )
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _simulate_training_decision(
        self,
        day_of_week: int,
        recovery: float,
        consecutive_hard_days: int,
    ) -> int:
        """
        Simulate a realistic training decision based on state.
        
        Args:
            day_of_week: Day of week (0=Monday).
            recovery: Current recovery score.
            consecutive_hard_days: Recent hard training days.
            
        Returns:
            Action index (0-5).
        """
        # Sunday is typically rest day
        if day_of_week == 6:
            return 0 if self.rng.random() < 0.7 else 1
        
        # Force rest if very depleted
        if recovery < 30:
            return 0
        
        # Limit consecutive hard days
        if consecutive_hard_days >= 3:
            return self.rng.choice([0, 1, 2])  # Rest or easy
        
        # Normal training distribution based on typical periodization
        if recovery > 70:
            # Fresh - can do hard training
            weights = [0.05, 0.10, 0.30, 0.25, 0.20, 0.10]
        elif recovery > 50:
            # Moderate - avoid hardest
            weights = [0.10, 0.20, 0.35, 0.20, 0.05, 0.10]
        else:
            # Fatigued - easy training
            weights = [0.25, 0.35, 0.25, 0.10, 0.00, 0.05]
        
        return self.rng.choice(6, p=weights)
    
    def _count_recent_hard_days(self, records: List[Dict], window: int = 3) -> int:
        """Count consecutive hard training days."""
        if not records:
            return 0
        
        recent = records[-window:]
        count = 0
        
        for r in reversed(recent):
            if r.get("action", 0) >= 3:  # Tempo, HIIT, or Strength
                count += 1
            else:
                break
        
        return count
    
    def _update_state(
        self,
        state: Dict[str, float],
        action: int,
        recovery_rate: float,
        fitness_responsiveness: float,
        baseline_hrv: float,
        baseline_rhr: float,
        hrv_variability: float,
    ) -> Dict[str, float]:
        """
        Update physiological state based on training action.
        
        Implements simplified PMC (Performance Management Chart) dynamics.
        """
        training_info = TRAINING_TYPES[action]
        strain = training_info["strain"]
        recovery_impact = training_info["recovery_impact"]
        
        # Update fatigue (fast response)
        fatigue_decay = 0.3
        state["fatigue"] = (
            state["fatigue"] * (1 - fatigue_decay) 
            + strain * fitness_responsiveness
        )
        
        # Update fitness (slow response)
        fitness_decay = 0.05
        fitness_gain = strain * 0.02 * fitness_responsiveness
        state["fitness"] = (
            state["fitness"] * (1 - fitness_decay)
            + fitness_gain
        )
        
        # Update recovery (affected by training load)
        recovery_change = recovery_impact * 10 + (10 - strain) * 0.5 * recovery_rate
        state["recovery"] = np.clip(
            state["recovery"] + recovery_change, 0, 100
        )
        
        # Update HRV (inversely related to fatigue)
        hrv_target = baseline_hrv - state["fatigue"] * 0.5 + state["fitness"] * 0.2
        hrv_noise = self.rng.normal(0, hrv_variability * 0.3)
        state["hrv"] = np.clip(
            hrv_target * 0.7 + state["hrv"] * 0.3 + hrv_noise,
            baseline_hrv * 0.4,
            baseline_hrv * 1.8,
        )
        
        # Update RHR (inversely related to HRV)
        rhr_target = baseline_rhr + state["fatigue"] * 0.3 - state["fitness"] * 0.1
        rhr_noise = self.rng.normal(0, 2)
        state["rhr"] = np.clip(
            rhr_target * 0.7 + state["rhr"] * 0.3 + rhr_noise,
            baseline_rhr * 0.8,
            baseline_rhr * 1.3,
        )
        
        return state
    
    def _generate_daily_record(
        self,
        user_id: int,
        date: pd.Timestamp,
        state: Dict[str, float],
        action: int,
        training_info: Dict,
        archetype: str,
        age: int,
        sex: str,
        baseline_hrv: float,
        baseline_rhr: float,
        hrv_variability: float,
    ) -> Dict:
        """Generate a single day's record with realistic noise."""
        
        # Sleep metrics (correlated with recovery)
        sleep_base = 6.5 + state["recovery"] / 50
        sleep_duration = np.clip(
            sleep_base + self.rng.normal(0, 0.75),
            4.0, 10.0
        )
        
        sleep_efficiency = np.clip(
            75 + state["recovery"] * 0.15 + self.rng.normal(0, 5),
            60, 98
        )
        
        rem_pct = np.clip(
            20 + self.rng.normal(0, 3),
            10, 35
        )
        
        deep_pct = np.clip(
            18 + state["recovery"] * 0.05 + self.rng.normal(0, 3),
            5, 30
        )
        
        # Respiratory rate (slightly elevated when fatigued)
        resp_rate = np.clip(
            13 + state["fatigue"] * 0.05 + self.rng.normal(0, 1),
            10, 20
        )
        
        # SpO2 (stays high unless very depleted)
        spo2 = np.clip(
            96 + self.rng.normal(0, 1),
            92, 100
        )
        
        # Skin temp deviation
        skin_temp_dev = np.clip(
            self.rng.normal(0, 0.3),
            -1.0, 1.5
        )
        
        return {
            "user_id": user_id,
            "date": date,
            "archetype": archetype,
            "age": age,
            "sex": sex,
            
            # Physiological metrics
            "hrv_rmssd": round(state["hrv"], 1),
            "resting_hr": round(state["rhr"], 1),
            "recovery_score": round(state["recovery"], 1),
            "respiratory_rate": round(resp_rate, 1),
            "spo2": round(spo2, 1),
            "skin_temp_deviation": round(skin_temp_dev, 2),
            
            # Sleep metrics
            "sleep_duration": round(sleep_duration, 2),
            "sleep_efficiency": round(sleep_efficiency, 1),
            "rem_sleep_pct": round(rem_pct, 1),
            "deep_sleep_pct": round(deep_pct, 1),
            
            # Training data
            "action": action,
            "action_name": training_info["name"],
            "strain_score": training_info["strain"] + self.rng.normal(0, 1),
            "activity_duration": training_info["duration"],
            
            # State (for debugging/analysis)
            "fitness_level": round(state["fitness"], 2),
            "fatigue_level": round(state["fatigue"], 2),
        }


def generate_synthetic_dataset(
    n_users: int = 100,
    n_days: int = 180,
    start_date: str = "2025-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Convenience function to generate a synthetic dataset.
    
    Args:
        n_users: Number of users to simulate.
        n_days: Number of days per user.
        start_date: Starting date for the dataset.
        seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with synthetic fitness data.
    """
    generator = SyntheticDataGenerator(seed=seed)
    return generator.generate_dataset(
        n_users=n_users,
        n_days=n_days,
        start_date=start_date,
    )


def generate_and_save_dataset(
    output_dir: str = "./data/raw",
    n_users: int = 100,
    n_days: int = 180,
    seed: int = 42,
) -> None:
    """
    Generate synthetic dataset and save to disk.
    
    Args:
        output_dir: Directory to save the dataset.
        n_users: Number of users to simulate.
        n_days: Number of days per user.
        seed: Random seed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = generate_synthetic_dataset(
        n_users=n_users,
        n_days=n_days,
        seed=seed,
    )
    
    filepath = output_dir / f"synthetic_data_{n_users}users_{n_days}days.csv"
    df.to_csv(filepath, index=False)
    
    logger.info(f"Saved synthetic dataset to {filepath}")
    print(f"Generated synthetic dataset: {len(df)} rows saved to {filepath}")


if __name__ == "__main__":
    # Quick test generation
    logging.basicConfig(level=logging.INFO)
    df = generate_synthetic_dataset(n_users=5, n_days=30)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
