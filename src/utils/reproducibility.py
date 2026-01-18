"""
Reproducibility utilities for ensuring consistent results.

Provides seed management and environment setup for reproducible experiments.
"""

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for CUDA
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    logger.info(f"Set random seed to {seed}")


def get_device(prefer_gpu: bool = True) -> str:
    """
    Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available.
        
    Returns:
        Device string ("cuda" or "cpu").
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if prefer_gpu and torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS")
    else:
        device = "cpu"
        logger.info("Using CPU")
    
    return device


def setup_experiment(
    experiment_name: str,
    seed: int = 42,
    log_level: str = "INFO",
) -> str:
    """
    Set up a reproducible experiment.
    
    Args:
        experiment_name: Name for logging.
        seed: Random seed.
        log_level: Logging level.
        
    Returns:
        Device string.
    """
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info(f"Setting up experiment: {experiment_name}")
    
    # Set seeds
    seed_everything(seed)
    
    # Get device
    device = get_device()
    
    return device


def log_environment_info() -> None:
    """Log information about the execution environment."""
    import sys
    import platform
    
    logger.info("=" * 50)
    logger.info("Environment Information")
    logger.info("=" * 50)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"NumPy version: {np.__version__}")
    
    if TORCH_AVAILABLE:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    logger.info("=" * 50)


class ExperimentContext:
    """
    Context manager for reproducible experiments.
    
    Usage:
        with ExperimentContext("my_experiment", seed=42) as ctx:
            # Your experiment code here
            results = train_and_evaluate()
    """
    
    def __init__(
        self,
        name: str,
        seed: int = 42,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ):
        """
        Initialize experiment context.
        
        Args:
            name: Experiment name.
            seed: Random seed.
            log_level: Logging level.
            log_file: Optional log file path.
        """
        self.name = name
        self.seed = seed
        self.log_level = log_level
        self.log_file = log_file
        self.device = None
    
    def __enter__(self):
        """Set up experiment."""
        # Set up logging
        handlers = [logging.StreamHandler()]
        if self.log_file:
            handlers.append(logging.FileHandler(self.log_file))
        
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
        
        logger.info(f"Starting experiment: {self.name}")
        
        # Set seeds
        seed_everything(self.seed)
        
        # Get device
        self.device = get_device()
        
        # Log environment
        log_environment_info()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up experiment."""
        if exc_type is not None:
            logger.error(f"Experiment failed with error: {exc_val}")
        else:
            logger.info(f"Experiment {self.name} completed successfully")
        
        return False  # Don't suppress exceptions
