"""
Training callbacks for the Adaptive Periodization Agent.

This module implements callbacks for logging, checkpointing, and early stopping
during training.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""
    
    def on_training_start(self, trainer: Any) -> None:
        """Called at the start of training."""
        pass
    
    def on_training_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass
    
    def on_episode_start(self, episode: int, trainer: Any) -> None:
        """Called at the start of each episode."""
        pass
    
    def on_episode_end(
        self,
        episode: int,
        metrics: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """
        Called at the end of each episode.
        
        Returns:
            True to continue training, False to stop.
        """
        return True
    
    def on_step(
        self,
        step: int,
        metrics: Dict[str, float],
        trainer: Any,
    ) -> None:
        """Called after each training step."""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """
    Early stopping callback based on validation metrics.
    
    Stops training when a monitored metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.01,
        monitor: str = "mean_reward",
        mode: str = "max",
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of episodes without improvement before stopping.
            min_delta: Minimum change to qualify as improvement.
            monitor: Metric to monitor.
            mode: 'max' or 'min' for optimization direction.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_value: Optional[float] = None
        self.episodes_without_improvement = 0
        self.best_episode = 0
    
    def on_training_start(self, trainer: Any) -> None:
        """Reset state at training start."""
        self.best_value = None
        self.episodes_without_improvement = 0
        self.best_episode = 0
    
    def on_episode_end(
        self,
        episode: int,
        metrics: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """Check for improvement and return whether to continue."""
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            return True
        
        # Check if this is an improvement
        is_improvement = False
        
        if self.best_value is None:
            is_improvement = True
        elif self.mode == "max":
            is_improvement = current_value > self.best_value + self.min_delta
        else:
            is_improvement = current_value < self.best_value - self.min_delta
        
        if is_improvement:
            self.best_value = current_value
            self.best_episode = episode
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        # Check if we should stop
        if self.episodes_without_improvement >= self.patience:
            logger.info(
                f"Early stopping triggered at episode {episode}. "
                f"Best {self.monitor}: {self.best_value:.4f} at episode {self.best_episode}"
            )
            return False
        
        return True


class CheckpointCallback(TrainingCallback):
    """
    Checkpoint callback for saving model weights.
    
    Saves model at regular intervals and keeps best model.
    """
    
    def __init__(
        self,
        save_dir: str,
        save_frequency: int = 100,
        save_best: bool = True,
        monitor: str = "mean_reward",
        mode: str = "max",
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_dir: Directory to save checkpoints.
            save_frequency: Save every N episodes.
            save_best: Whether to save best model separately.
            monitor: Metric to determine best model.
            mode: 'max' or 'min'.
        """
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        
        self.best_value: Optional[float] = None
        
        # Create directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_episode_end(
        self,
        episode: int,
        metrics: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """Save checkpoint if needed."""
        # Regular checkpoint
        if episode > 0 and episode % self.save_frequency == 0:
            path = self.save_dir / f"checkpoint_ep{episode}.pt"
            trainer.agent.save(path)
            logger.info(f"Saved checkpoint to {path}")
        
        # Best model
        if self.save_best:
            current_value = metrics.get(self.monitor)
            
            if current_value is not None:
                is_best = False
                
                if self.best_value is None:
                    is_best = True
                elif self.mode == "max" and current_value > self.best_value:
                    is_best = True
                elif self.mode == "min" and current_value < self.best_value:
                    is_best = True
                
                if is_best:
                    self.best_value = current_value
                    path = self.save_dir / "best_model.pt"
                    trainer.agent.save(path)
                    logger.info(
                        f"Saved best model (episode {episode}, "
                        f"{self.monitor}={current_value:.4f})"
                    )
        
        return True


class MetricsLoggerCallback(TrainingCallback):
    """
    Callback for logging training metrics.
    
    Logs to console, TensorBoard, and/or Weights & Biases.
    """
    
    def __init__(
        self,
        log_frequency: int = 10,
        log_to_tensorboard: bool = True,
        log_to_wandb: bool = False,
        tensorboard_dir: Optional[str] = None,
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_frequency: Log every N episodes.
            log_to_tensorboard: Enable TensorBoard logging.
            log_to_wandb: Enable Weights & Biases logging.
            tensorboard_dir: TensorBoard log directory.
        """
        self.log_frequency = log_frequency
        self.log_to_tensorboard = log_to_tensorboard
        self.log_to_wandb = log_to_wandb
        self.tensorboard_dir = tensorboard_dir
        
        self.writer = None
        self.episode_rewards: List[float] = []
        self.all_metrics: List[Dict[str, float]] = []
    
    def on_training_start(self, trainer: Any) -> None:
        """Initialize logging backends."""
        if self.log_to_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                
                log_dir = self.tensorboard_dir or "./runs"
                self.writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging to {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")
                self.log_to_tensorboard = False
        
        if self.log_to_wandb:
            try:
                import wandb
                
                if not wandb.run:
                    logger.warning("WandB not initialized, skipping")
                    self.log_to_wandb = False
            except ImportError:
                logger.warning("WandB not available")
                self.log_to_wandb = False
    
    def on_episode_end(
        self,
        episode: int,
        metrics: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """Log metrics at end of episode."""
        self.all_metrics.append(metrics)
        
        if "episode_reward" in metrics:
            self.episode_rewards.append(metrics["episode_reward"])
        
        # Log at specified frequency
        if episode > 0 and episode % self.log_frequency == 0:
            self._log_metrics(episode, metrics)
        
        return True
    
    def _log_metrics(self, episode: int, metrics: Dict[str, float]) -> None:
        """Log metrics to all backends."""
        # Console logging
        log_str = f"Episode {episode}"
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f" | {key}: {value:.4f}"
        logger.info(log_str)
        
        # TensorBoard
        if self.log_to_tensorboard and self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, episode)
            
            # Rolling average
            if len(self.episode_rewards) >= 10:
                rolling_avg = np.mean(self.episode_rewards[-10:])
                self.writer.add_scalar("train/rolling_reward", rolling_avg, episode)
        
        # Weights & Biases
        if self.log_to_wandb:
            import wandb
            wandb.log(metrics, step=episode)
    
    def on_training_end(self, trainer: Any) -> None:
        """Clean up logging backends."""
        if self.writer:
            self.writer.close()


class ProgressCallback(TrainingCallback):
    """
    Progress bar callback using tqdm.
    """
    
    def __init__(self, total_episodes: int):
        """
        Initialize progress callback.
        
        Args:
            total_episodes: Total number of episodes for progress bar.
        """
        self.total_episodes = total_episodes
        self.pbar = None
    
    def on_training_start(self, trainer: Any) -> None:
        """Create progress bar."""
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_episodes, desc="Training")
        except ImportError:
            pass
    
    def on_episode_end(
        self,
        episode: int,
        metrics: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """Update progress bar."""
        if self.pbar:
            self.pbar.update(1)
            
            # Update description with key metrics
            if "episode_reward" in metrics:
                self.pbar.set_postfix(reward=f"{metrics['episode_reward']:.2f}")
        
        return True
    
    def on_training_end(self, trainer: Any) -> None:
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()


class CallbackList(TrainingCallback):
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: List[TrainingCallback]):
        """
        Initialize callback list.
        
        Args:
            callbacks: List of callbacks to run.
        """
        self.callbacks = callbacks
    
    def on_training_start(self, trainer: Any) -> None:
        """Call all callbacks."""
        for cb in self.callbacks:
            cb.on_training_start(trainer)
    
    def on_training_end(self, trainer: Any) -> None:
        """Call all callbacks."""
        for cb in self.callbacks:
            cb.on_training_end(trainer)
    
    def on_episode_start(self, episode: int, trainer: Any) -> None:
        """Call all callbacks."""
        for cb in self.callbacks:
            cb.on_episode_start(episode, trainer)
    
    def on_episode_end(
        self,
        episode: int,
        metrics: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """Call all callbacks, stop if any returns False."""
        continue_training = True
        for cb in self.callbacks:
            if not cb.on_episode_end(episode, metrics, trainer):
                continue_training = False
        return continue_training
    
    def on_step(
        self,
        step: int,
        metrics: Dict[str, float],
        trainer: Any,
    ) -> None:
        """Call all callbacks."""
        for cb in self.callbacks:
            cb.on_step(step, metrics, trainer)
