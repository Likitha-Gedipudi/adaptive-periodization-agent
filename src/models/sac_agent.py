"""
SAC (Soft Actor-Critic) agent implementation for the Adaptive Periodization Agent.

This module provides a wrapper around stable-baselines3 SAC adapted for
discrete action spaces with safety constraints.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.models.networks import ActorNetwork, DualCritic, create_actor_critic

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        state_dim: int = 20,
        device: str = "cpu",
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size.
            state_dim: Dimension of state vectors.
            device: Device to place tensors on.
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pre-allocate tensors
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.float32)
        self.masks = torch.ones((capacity, 6), dtype=torch.bool)  # Action masks
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Add a transition to the buffer."""
        self.states[self.position] = torch.from_numpy(state).float()
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = torch.from_numpy(next_state).float()
        self.dones[self.position] = float(done)
        
        if action_mask is not None:
            self.masks[self.position] = torch.from_numpy(action_mask).bool()
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device),
            self.masks[indices].to(self.device),
        )
    
    def __len__(self) -> int:
        return self.size


class SACAgent:
    """
    Soft Actor-Critic agent for discrete action spaces.
    
    Implements SAC with entropy regularization for exploration.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 6,
        hidden_dims: List[int] = [256, 128, 64],
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 100000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: State dimension.
            action_dim: Number of actions.
            hidden_dims: Hidden layer dimensions.
            lr_actor: Actor learning rate.
            lr_critic: Critic learning rate.
            gamma: Discount factor.
            tau: Soft update coefficient.
            alpha: Entropy coefficient.
            auto_alpha: Whether to learn alpha automatically.
            buffer_size: Replay buffer size.
            batch_size: Training batch size.
            device: Device for training.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic = DualCritic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = DualCritic(state_dim, action_dim, hidden_dims).to(device)
        
        # Copy initial weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Entropy coefficient
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, device)
        
        # Tracking
        self.training_step = 0
    
    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current state.
            action_mask: Optional mask for valid actions.
            deterministic: If True, return greedy action.
            
        Returns:
            Selected action index.
        """
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            if action_mask is not None:
                mask_t = torch.from_numpy(action_mask).bool().unsqueeze(0).to(self.device)
            else:
                mask_t = None
            
            action, _ = self.actor.get_action(state_t, mask_t, deterministic)
            
        return action.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done, action_mask)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of training metrics.
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Sample from buffer
        states, actions, rewards, next_states, dones, masks = self.buffer.sample(
            self.batch_size
        )
        
        # Compute target Q-values
        with torch.no_grad():
            next_probs = self.actor(next_states, masks)
            next_log_probs = torch.log(next_probs + 1e-8)
            
            next_q1, next_q2 = self.critic_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            
            # V(s') = E[Q(s',a) - alpha * log(pi(a|s'))]
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            
            target_q = rewards + (1 - dones) * self.gamma * next_v
        
        # Update critics
        q1, q2 = self.critic(states)
        q1_values = q1.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2_values = q2.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        critic_loss = nn.functional.mse_loss(q1_values, target_q) + \
                      nn.functional.mse_loss(q2_values, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        probs = self.actor(states, masks)
        log_probs = torch.log(probs + 1e-8)
        
        with torch.no_grad():
            q1, q2 = self.critic(states)
            min_q = torch.min(q1, q2)
        
        # Policy loss: maximize E[Q(s,a) - alpha * log(pi(a|s))]
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (if automatic)
        alpha_loss = 0.0
        if self.auto_alpha:
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target network
        self._soft_update()
        
        self.training_step += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "q_mean": min_q.mean().item(),
            "entropy": -(probs * log_probs).sum(dim=-1).mean().item(),
        }
    
    def _soft_update(self) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save agent to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.auto_alpha else None,
            "training_step": self.training_step,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,
            },
        }, filepath)
        
        logger.info(f"Saved agent to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load agent from file."""
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        if checkpoint.get("log_alpha") is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp().item()
        
        self.training_step = checkpoint.get("training_step", 0)
        
        logger.info(f"Loaded agent from {filepath}")


def create_sac_agent(
    state_dim: int,
    action_dim: int = 6,
    config: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> SACAgent:
    """
    Factory function to create a configured SAC agent.
    
    Args:
        state_dim: State dimension.
        action_dim: Action dimension.
        config: Optional configuration dictionary.
        device: Device for training.
        
    Returns:
        Configured SACAgent instance.
    """
    if config is None:
        config = {}
    
    return SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.get("hidden_dims", [256, 128, 64]),
        lr_actor=config.get("learning_rate_actor", 3e-4),
        lr_critic=config.get("learning_rate_critic", 3e-4),
        gamma=config.get("gamma", 0.99),
        tau=config.get("tau", 0.005),
        alpha=config.get("alpha", 0.2),
        auto_alpha=config.get("auto_alpha", True),
        buffer_size=config.get("buffer_size", 100000),
        batch_size=config.get("batch_size", 256),
        device=device,
    )
