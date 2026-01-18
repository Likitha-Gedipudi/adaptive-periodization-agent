"""
Neural network architectures for the Adaptive Periodization Agent.

This module implements the Actor and Critic networks used by SAC
and other RL algorithms.
"""

import logging
from typing import List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    """
    Policy network (Actor) for discrete action space.
    
    Takes state as input and outputs action probabilities.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 6,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        """
        Initialize the Actor network.
        
        Args:
            state_dim: Dimension of state input.
            action_dim: Number of discrete actions (default 6).
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Smaller initial weights for output layer
        nn.init.orthogonal_(self.output.weight, gain=0.01)
    
    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_dim).
            action_mask: Optional boolean mask (batch_size, action_dim).
            
        Returns:
            Action probabilities (batch_size, action_dim).
        """
        features = self.features(state)
        logits = self.output(features)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set masked actions to large negative value
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~action_mask, mask_value)
        
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_action(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            state: State tensor.
            action_mask: Optional action mask.
            deterministic: If True, return greedy action.
            
        Returns:
            Tuple of (action, log_prob).
        """
        probs = self.forward(state, action_mask)
        
        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8)
        else:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob.squeeze(-1)
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        
        Args:
            state: State tensor.
            action: Action tensor.
            action_mask: Optional action mask.
            
        Returns:
            Tuple of (log_prob, entropy).
        """
        probs = self.forward(state, action_mask)
        dist = Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Value network (Critic) for estimating Q-values.
    
    Takes state and action as input, outputs Q-value for each action.
    For discrete actions, this outputs Q-values for all actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 6,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        """
        Initialize the Critic network.
        
        Args:
            state_dim: Dimension of state input.
            action_dim: Number of actions (output Q for each).
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_dim).
            
        Returns:
            Q-values for all actions (batch_size, action_dim).
        """
        features = self.features(state)
        q_values = self.output(features)
        return q_values
    
    def get_q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get Q-value for specific action.
        
        Args:
            state: State tensor.
            action: Action tensor.
            
        Returns:
            Q-values for the given actions.
        """
        q_values = self.forward(state)
        return q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)


class DualCritic(nn.Module):
    """
    Dual critic network (two Q-networks for reduced overestimation).
    
    Used in SAC to take the minimum of two Q-estimates.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 6,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        """
        Initialize dual critics.
        
        Args:
            state_dim: Dimension of state input.
            action_dim: Number of actions.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dims, dropout)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dims, dropout)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critics.
        
        Args:
            state: State tensor.
            
        Returns:
            Tuple of (Q1 values, Q2 values).
        """
        q1 = self.critic1(state)
        q2 = self.critic2(state)
        return q1, q2
    
    def get_min_q(self, state: torch.Tensor) -> torch.Tensor:
        """Get minimum Q-value across both critics."""
        q1, q2 = self.forward(state)
        return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    """
    State value network (V-function).
    
    Estimates value of being in a state.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        """
        Initialize Value network.
        
        Args:
            state_dim: Dimension of state input.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.features(state)
        value = self.output(features)
        return value.squeeze(-1)


def create_actor_critic(
    state_dim: int,
    action_dim: int = 6,
    hidden_dims: List[int] = [256, 128, 64],
    dropout: float = 0.2,
    device: str = "cpu",
) -> Tuple[ActorNetwork, DualCritic]:
    """
    Factory function to create actor and critic networks.
    
    Args:
        state_dim: State dimension.
        action_dim: Action dimension.
        hidden_dims: Hidden layer sizes.
        dropout: Dropout rate.
        device: Device to place networks on.
        
    Returns:
        Tuple of (actor, dual_critic).
    """
    actor = ActorNetwork(state_dim, action_dim, hidden_dims, dropout).to(device)
    critic = DualCritic(state_dim, action_dim, hidden_dims, dropout).to(device)
    
    logger.info(
        f"Created actor-critic networks: state_dim={state_dim}, "
        f"action_dim={action_dim}, hidden_dims={hidden_dims}"
    )
    
    return actor, critic


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
