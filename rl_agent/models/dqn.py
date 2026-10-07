"""
rl_agent/models/dqn.py
──────────────────────
Deep Q-Network architectures for packet flow threat classification:
1. DuelingDQN: Decouples state value V(s) and action advantage A(s, a)
   for superior stability and sample efficiency in discrete security actions.
2. StandardDQN: Standard multilayer perceptron Q-function network.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from rl_agent.action import NUM_ACTIONS
from rl_agent.state import STATE_DIM


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network Architecture.
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden_dims: Sequence[int] = (256, 256, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        act_layer = nn.ReLU if activation.lower() == "relu" else nn.GELU

        # Shared feature representation layers
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_layer())
            in_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)

        # Advantage stream: estimates A(s, a)
        last_hidden = hidden_dims[-1]
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, last_hidden),
            act_layer(),
            nn.Linear(last_hidden, action_dim),
        )

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, last_hidden),
            act_layer(),
            nn.Linear(last_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for stable gradient propagation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: (batch_size, state_dim)
        Output: Q-values (batch_size, action_dim)
        """
        features = self.feature_extractor(state)
        advantages = self.advantage_stream(features)
        values = self.value_stream(features)

        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))


class StandardDQN(nn.Module):
    """Standard Multilayer Perceptron Deep Q-Network."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden_dims: Sequence[int] = (256, 256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
