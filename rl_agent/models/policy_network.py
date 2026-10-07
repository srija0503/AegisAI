"""
rl_agent/models/policy_network.py
─────────────────────────────────
Actor-Critic Policy Network architecture for policy-gradient (PPO / A2C) methods.
Provides both action probability distribution logits (Actor) and state value estimation (Critic).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl_agent.action import NUM_ACTIONS
from rl_agent.state import STATE_DIM


class ActorCriticPolicy(nn.Module):
    """
    Shared-backbone Actor-Critic network for discrete network traffic decisions.
    Actor: outputs action logits.
    Critic: outputs scalar baseline state value V(s).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        # Shared trunk
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Actor head (action logits)
        self.actor_head = nn.Linear(in_dim, action_dim)

        # Critic head (state value)
        self.critic_head = nn.Linear(in_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        feat = self.trunk(state)
        logits = self.actor_head(feat)
        value = self.critic_head(feat)
        return logits, value

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples action, computes log probability, entropy, and value estimation.
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
