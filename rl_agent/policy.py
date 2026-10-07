"""
rl_agent/policy.py
──────────────────
Exploration strategies for Deep Q-Learning:
Epsilon-greedy with linear decay and confidence-aware action selection.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch

from rl_agent.action import NUM_ACTIONS


class EpsilonGreedyPolicy:
    """
    Manages epsilon decay schedule and action sampling for Q-value predictions.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        decay_steps: int = 10000,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.num_actions = num_actions
        self.step_count = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon value along linear decay trajectory."""
        if self.decay_steps <= 0 or self.step_count >= self.decay_steps:
            return self.epsilon_end
        progress = self.step_count / self.decay_steps
        return self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

    def step(self) -> None:
        """Advance decay step counter."""
        self.step_count += 1

    def select_action(self, q_values: torch.Tensor, explore: bool = True) -> int:
        """
        Choose action using epsilon-greedy heuristic.
        If explore=False, acts purely greedily (inference mode).
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        # Greedy choice
        if isinstance(q_values, torch.Tensor):
            return int(q_values.argmax(dim=-1).item())
        return int(np.argmax(q_values))

    def select_action_with_confidence(
        self, q_values: torch.Tensor
    ) -> tuple[int, float, np.ndarray]:
        """
        Returns greedy action, softmax confidence probability, and action probability vector.
        Used for real-time firewall classification and RAG triggering.
        """
        with torch.no_grad():
            if q_values.dim() == 1:
                q_values = q_values.unsqueeze(0)
            probs = torch.softmax(q_values, dim=-1).squeeze(0).cpu().numpy()
            best_action = int(np.argmax(probs))
            confidence = float(probs[best_action])
            return best_action, confidence, probs
