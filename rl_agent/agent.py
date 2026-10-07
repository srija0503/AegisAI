"""
rl_agent/agent.py
─────────────────
Deep Q-Network (Double DQN with Dueling architecture and Prioritized Replay) agent.
Processes network flow observations and outputs optimal defense decisions.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_agent.action import NUM_ACTIONS
from rl_agent.models.dqn import DuelingDQN
from rl_agent.policy import EpsilonGreedyPolicy
from rl_agent.replay_buffer import PrioritizedReplayBuffer, UniformReplayBuffer
from rl_agent.state import STATE_DIM


class DQNAgent:
    """
    Double Dueling Deep Q-Network Agent with Prioritized Experience Replay.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
        gamma: float = 0.99,
        learning_rate: float = 0.0001,
        batch_size: int = 64,
        target_update_freq: int = 200,
        tau: Optional[float] = None,  # None for hard update, float for soft Polyak
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        replay_capacity: int = 50000,
        use_per: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.use_per = use_per

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Policy & Exploration
        self.policy = EpsilonGreedyPolicy(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_steps=epsilon_decay_steps,
            num_actions=action_dim,
        )

        # Q-Networks (Online & Target)
        self.q_net = DuelingDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        # Optimizer & Loss
        self.optimizer = optim.AdamW(
            self.q_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.criterion = nn.SmoothL1Loss(reduction="none")

        # Experience Replay
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(capacity=replay_capacity)
        else:
            self.memory = UniformReplayBuffer(capacity=replay_capacity)

        self.train_step = 0

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action for state via policy."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        action = self.policy.select_action(q_values, explore=explore)
        if explore:
            self.policy.step()
        return action

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Record experience and perform learning step if ready."""
        self.memory.push(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            return self.learn()
        return None

    def learn(self) -> float:
        """Mini-batch Double DQN optimization step."""
        self.train_step += 1

        if self.use_per:
            assert isinstance(self.memory, PrioritizedReplayBuffer)
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(
                self.batch_size, device=str(self.device)
            )
        else:
            assert isinstance(self.memory, UniformReplayBuffer)
            states, actions, rewards, next_states, dones = self.memory.sample(
                self.batch_size, device=str(self.device)
            )
            weights = torch.ones_like(rewards)
            indices = []

        # Current Q-values: Q(s, a)
        q_eval = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target computation:
        # a* = argmax_a Q_online(s', a)
        # Target = r + gamma * (1 - done) * Q_target(s', a*)
        with torch.no_grad():
            next_q_online = self.q_net(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            q_target = rewards + (1.0 - dones) * self.gamma * next_q_target

        # Compute TD errors and weighted loss
        td_errors = q_eval - q_target
        loss_elements = self.criterion(q_eval, q_target)
        loss = (loss_elements * weights).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update PER priorities if enabled
        if self.use_per and indices:
            abs_errors = td_errors.detach().abs().cpu().numpy()
            self.memory.update_priorities(indices, abs_errors)

        # Update target network
        self._update_target_network()

        return float(loss.item())

    def _update_target_network(self) -> None:
        """Sync weights from online Q-net to target Q-net."""
        if self.tau is not None:
            # Soft Polyak update: theta_target = tau * theta + (1 - tau) * theta_target
            for target_param, online_param in zip(
                self.target_net.parameters(), self.q_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * online_param.data + (1.0 - self.tau) * target_param.data
                )
        elif self.train_step % self.target_update_freq == 0:
            # Periodic hard copy
            self.target_net.load_state_dict(self.q_net.state_dict())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Direct Q-value vector evaluation for state."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_t).squeeze(0).cpu().numpy()
