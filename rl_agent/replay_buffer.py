"""
rl_agent/replay_buffer.py
─────────────────────────
Replay memory buffers for Deep Q-Learning:
1. PrioritizedReplayBuffer (PER): SumTree-backed O(log N) prioritized sampling
   focusing learning on high-surprise security anomalies (large TD errors).
2. UniformReplayBuffer: Ring-buffer for standard uniform random mini-batch sampling.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from rl_agent.state import STATE_DIM


class SumTree:
    """
    Binary tree data structure where parent node value is the sum of its children.
    Enables O(log N) sample drawing and priority updates.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float) -> int:
        """Add new priority leaf at current pointer and update ancestors."""
        idx = self.data_pointer + self.capacity
        self.update(idx, priority)

        inserted_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return inserted_idx

    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority of a node and propagate delta up to the root."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] += change

    def get_leaf(self, value: float) -> tuple[int, float, int]:
        """
        Traverse tree to locate leaf node matching prefix sum value.
        Returns: (tree_idx, priority, data_idx)
        """
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity
        return idx, self.tree[idx], data_idx

    @property
    def total_priority(self) -> float:
        return float(self.tree[1])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer with proportional prioritization and
    importance sampling bias correction.
    """

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_annealing_steps: int = 20000,
        epsilon: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_annealing_steps = beta_annealing_steps
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self._step_count = 0

        # Preallocated ring storage
        self.states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a new experience transition with maximal current priority."""
        data_idx = self.tree.add(self.max_priority ** self.alpha)
        self.states[data_idx] = state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = done

    def sample(
        self, batch_size: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """
        Sample a prioritized mini-batch and compute importance-sampling weights.
        """
        assert len(self) >= batch_size, "Buffer has fewer transitions than batch_size"

        # Anneal beta towards 1.0
        self._step_count += 1
        fraction = min(1.0, self._step_count / max(1, self.beta_annealing_steps))
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

        segment = self.tree.total_priority / batch_size
        tree_indices = []
        data_indices = []
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            val = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.tree.get_leaf(val)
            tree_indices.append(tree_idx)
            data_indices.append(data_idx)
            priorities.append(priority)

        # Compute importance sampling weights: w_i = (N * P(i)) ^ (-beta)
        probs = np.array(priorities, dtype=np.float32) / (self.tree.total_priority + 1e-10)
        weights = (len(self) * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-10)  # Normalize by max weight

        # Convert to PyTorch tensors
        s = torch.from_numpy(self.states[data_indices]).float().to(device)
        a = torch.from_numpy(self.actions[data_indices]).long().to(device)
        r = torch.from_numpy(self.rewards[data_indices]).float().to(device)
        s_next = torch.from_numpy(self.next_states[data_indices]).float().to(device)
        d = torch.from_numpy(self.dones[data_indices]).float().to(device)
        w = torch.from_numpy(weights).float().to(device)

        return s, a, r, s_next, d, w, tree_indices

    def update_priorities(self, tree_indices: list[int], td_errors: np.ndarray) -> None:
        """Update priorities for sampled experiences using latest TD errors."""
        for tree_idx, td_err in zip(tree_indices, td_errors):
            p = (abs(float(td_err)) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, p)
            self.tree.update(tree_idx, p)

    def __len__(self) -> int:
        return self.tree.size


class UniformReplayBuffer:
    """Standard ring buffer for uniform experience replay."""

    def __init__(self, capacity: int = 50000) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(self.size, batch_size, replace=False)
        s = torch.from_numpy(self.states[indices]).float().to(device)
        a = torch.from_numpy(self.actions[indices]).long().to(device)
        r = torch.from_numpy(self.rewards[indices]).float().to(device)
        s_next = torch.from_numpy(self.next_states[indices]).float().to(device)
        d = torch.from_numpy(self.dones[indices]).float().to(device)
        return s, a, r, s_next, d

    def __len__(self) -> int:
        return self.size
