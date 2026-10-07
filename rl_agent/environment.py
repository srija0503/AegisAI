"""
rl_agent/environment.py
───────────────────────
Custom Gymnasium environment simulating live network flow traffic.
Presents the RL agent with state vectors (flow features) and evaluates actions
against ground-truth security labels using shaped rewards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from datasets.feature_extraction import FEATURE_DIM
from rl_agent.action import ACTION_NAMES, FirewallAction, NUM_ACTIONS
from rl_agent.reward import RewardCalculator, RewardConfig
from rl_agent.state import STATE_DIM, StateProcessor


class FirewallEnv(gym.Env):
    """
    Gymnasium environment simulating packet flow inspection at an edge firewall.
    Each step represents one incoming flow requiring an ALLOW, BLOCK, or TRIGGER_RAG decision.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        data_path: Optional[str] = None,
        max_steps_per_episode: int = 500,
        reward_config: Optional[RewardConfig] = None,
        shuffle_on_reset: bool = True,
    ) -> None:
        super().__init__()

        self.max_steps_per_episode = max_steps_per_episode
        self.shuffle_on_reset = shuffle_on_reset
        self.reward_calculator = RewardCalculator(reward_config)
        self.state_processor = StateProcessor()

        # Load dataset if provided as path
        if X is not None and y is not None:
            self.X = X.astype(np.float32)
            self.y = y.astype(np.int32)
        elif data_path:
            self.X, self.y = self._load_data(data_path)
        else:
            # Fallback default synthetic distribution for self-contained testing
            self.X = np.random.randn(1000, STATE_DIM).astype(np.float32)
            self.y = np.random.randint(0, 2, size=1000, dtype=np.int32)

        self.n_samples = len(self.X)

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )

        # Environment episode state
        self._current_step = 0
        self._sample_indices = np.arange(self.n_samples)
        self._ptr = 0

        # Episode telemetry
        self.episode_reward: float = 0.0
        self.tp: int = 0
        self.fp: int = 0
        self.tn: int = 0
        self.fn: int = 0
        self.rag_count: int = 0

    def _load_data(self, path_str: str) -> tuple[np.ndarray, np.ndarray]:
        """Load features and binary labels from Parquet or CSV."""
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Environment data file not found: {path}")

        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        from datasets.feature_extraction import df_to_tensors
        return df_to_tensors(df)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if self.shuffle_on_reset:
            self.np_random.shuffle(self._sample_indices)

        self._current_step = 0
        self._ptr = 0
        self.episode_reward = 0.0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.rag_count = 0

        obs = self._get_current_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        current_idx = self._sample_indices[self._ptr]
        is_attack = bool(self.y[current_idx])

        # Compute reward
        reward = self.reward_calculator.compute(action=action, is_attack=is_attack)
        self.episode_reward += reward

        # Update episode statistics
        act_enum = FirewallAction(action)
        if act_enum == FirewallAction.BLOCK:
            if is_attack:
                self.tp += 1
            else:
                self.fp += 1
        elif act_enum == FirewallAction.ALLOW:
            if not is_attack:
                self.tn += 1
            else:
                self.fn += 1
        elif act_enum == FirewallAction.TRIGGER_RAG:
            self.rag_count += 1
            if is_attack:
                self.tp += 1
            else:
                self.tn += 1

        self._current_step += 1
        self._ptr = (self._ptr + 1) % self.n_samples

        terminated = self._current_step >= self.max_steps_per_episode
        truncated = False

        next_obs = self._get_current_obs()
        info = self._get_info()
        info["step_reward"] = reward
        info["ground_truth"] = int(is_attack)
        info["action_name"] = ACTION_NAMES[action]

        return next_obs, reward, terminated, truncated, info

    def _get_current_obs(self) -> np.ndarray:
        idx = self._sample_indices[self._ptr]
        return self.state_processor.process(self.X[idx])

    def _get_info(self) -> dict[str, Any]:
        total_eval = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / total_eval if total_eval > 0 else 0.0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

        return {
            "step": self._current_step,
            "episode_reward": round(self.episode_reward, 3),
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "rag_count": self.rag_count,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
