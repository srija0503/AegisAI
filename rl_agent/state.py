"""
rl_agent/state.py
─────────────────
Defines the state space representation (41 normalized network flow features)
compatible with the NSL-KDD dataset schema and the feature extractor.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from datasets.feature_extraction import FEATURE_DIM, FEATURE_NAMES

STATE_DIM = FEATURE_DIM  # 41


class StateProcessor:
    """
    Validates, clips, and normalizes state observation vectors for the RL agent.
    """

    def __init__(
        self,
        clip_min: float = -5.0,
        clip_max: float = 5.0,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.dtype = dtype

    def process(self, raw_state: np.ndarray) -> np.ndarray:
        """Sanitizes raw array: handles NaNs/Infs and enforces dimension and range."""
        state = np.nan_to_num(raw_state, nan=0.0, posinf=self.clip_max, neginf=self.clip_min)
        if len(state) != STATE_DIM:
            # Pad or truncate to match STATE_DIM exactly
            padded = np.zeros(STATE_DIM, dtype=self.dtype)
            copy_len = min(len(state), STATE_DIM)
            padded[:copy_len] = state[:copy_len]
            state = padded
        return np.clip(state.astype(self.dtype), self.clip_min, self.clip_max)

    def to_tensor(self, state: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Converts state array to torch Tensor of shape (1, STATE_DIM)."""
        clean = self.process(state)
        return torch.from_numpy(clean).unsqueeze(0).to(device)
