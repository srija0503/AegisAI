"""
rl_agent/inference.py
─────────────────────
Real-time, low-latency inference engine for the RL Firewall Agent.
Plugs directly into the FirewallController to classify live incoming flows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from rl_agent.action import ACTION_NAMES, FirewallAction
from rl_agent.models.dqn import DuelingDQN
from rl_agent.state import STATE_DIM, StateProcessor


@dataclass
class RLInferenceResult:
    action: FirewallAction
    action_name: str
    confidence: float
    q_values: list[float]
    probabilities: list[float]
    anomaly_score: float
    latency_ms: float

    @property
    def is_attack(self) -> bool:
        return self.action == FirewallAction.BLOCK

    @property
    def is_uncertain(self) -> bool:
        return self.action == FirewallAction.TRIGGER_RAG


class RLInferenceEngine:
    """
    Inference wrapper optimized for microsecond-scale edge firewall execution.
    """

    def __init__(
        self,
        model_or_path: Optional[Union[nn.Module, str]] = None,
        confidence_threshold: float = 0.80,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.processor = StateProcessor()

        if isinstance(model_or_path, nn.Module):
            self.model = model_or_path.to(self.device)
        elif isinstance(model_or_path, str):
            self.model = DuelingDQN().to(self.device)
            from rl_agent.checkpoint import CheckpointManager
            CheckpointManager().load(self.model, model_or_path, device=str(self.device))
        else:
            self.model = DuelingDQN().to(self.device)

        self.model.eval()

    def predict(self, feature_vector: np.ndarray) -> RLInferenceResult:
        """
        Classify single flow feature vector.
        """
        t0 = time.perf_counter()

        clean_state = self.processor.process(feature_vector)
        tensor_state = torch.from_numpy(clean_state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_vals = self.model(tensor_state)
            probs = torch.softmax(q_vals, dim=-1).squeeze(0)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        q_list = [float(x) for x in q_vals.squeeze(0).cpu().numpy()]
        p_list = [float(x) for x in probs.cpu().numpy()]

        best_action_idx = int(np.argmax(p_list))
        confidence = p_list[best_action_idx]

        # Anomaly score correlates with attack probability (BLOCK=1, TRIGGER_RAG=2)
        anomaly_score = p_list[int(FirewallAction.BLOCK)] + (0.5 * p_list[int(FirewallAction.TRIGGER_RAG)])

        # If model is uncertain, escalate to TRIGGER_RAG
        if confidence < self.confidence_threshold and best_action_idx != int(FirewallAction.BLOCK):
            action = FirewallAction.TRIGGER_RAG
        else:
            action = FirewallAction(best_action_idx)

        return RLInferenceResult(
            action=action,
            action_name=ACTION_NAMES[int(action)],
            confidence=round(confidence, 4),
            q_values=[round(q, 4) for q in q_list],
            probabilities=[round(p, 4) for p in p_list],
            anomaly_score=round(min(1.0, anomaly_score), 4),
            latency_ms=round(latency_ms, 4),
        )
