"""
federated_learning/client.py
────────────────────────────
Edge Node Federated Learning Client.
Maintains local DQNAgent, trains on local network traffic without revealing
raw packet payloads, and exports serialized weights for PQC transmission.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from federated_learning.model_serializer import ModelSerializer
from rl_agent.models.dqn import DuelingDQN

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Client agent running on edge firewall node (node_01, node_02, node_03).
    """

    def __init__(
        self,
        client_id: str,
        agent: Optional[Any] = None,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.device = torch.device(device)
        self.agent = agent

        if model is not None:
            self.model: nn.Module = model
        elif agent is not None and hasattr(agent, "q_net"):
            self.model: nn.Module = agent.q_net
        else:
            self.model: nn.Module = DuelingDQN().to(self.device)

        self.num_local_samples = 0
        self.local_epochs_completed = 0

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Extracts model state_dict."""
        return {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, new_weights: dict[str, torch.Tensor]) -> None:
        """Applies updated global weights to local model and target networks."""
        cpu_weights = {k: v.to(self.device) for k, v in new_weights.items()}
        self.model.load_state_dict(cpu_weights)

        # Update DQNAgent target net as well if agent attached
        if self.agent and hasattr(self.agent, "target_net"):
            self.agent.target_net.load_state_dict(cpu_weights)

        logger.info(f"Client '{self.client_id}' updated local model with new global weights.")

    def train_on_batch(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        """
        Executes local supervised or Q-learning gradient step on local batches.
        """
        self.model.train()
        optimizer = getattr(self.agent, "optimizer", None)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        s_t = torch.from_numpy(states).float().to(self.device)
        a_t = torch.from_numpy(actions).long().to(self.device)
        target_t = torch.from_numpy(targets).float().to(self.device)

        optimizer.zero_grad()
        q_vals = self.model(s_t)
        chosen_q = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)

        loss = criterion(chosen_q, target_t)
        loss.backward()
        optimizer.step()

        self.num_local_samples += len(states)
        self.local_epochs_completed += 1
        return float(loss.item())

    def export_update(self) -> dict[str, Any]:
        """
        Packages model weights and metadata for transmission to central aggregator.
        """
        weights = self.get_weights()
        serialized = ModelSerializer.serialize_weights(weights)
        checksum = ModelSerializer.compute_checksum(serialized)

        return {
            "client_id": self.client_id,
            "weights": weights,
            "serialized_bytes": serialized,
            "checksum": checksum,
            "num_samples": max(1, self.num_local_samples),
            "epochs": self.local_epochs_completed,
        }
