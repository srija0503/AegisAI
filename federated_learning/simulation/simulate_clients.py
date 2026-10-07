"""
federated_learning/simulation/simulate_clients.py
─────────────────────────────────────────────────
Generates simulated edge firewall nodes with heterogeneous (non-IID)
traffic distributions to model real-world distributed network environments.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from federated_learning.client import FederatedClient
from rl_agent.action import NUM_ACTIONS
from rl_agent.models.dqn import DuelingDQN
from rl_agent.state import STATE_DIM

logger = logging.getLogger(__name__)


def generate_synthetic_flow_data(
    num_samples: int = 200,
    state_dim: int = STATE_DIM,
    attack_bias: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic (states, actions, Q-targets) for local client training.
    """
    rng = np.random.RandomState(seed)
    states = rng.randn(num_samples, state_dim).astype(np.float32)

    # Actions: 0=ALLOW, 1=BLOCK, 2=DROP, 3=TRIGGER_RAG
    actions = rng.choice(NUM_ACTIONS, size=num_samples).astype(np.int64)

    # Simulated Q-targets
    targets = rng.uniform(0.1, 1.0, size=num_samples).astype(np.float32)
    # Higher targets for blocking attack traffic
    targets[actions == 1] += attack_bias

    return states, actions, targets


def create_simulated_clients(
    node_ids: tuple[str, ...] = ("node_01", "node_02", "node_03"),
    state_dim: int = STATE_DIM,
    action_dim: int = NUM_ACTIONS,
) -> dict[str, FederatedClient]:
    """
    Instantiates simulated edge firewall nodes with independent DQNs.
    """
    clients: dict[str, FederatedClient] = {}

    for i, nid in enumerate(node_ids):
        model = DuelingDQN(state_dim=state_dim, action_dim=action_dim)
        client = FederatedClient(client_id=nid, model=model)

        # Generate unique local traffic patterns for each node
        states, actions, targets = generate_synthetic_flow_data(
            num_samples=150 + (i * 50),
            state_dim=state_dim,
            attack_bias=0.2 * (i + 1),
            seed=42 + i,
        )

        # Train local step
        loss = client.train_on_batch(states, actions, targets)
        logger.info(f"Initialized client '{nid}' with {len(states)} samples (loss: {loss:.4f}).")

        clients[nid] = client

    return clients
