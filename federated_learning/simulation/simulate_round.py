"""
federated_learning/simulation/simulate_round.py
───────────────────────────────────────────────
Simulates an end-to-end Federated Learning round across simulated edge nodes.
Validates weight transmission, aggregation, and client model synchronization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from federated_learning.coordinator import FederatedCoordinator
from federated_learning.server import FederatedServer
from federated_learning.simulation.simulate_clients import create_simulated_clients
from rl_agent.models.dqn import DuelingDQN

logger = logging.getLogger(__name__)


def run_simulated_fl_round(
    strategy: str = "fedavg",
    num_nodes: int = 3,
) -> dict[str, Any]:
    """
    Executes a complete standalone federated round simulation.
    Returns summary metrics.
    """
    global_model = DuelingDQN()
    server = FederatedServer(global_model=global_model, strategy=strategy)

    node_ids = tuple(f"node_{i+1:02d}" for i in range(num_nodes))
    clients = create_simulated_clients(node_ids=node_ids)

    coordinator = FederatedCoordinator(server=server, clients=clients)

    fl_round = coordinator.run_round()

    return {
        "round_id": fl_round.round_id,
        "status": fl_round.status.value,
        "duration": fl_round.duration_seconds,
        "participating_clients": fl_round.participating_clients,
        "metrics": fl_round.metrics,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_simulated_fl_round()
    print("Simulation Result:", res)
