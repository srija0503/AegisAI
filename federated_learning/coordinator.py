"""
federated_learning/coordinator.py
─────────────────────────────────
Federated Learning Coordinator.
Orchestrates communication, synchronization, and round scheduling between
distributed edge firewall nodes and the central aggregation server.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from federated_learning.client import FederatedClient
from federated_learning.client_selection import ClientSelector
from federated_learning.rounds import FLRound
from federated_learning.server import FederatedServer

logger = logging.getLogger(__name__)


class FederatedCoordinator:
    """
    Coordinates distributed FL training campaigns across edge nodes.
    """

    def __init__(
        self,
        server: Optional[FederatedServer] = None,
        clients: Optional[dict[str, FederatedClient]] = None,
    ) -> None:
        self.server = server or FederatedServer()
        self.clients: dict[str, FederatedClient] = clients or {}

        # Register existing clients with server
        for cid in self.clients.keys():
            self.server.register_client(cid)

    def add_client(self, client: FederatedClient) -> None:
        self.clients[client.client_id] = client
        self.server.register_client(client.client_id)

    def run_round(self, client_fraction: float = 1.0) -> FLRound:
        """
        Executes a single end-to-end synchronized FL round:
        1. Selects participating clients
        2. Distributes global weights to selected clients
        3. Collects client updates
        4. Aggregates updates on server
        5. Re-distributes improved global weights
        """
        all_client_ids = list(self.clients.keys())
        if not all_client_ids:
            raise RuntimeError("Cannot run FL round: No registered clients.")

        selected_ids = self.server.client_selector.select_clients(
            available_clients=all_client_ids,
            fraction=client_fraction,
        )

        global_weights = self.server.get_global_weights()

        # Distribute global model & collect updates
        updates: list[dict[str, Any]] = []
        for cid in selected_ids:
            client = self.clients[cid]
            # Sync to current global weights before round
            client.set_weights(global_weights)
            # Export local update
            updates.append(client.export_update())

        # Process round on server
        fl_round = self.server.process_round(updates)

        # Distribute new global model back to participating clients
        new_global_weights = self.server.get_global_weights()
        for cid in selected_ids:
            self.clients[cid].set_weights(new_global_weights)

        logger.info(f"Synchronized global model to {len(selected_ids)} clients after Round {fl_round.round_id}.")
        return fl_round

    def run_campaign(self, num_rounds: int = 3) -> list[FLRound]:
        """Runs multiple consecutive training rounds."""
        results: list[FLRound] = []
        for r in range(num_rounds):
            fl_round = self.run_round()
            results.append(fl_round)
        return results

    def get_status(self) -> dict[str, Any]:
        return {
            "total_clients": len(self.clients),
            "registered_nodes": list(self.clients.keys()),
            "rounds_summary": self.server.round_manager.get_summary(),
            "strategy": self.server.aggregator.strategy,
        }
