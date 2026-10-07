"""
federated_learning/server.py
────────────────────────────
Central Federated Learning Aggregation Server.
Maintains global DQN model weights, coordinates training rounds, selects edge nodes,
validates client updates, and distributes improved defense policies.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from federated_learning.aggregator import ModelAggregator
from federated_learning.client_selection import ClientSelector, SelectionStrategy
from federated_learning.rounds import FLRound, RoundManager
from federated_learning.weight_validator import WeightValidator
from rl_agent.models.dqn import DuelingDQN

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Central server coordinating federated rounds across edge firewall nodes.
    """

    def __init__(
        self,
        global_model: Optional[nn.Module] = None,
        strategy: str = "fedavg",
        client_selector: Optional[ClientSelector] = None,
        validator: Optional[WeightValidator] = None,
    ) -> None:
        if global_model is not None:
            self.global_model: nn.Module = global_model
        else:
            self.global_model: nn.Module = DuelingDQN()

        self.global_weights: dict[str, torch.Tensor] = {
            k: v.detach().clone().cpu() for k, v in self.global_model.state_dict().items()
        }

        self.validator: WeightValidator = validator if validator is not None else WeightValidator()
        self.aggregator: ModelAggregator = ModelAggregator(strategy=strategy, validator=self.validator)
        self.client_selector: ClientSelector = (
            client_selector if client_selector is not None else ClientSelector(strategy=SelectionStrategy.ALL)
        )
        self.round_manager: RoundManager = RoundManager()

        self.registered_clients: set[str] = set()

    def register_client(self, client_id: str) -> None:
        self.registered_clients.add(client_id)
        logger.info(f"Registered edge node '{client_id}'. Total clients: {len(self.registered_clients)}")

    def get_global_weights(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.global_weights.items()}

    def process_round(
        self,
        client_updates: list[dict[str, Any]],
    ) -> FLRound:
        """
        Executes a single federated learning aggregation round:
        1. Initiates round tracking
        2. Aggregates and validates received client updates
        3. Updates global model weights
        4. Closes round with metrics
        """
        participating = [str(u.get("client_id", "unknown")) for u in client_updates]
        fl_round = self.round_manager.start_new_round(selected_clients=participating)

        try:
            new_weights, meta = self.aggregator.aggregate_client_updates(
                client_updates=client_updates,
                current_global_weights=self.global_weights,
            )

            # Update server's global model
            self.global_weights = new_weights
            self.global_model.load_state_dict(new_weights)

            fl_round.participating_clients = participating
            fl_round.total_samples = meta.get("total_samples", 0)
            fl_round.complete(metrics=meta)

            logger.info(
                f"Round {fl_round.round_id} successfully completed. "
                f"Accepted: {meta['accepted_clients']}, Rejected: {meta['rejected_clients']}."
            )
            return fl_round

        except Exception as e:
            logger.error(f"Round {fl_round.round_id} failed: {e}")
            fl_round.fail(str(e))
            raise
