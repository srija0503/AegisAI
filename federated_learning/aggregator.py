"""
federated_learning/aggregator.py
────────────────────────────────
Top-level Model Aggregator coordinating validation, outlier rejection,
and multi-strategy weight aggregation across distributed edge firewall nodes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from federated_learning.aggregation.fedavg import FedAvg
from federated_learning.aggregation.weighted_fedavg import WeightedFedAvg
from federated_learning.weight_validator import ValidationReport, WeightValidator

logger = logging.getLogger(__name__)


class ModelAggregator:
    """
    Central server aggregation engine.
    """

    def __init__(
        self,
        strategy: str = "fedavg",
        validator: Optional[WeightValidator] = None,
        server_momentum: float = 0.9,
    ) -> None:
        self.strategy = strategy.lower()
        self.validator = validator or WeightValidator()
        self.fedavg = FedAvg()
        self.weighted_fedavg = WeightedFedAvg(server_momentum=server_momentum)
        self.rejected_clients: list[tuple[str, str]] = []

    def aggregate_client_updates(
        self,
        client_updates: list[dict[str, Any]],  # each: {"client_id": str, "weights": state_dict, "num_samples": int, ...}
        current_global_weights: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Validates all incoming client updates, rejects poisoned or malformed updates,
        and aggregates the remainder.
        Returns: (new_global_weights, aggregation_metadata)
        """
        if not client_updates:
            raise ValueError("No client updates provided to aggregator.")

        self.rejected_clients.clear()
        valid_updates: list[tuple[dict[str, torch.Tensor], int]] = []
        weighted_updates: list[tuple[dict[str, torch.Tensor], float]] = []

        for update in client_updates:
            client_id = update.get("client_id", "unknown_node")
            weights = update.get("weights", {})
            num_samples = int(update.get("num_samples", 1))
            custom_weight = float(update.get("weight", num_samples))

            # Validate weights against global model
            report = self.validator.validate_weights(weights, current_global_weights)
            if not report.is_valid:
                logger.warning(f"Rejected update from {client_id}: {report.reason}")
                self.rejected_clients.append((client_id, report.reason))
                continue

            valid_updates.append((weights, num_samples))
            weighted_updates.append((weights, custom_weight))

        if not valid_updates:
            raise RuntimeError(
                f"All {len(client_updates)} client updates were rejected by WeightValidator."
            )

        # Aggregate based on selected strategy
        if self.strategy == "weighted_fedavg":
            new_global_weights = self.weighted_fedavg.aggregate(
                weighted_updates,
                current_global_weights=current_global_weights,
            )
        else:
            new_global_weights = self.fedavg.aggregate(valid_updates)

        meta = {
            "total_clients": len(client_updates),
            "accepted_clients": len(valid_updates),
            "rejected_clients": len(self.rejected_clients),
            "rejections": list(self.rejected_clients),
            "strategy": self.strategy,
            "total_samples": sum(s for _, s in valid_updates),
        }

        return new_global_weights, meta
