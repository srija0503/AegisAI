"""
federated_learning/aggregation/fedavg.py
────────────────────────────────────────
Federated Averaging (FedAvg) aggregation algorithm.
Computes sample-weighted average across participating edge client weights.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FedAvg:
    """
    Standard Federated Averaging implementation.
    """

    @staticmethod
    def aggregate(
        client_updates: list[tuple[dict[str, torch.Tensor], int]],
    ) -> dict[str, torch.Tensor]:
        """
        Aggregates a list of (client_weights, num_samples) tuples.
        Returns the averaged state_dict.
        """
        if not client_updates:
            raise ValueError("Cannot aggregate empty client updates list.")

        # If only one client, return its weights
        if len(client_updates) == 1:
            return {k: v.clone() for k, v in client_updates[0][0].items()}

        total_samples = sum(num_samples for _, num_samples in client_updates)
        if total_samples <= 0:
            # Fallback to uniform weighting
            weights_per_client = [1.0 / len(client_updates)] * len(client_updates)
        else:
            weights_per_client = [num_samples / total_samples for _, num_samples in client_updates]

        first_weights = client_updates[0][0]
        aggregated: dict[str, torch.Tensor] = {}

        for key in first_weights.keys():
            # Accumulate on CPU float for precision
            stacked = torch.zeros_like(first_weights[key], dtype=torch.float32)
            for (c_weights, _), alpha in zip(client_updates, weights_per_client):
                c_tensor = c_weights[key].to(torch.float32)
                stacked += alpha * c_tensor

            # Restore original dtype if needed
            aggregated[key] = stacked.to(first_weights[key].dtype)

        logger.info(
            f"FedAvg aggregated {len(client_updates)} client models across {total_samples} samples."
        )
        return aggregated
