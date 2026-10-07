"""
federated_learning/aggregation/weighted_fedavg.py
─────────────────────────────────────────────────
Weighted Federated Averaging with reputation weighting, quality metrics,
and server momentum / relaxation parameters.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


class WeightedFedAvg:
    """
    Weighted FedAvg incorporating edge client trust scores and server momentum.
    """

    def __init__(self, server_momentum: float = 0.9, server_lr: float = 1.0) -> None:
        self.server_momentum = server_momentum
        self.server_lr = server_lr
        self._velocity: dict[str, torch.Tensor] = {}

    def aggregate(
        self,
        client_updates: list[tuple[dict[str, torch.Tensor], float]],  # (weights, custom_weight)
        current_global_weights: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Aggregates client weights weighted by custom importance factors (e.g. sample count * reputation).
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation.")

        total_weight = sum(w for _, w in client_updates)
        if total_weight <= 0:
            normalized_alphas = [1.0 / len(client_updates)] * len(client_updates)
        else:
            normalized_alphas = [w / total_weight for _, w in client_updates]

        first_weights = client_updates[0][0]
        agg_pseudo_gradient: dict[str, torch.Tensor] = {}

        for key in first_weights.keys():
            accum = torch.zeros_like(first_weights[key], dtype=torch.float32)
            for (c_weights, _), alpha in zip(client_updates, normalized_alphas):
                accum += alpha * c_weights[key].to(torch.float32)

            # Apply momentum if global weights provided
            if current_global_weights is not None and key in current_global_weights:
                g_tensor = current_global_weights[key].to(torch.float32)
                pseudo_grad = accum - g_tensor

                if key not in self._velocity:
                    self._velocity[key] = torch.zeros_like(pseudo_grad)

                self._velocity[key] = (
                    self.server_momentum * self._velocity[key] + pseudo_grad
                )
                accum = g_tensor + (self.server_lr * self._velocity[key])

            agg_pseudo_gradient[key] = accum.to(first_weights[key].dtype)

        logger.info(f"WeightedFedAvg aggregated {len(client_updates)} client models.")
        return agg_pseudo_gradient

    def reset_momentum(self) -> None:
        self._velocity.clear()
