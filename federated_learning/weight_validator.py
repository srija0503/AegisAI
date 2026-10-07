"""
federated_learning/weight_validator.py
──────────────────────────────────────
Byzantine-resilient validator for client model weights in Federated Learning.
Detects NaN/Inf values, dimension mismatches, model poisoning attacks,
and anomalous weight divergence norms before server aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report detailing weight update validity."""
    is_valid: bool
    reason: str
    l2_norm_diff: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    layer_count: int = 0


class WeightValidator:
    """
    Validates client updates against the reference global model.
    """

    def __init__(
        self,
        max_l2_norm_diff: float = 50.0,
        max_absolute_weight: float = 100.0,
    ) -> None:
        self.max_l2_norm_diff = max_l2_norm_diff
        self.max_absolute_weight = max_absolute_weight

    def validate_weights(
        self,
        client_weights: dict[str, Union[torch.Tensor, np.ndarray]],
        global_weights: Optional[dict[str, Union[torch.Tensor, np.ndarray]]] = None,
    ) -> ValidationReport:
        """
        Validates client weights against sanity checks and optional global baseline.
        """
        if not client_weights:
            return ValidationReport(is_valid=False, reason="Empty weight dictionary")

        total_sq_diff = 0.0

        for layer_name, tensor in client_weights.items():
            # Convert to numpy for analysis
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                arr = tensor
            else:
                return ValidationReport(
                    is_valid=False,
                    reason=f"Invalid tensor type for layer '{layer_name}': {type(tensor)}",
                )

            # 1. NaN and Inf checks
            nans = int(np.isnan(arr).sum())
            infs = int(np.isinf(arr).sum())
            if nans > 0 or infs > 0:
                return ValidationReport(
                    is_valid=False,
                    reason=f"Layer '{layer_name}' contains {nans} NaNs and {infs} Infs",
                    nan_count=nans,
                    inf_count=infs,
                )

            # 2. Extreme weight magnitude check
            max_val = float(np.abs(arr).max()) if arr.size > 0 else 0.0
            if max_val > self.max_absolute_weight:
                return ValidationReport(
                    is_valid=False,
                    reason=f"Layer '{layer_name}' has extreme value {max_val:.2f} > {self.max_absolute_weight}",
                )

            # 3. Layer dimension & norm check against global weights if provided
            if global_weights is not None:
                if layer_name not in global_weights:
                    return ValidationReport(
                        is_valid=False,
                        reason=f"Unknown layer '{layer_name}' not present in global model",
                    )

                g_tensor = global_weights[layer_name]
                g_arr = g_tensor.detach().cpu().numpy() if isinstance(g_tensor, torch.Tensor) else g_tensor

                if arr.shape != g_arr.shape:
                    return ValidationReport(
                        is_valid=False,
                        reason=f"Shape mismatch in layer '{layer_name}': client {arr.shape} vs global {g_arr.shape}",
                    )

                diff = arr - g_arr
                total_sq_diff += float(np.sum(diff ** 2))

        l2_diff = float(np.sqrt(total_sq_diff))

        if global_weights is not None and l2_diff > self.max_l2_norm_diff:
            return ValidationReport(
                is_valid=False,
                reason=f"L2 update norm {l2_diff:.2f} exceeds threshold {self.max_l2_norm_diff:.2f} (suspected poisoning)",
                l2_norm_diff=l2_diff,
                layer_count=len(client_weights),
            )

        return ValidationReport(
            is_valid=True,
            reason="Weights passed all sanity and norm validation checks",
            l2_norm_diff=l2_diff,
            layer_count=len(client_weights),
        )

    def clip_weight_update(
        self,
        client_weights: dict[str, torch.Tensor],
        global_weights: dict[str, torch.Tensor],
        max_norm: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Clips the client update delta if it exceeds maximum norm (differential privacy / robustness).
        """
        threshold = max_norm or self.max_l2_norm_diff
        total_sq_diff = 0.0

        for k in client_weights:
            diff = client_weights[k] - global_weights[k]
            total_sq_diff += float(torch.sum(diff ** 2).item())

        l2_norm = np.sqrt(total_sq_diff)
        if l2_norm <= threshold or l2_norm < 1e-9:
            return client_weights

        scale = threshold / l2_norm
        clipped: dict[str, torch.Tensor] = {}
        for k in client_weights:
            diff = client_weights[k] - global_weights[k]
            clipped[k] = global_weights[k] + (diff * scale)

        logger.info(f"Clipped client weight update from norm {l2_norm:.2f} to {threshold:.2f}")
        return clipped
