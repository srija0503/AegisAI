"""
federated_learning/client_selection.py
──────────────────────────────────────
Client selection algorithms for distributed edge training rounds.
Supports random sampling, round-robin, and tiered reputation/resource-aware selection.
"""

from __future__ import annotations

import logging
import random
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SelectionStrategy(str, Enum):
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    RESOURCE_AWARE = "resource_aware"
    ALL = "all"


class ClientSelector:
    """
    Selects edge nodes for participation in federated training rounds.
    """

    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.ALL, seed: int = 42) -> None:
        self.strategy = strategy
        self._rng = random.Random(seed)
        self._rr_index = 0

    def select_clients(
        self,
        available_clients: list[str],
        fraction: float = 1.0,
        min_clients: int = 1,
        client_metadata: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[str]:
        """
        Selects a subset of available edge nodes.
        """
        if not available_clients:
            return []

        target_count = max(min_clients, int(len(available_clients) * fraction))
        target_count = min(target_count, len(available_clients))

        if self.strategy == SelectionStrategy.ALL or target_count == len(available_clients):
            return list(available_clients)

        if self.strategy == SelectionStrategy.RANDOM:
            return self._rng.sample(available_clients, target_count)

        if self.strategy == SelectionStrategy.ROUND_ROBIN:
            selected = []
            n = len(available_clients)
            for _ in range(target_count):
                selected.append(available_clients[self._rr_index % n])
                self._rr_index += 1
            return selected

        if self.strategy == SelectionStrategy.RESOURCE_AWARE and client_metadata:
            # Score clients by low latency and high battery/stability
            def score(cid: str) -> float:
                meta = client_metadata.get(cid, {})
                samples = meta.get("samples_collected", 100)
                latency = meta.get("latency_ms", 50.0)
                return float(samples) / (latency + 1.0)

            sorted_clients = sorted(available_clients, key=score, reverse=True)
            return sorted_clients[:target_count]

        # Default fallback: random
        return self._rng.sample(available_clients, target_count)
