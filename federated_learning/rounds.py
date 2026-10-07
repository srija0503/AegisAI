"""
federated_learning/rounds.py
────────────────────────────
Round lifecycle and state tracking for Federated Learning iterations.
Maintains history of round metrics, convergence, and participating edge nodes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RoundStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FLRound:
    round_id: int
    status: RoundStatus = RoundStatus.PENDING
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    selected_clients: list[str] = field(default_factory=list)
    participating_clients: list[str] = field(default_factory=list)
    total_samples: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def complete(self, metrics: Optional[dict[str, Any]] = None) -> None:
        self.status = RoundStatus.COMPLETED
        self.end_time = time.time()
        if metrics:
            self.metrics.update(metrics)

    def fail(self, reason: str) -> None:
        self.status = RoundStatus.FAILED
        self.end_time = time.time()
        self.error = reason

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return round(self.end_time - self.start_time, 3)
        return round(time.time() - self.start_time, 3)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["duration"] = self.duration_seconds
        return d


class RoundManager:
    """
    Manages the sequential execution of Federated Learning rounds.
    """

    def __init__(self, max_rounds: int = 50) -> None:
        self.max_rounds = max_rounds
        self.current_round_num = 0
        self.history: list[FLRound] = []

    def start_new_round(self, selected_clients: list[str]) -> FLRound:
        self.current_round_num += 1
        new_round = FLRound(
            round_id=self.current_round_num,
            status=RoundStatus.RUNNING,
            selected_clients=selected_clients,
        )
        self.history.append(new_round)
        logger.info(f"Started FL Round {self.current_round_num} with {len(selected_clients)} clients.")
        return new_round

    def get_current_round(self) -> Optional[FLRound]:
        return self.history[-1] if self.history else None

    def get_round_history(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.history]

    def get_summary(self) -> dict[str, Any]:
        completed = [r for r in self.history if r.status == RoundStatus.COMPLETED]
        failed = [r for r in self.history if r.status == RoundStatus.FAILED]
        return {
            "total_rounds_run": len(self.history),
            "completed_rounds": len(completed),
            "failed_rounds": len(failed),
            "current_round": self.current_round_num,
        }
