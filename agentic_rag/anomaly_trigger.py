"""
agentic_rag/anomaly_trigger.py
──────────────────────────────
Evaluates whether traffic anomalies flagged by the RL Agent warrant
an Agentic RAG investigation loop. Provides anomaly thresholding,
flow deduplication, and rate-limiting to prevent engine saturation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TriggerStats:
    total_evaluations: int = 0
    total_triggers: int = 0
    suppressed_by_threshold: int = 0
    suppressed_by_cooldown: int = 0
    suppressed_by_rate_limit: int = 0


class AnomalyTrigger:
    """
    Decides when to escalate an anomalous network flow to the Agentic RAG loop.
    """

    def __init__(
        self,
        threshold: float = 0.65,
        cooldown_seconds: float = 30.0,
        max_triggers_per_minute: int = 30,
    ) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_triggers_per_minute = max_triggers_per_minute
        self.stats = TriggerStats()

        self._lock = threading.Lock()
        # Track last trigger per signature: (src_ip, dst_port, proto) -> timestamp
        self._last_trigger_time: dict[str, float] = {}
        # Sliding minute counter
        self._minute_triggers: list[float] = []

    def should_trigger(self, flow_features: dict[str, Any], anomaly_score: float) -> bool:
        """
        Determines whether to trigger the RAG loop for this flow.
        """
        with self._lock:
            self.stats.total_evaluations += 1
            now = time.time()

            # 1. Anomaly threshold check
            if anomaly_score < self.threshold:
                self.stats.suppressed_by_threshold += 1
                return False

            # 2. Global rate-limiting check
            self._minute_triggers = [t for t in self._minute_triggers if now - t < 60.0]
            if len(self._minute_triggers) >= self.max_triggers_per_minute:
                self.stats.suppressed_by_rate_limit += 1
                logger.warning("RAG trigger rate limit reached (60s window). Escaping.")
                return False

            # 3. Deduplication / Cooldown check per flow signature
            src_ip = flow_features.get("src_ip", "unknown")
            dst_port = flow_features.get("dst_port", 0)
            proto = flow_features.get("protocol", "IP")
            sig = f"{src_ip}:{dst_port}:{proto}"

            last_time = self._last_trigger_time.get(sig, 0.0)
            if now - last_time < self.cooldown_seconds:
                self.stats.suppressed_by_cooldown += 1
                return False

            # Approved for RAG execution
            self._last_trigger_time[sig] = now
            self._minute_triggers.append(now)
            self.stats.total_triggers += 1
            logger.info(
                f"Agentic RAG triggered for {sig} (anomaly_score={anomaly_score:.3f} >= {self.threshold})."
            )
            return True

    def reset(self) -> None:
        with self._lock:
            self._last_trigger_time.clear()
            self._minute_triggers.clear()
            self.stats = TriggerStats()
