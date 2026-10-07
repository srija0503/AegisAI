"""
agentic_rag/rag_memory.py
─────────────────────────
In-memory incident ledger and telemetry buffer for Agentic RAG.
Maintains recent threat history, past generated rules, and IP reputations
to provide longitudinal context for repeated or multi-stage intrusions.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agentic_rag.threat_reasoner import ReasoningDecision


@dataclass
class IncidentRecord:
    incident_id: str
    timestamp: str
    src_ip: str
    dst_ip: str
    dst_port: Optional[int]
    protocol: str
    anomaly_score: float
    threat_name: str
    category: str
    action: str
    applied_rules: list[str]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RAGMemory:
    """
    Episodic memory tracking recent intrusions and decisions.
    """

    def __init__(self, max_history: int = 500) -> None:
        self.max_history = max_history
        self._history: deque[IncidentRecord] = deque(maxlen=max_history)
        self._ip_index: dict[str, list[IncidentRecord]] = {}
        self._lock = threading.Lock()

    def record_incident(
        self,
        flow: dict[str, Any],
        anomaly_score: float,
        decision: ReasoningDecision,
        rules: list[dict[str, Any]],
    ) -> IncidentRecord:
        with self._lock:
            src_ip = str(flow.get("src_ip", "0.0.0.0"))
            inc_id = f"inc_{int(time.time() * 1000)}_{src_ip}"
            record = IncidentRecord(
                incident_id=inc_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                src_ip=src_ip,
                dst_ip=str(flow.get("dst_ip", "0.0.0.0")),
                dst_port=flow.get("dst_port"),
                protocol=str(flow.get("protocol", "TCP")),
                anomaly_score=anomaly_score,
                threat_name=decision.threat_name,
                category=decision.threat_category,
                action=decision.recommended_action,
                applied_rules=[r.get("rule_id", "") for r in rules],
            )
            self._history.append(record)

            if src_ip not in self._ip_index:
                self._ip_index[src_ip] = []
            self._ip_index[src_ip].append(record)
            # Keep IP index trimmed
            if len(self._ip_index[src_ip]) > 20:
                self._ip_index[src_ip].pop(0)

            return record

    def get_recent_incidents(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._history)[-limit:]
            return [it.to_dict() for it in items]

    def get_history_for_ip(self, src_ip: str) -> list[dict[str, Any]]:
        with self._lock:
            items = self._ip_index.get(src_ip, [])
            return [it.to_dict() for it in items]

    def count(self) -> int:
        with self._lock:
            return len(self._history)

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._ip_index.clear()
