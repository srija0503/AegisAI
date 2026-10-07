"""
firewall/emergency_block.py
───────────────────────────
Emergency circuit breaker for automated DDoS mitigation and PPS spike containment.
Temporarily suspends or isolates high-rate traffic when packet rates breach safe thresholds.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class SpikeRecord:
    target: str
    triggered_at: float
    expires_at: float
    peak_pps: float
    reason: str


class EmergencyCircuitBreaker:
    """
    Monitors sliding-window packet arrival rates globally and per source IP.
    When packet-per-second (PPS) thresholds are breached, automatically engages
    an emergency block for a designated duration.
    """

    def __init__(
        self,
        enabled: bool = True,
        threshold_pps: float = 10000.0,
        per_ip_threshold_pps: Optional[float] = None,
        block_duration_seconds: float = 300.0,
        window_seconds: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self.threshold_pps = threshold_pps
        # Default per-IP threshold to 20% of global threshold or at least 200 PPS
        self.per_ip_threshold_pps = per_ip_threshold_pps or max(200.0, threshold_pps * 0.2)
        self.block_duration_seconds = block_duration_seconds
        self.window_seconds = window_seconds

        self._lock = threading.Lock()

        # Timestamps of recent packets for rate calculation
        self._global_packet_times: deque[float] = deque()
        self._ip_packet_times: dict[str, deque[float]] = defaultdict(deque)

        # Active emergency blocks
        self._active_ip_blocks: dict[str, SpikeRecord] = {}
        self._global_block: Optional[SpikeRecord] = None

        # Statistics
        self.total_emergency_triggers: int = 0
        self.ip_blocks_triggered: int = 0

    def record_packet(self, src_ip: str, timestamp: Optional[float] = None) -> bool:
        """
        Record the arrival of a packet from `src_ip`.
        Returns True if this packet caused an emergency block to trigger.
        """
        if not self.enabled:
            return False

        now = timestamp or time.time()
        cutoff = now - self.window_seconds
        triggered = False

        with self._lock:
            # 1. Update global sliding window
            self._global_packet_times.append(now)
            while self._global_packet_times and self._global_packet_times[0] < cutoff:
                self._global_packet_times.popleft()

            global_pps = len(self._global_packet_times) / self.window_seconds
            if global_pps >= self.threshold_pps:
                if not self._is_global_active_unlocked(now):
                    self._global_block = SpikeRecord(
                        target="GLOBAL",
                        triggered_at=now,
                        expires_at=now + self.block_duration_seconds,
                        peak_pps=global_pps,
                        reason=f"Global PPS spike ({global_pps:.0f} >= {self.threshold_pps:.0f})",
                    )
                    self.total_emergency_triggers += 1
                    triggered = True

            # 2. Update per-IP sliding window
            q = self._ip_packet_times[src_ip]
            q.append(now)
            while q and q[0] < cutoff:
                q.popleft()

            ip_pps = len(q) / self.window_seconds
            if ip_pps >= self.per_ip_threshold_pps:
                existing = self._active_ip_blocks.get(src_ip)
                if not existing or now > existing.expires_at:
                    self._active_ip_blocks[src_ip] = SpikeRecord(
                        target=src_ip,
                        triggered_at=now,
                        expires_at=now + self.block_duration_seconds,
                        peak_pps=ip_pps,
                        reason=f"Per-IP PPS spike ({ip_pps:.0f} >= {self.per_ip_threshold_pps:.0f})",
                    )
                    self.ip_blocks_triggered += 1
                    triggered = True

            # Periodic cleanup of idle IP tracking deques
            if len(self._ip_packet_times) > 5000:
                self._cleanup_idle_ips_unlocked(cutoff)

            return triggered

    def is_blocked(self, src_ip: str, now: Optional[float] = None) -> tuple[bool, Optional[str]]:
        """
        Check if src_ip is currently blocked under emergency containment.
        Returns: (True, reason) or (False, None).
        """
        if not self.enabled:
            return False, None

        current = now or time.time()
        with self._lock:
            # Check global emergency
            if self._is_global_active_unlocked(current):
                return True, f"Global Emergency Lockdown: {self._global_block.reason}"  # type: ignore

            # Check per-IP block
            record = self._active_ip_blocks.get(src_ip)
            if record:
                if current <= record.expires_at:
                    return True, f"Emergency IP Quarantine: {record.reason}"
                else:
                    del self._active_ip_blocks[src_ip]

            return False, None

    def _is_global_active_unlocked(self, now: float) -> bool:
        if self._global_block is None:
            return False
        if now <= self._global_block.expires_at:
            return True
        self._global_block = None
        return False

    def is_global_emergency_active(self) -> bool:
        """True if global firewall emergency quarantine is currently engaged."""
        now = time.time()
        with self._lock:
            return self._is_global_active_unlocked(now)

    def active_quarantine_count(self) -> int:
        """Number of source IPs currently in emergency quarantine."""
        now = time.time()
        with self._lock:
            active = [ip for ip, rec in self._active_ip_blocks.items() if now <= rec.expires_at]
            return len(active)

    def reset(self) -> None:
        """Reset all circuit breaker states."""
        with self._lock:
            self._global_packet_times.clear()
            self._ip_packet_times.clear()
            self._active_ip_blocks.clear()
            self._global_block = None

    def _cleanup_idle_ips_unlocked(self, cutoff: float) -> None:
        """Purge IP deques with no packets in current window."""
        to_del = [ip for ip, q in self._ip_packet_times.items() if not q or q[-1] < cutoff]
        for ip in to_del:
            del self._ip_packet_times[ip]
