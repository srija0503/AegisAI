"""
firewall/flow_manager.py
────────────────────────
Bidirectional network flow tracking with timeout eviction, capacity bounds,
and statistical aggregation for downstream feature extraction.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from firewall.packet_parser import ParsedPacket

# Standard port-to-service mapping
PORT_SERVICE_MAP: dict[int, str] = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain", 67: "bootps", 68: "bootpc", 69: "tftp", 80: "http",
    110: "pop_3", 119: "nntp", 123: "ntp", 135: "epmap", 137: "netbios_ns",
    138: "netbios_dgm", 139: "netbios_ssn", 143: "imap4", 161: "snmp",
    389: "ldap", 443: "https", 445: "microsoft_ds", 465: "smtps",
    993: "imaps", 995: "pop3s", 1433: "ms_sql_s", 1521: "oracle",
    3306: "mysql", 3389: "rdp", 5432: "postgresql", 6379: "redis",
    8080: "http_proxy", 8443: "https_alt",
}


@dataclass
class NetworkFlow:
    """
    Tracks state and aggregated metrics for a bidirectional flow.
    Canonical key ensures both forward (A->B) and reverse (B->A) packets update this flow.
    """

    key: tuple[str, str, int, int, str]
    init_src_ip: str
    init_dst_ip: str
    init_src_port: int
    init_dst_port: int
    protocol: str
    service: str = "other"

    start_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Directional counters
    forward_pkts: int = 0
    backward_pkts: int = 0
    forward_bytes: int = 0
    backward_bytes: int = 0

    # TCP flag counts
    syn_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    ack_count: int = 0
    urg_count: int = 0
    flag_str: str = "OTH"

    # Connection state
    state: str = "INIT"  # INIT, SYN_SENT, ESTABLISHED, FIN_WAIT, CLOSED, RESET

    # Inter-arrival timing
    last_forward_time: Optional[float] = None
    last_backward_time: Optional[float] = None
    fwd_iats: list[float] = field(default_factory=list)
    bwd_iats: list[float] = field(default_factory=list)

    @property
    def total_packets(self) -> int:
        return self.forward_pkts + self.backward_pkts

    @property
    def total_bytes(self) -> int:
        return self.forward_bytes + self.backward_bytes

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.start_time)

    def is_forward(self, pkt: ParsedPacket) -> bool:
        """True if packet direction matches the initial connection direction."""
        return (pkt.src_ip == self.init_src_ip and pkt.src_port == self.init_src_port)

    def update(self, pkt: ParsedPacket) -> None:
        """Update flow metrics with a new packet."""
        now = pkt.timestamp or time.time()
        self.last_seen = now
        self.flag_str = pkt.flag_str

        # TCP flags
        if pkt.tcp_flags.get("SYN"):
            self.syn_count += 1
        if pkt.tcp_flags.get("FIN"):
            self.fin_count += 1
        if pkt.tcp_flags.get("RST"):
            self.rst_count += 1
        if pkt.tcp_flags.get("ACK"):
            self.ack_count += 1
        if pkt.tcp_flags.get("URG"):
            self.urg_count += 1

        # State transition for TCP
        if self.protocol == "tcp":
            if pkt.is_rst():
                self.state = "RESET"
            elif pkt.is_fin():
                self.state = "FIN_WAIT" if self.fin_count == 1 else "CLOSED"
            elif pkt.is_syn_only() and self.state == "INIT":
                self.state = "SYN_SENT"
            elif pkt.is_syn_ack() and self.state == "SYN_SENT":
                self.state = "ESTABLISHED"
            elif self.ack_count > 0 and self.state in ("INIT", "SYN_SENT"):
                self.state = "ESTABLISHED"
        else:
            self.state = "ACTIVE"

        # Direction-based stats
        if self.is_forward(pkt):
            if self.last_forward_time is not None:
                iat = max(0.0, now - self.last_forward_time)
                if len(self.fwd_iats) < 50:
                    self.fwd_iats.append(iat)
            self.last_forward_time = now
            self.forward_pkts += 1
            self.forward_bytes += pkt.packet_len
        else:
            if self.last_backward_time is not None:
                iat = max(0.0, now - self.last_backward_time)
                if len(self.bwd_iats) < 50:
                    self.bwd_iats.append(iat)
            self.last_backward_time = now
            self.backward_pkts += 1
            self.backward_bytes += pkt.packet_len


class FlowManager:
    """
    Tracks and manages concurrent network flows.
    Thread-safe, bounded capacity, and automatic timeout cleanup.
    """

    def __init__(
        self,
        timeout_seconds: float = 120.0,
        max_flows: int = 100000,
        cleanup_interval_seconds: float = 30.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_flows = max_flows
        self.cleanup_interval_seconds = cleanup_interval_seconds

        self._flows: dict[tuple[str, str, int, int, str], NetworkFlow] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

        # Telemetry metrics
        self.total_flows_created: int = 0
        self.total_flows_expired: int = 0
        self.total_packets_processed: int = 0

    def update(self, pkt: ParsedPacket) -> NetworkFlow:
        """
        Record a packet into its corresponding flow. Creates a new flow if absent.
        Periodically triggers timeout sweep and enforces capacity limits.
        """
        key = pkt.canonical_flow_key

        with self._lock:
            self.total_packets_processed += 1
            flow = self._flows.get(key)

            if flow is None:
                # Capacity eviction if limit reached
                if len(self._flows) >= self.max_flows:
                    self._evict_oldest_flow_unlocked()

                service = (
                    PORT_SERVICE_MAP.get(pkt.dst_port)
                    or PORT_SERVICE_MAP.get(pkt.src_port)
                    or "other"
                )

                flow = NetworkFlow(
                    key=key,
                    init_src_ip=pkt.src_ip,
                    init_dst_ip=pkt.dst_ip,
                    init_src_port=pkt.src_port,
                    init_dst_port=pkt.dst_port,
                    protocol=pkt.protocol,
                    service=service,
                    start_time=pkt.timestamp,
                    last_seen=pkt.timestamp,
                )
                self._flows[key] = flow
                self.total_flows_created += 1

            flow.update(pkt)

            # Check if scheduled cleanup is due
            now = time.time()
            if now - self._last_cleanup >= self.cleanup_interval_seconds:
                self._cleanup_expired_unlocked(now)

            return flow

    def get_flow(self, key: tuple[str, str, int, int, str]) -> Optional[NetworkFlow]:
        """Lookup an active flow by its canonical key."""
        with self._lock:
            return self._flows.get(key)

    def cleanup_expired_flows(self, now: Optional[float] = None) -> int:
        """Explicitly purge flows that have been inactive longer than timeout_seconds."""
        current_time = now or time.time()
        with self._lock:
            return self._cleanup_expired_unlocked(current_time)

    def _cleanup_expired_unlocked(self, current_time: float) -> int:
        """Internal unlocked cleanup helper."""
        expired_keys = [
            k for k, flow in self._flows.items()
            if current_time - flow.last_seen > self.timeout_seconds
        ]
        for k in expired_keys:
            del self._flows[k]

        count = len(expired_keys)
        self.total_flows_expired += count
        self._last_cleanup = current_time
        return count

    def _evict_oldest_flow_unlocked(self) -> None:
        """Evict the single flow with the oldest last_seen timestamp."""
        if not self._flows:
            return
        oldest_key = min(self._flows.keys(), key=lambda k: self._flows[k].last_seen)
        del self._flows[oldest_key]
        self.total_flows_expired += 1

    def active_flow_count(self) -> int:
        """Number of active flows currently tracked."""
        with self._lock:
            return len(self._flows)

    def get_all_flows(self) -> list[NetworkFlow]:
        """Return a snapshot list of all tracked flows."""
        with self._lock:
            return list(self._flows.values())

    def clear(self) -> None:
        """Reset all tracked flows and counters."""
        with self._lock:
            self._flows.clear()
            self.total_flows_created = 0
            self.total_flows_expired = 0
            self.total_packets_processed = 0
