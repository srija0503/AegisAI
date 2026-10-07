"""
simulation/benign_traffic.py
────────────────────────────
Simulates realistic, legitimate enterprise network traffic:
- HTTPS (TCP 443) and HTTP (TCP 80) web traffic
- DNS (UDP 53) hostname resolution queries
- SSH (TCP 22) internal administrative sessions
- ICMP Echo (ping) health monitoring
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from simulation.traffic_generator import BaseTrafficGenerator

try:
    from scapy.layers.inet import ICMP, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class BenignTrafficGenerator(BaseTrafficGenerator):
    """
    Generates varied benign network traffic reflecting legitimate user activity.
    """

    SERVICES = [
        {"proto": "tcp", "dport": 443, "service": "https", "weight": 55},
        {"proto": "tcp", "dport": 80, "service": "http", "weight": 25},
        {"proto": "udp", "dport": 53, "service": "dns", "weight": 15},
        {"proto": "tcp", "dport": 22, "service": "ssh", "weight": 5},
    ]

    BENIGN_SUBNETS = [
        "10.0.1.",
        "10.0.2.",
        "192.168.1.",
        "172.16.5.",
    ]

    def __init__(
        self,
        target_ip: str = "192.168.1.100",
        rate_pps: Optional[float] = None,
        seed: Optional[int] = None,
        use_scapy: bool = True,
    ) -> None:
        super().__init__(target_ip=target_ip, rate_pps=rate_pps, seed=seed)
        self.use_scapy = use_scapy and SCAPY_AVAILABLE

    def _sample_service(self) -> dict[str, Any]:
        weights = [s["weight"] for s in self.SERVICES]
        return self.random.choices(self.SERVICES, weights=weights, k=1)[0]

    def _random_benign_ip(self) -> str:
        subnet = self.random.choice(self.BENIGN_SUBNETS)
        host = self.random.randint(10, 250)
        return f"{subnet}{host}"

    def generate_packet(self) -> Any:
        srv = self._sample_service()
        src_ip = self._random_benign_ip()
        src_port = self.random.randint(32768, 61000)
        dst_port = srv["dport"]
        proto = srv["proto"]
        now = time.time()

        if self.use_scapy:
            if proto == "tcp":
                # Simulated normal TCP handshake/data packet (ACK/PSH)
                flags = self.random.choice(["SA", "PA", "A", "FA"])
                payload = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n" if dst_port in (80, 443) else b"\x00" * 32
                pkt = (
                    IP(src=src_ip, dst=self.target_ip, ttl=64)
                    / TCP(sport=src_port, dport=dst_port, flags=flags)
                    / payload
                )
            elif proto == "udp":
                payload = b"\x12\x34\x01\x00\x00\x01" + b"\x00" * 20
                pkt = IP(src=src_ip, dst=self.target_ip, ttl=64) / UDP(sport=src_port, dport=dst_port) / payload
            else:
                pkt = IP(src=src_ip, dst=self.target_ip, ttl=64) / ICMP()
            pkt.time = now
            return pkt

        # Fast dictionary fallback
        return {
            "src_ip": src_ip,
            "dst_ip": self.target_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": proto,
            "packet_len": self.random.randint(64, 1460),
            "payload_len": self.random.randint(0, 1024),
            "tcp_flags": {"SYN": False, "ACK": True, "FIN": False, "RST": False},
            "flag_str": "SF",
            "timestamp": now,
            "is_attack": False,
        }
