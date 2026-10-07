"""
simulation/anomaly_generator.py
───────────────────────────────
Simulates zero-day anomalies and statistical outliers:
- Xmas scans (SYN + FIN + URG + PSH illegal flag combinations)
- Covert channel tunneling (DNS tunneling payloads)
- Feature mutations designed to trigger low-confidence RL uncertainty
  and activate Member 2's Agentic RAG workflow.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional

from simulation.traffic_generator import BaseTrafficGenerator

try:
    from scapy.layers.inet import IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class AnomalyTrafficGenerator(BaseTrafficGenerator):
    """
    Produces zero-day and out-of-distribution traffic anomalies.
    """

    def __init__(
        self,
        target_ip: str = "192.168.1.100",
        rate_pps: Optional[float] = None,
        seed: Optional[int] = None,
        use_scapy: bool = True,
    ) -> None:
        super().__init__(target_ip=target_ip, rate_pps=rate_pps, seed=seed)
        self.use_scapy = use_scapy and SCAPY_AVAILABLE

    def generate_packet(self) -> Any:
        now = time.time()
        anomaly_kind = self.random.choice(["xmas_scan", "dns_tunnel", "protocol_evasion"])

        if anomaly_kind == "xmas_scan":
            return self._generate_xmas(now)
        elif anomaly_kind == "dns_tunnel":
            return self._generate_dns_tunnel(now)
        else:
            return self._generate_protocol_evasion(now)

    def _generate_xmas(self, timestamp: float) -> Any:
        """Illegal TCP flags: FIN + URG + PSH (RFC 793 violation used by scanners)."""
        src_ip = f"198.51.100.{self.random.randint(2, 254)}"
        src_port = self.random.randint(1024, 65535)
        dst_port = self.random.choice([80, 443, 8080])

        if self.use_scapy:
            pkt = IP(src=src_ip, dst=self.target_ip, ttl=40) / TCP(sport=src_port, dport=dst_port, flags="FPU")
            pkt.time = timestamp
            return pkt

        return {
            "src_ip": src_ip,
            "dst_ip": self.target_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": "tcp",
            "packet_len": 54,
            "payload_len": 0,
            "tcp_flags": {"SYN": False, "FIN": True, "PSH": True, "URG": True, "ACK": False, "RST": False},
            "flag_str": "OTH",
            "timestamp": timestamp,
            "is_attack": True,
            "attack_type": "zero_day_xmas",
        }

    def _generate_dns_tunnel(self, timestamp: float) -> Any:
        """High-entropy large DNS query simulating data exfiltration."""
        src_ip = f"10.0.1.{self.random.randint(20, 200)}"
        src_port = self.random.randint(30000, 60000)

        # 60-char encoded payload domain
        random_subdomain = "".join(self.random.choices("abcdef0123456789", k=32))
        payload = f"{random_subdomain}.exfil.attacker.com".encode()

        if self.use_scapy:
            pkt = IP(src=src_ip, dst=self.target_ip, ttl=64) / UDP(sport=src_port, dport=53) / payload
            pkt.time = timestamp
            return pkt

        return {
            "src_ip": src_ip,
            "dst_ip": self.target_ip,
            "src_port": src_port,
            "dst_port": 53,
            "protocol": "udp",
            "packet_len": len(payload) + 28,
            "payload_len": len(payload),
            "tcp_flags": {},
            "flag_str": "SF",
            "timestamp": timestamp,
            "is_attack": True,
            "attack_type": "zero_day_dns_tunnel",
        }

    def _generate_protocol_evasion(self, timestamp: float) -> Any:
        """Unusual port/protocol combination with abnormal fragment characteristics."""
        src_ip = f"203.0.113.{self.random.randint(50, 100)}"
        src_port = 80  # Web port used as source
        dst_port = self.random.randint(40000, 50000)

        if self.use_scapy:
            pkt = IP(src=src_ip, dst=self.target_ip, ttl=15) / TCP(sport=src_port, dport=dst_port, flags="S")
            pkt.time = timestamp
            return pkt

        return {
            "src_ip": src_ip,
            "dst_ip": self.target_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": "tcp",
            "packet_len": 256,
            "payload_len": 200,
            "tcp_flags": {"SYN": True, "ACK": False, "FIN": False, "RST": False},
            "flag_str": "S0",
            "timestamp": timestamp,
            "is_attack": True,
            "attack_type": "zero_day_evasion",
        }
