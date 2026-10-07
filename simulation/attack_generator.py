"""
simulation/attack_generator.py
──────────────────────────────
Simulates active, malicious cyber attack vectors:
1. SYN Flood (DoS / DDoS)
2. TCP Port Sweep (Reconnaissance)
3. Brute Force SSH/FTP
4. Land Attack (Spoofed loopback packet: src_ip == dst_ip)
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from simulation.traffic_generator import BaseTrafficGenerator

try:
    from scapy.layers.inet import IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class AttackTrafficGenerator(BaseTrafficGenerator):
    """
    Generates synthetic attack traffic streams according to specified attack type.
    """

    ATTACK_TYPES = ["syn_flood", "port_scan", "brute_force", "land_attack"]

    SCAN_PORTS = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 1433, 3306, 3389, 8080]

    def __init__(
        self,
        attack_type: str = "syn_flood",
        attacker_ip: Optional[str] = "203.0.113.15",
        target_ip: str = "192.168.1.100",
        rate_pps: Optional[float] = None,
        seed: Optional[int] = None,
        use_scapy: bool = True,
    ) -> None:
        super().__init__(target_ip=target_ip, rate_pps=rate_pps, seed=seed)
        self.attack_type = attack_type.lower()
        self.attacker_ip = attacker_ip or "203.0.113.15"
        self.use_scapy = use_scapy and SCAPY_AVAILABLE
        self._scan_ptr = 0

    def generate_packet(self) -> Any:
        now = time.time()

        if self.attack_type == "syn_flood":
            return self._generate_syn_flood(now)
        elif self.attack_type == "port_scan":
            return self._generate_port_scan(now)
        elif self.attack_type == "brute_force":
            return self._generate_brute_force(now)
        elif self.attack_type == "land_attack":
            return self._generate_land_attack(now)
        else:
            return self._generate_syn_flood(now)

    def _generate_syn_flood(self, timestamp: float) -> Any:
        """High frequency TCP SYN packets with randomized spoofed IPs."""
        src_ip = f"198.51.100.{self.random.randint(2, 254)}"
        src_port = self.random.randint(1024, 65535)
        dst_port = 80  # Target web server

        if self.use_scapy:
            pkt = IP(src=src_ip, dst=self.target_ip, ttl=64) / TCP(sport=src_port, dport=dst_port, flags="S")
            pkt.time = timestamp
            return pkt

        return {
            "src_ip": src_ip,
            "dst_ip": self.target_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": "tcp",
            "packet_len": 64,
            "payload_len": 0,
            "tcp_flags": {"SYN": True, "ACK": False, "FIN": False, "RST": False},
            "flag_str": "S0",
            "timestamp": timestamp,
            "is_attack": True,
            "attack_type": "syn_flood",
        }

    def _generate_port_scan(self, timestamp: float) -> Any:
        """Sequential port probe from single attacker."""
        dst_port = self.SCAN_PORTS[self._scan_ptr % len(self.SCAN_PORTS)]
        self._scan_ptr += 1
        src_port = self.random.randint(30000, 60000)

        if self.use_scapy:
            # Half-open SYN scan
            pkt = IP(src=self.attacker_ip, dst=self.target_ip, ttl=50) / TCP(sport=src_port, dport=dst_port, flags="S")
            pkt.time = timestamp
            return pkt

        return {
            "src_ip": self.attacker_ip,
            "dst_ip": self.target_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": "tcp",
            "packet_len": 60,
            "payload_len": 0,
            "tcp_flags": {"SYN": True, "ACK": False, "FIN": False, "RST": False},
            "flag_str": "S0",
            "timestamp": timestamp,
            "is_attack": True,
            "attack_type": "port_scan",
        }

    def _generate_brute_force(self, timestamp: float) -> Any:
        """High volume connection attempts to SSH port 22."""
        src_port = self.random.randint(1024, 65535)

        if self.use_scapy:
            pkt = (
                IP(src=self.attacker_ip, dst=self.target_ip, ttl=64)
                / TCP(sport=src_port, dport=22, flags="PA")
                / b"SSH-2.0-OpenSSH_8.2p1\r\n"
            )
            pkt.time = timestamp
            return pkt

        return {
            "src_ip": self.attacker_ip,
            "dst_ip": self.target_ip,
            "src_port": src_port,
            "dst_port": 22,
            "protocol": "tcp",
            "packet_len": 128,
            "payload_len": 64,
            "tcp_flags": {"SYN": False, "ACK": True, "PSH": True, "FIN": False, "RST": False},
            "flag_str": "SF",
            "timestamp": timestamp,
            "is_attack": True,
            "attack_type": "brute_force",
        }

    def _generate_land_attack(self, timestamp: float) -> Any:
        """Classic Land attack: source IP equals destination target IP."""
        port = 80
        if self.use_scapy:
            pkt = IP(src=self.target_ip, dst=self.target_ip, ttl=64) / TCP(sport=port, dport=port, flags="S")
            pkt.time = timestamp
            return pkt

        return {
            "src_ip": self.target_ip,
            "dst_ip": self.target_ip,
            "src_port": port,
            "dst_port": port,
            "protocol": "tcp",
            "packet_len": 64,
            "payload_len": 0,
            "tcp_flags": {"SYN": True, "ACK": False, "FIN": False, "RST": False},
            "flag_str": "S0",
            "timestamp": timestamp,
            "is_attack": True,
            "attack_type": "land_attack",
        }
