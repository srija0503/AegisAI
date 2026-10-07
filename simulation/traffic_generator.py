"""
simulation/traffic_generator.py
───────────────────────────────
Base classes and interfaces for generating network traffic streams.
Supports streaming Scapy packets or ParsedPacket dictionaries with
configurable arrival rates, burst intervals, and source topologies.
"""

from __future__ import annotations

import abc
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional


@dataclass
class FlowSpec:
    """Specification describing a simulated network connection flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str       # "tcp", "udp", "icmp"
    service: str        # "http", "dns", "ssh", etc.
    packet_count: int = 10
    total_bytes: int = 5000
    is_attack: bool = False
    attack_type: Optional[str] = None
    flags: list[str] = field(default_factory=lambda: ["SYN", "ACK"])


class BaseTrafficGenerator(abc.ABC):
    """
    Abstract base class for all traffic pattern generators.
    """

    def __init__(
        self,
        target_ip: str = "192.168.1.100",
        rate_pps: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.target_ip = target_ip
        self.rate_pps = rate_pps
        self.random = random.Random(seed)
        self.generated_count: int = 0

    @abc.abstractmethod
    def generate_packet(self) -> Any:
        """Produce the next simulated network packet."""
        pass

    def stream(self, count: Optional[int] = None) -> Iterator[Any]:
        """
        Yields packets continuously up to `count` (or infinitely if count is None).
        Paces generation if rate_pps is specified.
        """
        delay = (1.0 / self.rate_pps) if self.rate_pps and self.rate_pps > 0 else 0.0
        generated = 0

        while count is None or generated < count:
            pkt = self.generate_packet()
            self.generated_count += 1
            generated += 1
            yield pkt
            if delay > 0:
                time.sleep(delay)

    def generate_batch(self, size: int) -> list[Any]:
        """Generate a fixed list of packets."""
        return [self.generate_packet() for _ in range(size)]
