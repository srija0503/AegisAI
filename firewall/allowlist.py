"""
firewall/allowlist.py
─────────────────────
Thread-safe IP and CIDR subnet allowlist manager with support for RFC1918
private network whitelisting and file persistence.
"""

from __future__ import annotations

import ipaddress
import threading
from pathlib import Path
from typing import List, Optional, Set, Union


class AllowList:
    """
    Manages trusted IPv4/IPv6 addresses and CIDR subnets that bypass
    firewall inspection or RL classification.
    """

    DEFAULT_PRIVATE_RANGES = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "::1/128",
        "fc00::/7",
    ]

    def __init__(
        self,
        always_allow_private: bool = True,
        private_ranges: Optional[list[str]] = None,
        filepath: Optional[str] = None,
    ) -> None:
        self.always_allow_private = always_allow_private
        self.filepath = Path(filepath) if filepath else None

        self._exact_ips: set[str] = set()
        self._networks: list[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = []
        self._lock = threading.RLock()

        # Load private ranges if configured
        if self.always_allow_private:
            ranges = private_ranges or self.DEFAULT_PRIVATE_RANGES
            for r in ranges:
                self.add(r)

        # Load from file if specified and exists
        if self.filepath and self.filepath.exists():
            self.load_from_file(str(self.filepath))

    def add(self, item: str) -> bool:
        """Add an IP address or CIDR range to the allowlist."""
        item = item.strip()
        if not item or item.startswith("#"):
            return False

        with self._lock:
            try:
                if "/" in item:
                    net = ipaddress.ip_network(item, strict=False)
                    if net not in self._networks:
                        self._networks.append(net)
                    return True
                else:
                    addr = ipaddress.ip_address(item)
                    self._exact_ips.add(str(addr))
                    return True
            except ValueError:
                return False

    def remove(self, item: str) -> bool:
        """Remove an IP address or CIDR range from the allowlist."""
        item = item.strip()
        with self._lock:
            try:
                if "/" in item:
                    net = ipaddress.ip_network(item, strict=False)
                    if net in self._networks:
                        self._networks.remove(net)
                        return True
                else:
                    addr = str(ipaddress.ip_address(item))
                    if addr in self._exact_ips:
                        self._exact_ips.remove(addr)
                        return True
            except ValueError:
                pass
            return False

    def is_allowed(self, ip_str: str) -> bool:
        """Check if an IP address matches any allowlist entry or private range."""
        ip_str = ip_str.strip()
        with self._lock:
            if ip_str in self._exact_ips:
                return True

            try:
                addr = ipaddress.ip_address(ip_str)
                # Check CIDR networks
                for net in self._networks:
                    if addr in net:
                        return True
            except ValueError:
                return False

            return False

    def count(self) -> int:
        """Total distinct exact IPs and subnet entries in allowlist."""
        with self._lock:
            return len(self._exact_ips) + len(self._networks)

    def load_from_file(self, filepath: Optional[str] = None) -> int:
        """Load allowlist entries from a text file (one IP/CIDR per line)."""
        path = Path(filepath) if filepath else self.filepath
        if not path or not path.exists():
            return 0

        loaded = 0
        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if self.add(line):
                            loaded += 1
        return loaded

    def save_to_file(self, filepath: Optional[str] = None) -> None:
        """Persist current allowlist to text file."""
        path = Path(filepath) if filepath else self.filepath
        if not path:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Firewall Allowlist\n")
                for net in self._networks:
                    f.write(f"{net}\n")
                for ip in sorted(self._exact_ips):
                    f.write(f"{ip}\n")
