"""
firewall/blocklist.py
─────────────────────
Thread-safe IP and CIDR subnet blocklist with support for expiration,
automatic RL-triggered blocking, and file persistence.
"""

from __future__ import annotations

import ipaddress
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class BlockEntry:
    target: str  # IP or CIDR
    reason: str
    added_at: float
    expires_at: Optional[float] = None  # None = permanent

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class BlockList:
    """
    Manages forbidden IPv4/IPv6 addresses and CIDR subnets.
    Supports permanent rules, TTL-based dynamic blocks, and auto-blocking by ML score.
    """

    def __init__(
        self,
        auto_block_threshold: float = 0.95,
        default_ttl_seconds: Optional[float] = None,
        filepath: Optional[str] = None,
    ) -> None:
        self.auto_block_threshold = auto_block_threshold
        self.default_ttl_seconds = default_ttl_seconds
        self.filepath = Path(filepath) if filepath else None

        self._entries: dict[str, BlockEntry] = {}
        self._networks: dict[Union[ipaddress.IPv4Network, ipaddress.IPv6Network], BlockEntry] = {}
        self._lock = threading.RLock()

        if self.filepath and self.filepath.exists():
            self.load_from_file(str(self.filepath))

    def add(
        self,
        item: str,
        reason: str = "Manual Block",
        ttl_seconds: Optional[float] = None,
    ) -> bool:
        """Add an IP address or CIDR range to the blocklist."""
        item = item.strip()
        if not item or item.startswith("#"):
            return False

        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        expires_at = (now + ttl) if ttl else None
        entry = BlockEntry(target=item, reason=reason, added_at=now, expires_at=expires_at)

        with self._lock:
            try:
                if "/" in item:
                    net = ipaddress.ip_network(item, strict=False)
                    self._networks[net] = entry
                    return True
                else:
                    addr = str(ipaddress.ip_address(item))
                    self._entries[addr] = entry
                    return True
            except ValueError:
                return False

    def remove(self, item: str) -> bool:
        """Remove an item from the blocklist."""
        item = item.strip()
        with self._lock:
            try:
                if "/" in item:
                    net = ipaddress.ip_network(item, strict=False)
                    if net in self._networks:
                        del self._networks[net]
                        return True
                else:
                    addr = str(ipaddress.ip_address(item))
                    if addr in self._entries:
                        del self._entries[addr]
                        return True
            except ValueError:
                pass
            return False

    def is_blocked(self, ip_str: str) -> tuple[bool, Optional[str]]:
        """
        Check if an IP is currently blocked.
        Returns:
            (True, reason) if blocked and active.
            (False, None) if not blocked or block has expired.
        """
        ip_str = ip_str.strip()
        now = time.time()

        with self._lock:
            # 1. Exact IP check
            entry = self._entries.get(ip_str)
            if entry:
                if entry.is_expired:
                    del self._entries[ip_str]
                else:
                    return True, entry.reason

            # 2. Subnet checks
            try:
                addr = ipaddress.ip_address(ip_str)
                expired_nets = []
                for net, n_entry in self._networks.items():
                    if n_entry.is_expired:
                        expired_nets.append(net)
                        continue
                    if addr in net:
                        return True, n_entry.reason

                for net in expired_nets:
                    del self._networks[net]
            except ValueError:
                return False, None

            return False, None

    def auto_block(
        self,
        ip_str: str,
        confidence: float,
        reason: str = "RL High-Confidence Threat",
        ttl_seconds: Optional[float] = 300.0,
    ) -> bool:
        """
        Conditionally block an IP if its threat confidence score exceeds auto_block_threshold.
        """
        if confidence >= self.auto_block_threshold:
            full_reason = f"{reason} (confidence={confidence:.2f})"
            return self.add(ip_str, reason=full_reason, ttl_seconds=ttl_seconds)
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired block entries."""
        with self._lock:
            removed = 0
            # Expired exact entries
            expired_ips = [k for k, v in self._entries.items() if v.is_expired]
            for k in expired_ips:
                del self._entries[k]
                removed += 1

            # Expired networks
            expired_nets = [k for k, v in self._networks.items() if v.is_expired]
            for k in expired_nets:
                del self._networks[k]
                removed += 1

            return removed

    def count(self) -> int:
        """Count active (non-expired) block entries."""
        with self._lock:
            self.cleanup_expired()
            return len(self._entries) + len(self._networks)

    def load_from_file(self, filepath: Optional[str] = None) -> int:
        """Load blocklist entries from file (format: IP/CIDR [optional reason])."""
        path = Path(filepath) if filepath else self.filepath
        if not path or not path.exists():
            return 0

        loaded = 0
        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split(maxsplit=1)
                        item = parts[0]
                        reason = parts[1] if len(parts) > 1 else "File Block"
                        if self.add(item, reason=reason):
                            loaded += 1
        return loaded

    def save_to_file(self, filepath: Optional[str] = None) -> None:
        """Persist blocklist entries to file."""
        path = Path(filepath) if filepath else self.filepath
        if not path:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.cleanup_expired()
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Firewall Blocklist\n")
                for net, entry in self._networks.items():
                    f.write(f"{net} {entry.reason}\n")
                for ip, entry in sorted(self._entries.items()):
                    f.write(f"{ip} {entry.reason}\n")
