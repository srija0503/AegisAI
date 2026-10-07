"""
firewall/rule_engine.py
───────────────────────
Rule definitions and evaluation engine for priority-ordered pattern matching.
Supports IP ranges (CIDR), port ranges/lists, protocol filters, and TCP flags.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Set, Tuple, Union

from firewall.packet_parser import ParsedPacket


class RuleAction(str, Enum):
    """Actions resulting from firewall rule evaluation."""
    ALLOW = "allow"
    BLOCK = "block"
    DROP = "drop"
    ALERT = "alert"
    TRIGGER_RAG = "trigger_rag"


@dataclass
class FirewallRule:
    """
    Firewall filtering rule definition.
    Lower priority value = higher evaluation precedence (e.g., priority 1 runs before 100).
    """

    rule_id: str
    priority: int
    action: RuleAction
    src_ip: Optional[str] = None       # e.g. "192.168.1.50" or "10.0.0.0/8"
    dst_ip: Optional[str] = None       # e.g. "0.0.0.0/0"
    src_port: Optional[Union[int, list[int], tuple[int, int]]] = None
    dst_port: Optional[Union[int, list[int], tuple[int, int]]] = None
    protocol: Optional[str] = None     # "tcp", "udp", "icmp", "any"
    tcp_flags: Optional[list[str]] = None  # e.g. ["SYN"]
    description: str = ""
    created_by: str = "admin"          # "admin", "rag_agent", "federated_policy"
    enabled: bool = True
    hit_count: int = 0

    # Cached parsed networks for performance
    _src_net: Optional[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = field(init=False, default=None)
    _dst_net: Optional[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if isinstance(self.action, str):
            self.action = RuleAction(self.action.lower())

        if self.src_ip and self.src_ip != "any":
            try:
                self._src_net = ipaddress.ip_network(self.src_ip, strict=False)
            except ValueError:
                self._src_net = None

        if self.dst_ip and self.dst_ip != "any":
            try:
                self._dst_net = ipaddress.ip_network(self.dst_ip, strict=False)
            except ValueError:
                self._dst_net = None

    def matches(self, pkt: ParsedPacket) -> bool:
        """Evaluate if the parsed packet satisfies all criteria of this rule."""
        if not self.enabled:
            return False

        # 1. Protocol match
        if self.protocol and self.protocol.lower() not in ("any", "*"):
            if pkt.protocol.lower() != self.protocol.lower():
                return False

        # 2. Source IP / Subnet
        if self._src_net:
            try:
                if ipaddress.ip_address(pkt.src_ip) not in self._src_net:
                    return False
            except ValueError:
                return False

        # 3. Destination IP / Subnet
        if self._dst_net:
            try:
                if ipaddress.ip_address(pkt.dst_ip) not in self._dst_net:
                    return False
            except ValueError:
                return False

        # 4. Source Port
        if self.src_port is not None:
            if not self._match_port(pkt.src_port, self.src_port):
                return False

        # 5. Destination Port
        if self.dst_port is not None:
            if not self._match_port(pkt.dst_port, self.dst_port):
                return False

        # 6. TCP Flags
        if self.tcp_flags and pkt.protocol == "tcp":
            for flag in self.tcp_flags:
                if not pkt.tcp_flags.get(flag.upper(), False):
                    return False

        return True

    @staticmethod
    def _match_port(pkt_port: int, criterion: Union[int, list[int], tuple[int, int]]) -> bool:
        if isinstance(criterion, int):
            return pkt_port == criterion
        if isinstance(criterion, (list, set)):
            return pkt_port in criterion
        if isinstance(criterion, tuple) and len(criterion) == 2:
            return criterion[0] <= pkt_port <= criterion[1]
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize rule to dictionary for YAML persistence."""
        return {
            "rule_id": self.rule_id,
            "priority": self.priority,
            "action": self.action.value,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "tcp_flags": self.tcp_flags,
            "description": self.description,
            "created_by": self.created_by,
            "enabled": self.enabled,
            "hit_count": self.hit_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FirewallRule":
        """Deserialize rule from dictionary."""
        return cls(
            rule_id=str(data["rule_id"]),
            priority=int(data.get("priority", 100)),
            action=RuleAction(data["action"]),
            src_ip=data.get("src_ip"),
            dst_ip=data.get("dst_ip"),
            src_port=data.get("src_port"),
            dst_port=data.get("dst_port"),
            protocol=data.get("protocol"),
            tcp_flags=data.get("tcp_flags"),
            description=data.get("description", ""),
            created_by=data.get("created_by", "admin"),
            enabled=data.get("enabled", True),
            hit_count=data.get("hit_count", 0),
        )


class RuleEngine:
    """
    Evaluates packets against a priority-ordered collection of FirewallRules.
    Falls back to a default policy if no rules match.
    """

    def __init__(self, default_policy: str = "allow") -> None:
        self.default_action = RuleAction.ALLOW if default_policy.lower() == "allow" else RuleAction.BLOCK
        self._rules: list[FirewallRule] = []

    def set_rules(self, rules: list[FirewallRule]) -> None:
        """Replace active rule set and sort in priority order (ascending priority number)."""
        self._rules = sorted(rules, key=lambda r: r.priority)

    def evaluate(self, pkt: ParsedPacket) -> tuple[RuleAction, Optional[FirewallRule]]:
        """
        Evaluate packet against active rules in priority sequence.
        Returns:
            (RuleAction, matched_rule) on match.
            (default_action, None) if no rule matches.
        """
        for rule in self._rules:
            if rule.matches(pkt):
                rule.hit_count += 1
                return rule.action, rule

        return self.default_action, None
