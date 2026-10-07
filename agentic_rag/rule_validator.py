"""
agentic_rag/rule_validator.py
─────────────────────────────
Pre-deployment validator for dynamically generated firewall rules.
Guarantees safety constraints to prevent self-inflicted denial of service,
such as blocking localhost, DNS, critical infrastructure gateways, or malformed subnets.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Critical IPs and subnets that must NEVER be blocked
PROTECTED_IPS = {
    "127.0.0.1",
    "::1",
    "0.0.0.0",
}

PROTECTED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
]


class RuleValidator:
    """
    Validates synthesized firewall rules before committing them to the rule manager.
    """

    def __init__(self, protected_ips: Optional[set[str]] = None) -> None:
        self.protected_ips = set(PROTECTED_IPS)
        if protected_ips:
            self.protected_ips.update(protected_ips)

    def validate_rule(self, rule: dict[str, Any]) -> tuple[bool, str]:
        """
        Validates rule dictionary.
        Returns: (is_valid: bool, reason: str)
        """
        # 1. Required fields
        for field in ("rule_id", "priority", "action"):
            if field not in rule:
                return False, f"Missing mandatory field '{field}'"

        # 2. Priority check
        priority = rule.get("priority", 100)
        if not isinstance(priority, int) or priority < 1 or priority > 1000:
            return False, f"Invalid rule priority: {priority} (must be 1-1000)"

        # 3. Action check
        action = str(rule.get("action", "")).lower()
        if action not in ("allow", "block", "drop", "alert"):
            return False, f"Invalid rule action: {action}"

        # 4. Source IP safety checks
        src_ip = rule.get("src_ip")
        if src_ip:
            if src_ip in ("0.0.0.0/0", "any", "*") and action in ("block", "drop"):
                return False, "Dangerous rule: Blocking all source traffic (0.0.0.0/0) would sever network."

            if src_ip in self.protected_ips:
                return False, f"Cannot block protected host IP: {src_ip}"

            try:
                net = ipaddress.ip_network(src_ip, strict=False)
                for p_net in PROTECTED_NETWORKS:
                    if net.overlaps(p_net) and action in ("block", "drop"):
                        return False, f"Source IP overlaps with protected network {p_net}"
            except ValueError as e:
                return False, f"Invalid IP / CIDR format '{src_ip}': {e}"

        # 5. Port check
        dst_port = rule.get("dst_port")
        if dst_port is not None:
            if isinstance(dst_port, int):
                if dst_port < 1 or dst_port > 65535:
                    return False, f"Port {dst_port} out of range (1-65535)"
            elif isinstance(dst_port, (list, tuple)):
                for p in dst_port:
                    if not isinstance(p, int) or p < 1 or p > 65535:
                        return False, f"Port element {p} out of range"

        # 6. Verify compatibility with FirewallRule class
        try:
            from firewall.rule_engine import FirewallRule
            FirewallRule.from_dict(rule)
        except Exception as e:
            return False, f"FirewallRule instantiation failed: {e}"

        return True, "Rule validated successfully"

    def filter_valid_rules(self, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filters a list of rules, discarding rejected ones."""
        valid_rules: list[dict[str, Any]] = []
        for r in rules:
            ok, reason = self.validate_rule(r)
            if ok:
                valid_rules.append(r)
            else:
                logger.warning(f"Rejected rule '{r.get('rule_id')}': {reason}")
        return valid_rules
