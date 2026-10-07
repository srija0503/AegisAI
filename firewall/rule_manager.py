"""
firewall/rule_manager.py
────────────────────────
Manages dynamic firewall rules, YAML serialization/deserialization, and
provides dynamic rule update APIs for Member 2's Agentic RAG integration.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, List, Optional

import yaml

from firewall.rule_engine import FirewallRule, RuleAction, RuleEngine


class RuleManager:
    """
    Thread-safe rule configuration manager. Synchronizes in-memory rules with
    the RuleEngine and persists updates to YAML.
    """

    def __init__(
        self,
        engine: RuleEngine,
        rules_file: Optional[str] = "./config/firewall_rules.yaml",
    ) -> None:
        self.engine = engine
        self.rules_file = Path(rules_file) if rules_file else None
        self._rules_by_id: dict[str, FirewallRule] = {}
        self._lock = threading.RLock()

        if self.rules_file and self.rules_file.exists():
            self.load()

    def add_rule(self, rule: FirewallRule, save: bool = True) -> bool:
        """
        Add or overwrite a rule. Updates rule engine immediately.
        Agentic RAG calls this when new mitigation rules are generated.
        """
        with self._lock:
            self._rules_by_id[rule.rule_id] = rule
            self._sync_engine()
            if save and self.rules_file:
                self.save()
            return True

    def remove_rule(self, rule_id: str, save: bool = True) -> bool:
        """Remove a rule by its ID."""
        with self._lock:
            if rule_id in self._rules_by_id:
                del self._rules_by_id[rule_id]
                self._sync_engine()
                if save and self.rules_file:
                    self.save()
                return True
            return False

    def get_rule(self, rule_id: str) -> Optional[FirewallRule]:
        """Fetch a rule by ID."""
        with self._lock:
            return self._rules_by_id.get(rule_id)

    def list_rules(self) -> list[FirewallRule]:
        """List all rules ordered by priority."""
        with self._lock:
            return sorted(self._rules_by_id.values(), key=lambda r: r.priority)

    def enable_rule(self, rule_id: str, enabled: bool = True, save: bool = True) -> bool:
        """Enable or disable an existing rule."""
        with self._lock:
            rule = self._rules_by_id.get(rule_id)
            if rule:
                rule.enabled = enabled
                self._sync_engine()
                if save and self.rules_file:
                    self.save()
                return True
            return False

    def _sync_engine(self) -> None:
        """Propagate current rule list into RuleEngine."""
        self.engine.set_rules(list(self._rules_by_id.values()))

    def load(self, filepath: Optional[str] = None) -> int:
        """Load rules from YAML file."""
        path = Path(filepath) if filepath else self.rules_file
        if not path or not path.exists():
            return 0

        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            rules_data = data.get("rules", [])
            loaded = 0
            self._rules_by_id.clear()

            for item in rules_data:
                try:
                    rule = FirewallRule.from_dict(item)
                    self._rules_by_id[rule.rule_id] = rule
                    loaded += 1
                except Exception:
                    pass

            self._sync_engine()
            return loaded

    def save(self, filepath: Optional[str] = None) -> None:
        """Persist current rules to YAML file."""
        path = Path(filepath) if filepath else self.rules_file
        if not path:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "version": "1.0",
                "rules": [rule.to_dict() for rule in self.list_rules()],
            }
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, indent=2)
