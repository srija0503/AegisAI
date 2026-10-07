"""
rl_agent/action.py
──────────────────
Defines the discrete action space for the Reinforcement Learning firewall agent.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict


class FirewallAction(IntEnum):
    """
    Action choices available to the RL agent for each inspected network flow.
    """
    ALLOW = 0         # Permit benign traffic through the firewall
    BLOCK = 1         # Drop/terminate malicious traffic
    TRIGGER_RAG = 2   # Escalate uncertain/anomalous flow to Agentic RAG (Member 2)


ACTION_NAMES: dict[int, str] = {
    FirewallAction.ALLOW: "ALLOW",
    FirewallAction.BLOCK: "BLOCK",
    FirewallAction.TRIGGER_RAG: "TRIGGER_RAG",
}

NUM_ACTIONS = len(FirewallAction)  # 3
