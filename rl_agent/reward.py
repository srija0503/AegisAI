"""
rl_agent/reward.py
──────────────────
Reward engineering module for the RL firewall agent.
Penalizes security breaches (False Negatives) and operational disruption (False Positives),
while incentivizing accurate threat neutralization and smart RAG escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from rl_agent.action import FirewallAction


@dataclass
class RewardConfig:
    """Configurable reward coefficients mirroring security risk profiles."""
    true_positive: float = 1.0     # Blocked an actual attack
    true_negative: float = 0.2     # Allowed benign traffic
    false_positive: float = -1.0   # Blocked legitimate traffic (usability impact)
    false_negative: float = -2.0   # Allowed an attack (critical security failure)
    rag_trigger_cost: float = -0.1 # Modest computational overhead cost for RAG lookup
    rag_true_reward: float = 0.8   # Reward when RAG correctly resolves an ambiguous attack
    latency_penalty: float = -0.05 # Penalty per millisecond exceeding latency budget


class RewardCalculator:
    """Computes shaped rewards based on agent action, ground truth, and execution latency."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        action: int | FirewallAction,
        is_attack: bool | int,
        latency_ms: float = 0.0,
        is_ambiguous_zero_day: bool = False,
    ) -> float:
        """
        Calculate total scalar reward for a step.

        Parameters:
            action: Selected FirewallAction (0=ALLOW, 1=BLOCK, 2=TRIGGER_RAG)
            is_attack: 1/True if malicious, 0/False if benign
            latency_ms: Decision time in milliseconds
            is_ambiguous_zero_day: True if this sample is an unknown/unseen zero-day variant
        """
        action = FirewallAction(action)
        is_attack = bool(is_attack)
        r = 0.0

        if action == FirewallAction.ALLOW:
            if not is_attack:
                r += self.config.true_negative
            else:
                r += self.config.false_negative  # Critical breach

        elif action == FirewallAction.BLOCK:
            if is_attack:
                r += self.config.true_positive
            else:
                r += self.config.false_positive  # Interrupted valid user

        elif action == FirewallAction.TRIGGER_RAG:
            # Base exploration/API query cost
            r += self.config.rag_trigger_cost
            if is_ambiguous_zero_day or is_attack:
                # Successfully caught an anomaly or attack needing threat intel
                r += self.config.rag_true_reward
            else:
                # Unnecessary RAG query for obvious benign traffic
                r += -0.2

        # Optional latency penalty if time exceeds budget (e.g., > 5.0 ms)
        if latency_ms > 5.0:
            excess = (latency_ms - 5.0)
            r += self.config.latency_penalty * excess

        return float(r)
