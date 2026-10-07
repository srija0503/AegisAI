"""
firewall/traffic_classifier.py
──────────────────────────────
Traffic decision pipeline orchestrating Allowlist, Blocklist, Emergency Breaker,
Rule Engine, and the Reinforcement Learning model.
Detects zero-day anomalies and triggers the Agentic RAG loop when confidence is low.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from firewall.allowlist import AllowList
from firewall.blocklist import BlockList
from firewall.emergency_block import EmergencyCircuitBreaker
from firewall.flow_manager import NetworkFlow
from firewall.packet_parser import ParsedPacket
from firewall.rule_engine import FirewallRule, RuleAction, RuleEngine


@dataclass
class ClassificationDecision:
    """The final disposition determined for an inspected packet/flow."""

    action: RuleAction
    source: str  # "emergency", "allowlist", "blocklist", "rule_engine", "rl_agent", "default_policy"
    confidence: float = 1.0
    anomaly_score: float = 0.0
    rule_matched: Optional[FirewallRule] = None
    trigger_rag: bool = False
    explanation: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def is_blocked(self) -> bool:
        return self.action in (RuleAction.BLOCK, RuleAction.DROP)

    @property
    def is_allowed(self) -> bool:
        return self.action == RuleAction.ALLOW


class TrafficClassifier:
    """
    Decides the action for incoming network traffic by combining:
    1. Emergency Circuit Breaker (Anti-DDoS)
    2. IP Allowlist (Immediate Fast-Path Bypass)
    3. IP Blocklist (Immediate Fast-Path Drop)
    4. Deterministic Rule Engine
    5. RL Agent Model Inference (with uncertainty estimation & RAG triggering)
    """

    def __init__(
        self,
        mode: str = "hybrid",  # "rule" | "rl" | "hybrid"
        confidence_threshold: float = 0.80,
        auto_block_threshold: float = 0.95,
        rl_model_path: Optional[str] = None,
        allowlist: Optional[AllowList] = None,
        blocklist: Optional[BlockList] = None,
        rule_engine: Optional[RuleEngine] = None,
        emergency_breaker: Optional[EmergencyCircuitBreaker] = None,
        default_policy: str = "allow",
    ) -> None:
        self.mode = mode.lower()
        self.confidence_threshold = confidence_threshold
        self.auto_block_threshold = auto_block_threshold
        self.rl_model_path = Path(rl_model_path) if rl_model_path else None
        self.default_action = RuleAction.ALLOW if default_policy.lower() == "allow" else RuleAction.BLOCK

        self.allowlist = allowlist or AllowList()
        self.blocklist = blocklist or BlockList(auto_block_threshold=auto_block_threshold)
        self.rule_engine = rule_engine or RuleEngine(default_policy=default_policy)
        self.emergency_breaker = emergency_breaker or EmergencyCircuitBreaker()

        # RL model instance (optional until Phase 3)
        self._rl_model: Optional[torch.nn.Module] = None
        self._load_rl_model_if_available()

    def set_rl_model(self, model: torch.nn.Module) -> None:
        """Register trained RL model instance for real-time inference."""
        self._rl_model = model
        self._rl_model.eval()

    def _load_rl_model_if_available(self) -> None:
        if self.rl_model_path and self.rl_model_path.exists():
            try:
                # Load checkpoint if compatible
                checkpoint = torch.load(self.rl_model_path, map_location="cpu")
                if isinstance(checkpoint, torch.nn.Module):
                    self.set_rl_model(checkpoint)
                elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    # Model architecture instantiated in Phase 3
                    pass
            except Exception:
                pass

    def classify(
        self,
        pkt: ParsedPacket,
        flow: NetworkFlow,
        feature_vector: np.ndarray,
    ) -> ClassificationDecision:
        """
        Evaluate packet/flow through defense layers in prioritized sequence.
        """
        now = pkt.timestamp or time.time()

        # ─── Layer 1: Emergency Circuit Breaker ──────────────────────────────
        if self.emergency_breaker.enabled:
            is_emer_blocked, emer_reason = self.emergency_breaker.is_blocked(pkt.src_ip, now=now)
            if is_emer_blocked:
                return ClassificationDecision(
                    action=RuleAction.DROP,
                    source="emergency",
                    confidence=1.0,
                    explanation=emer_reason or "Emergency Circuit Breaker Triggered",
                    timestamp=now,
                )

        # ─── Layer 2: Allowlist (Fast Bypass) ─────────────────────────────────
        if self.allowlist.is_allowed(pkt.src_ip):
            return ClassificationDecision(
                action=RuleAction.ALLOW,
                source="allowlist",
                confidence=1.0,
                explanation=f"Source {pkt.src_ip} in allowlist",
                timestamp=now,
            )

        # ─── Layer 3: Blocklist (Fast Drop) ───────────────────────────────────
        is_blocked, block_reason = self.blocklist.is_blocked(pkt.src_ip)
        if is_blocked:
            return ClassificationDecision(
                action=RuleAction.BLOCK,
                source="blocklist",
                confidence=1.0,
                explanation=f"Source {pkt.src_ip} in blocklist: {block_reason}",
                timestamp=now,
            )

        # ─── Layer 4: Deterministic Rules (in 'rule' or 'hybrid' mode) ────────
        if self.mode in ("rule", "hybrid"):
            rule_action, matched_rule = self.rule_engine.evaluate(pkt)
            if matched_rule is not None:
                # High-priority explicit rule matched
                is_rag_rule = rule_action == RuleAction.TRIGGER_RAG
                return ClassificationDecision(
                    action=rule_action,
                    source="rule_engine",
                    confidence=1.0,
                    rule_matched=matched_rule,
                    trigger_rag=is_rag_rule,
                    explanation=f"Matched rule {matched_rule.rule_id}: {matched_rule.description}",
                    timestamp=now,
                )

        # In rule-only mode with no rule match: default policy
        if self.mode == "rule":
            return ClassificationDecision(
                action=self.default_action,
                source="default_policy",
                confidence=1.0,
                explanation=f"Default firewall policy ({self.default_action.value})",
                timestamp=now,
            )

        # ─── Layer 5: RL Model Inference / Anomaly Scoring ───────────────────
        return self._evaluate_ml(pkt, flow, feature_vector, now)

    def _evaluate_ml(
        self,
        pkt: ParsedPacket,
        flow: NetworkFlow,
        vec: np.ndarray,
        now: float,
    ) -> ClassificationDecision:
        """Inference via RL model or baseline heuristic anomaly estimator."""
        confidence: float
        is_attack: bool
        anomaly_score: float

        if self._rl_model is not None:
            # Model inference
            try:
                with torch.no_grad():
                    t_vec = torch.from_numpy(vec).float().unsqueeze(0)
                    q_values = self._rl_model(t_vec)
                    probs = torch.softmax(q_values, dim=1).squeeze(0)
                    # Assuming Action 0 = ALLOW (normal), Action 1 = BLOCK (attack)
                    allow_prob = float(probs[0])
                    block_prob = float(probs[1]) if probs.shape[0] > 1 else (1.0 - allow_prob)

                    is_attack = block_prob > allow_prob
                    confidence = max(allow_prob, block_prob)
                    anomaly_score = block_prob
            except Exception:
                is_attack, confidence, anomaly_score = self._heuristic_anomaly_score(pkt, flow)
        else:
            # Baseline heuristic until Phase 3 trains the DQN
            is_attack, confidence, anomaly_score = self._heuristic_anomaly_score(pkt, flow)

        # Low confidence uncertainty detection -> Trigger Agentic RAG!
        if confidence < self.confidence_threshold:
            return ClassificationDecision(
                action=RuleAction.TRIGGER_RAG,
                source="rl_agent",
                confidence=confidence,
                anomaly_score=anomaly_score,
                trigger_rag=True,
                explanation=(
                    f"Ambiguous traffic pattern: confidence {confidence:.2f} "
                    f"below threshold {self.confidence_threshold:.2f}. Escalating to RAG."
                ),
                timestamp=now,
            )

        # High confidence attack -> Auto-block trigger check
        if is_attack:
            if confidence >= self.auto_block_threshold:
                self.blocklist.auto_block(
                    pkt.src_ip,
                    confidence=confidence,
                    reason=f"Auto-blocked by RL Classifier (anomaly={anomaly_score:.2f})",
                )

            return ClassificationDecision(
                action=RuleAction.BLOCK,
                source="rl_agent",
                confidence=confidence,
                anomaly_score=anomaly_score,
                trigger_rag=False,
                explanation=f"RL Agent detected attack with {confidence:.1%} confidence",
                timestamp=now,
            )

        # Confident normal traffic
        return ClassificationDecision(
            action=RuleAction.ALLOW,
            source="rl_agent",
            confidence=confidence,
            anomaly_score=anomaly_score,
            trigger_rag=False,
            explanation=f"RL Agent classified normal traffic with {confidence:.1%} confidence",
            timestamp=now,
        )

    def _heuristic_anomaly_score(
        self,
        pkt: ParsedPacket,
        flow: NetworkFlow,
    ) -> tuple[bool, float, float]:
        """
        Reliable baseline heuristic when RL model checkpoint is not yet present.
        Returns: (is_attack: bool, confidence: float, anomaly_score: float).
        """
        anomaly = 0.0

        # TCP Flag anomalies
        if pkt.tcp_flags.get("SYN") and pkt.tcp_flags.get("FIN"):
            anomaly += 0.85  # SYN+FIN illegal scan
        if pkt.is_syn_only() and flow.syn_count > 10 and flow.ack_count == 0:
            anomaly += 0.75  # SYN flood pattern
        if pkt.flag_str in ("S0", "REJ", "RSTO"):
            anomaly += 0.40

        # Land attack (src IP == dst IP)
        if pkt.src_ip == pkt.dst_ip:
            anomaly += 0.95

        # Normal established connections
        if flow.state == "ESTABLISHED" and flow.total_packets > 3:
            anomaly = max(0.0, anomaly - 0.3)

        anomaly = min(1.0, max(0.0, anomaly))

        if anomaly >= 0.70:
            is_attack = True
            confidence = anomaly
        elif anomaly <= 0.30:
            is_attack = False
            confidence = 1.0 - anomaly
        else:
            # Ambiguous zone
            is_attack = anomaly >= 0.50
            confidence = max(anomaly, 1.0 - anomaly)  # e.g., 0.55 or 0.60 < 0.80

        return is_attack, confidence, anomaly
