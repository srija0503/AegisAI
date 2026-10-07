"""
firewall/firewall_controller.py
───────────────────────────────
Main orchestration engine for the Quantum-Secure Federated Firewall.
Coordinates packet capture, parsing, stateful flow tracking, feature extraction,
rule evaluation, and reinforcement learning classification.
Serves as the integration hub for Member 2 (Agentic RAG) and Member 3 (Federated Learning).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from firewall.allowlist import AllowList
from firewall.blocklist import BlockList
from firewall.emergency_block import EmergencyCircuitBreaker
from firewall.feature_extractor import FirewallFeatureExtractor
from firewall.flow_manager import FlowManager, NetworkFlow
from firewall.packet_capture import PacketCapture
from firewall.packet_parser import PacketParser, ParsedPacket
from firewall.rule_engine import RuleAction, RuleEngine
from firewall.rule_manager import RuleManager
from firewall.traffic_classifier import ClassificationDecision, TrafficClassifier

logger = logging.getLogger(__name__)


@dataclass
class FirewallMetrics:
    total_packets_inspected: int = 0
    packets_allowed: int = 0
    packets_blocked: int = 0
    packets_dropped: int = 0
    rag_triggers: int = 0
    emergency_drops: int = 0
    decisions_by_source: dict[str, int] = field(default_factory=lambda: {
        "emergency": 0,
        "allowlist": 0,
        "blocklist": 0,
        "rule_engine": 0,
        "rl_agent": 0,
        "default_policy": 0,
    })
    start_time: float = field(default_factory=time.time)


class FirewallController:
    """
    Top-level Firewall Orchestrator.
    Controls the entire traffic ingestion, analysis, and enforcement loop.
    """

    def __init__(
        self,
        config_path: Optional[str] = "./config/firewall.yaml",
        config_dict: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = self._load_config(config_path, config_dict)
        self.metrics = FirewallMetrics()
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Member 2 Callback hook: signature (flow_features: dict, anomaly_score: float) -> dict
        self._rag_callback: Optional[Callable[[dict[str, Any], float], Any]] = None

        # ─── 1. Subsystems Initialization ─────────────────────────────────────
        # Parser
        self.parser = PacketParser()

        # Capture
        cap_cfg = self.config.get("capture", {})
        self.capture = PacketCapture(
            interface=cap_cfg.get("interface", "eth0"),
            bpf_filter=cap_cfg.get("filter", "ip"),
            promisc=cap_cfg.get("promisc", True),
            snaplen=cap_cfg.get("snaplen", 65535),
        )

        # Flow Manager
        flow_cfg = self.config.get("flow", {})
        self.flow_manager = FlowManager(
            timeout_seconds=flow_cfg.get("timeout_seconds", 120.0),
            max_flows=flow_cfg.get("max_flows", 100000),
            cleanup_interval_seconds=flow_cfg.get("cleanup_interval_seconds", 30.0),
        )

        # Feature Extractor
        feat_cfg = self.config.get("feature_extraction", {})
        self.feature_extractor = FirewallFeatureExtractor(
            window_size=feat_cfg.get("window_size", 10),
        )

        # Allowlist
        allow_cfg = self.config.get("allowlist", {})
        self.allowlist = AllowList(
            always_allow_private=allow_cfg.get("always_allow_private", True),
            private_ranges=allow_cfg.get("private_ranges"),
            filepath=allow_cfg.get("file"),
        )

        # Blocklist
        block_cfg = self.config.get("blocklist", {})
        self.blocklist = BlockList(
            auto_block_threshold=block_cfg.get("auto_block_threshold", 0.95),
            filepath=block_cfg.get("file"),
        )

        # Emergency Circuit Breaker
        emer_cfg = self.config.get("emergency", {})
        self.emergency_breaker = EmergencyCircuitBreaker(
            enabled=emer_cfg.get("enabled", True),
            threshold_pps=emer_cfg.get("threshold_pps", 10000.0),
            block_duration_seconds=emer_cfg.get("block_duration_seconds", 300.0),
        )

        # Rules Engine & Manager
        rule_cfg = self.config.get("rules", {})
        default_policy = rule_cfg.get("default_policy", "allow")
        self.rule_engine = RuleEngine(default_policy=default_policy)
        self.rule_manager = RuleManager(
            engine=self.rule_engine,
            rules_file=rule_cfg.get("rules_file", "./config/firewall_rules.yaml"),
        )

        # Traffic Classifier
        class_cfg = self.config.get("classifier", {})
        self.classifier = TrafficClassifier(
            mode=class_cfg.get("mode", "hybrid"),
            confidence_threshold=class_cfg.get("confidence_threshold", 0.80),
            auto_block_threshold=block_cfg.get("auto_block_threshold", 0.95),
            rl_model_path=class_cfg.get("rl_model_path"),
            allowlist=self.allowlist,
            blocklist=self.blocklist,
            rule_engine=self.rule_engine,
            emergency_breaker=self.emergency_breaker,
            default_policy=default_policy,
        )

    def _load_config(
        self,
        config_path: Optional[str],
        config_dict: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Load configuration from dictionary or YAML file."""
        if config_dict:
            return config_dict

        if config_path:
            p = Path(config_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

        return {}

    def register_rag_hook(self, callback: Callable[[dict[str, Any], float], Any]) -> None:
        """
        Interface hook for Member 2 (Agentic RAG).
        When an ambiguous or zero-day anomaly is encountered, this callback is fired:
        callback(flow_features: dict, anomaly_score: float) -> response_dict
        """
        self._rag_callback = callback
        logger.info("Agentic RAG hook successfully registered.")

    def process_packet(self, raw_pkt: Any) -> Optional[ClassificationDecision]:
        """
        Main inspection pipeline for a single packet.
        Returns the ClassificationDecision, or None if packet could not be parsed.
        """
        # Step 1: Parse packet
        pkt = self.parser.parse(raw_pkt)
        if pkt is None:
            return None

        # Step 2: Emergency PPS tracker
        self.emergency_breaker.record_packet(pkt.src_ip, timestamp=pkt.timestamp)

        # Step 3: Update stateful bidirectional flow
        flow = self.flow_manager.update(pkt)

        # Step 4: Extract 41-dim feature vector
        features = self.feature_extractor.extract_vector(flow)

        # Step 5: Classify traffic
        decision = self.classifier.classify(pkt, flow, features)

        # Step 6: Trigger Agentic RAG if uncertain/anomalous
        if decision.trigger_rag:
            self._handle_rag_trigger(flow, decision)

        # Step 7: Update metrics
        self._record_metrics(decision)

        return decision

    def _handle_rag_trigger(self, flow: NetworkFlow, decision: ClassificationDecision) -> None:
        """Constructs threat payload and notifies registered RAG hook."""
        self.metrics.rag_triggers += 1
        if self._rag_callback:
            rag_features = self.feature_extractor.extract_rag_features(
                flow=flow,
                anomaly_score=decision.anomaly_score,
                extra={"decision_confidence": decision.confidence},
            )
            try:
                # Member 2 RAG execution
                rag_result = self._rag_callback(rag_features, decision.anomaly_score)
                # If RAG returned new rules, they can be dynamically registered
                if isinstance(rag_result, dict) and "rules" in rag_result:
                    for rule_dict in rag_result["rules"]:
                        try:
                            from firewall.rule_engine import FirewallRule
                            rule = FirewallRule.from_dict(rule_dict)
                            self.rule_manager.add_rule(rule)
                        except Exception as e:
                            logger.warning(f"Failed to add dynamic RAG rule: {e}")
            except Exception as e:
                logger.error(f"Error in RAG callback: {e}")

    def _record_metrics(self, decision: ClassificationDecision) -> None:
        """Thread-safe metric updates."""
        with self._lock:
            self.metrics.total_packets_inspected += 1
            if decision.action == RuleAction.ALLOW:
                self.metrics.packets_allowed += 1
            elif decision.action == RuleAction.BLOCK:
                self.metrics.packets_blocked += 1
            elif decision.action == RuleAction.DROP:
                self.metrics.packets_dropped += 1
                if decision.source == "emergency":
                    self.metrics.emergency_drops += 1

            src = decision.source
            if src in self.metrics.decisions_by_source:
                self.metrics.decisions_by_source[src] += 1
            else:
                self.metrics.decisions_by_source[src] = 1

    def run_pcap(self, pcap_path: str, rate_pps: Optional[float] = None) -> FirewallMetrics:
        """Process an entire PCAP file synchronously and return resulting metrics."""
        self.capture.start_pcap_replay(pcap_path, rate_pps=rate_pps, loop=False)
        for raw_pkt in self.capture.stream_packets():
            self.process_packet(raw_pkt)
        return self.get_metrics()

    def start(self) -> None:
        """Start live packet processing in background worker."""
        if self._running:
            return
        self._running = True
        self.capture.start_live()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop firewall capture and background worker."""
        self._running = False
        self.capture.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    def _worker_loop(self) -> None:
        while self._running:
            pkt = self.capture.get_packet(timeout=0.1)
            if pkt is not None:
                self.process_packet(pkt)

    def get_metrics(self) -> FirewallMetrics:
        """Return a snapshot of firewall runtime metrics."""
        with self._lock:
            return FirewallMetrics(
                total_packets_inspected=self.metrics.total_packets_inspected,
                packets_allowed=self.metrics.packets_allowed,
                packets_blocked=self.metrics.packets_blocked,
                packets_dropped=self.metrics.packets_dropped,
                rag_triggers=self.metrics.rag_triggers,
                emergency_drops=self.metrics.emergency_drops,
                decisions_by_source=dict(self.metrics.decisions_by_source),
                start_time=self.metrics.start_time,
            )
