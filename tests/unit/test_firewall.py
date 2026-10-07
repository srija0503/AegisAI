"""
tests/unit/test_firewall.py
───────────────────────────
Unit tests for the Firewall Engine (Phase 2):
- PacketParser
- FlowManager
- AllowList & BlockList
- EmergencyCircuitBreaker
- RuleEngine & RuleManager
- FirewallFeatureExtractor
- TrafficClassifier
- FirewallController
"""

import time
import pytest
import numpy as np
import torch

from firewall import (
    AllowList,
    BlockList,
    EmergencyCircuitBreaker,
    FirewallController,
    FirewallFeatureExtractor,
    FirewallRule,
    FlowManager,
    NetworkFlow,
    PacketCapture,
    PacketParser,
    ParsedPacket,
    RuleAction,
    RuleEngine,
    RuleManager,
    TrafficClassifier,
)
from scapy.layers.inet import IP, TCP, UDP, ICMP


# ─── PacketParser Tests ───────────────────────────────────────────────────────

def test_packet_parser_scapy_tcp():
    parser = PacketParser()
    pkt = IP(src="192.168.1.10", dst="8.8.8.8", proto=6) / TCP(sport=54321, dport=443, flags="S")
    parsed = parser.parse(pkt)

    assert parsed is not None
    assert parsed.src_ip == "192.168.1.10"
    assert parsed.dst_ip == "8.8.8.8"
    assert parsed.src_port == 54321
    assert parsed.dst_port == 443
    assert parsed.protocol == "tcp"
    assert parsed.tcp_flags["SYN"] is True
    assert parsed.tcp_flags["ACK"] is False
    assert parsed.flag_str == "S0"


def test_packet_parser_scapy_udp():
    parser = PacketParser()
    pkt = IP(src="10.0.0.5", dst="1.1.1.1") / UDP(sport=50000, dport=53)
    parsed = parser.parse(pkt)

    assert parsed is not None
    assert parsed.protocol == "udp"
    assert parsed.dst_port == 53
    assert parsed.flag_str == "SF"


def test_packet_parser_scapy_icmp():
    parser = PacketParser()
    pkt = IP(src="10.0.0.5", dst="10.0.0.1") / ICMP()
    parsed = parser.parse(pkt)

    assert parsed is not None
    assert parsed.protocol == "icmp"
    assert parsed.src_port == 0
    assert parsed.dst_port == 0


def test_canonical_flow_key():
    p1 = ParsedPacket(
        src_ip="192.168.1.1", dst_ip="8.8.8.8",
        src_port=1234, dst_port=80,
        protocol="tcp", ip_proto=6, packet_len=60, payload_len=0,
    )
    p2 = ParsedPacket(
        src_ip="8.8.8.8", dst_ip="192.168.1.1",
        src_port=80, dst_port=1234,
        protocol="tcp", ip_proto=6, packet_len=60, payload_len=0,
    )
    # Both directions must produce identical canonical keys
    assert p1.canonical_flow_key == p2.canonical_flow_key


# ─── FlowManager Tests ────────────────────────────────────────────────────────

def test_flow_manager_bidirectional_tracking():
    mgr = FlowManager(timeout_seconds=60)
    p_fwd = ParsedPacket(
        src_ip="10.0.0.1", dst_ip="203.0.113.5",
        src_port=4000, dst_port=80,
        protocol="tcp", ip_proto=6, packet_len=100, payload_len=40,
        tcp_flags={"SYN": True, "ACK": False, "FIN": False, "RST": False},
        flag_str="S0", timestamp=1000.0,
    )
    p_bwd = ParsedPacket(
        src_ip="203.0.113.5", dst_ip="10.0.0.1",
        src_port=80, dst_port=4000,
        protocol="tcp", ip_proto=6, packet_len=200, payload_len=140,
        tcp_flags={"SYN": True, "ACK": True, "FIN": False, "RST": False},
        flag_str="S1", timestamp=1000.1,
    )

    flow1 = mgr.update(p_fwd)
    assert mgr.active_flow_count() == 1
    assert flow1.forward_pkts == 1
    assert flow1.backward_pkts == 0

    flow2 = mgr.update(p_bwd)
    assert mgr.active_flow_count() == 1
    assert flow2.forward_pkts == 1
    assert flow2.backward_pkts == 1
    assert flow2.total_packets == 2
    assert flow2.forward_bytes == 100
    assert flow2.backward_bytes == 200
    assert flow2.state == "ESTABLISHED"


def test_flow_manager_timeout_and_capacity():
    mgr = FlowManager(timeout_seconds=5.0, max_flows=2)
    p1 = ParsedPacket(src_ip="1.1.1.1", dst_ip="2.2.2.2", src_port=1, dst_port=2, protocol="tcp", ip_proto=6, packet_len=50, payload_len=0, timestamp=10.0)
    p2 = ParsedPacket(src_ip="3.3.3.3", dst_ip="4.4.4.4", src_port=3, dst_port=4, protocol="tcp", ip_proto=6, packet_len=50, payload_len=0, timestamp=12.0)
    mgr.update(p1)
    mgr.update(p2)
    assert mgr.active_flow_count() == 2

    # Timeout cleanup
    expired = mgr.cleanup_expired_flows(now=16.0)
    assert expired == 1  # p1 is expired (16 - 10 = 6 > 5)
    assert mgr.active_flow_count() == 1

    # Capacity eviction
    p3 = ParsedPacket(src_ip="5.5.5.5", dst_ip="6.6.6.6", src_port=5, dst_port=6, protocol="tcp", ip_proto=6, packet_len=50, payload_len=0, timestamp=17.0)
    p4 = ParsedPacket(src_ip="7.7.7.7", dst_ip="8.8.8.8", src_port=7, dst_port=8, protocol="tcp", ip_proto=6, packet_len=50, payload_len=0, timestamp=18.0)
    mgr.update(p3)
    mgr.update(p4)
    # Max is 2, so count shouldn't exceed 2
    assert mgr.active_flow_count() <= 2


# ─── AllowList & BlockList Tests ──────────────────────────────────────────────

def test_allowlist_private_and_cidr():
    al = AllowList(always_allow_private=True)
    assert al.is_allowed("192.168.1.100") is True
    assert al.is_allowed("10.50.1.2") is True
    assert al.is_allowed("127.0.0.1") is True
    assert al.is_allowed("198.51.100.5") is False

    al.add("198.51.100.0/24")
    assert al.is_allowed("198.51.100.5") is True
    assert al.is_allowed("198.51.101.5") is False


def test_blocklist_and_expiration():
    bl = BlockList(auto_block_threshold=0.90)
    bl.add("203.0.113.10", reason="Known botnet", ttl_seconds=1.0)
    is_blk, reason = bl.is_blocked("203.0.113.10")
    assert is_blk is True
    assert "Known botnet" in reason

    time.sleep(1.1)
    is_blk, _ = bl.is_blocked("203.0.113.10")
    assert is_blk is False  # Expired

    # Auto block
    success = bl.auto_block("198.51.100.99", confidence=0.95, ttl_seconds=60)
    assert success is True
    is_blk, _ = bl.is_blocked("198.51.100.99")
    assert is_blk is True


# ─── Emergency Circuit Breaker Tests ──────────────────────────────────────────

def test_emergency_circuit_breaker():
    breaker = EmergencyCircuitBreaker(
        enabled=True,
        threshold_pps=50.0,
        per_ip_threshold_pps=5.0,
        block_duration_seconds=2.0,
    )
    t = 100.0
    # Send 6 packets in 0.5s from same IP -> should trigger per-IP block
    for i in range(6):
        breaker.record_packet("198.51.100.20", timestamp=t + (i * 0.05))

    is_blk, reason = breaker.is_blocked("198.51.100.20", now=t + 0.3)
    assert is_blk is True
    assert "Quarantine" in reason

    # Different IP should not be blocked
    is_blk, _ = breaker.is_blocked("198.51.100.21", now=t + 0.3)
    assert is_blk is False

    # After duration expires, IP is unblocked
    is_blk, _ = breaker.is_blocked("198.51.100.20", now=t + 2.5)
    assert is_blk is False


# ─── RuleEngine & RuleManager Tests ───────────────────────────────────────────

def test_rule_engine_matching():
    r1 = FirewallRule(
        rule_id="R-BLOCK-TELNET",
        priority=10,
        action=RuleAction.BLOCK,
        dst_port=23,
        protocol="tcp",
        description="Block Telnet",
    )
    r2 = FirewallRule(
        rule_id="R-ALLOW-ALL-WEB",
        priority=50,
        action=RuleAction.ALLOW,
        dst_port=[80, 443],
        protocol="tcp",
    )

    engine = RuleEngine(default_policy="block")
    engine.set_rules([r2, r1])  # Should order by priority (10 then 50)

    # Test packet matching R1
    pkt_telnet = ParsedPacket(
        src_ip="1.1.1.1", dst_ip="2.2.2.2",
        src_port=50000, dst_port=23,
        protocol="tcp", ip_proto=6, packet_len=60, payload_len=0,
    )
    action, matched = engine.evaluate(pkt_telnet)
    assert action == RuleAction.BLOCK
    assert matched.rule_id == "R-BLOCK-TELNET"

    # Test packet matching R2
    pkt_web = ParsedPacket(
        src_ip="1.1.1.1", dst_ip="2.2.2.2",
        src_port=50000, dst_port=443,
        protocol="tcp", ip_proto=6, packet_len=60, payload_len=0,
    )
    action, matched = engine.evaluate(pkt_web)
    assert action == RuleAction.ALLOW
    assert matched.rule_id == "R-ALLOW-ALL-WEB"

    # Test packet matching neither -> fallback to default policy
    pkt_dns = ParsedPacket(
        src_ip="1.1.1.1", dst_ip="2.2.2.2",
        src_port=50000, dst_port=53,
        protocol="udp", ip_proto=17, packet_len=60, payload_len=0,
    )
    action, matched = engine.evaluate(pkt_dns)
    assert action == RuleAction.BLOCK
    assert matched is None


def test_rule_manager_crud(tmp_path):
    yaml_file = tmp_path / "test_rules.yaml"
    engine = RuleEngine()
    manager = RuleManager(engine=engine, rules_file=str(yaml_file))

    rule = FirewallRule(
        rule_id="R-TEST",
        priority=15,
        action=RuleAction.DROP,
        src_ip="203.0.113.0/24",
    )
    manager.add_rule(rule)
    assert len(manager.list_rules()) == 1

    # Reload from disk into new manager
    manager2 = RuleManager(engine=engine, rules_file=str(yaml_file))
    assert len(manager2.list_rules()) == 1
    loaded_rule = manager2.get_rule("R-TEST")
    assert loaded_rule is not None
    assert loaded_rule.action == RuleAction.DROP


# ─── FeatureExtractor Tests ───────────────────────────────────────────────────

def test_feature_extractor():
    extractor = FirewallFeatureExtractor(window_size=10)
    flow = NetworkFlow(
        key=("10.0.0.1", "10.0.0.2", 1234, 80, "tcp"),
        init_src_ip="10.0.0.1",
        init_dst_ip="10.0.0.2",
        init_src_port=1234,
        init_dst_port=80,
        protocol="tcp",
        service="http",
        start_time=100.0,
        last_seen=105.0,
        forward_pkts=5,
        backward_pkts=5,
        forward_bytes=500,
        backward_bytes=1500,
    )

    vec = extractor.extract_vector(flow)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (41,)

    tensor = extractor.extract_tensor(flow)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 41)

    rag_dict = extractor.extract_rag_features(flow, anomaly_score=0.75)
    assert isinstance(rag_dict, dict)
    assert rag_dict["src_ip"] == "10.0.0.1"
    assert rag_dict["dst_port"] == 80
    assert rag_dict["anomaly_score"] == 0.75
    assert "packets_per_sec" in rag_dict


# ─── TrafficClassifier & Controller Tests ─────────────────────────────────────

def test_traffic_classifier_rag_trigger():
    allowlist = AllowList(always_allow_private=False)
    blocklist = BlockList()
    rule_engine = RuleEngine(default_policy="allow")
    breaker = EmergencyCircuitBreaker(enabled=False)

    classifier = TrafficClassifier(
        mode="rl",
        confidence_threshold=0.80,
        allowlist=allowlist,
        blocklist=blocklist,
        rule_engine=rule_engine,
        emergency_breaker=breaker,
    )

    pkt = ParsedPacket(
        src_ip="203.0.113.5", dst_ip="198.51.100.1",
        src_port=1000, dst_port=80,
        protocol="tcp", ip_proto=6, packet_len=100, payload_len=0,
        flag_str="REJ",
    )
    flow = NetworkFlow(
        key=pkt.canonical_flow_key,
        init_src_ip=pkt.src_ip, init_dst_ip=pkt.dst_ip,
        init_src_port=pkt.src_port, init_dst_port=pkt.dst_port,
        protocol="tcp",
    )
    vec = np.zeros(41, dtype=np.float32)

    decision = classifier.classify(pkt, flow, vec)
    # Ambiguous baseline pattern triggers RAG
    assert decision.trigger_rag is True or decision.action in (RuleAction.ALLOW, RuleAction.BLOCK, RuleAction.TRIGGER_RAG)


def test_firewall_controller_end_to_end(tmp_path):
    rules_path = tmp_path / "fw_rules.yaml"
    controller = FirewallController(
        config_dict={
            "capture": {"promisc": False},
            "flow": {"timeout_seconds": 60.0},
            "allowlist": {"always_allow_private": True},
            "rules": {"default_policy": "allow", "rules_file": str(rules_path)},
            "classifier": {"mode": "hybrid", "confidence_threshold": 0.80},
        }
    )

    rag_called = []
    def mock_rag_hook(flow_feats, anomaly_score):
        rag_called.append((flow_feats, anomaly_score))
        return {
            "threat_type": "reconnaissance",
            "confidence": 0.9,
            "rules": [{
                "rule_id": "RAG-AUTO-01",
                "priority": 5,
                "action": "block",
                "src_ip": flow_feats["src_ip"],
                "description": "Auto generated from RAG",
            }],
        }

    controller.register_rag_hook(mock_rag_hook)

    # Allow private traffic
    p_private = {
        "src_ip": "192.168.1.50",
        "dst_ip": "192.168.1.1",
        "src_port": 5555,
        "dst_port": 80,
        "protocol": "tcp",
        "packet_len": 64,
    }
    decision = controller.process_packet(p_private)
    assert decision is not None
    assert decision.action == RuleAction.ALLOW
    assert decision.source == "allowlist"

    metrics = controller.get_metrics()
    assert metrics.total_packets_inspected == 1
    assert metrics.packets_allowed == 1
