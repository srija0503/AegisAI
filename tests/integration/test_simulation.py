"""
tests/integration/test_simulation.py
────────────────────────────────────
Integration tests for Phase 4 (Simulation + Multi-Node Integration):
- Traffic generators (benign, attack, anomaly)
- EdgeNodeSimulator end-to-end execution
- MultiNodeNetworkSimulator distributed federation
- Member 2 Agentic RAG dynamic rule response hook
- Member 3 Federated Learning payload export
"""

import pytest

from simulation import (
    AnomalyTrafficGenerator,
    AttackTrafficGenerator,
    BenignTrafficGenerator,
    EdgeNodeSimulator,
    MultiNodeNetworkSimulator,
)


def test_benign_traffic_generator():
    gen = BenignTrafficGenerator(target_ip="192.168.1.100", use_scapy=False)
    batch = gen.generate_batch(30)
    assert len(batch) == 30

    for pkt in batch:
        assert pkt["dst_ip"] == "192.168.1.100"
        assert pkt["is_attack"] is False
        assert pkt["protocol"] in ("tcp", "udp", "icmp")
        assert pkt["dst_port"] in (80, 443, 53, 22)


def test_attack_traffic_generator():
    for atk in ["syn_flood", "port_scan", "brute_force", "land_attack"]:
        gen = AttackTrafficGenerator(attack_type=atk, target_ip="192.168.1.100", use_scapy=False)
        pkt = gen.generate_packet()
        assert pkt["is_attack"] is True
        assert pkt["attack_type"] == atk

    # Test Land attack specifics
    land_gen = AttackTrafficGenerator(attack_type="land_attack", target_ip="192.168.1.100", use_scapy=False)
    p_land = land_gen.generate_packet()
    assert p_land["src_ip"] == p_land["dst_ip"] == "192.168.1.100"


def test_anomaly_traffic_generator():
    gen = AnomalyTrafficGenerator(target_ip="192.168.1.100", use_scapy=False)
    batch = gen.generate_batch(15)
    assert len(batch) == 15

    for pkt in batch:
        assert pkt["is_attack"] is True
        assert "zero_day" in pkt["attack_type"]


def test_edge_node_simulator_end_to_end(tmp_path):
    sim = EdgeNodeSimulator(
        node_id="test_edge_node",
        target_ip="192.168.1.100",
    )

    # Run simulation wave
    report = sim.run_simulation(
        num_benign=50,
        num_attacks=40,
        num_anomalies=15,
        attack_type="syn_flood",
    )

    assert report.node_id == "test_edge_node"
    assert report.total_packets >= 105
    assert report.accuracy >= 0.70
    assert report.f1_score >= 0.70
    assert report.blocked_count + report.dropped_count + report.rag_triggers > 0
    assert report.allowed_count > 0

    # RAG should have been triggered for zero-day anomalies
    assert report.rag_triggers > 0

    # Member 3 FL contract check
    fl_payload = sim.export_fl_weights(round_num=1)
    assert fl_payload["node_id"] == "test_edge_node"
    assert fl_payload["round"] == 1
    assert "model_state_dict" in fl_payload
    assert "accuracy" in fl_payload["metrics"]


def test_multi_node_network_simulator():
    sim = MultiNodeNetworkSimulator(
        node_ids=["branch_east", "branch_west", "datacenter_core"]
    )

    summary = sim.run_distributed_simulation(
        benign_per_node=30,
        attacks_per_node=20,
        anomalies_per_node=10,
    )

    assert len(summary.nodes) == 3
    assert summary.total_packets_inspected >= 180
    assert summary.total_attacks_blocked > 0
    assert summary.total_rag_escalations > 0
    assert summary.global_average_accuracy >= 0.70

    # Federated learning payloads exported for all 3 nodes
    assert len(summary.federated_payloads) == 3
    for payload in summary.federated_payloads:
        assert "model_state_dict" in payload
        assert "node_id" in payload
