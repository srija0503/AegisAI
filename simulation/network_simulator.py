"""
simulation/network_simulator.py
───────────────────────────────
Multi-node network simulator coordinating a federation of edge firewall nodes.
Simulates distributed enterprise branches under concurrent attack waves and
produces the multi-node federated training payload for Member 3's FL aggregation server.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from simulation.node_simulator import EdgeNodeSimulator, NodeSimulationReport

logger = logging.getLogger(__name__)


@dataclass
class NetworkSimulationSummary:
    nodes: dict[str, NodeSimulationReport]
    total_packets_inspected: int
    total_attacks_blocked: int
    total_rag_escalations: int
    global_average_accuracy: float
    global_average_f1: float
    federated_payloads: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_packets_inspected": self.total_packets_inspected,
            "total_attacks_blocked": self.total_attacks_blocked,
            "total_rag_escalations": self.total_rag_escalations,
            "global_average_accuracy": round(self.global_average_accuracy, 4),
            "global_average_f1": round(self.global_average_f1, 4),
            "nodes": {nid: report.to_dict() for nid, report in self.nodes.items()},
        }


class MultiNodeNetworkSimulator:
    """
    Coordinates multi-edge firewall network topologies.
    """

    def __init__(
        self,
        node_ids: Optional[list[str]] = None,
        config_path: str = "./config/firewall.yaml",
        model_path: Optional[str] = "./rl_agent/models/checkpoints/best_model.pt",
    ) -> None:
        self.node_ids = node_ids or [
            "edge_node_branch_a",
            "edge_node_branch_b",
            "edge_node_datacenter",
        ]

        self.nodes: dict[str, EdgeNodeSimulator] = {
            nid: EdgeNodeSimulator(
                node_id=nid,
                config_path=config_path,
                model_path=model_path,
                target_ip=f"10.0.{i+1}.10",
            )
            for i, nid in enumerate(self.node_ids)
        }

    def run_distributed_simulation(
        self,
        benign_per_node: int = 150,
        attacks_per_node: int = 100,
        anomalies_per_node: int = 40,
        attack_types: Optional[list[str]] = None,
    ) -> NetworkSimulationSummary:
        """
        Executes attack and benign traffic streams against all participating edge nodes.
        """
        types = attack_types or ["syn_flood", "port_scan", "brute_force"]
        reports: dict[str, NodeSimulationReport] = {}
        fl_payloads: list[dict[str, Any]] = []

        total_pkts = 0
        total_blocked = 0
        total_rag = 0
        f1_list = []
        acc_list = []

        for i, (nid, simulator) in enumerate(self.nodes.items()):
            atk_type = types[i % len(types)]
            report = simulator.run_simulation(
                num_benign=benign_per_node,
                num_attacks=attacks_per_node,
                num_anomalies=anomalies_per_node,
                attack_type=atk_type,
            )
            reports[nid] = report
            total_pkts += report.total_packets
            total_blocked += (report.blocked_count + report.dropped_count + report.rag_triggers)
            total_rag += report.rag_triggers
            f1_list.append(report.f1_score)
            acc_list.append(report.accuracy)

            # Export local model weights for Member 3
            fl_payloads.append(simulator.export_fl_weights(round_num=1))

        avg_acc = float(sum(acc_list) / len(acc_list)) if acc_list else 0.0
        avg_f1 = float(sum(f1_list) / len(f1_list)) if f1_list else 0.0

        return NetworkSimulationSummary(
            nodes=reports,
            total_packets_inspected=total_pkts,
            total_attacks_blocked=total_blocked,
            total_rag_escalations=total_rag,
            global_average_accuracy=avg_acc,
            global_average_f1=avg_f1,
            federated_payloads=fl_payloads,
        )
