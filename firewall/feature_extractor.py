"""
firewall/feature_extractor.py
─────────────────────────────
Extracts feature vectors from live network flows for:
1. RL Agent inference: 41-dimensional normalized vector (matching NSL-KDD schema).
2. Agentic RAG (Member 2 interface): Rich metadata dictionary detailing
   threat indicators, connection statistics, and behavioral context.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from datasets.feature_extraction import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FlowStats,
    OnlineFeatureExtractor,
)
from firewall.flow_manager import NetworkFlow
from firewall.packet_parser import ParsedPacket


class FirewallFeatureExtractor:
    """
    Transforms NetworkFlow instances into vector representations for RL
    and dictionary representations for RAG threat intelligence.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._online_extractor = OnlineFeatureExtractor(window_size=window_size)
        self._recent_stats: list[FlowStats] = []

    def _flow_to_stats(self, flow: NetworkFlow) -> FlowStats:
        """Convert a NetworkFlow into the FlowStats dataclass expected by OnlineFeatureExtractor."""
        return FlowStats(
            src_ip=flow.init_src_ip,
            dst_ip=flow.init_dst_ip,
            src_port=flow.init_src_port,
            dst_port=flow.init_dst_port,
            protocol=flow.protocol,
            service=flow.service,
            flag=flow.flag_str,
            start_time=flow.start_time,
            last_time=flow.last_seen,
            src_bytes=flow.forward_bytes,
            dst_bytes=flow.backward_bytes,
            packet_count=flow.total_packets,
            syn_count=flow.syn_count,
            fin_count=flow.fin_count,
            rst_count=flow.rst_count,
        )

    def extract_vector(self, flow: NetworkFlow) -> np.ndarray:
        """
        Extract the 41-dimensional NSL-KDD feature vector (float32, shape (41,)).
        """
        stats = self._flow_to_stats(flow)
        vec = self._online_extractor.extract_from_flow(stats, self._recent_stats)

        # Maintain rolling window for rate computations
        self._recent_stats.append(stats)
        if len(self._recent_stats) > self.window_size * 2:
            self._recent_stats = self._recent_stats[-self.window_size:]

        return vec

    def extract_tensor(self, flow: NetworkFlow, device: str = "cpu") -> torch.Tensor:
        """
        Extract features as a PyTorch Tensor of shape (1, 41) for model inference.
        """
        vec = self.extract_vector(flow)
        return torch.from_numpy(vec).unsqueeze(0).to(device)

    def extract_rag_features(
        self,
        flow: NetworkFlow,
        anomaly_score: float = 0.0,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Produces the rich flow_features dictionary contract for Member 2's Agentic RAG:
        `RAGOrchestrator.analyze(flow_features: dict, anomaly_score: float)`
        """
        duration = flow.duration
        pps = (flow.total_packets / duration) if duration > 0.001 else float(flow.total_packets)
        bps = (flow.total_bytes / duration) if duration > 0.001 else float(flow.total_bytes)

        features: dict[str, Any] = {
            "src_ip": flow.init_src_ip,
            "dst_ip": flow.init_dst_ip,
            "src_port": flow.init_src_port,
            "dst_port": flow.init_dst_port,
            "protocol": flow.protocol.upper(),
            "service": flow.service,
            "duration": round(duration, 4),
            "forward_packets": flow.forward_pkts,
            "backward_packets": flow.backward_pkts,
            "total_packets": flow.total_packets,
            "forward_bytes": flow.forward_bytes,
            "backward_bytes": flow.backward_bytes,
            "total_bytes": flow.total_bytes,
            "packets_per_sec": round(pps, 2),
            "bytes_per_sec": round(bps, 2),
            "tcp_flags": {
                "syn": flow.syn_count,
                "fin": flow.fin_count,
                "rst": flow.rst_count,
                "ack": flow.ack_count,
                "urg": flow.urg_count,
            },
            "kdd_flag": flow.flag_str,
            "connection_state": flow.state,
            "anomaly_score": round(float(anomaly_score), 4),
            "timestamp": flow.last_seen,
        }

        if extra:
            features.update(extra)

        return features
