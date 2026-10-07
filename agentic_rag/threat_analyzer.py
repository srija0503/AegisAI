"""
agentic_rag/threat_analyzer.py
──────────────────────────────
Heuristic and telemetry-based threat analyzer for packet flow features.
Detects attack classes (SYN flood, port scan, brute force, zero-day anomalies)
and computes initial indicators of compromise for the RAG reasoner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ThreatAssessment:
    """Pre-reasoning behavioral threat analysis."""
    threat_name: str
    category: str              # DoS, Reconnaissance, Brute Force, Exploit, Anomaly
    severity: str              # CRITICAL, HIGH, MEDIUM, LOW
    confidence: float          # 0.0 to 1.0
    key_indicators: list[str]
    suggested_action: str      # block, drop, alert
    protocol: str
    src_ip: str
    dst_port: Optional[int] = None


class ThreatAnalyzer:
    """
    Analyzes raw flow metrics into security indicators.
    """

    def analyze_flow(self, flow: dict[str, Any], anomaly_score: float) -> ThreatAssessment:
        src_ip = str(flow.get("src_ip", "0.0.0.0"))
        dst_port = flow.get("dst_port")
        proto = str(flow.get("protocol", "TCP")).upper()
        pps = float(flow.get("packets_per_sec", 0.0))
        bps = float(flow.get("bytes_per_sec", 0.0))
        duration = float(flow.get("duration", 0.0))
        kdd_flag = str(flow.get("kdd_flag", ""))
        tcp_flags = flow.get("tcp_flags", {})

        syn = tcp_flags.get("syn", 0)
        ack = tcp_flags.get("ack", 0)
        rst = tcp_flags.get("rst", 0)
        fin = tcp_flags.get("fin", 0)

        indicators: list[str] = []

        # 1. SYN Flood Detection
        if proto == "TCP" and syn > 10 and ack == 0:
            indicators.append(f"High SYN/ACK disparity ({syn} SYN, 0 ACK)")
            if pps > 500:
                indicators.append(f"Volumetric packet rate: {pps} pps")
            return ThreatAssessment(
                threat_name="TCP SYN Flood (DoS / T1498)",
                category="DoS",
                severity="HIGH" if pps < 1000 else "CRITICAL",
                confidence=min(1.0, 0.70 + (pps / 5000.0)),
                key_indicators=indicators,
                suggested_action="drop",
                protocol=proto,
                src_ip=src_ip,
                dst_port=dst_port,
            )

        # 2. Reconnaissance / Port Scan
        if kdd_flag in ("S0", "REJ", "RSTO") or (rst > 5 and duration < 1.0):
            indicators.append(f"Connection state rejected/reset ({kdd_flag})")
            if duration < 0.5:
                indicators.append("Ultra-short connection probe duration")
            return ThreatAssessment(
                threat_name="Port Reconnaissance / SYN Sweep (T1046)",
                category="Reconnaissance",
                severity="MEDIUM",
                confidence=0.85,
                key_indicators=indicators,
                suggested_action="block",
                protocol=proto,
                src_ip=src_ip,
                dst_port=dst_port,
            )

        # 3. High Risk Exploited Ports
        if dst_port == 445:
            indicators.append("Targeting Microsoft SMB port 445 (EternalBlue vector)")
            return ThreatAssessment(
                threat_name="SMB Lateral Movement / Remote Exploit",
                category="Exploit",
                severity="CRITICAL",
                confidence=0.90,
                key_indicators=indicators,
                suggested_action="block",
                protocol=proto,
                src_ip=src_ip,
                dst_port=dst_port,
            )
        elif dst_port == 3389 and rst > 3:
            indicators.append("Targeting RDP port 3389 with repeated resets")
            return ThreatAssessment(
                threat_name="RDP Authentication Brute Force (T1110)",
                category="Brute Force",
                severity="HIGH",
                confidence=0.88,
                key_indicators=indicators,
                suggested_action="block",
                protocol=proto,
                src_ip=src_ip,
                dst_port=dst_port,
            )
        elif dst_port == 22 and (syn > 5 or rst > 3):
            indicators.append("Targeting SSH port 22 with burst connections")
            return ThreatAssessment(
                threat_name="SSH Brute Force Intrusion",
                category="Brute Force",
                severity="HIGH",
                confidence=0.85,
                key_indicators=indicators,
                suggested_action="block",
                protocol=proto,
                src_ip=src_ip,
                dst_port=dst_port,
            )

        # 4. Volumetric Anomaly
        if pps > 2000.0 or bps > 5000000.0:
            indicators.append(f"Excessive traffic volume: {pps} pps, {bps} bps")
            return ThreatAssessment(
                threat_name="Volumetric Bandwidth Flooding Attack",
                category="DoS",
                severity="CRITICAL",
                confidence=0.92,
                key_indicators=indicators,
                suggested_action="drop",
                protocol=proto,
                src_ip=src_ip,
                dst_port=dst_port,
            )

        # 5. Default Zero-Day / Novel Anomaly
        indicators.append(f"Unclassified statistical deviation (anomaly_score: {anomaly_score:.3f})")
        return ThreatAssessment(
            threat_name="Zero-Day Suspicious Behavioral Anomaly",
            category="Anomaly",
            severity="MEDIUM" if anomaly_score < 0.80 else "HIGH",
            confidence=float(anomaly_score),
            key_indicators=indicators,
            suggested_action="alert",
            protocol=proto,
            src_ip=src_ip,
            dst_port=dst_port,
        )
