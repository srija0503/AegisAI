"""
agentic_rag/tools/threat_intel.py
─────────────────────────────────
RAG Tool: Queries threat intelligence feeds, MITRE ATT&CK techniques,
and indicators of compromise (IoCs) for real-time anomaly contextualization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from knowledge_base.vector_store.search import VectorSearchEngine

logger = logging.getLogger(__name__)


class ThreatIntelTool:
    """
    Tool querying MITRE ATT&CK tactics, malicious port activities,
    and attack signatures.
    """

    def __init__(self, search_engine: Optional[VectorSearchEngine] = None) -> None:
        self.search_engine = search_engine or VectorSearchEngine()

    def query_intel(
        self,
        query: str,
        protocol: Optional[str] = None,
        port: Optional[int] = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search threat intelligence documents."""
        results = self.search_engine.search(
            query=query,
            protocol=protocol,
            port=port,
            top_k=top_k,
        )
        return [r.to_dict() for r in results]

    def check_suspicious_port(self, port: int, protocol: str = "TCP") -> dict[str, Any]:
        """Provides quick risk analysis on commonly targeted network service ports."""
        high_risk_ports = {
            22: {"service": "SSH", "common_attacks": ["Brute Force (T1110)", "Credential Stuffing"], "risk": "HIGH"},
            23: {"service": "Telnet", "common_attacks": ["Mirai Botnet", "Cleartext Sniffing"], "risk": "CRITICAL"},
            25: {"service": "SMTP", "common_attacks": ["Open Relay Abuse", "Spam Flooding"], "risk": "MEDIUM"},
            53: {"service": "DNS", "common_attacks": ["DNS Amplification (T1498)", "DNS Tunneling"], "risk": "HIGH"},
            80: {"service": "HTTP", "common_attacks": ["Slowloris", "Web Application Attacks"], "risk": "HIGH"},
            135: {"service": "MSRPC", "common_attacks": ["Zerologon", "Lateral Movement"], "risk": "CRITICAL"},
            139: {"service": "NetBIOS", "common_attacks": ["SMB Exploits", "Null Session Recon"], "risk": "CRITICAL"},
            443: {"service": "HTTPS", "common_attacks": ["SSL Renegotiation DoS", "Encrypted C2"], "risk": "HIGH"},
            445: {"service": "SMB", "common_attacks": ["EternalBlue (CVE-2017-0144)", "Ransomware Lateral Movement"], "risk": "CRITICAL"},
            3389: {"service": "RDP", "common_attacks": ["BlueKeep (CVE-2019-0708)", "RDP Brute Force"], "risk": "CRITICAL"},
        }
        info = high_risk_ports.get(port, {"service": "Custom/Unknown", "common_attacks": ["General Port Scan"], "risk": "MEDIUM" if port < 1024 else "LOW"})
        return {
            "port": port,
            "protocol": protocol.upper(),
            "service": info["service"],
            "risk_level": info["risk"],
            "common_attacks": info["common_attacks"],
        }

    def get_tool_metadata(self) -> dict[str, Any]:
        return {
            "name": "threat_intel",
            "description": "Queries MITRE ATT&CK tactics, IoCs, and common target port risk profiles.",
            "parameters": {
                "query": "string (attack tactic or signature terms)",
                "port": "optional integer port number",
            },
        }
