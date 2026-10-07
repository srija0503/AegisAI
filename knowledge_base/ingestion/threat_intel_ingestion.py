"""
knowledge_base/ingestion/threat_intel_ingestion.py
─────────────────────────────────────────────────
Ingestion engine for Threat Intelligence feeds, including MITRE ATT&CK techniques,
CISA Known Exploited Vulnerabilities (KEV), and network intrusion patterns (SYN floods,
port scanning, DNS amplification, botnets).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge_base.processing.metadata import DocumentMetadata, SeverityLevel, ThreatSource
from knowledge_base.vector_store.index import VectorIndex

logger = logging.getLogger(__name__)


# Curated network attack techniques & signatures (MITRE ATT&CK & Real-world threat intel)
DEFAULT_NETWORK_INTEL_RECORDS: list[dict[str, Any]] = [
    {
        "title": "MITRE T1498: Network Denial of Service (SYN Flood / UDP Reflection)",
        "description": "Adversaries may perform Network Denial of Service (DoS) attacks to degrade or disrupt the availability of targeted systems. In SYN flood attacks, an attacker rapidly transmits TCP SYN requests without completing the 3-way handshake (missing ACK), exhausting OS socket connection tables (SYN backlog queue). In UDP amplification, spoofed requests trigger massive responses.",
        "cvss_score": 7.5,
        "severity": "HIGH",
        "affected_protocols": ["TCP", "UDP"],
        "affected_ports": [80, 443, 53, 123],
        "attack_vector": "NETWORK",
        "mitigation": "Enable SYN cookies at the TCP stack. Deploy rate-limiting per source IP (threshold 500 pps). Drop half-open connections after 3 seconds. Block unverified external UDP traffic to service ports.",
        "tags": ["dos", "ddos", "syn-flood", "amplification", "mitre-t1498"],
    },
    {
        "title": "MITRE T1046: Network Service Discovery (Port Scanning / SYN Sweep)",
        "description": "Adversaries may attempt to get a listing of services running on remote hosts, including using tools like Nmap, Masscan, or custom stealth scanners. Characterized by high connection attempt rates to sequential or random destination ports with SYN or FIN flags and minimal or zero payload data.",
        "cvss_score": 5.3,
        "severity": "MEDIUM",
        "affected_protocols": ["TCP", "UDP"],
        "affected_ports": [21, 22, 23, 25, 80, 443, 445, 3389, 8080],
        "attack_vector": "NETWORK",
        "mitigation": "Detect and quarantine source IPs connecting to more than 15 unique ports within a 1-second window. Deploy honeypot ports to trigger immediate emergency blocklist additions.",
        "tags": ["reconnaissance", "port-scan", "syn-scan", "nmap", "mitre-t1046"],
    },
    {
        "title": "MITRE T1110: Brute Force Authentication (SSH / RDP / Telnet)",
        "description": "Adversaries may use brute force techniques to attempt credential guessing across remote management protocols. Characterized by repetitive TCP connections to ports 22, 3389, or 23 from a single IP or distributed botnet, yielding high RST counts or brief session durations with constant packet sizes.",
        "cvss_score": 8.1,
        "severity": "HIGH",
        "affected_protocols": ["TCP"],
        "affected_ports": [22, 3389, 23],
        "attack_vector": "NETWORK",
        "mitigation": "Enforce maximum 5 connection attempts per minute per source IP. Auto-block failing source IPs for 3600 seconds. Require MFA and key-based authentication.",
        "tags": ["credential-access", "brute-force", "ssh", "rdp", "mitre-t1110"],
    },
    {
        "title": "Zero-Day Attack Pattern: DNS Tunneling & C2 Exfiltration",
        "description": "Command and control communication or sensitive data exfiltration cloaked inside standard DNS query/response structures. Characterized by abnormal volume of high-entropy subdomain lookups directed to external authoritative name servers over UDP port 53, with TXT and NULL record queries.",
        "cvss_score": 8.6,
        "severity": "HIGH",
        "affected_protocols": ["UDP"],
        "affected_ports": [53],
        "attack_vector": "NETWORK",
        "mitigation": "Enforce DNS inspection on perimeter firewalls. Block direct external outbound UDP 53 from internal endpoints; restrict outbound DNS strictly to designated internal recursive resolvers.",
        "tags": ["zero-day", "dns-tunneling", "c2", "exfiltration"],
    },
    {
        "title": "Zero-Day Attack Pattern: HTTP Slowloris / Low-and-Slow Attack",
        "description": "A denial of service attack where an attacker sends partial HTTP headers continuously at slow intervals without ever completing the request, monopolizing web server thread pools and connection sockets while maintaining minimal bandwidth consumption, bypassing basic volumetric DDoS threshold detectors.",
        "cvss_score": 7.5,
        "severity": "HIGH",
        "affected_protocols": ["TCP", "HTTP"],
        "affected_ports": [80, 443, 8080],
        "attack_vector": "NETWORK",
        "mitigation": "Enforce minimum transfer rate requirements and aggressive HTTP header read timeouts (max 5 seconds). Limit maximum concurrent active connections per individual source IP address.",
        "tags": ["slowloris", "low-and-slow", "dos", "http"],
    },
    {
        "title": "Mirai Botnet Telnet/HTTP Spread Signatures",
        "description": "Automated malware propagation scanning default credentials on IoT ports (23, 2323, 80, 8080). Sends rapid burst TCP SYN probes followed by automated telnet login sequences to recruit devices into massive distributed DDoS swarms.",
        "cvss_score": 9.1,
        "severity": "CRITICAL",
        "affected_protocols": ["TCP"],
        "affected_ports": [23, 2323, 8080],
        "attack_vector": "NETWORK",
        "mitigation": "Immediately isolate internal devices originating port 23/2323 outbound traffic. Block all external WAN access to IoT management interfaces.",
        "tags": ["mirai", "botnet", "iot", "worm"],
    },
]


class ThreatIntelIngestionEngine:
    """
    Ingests threat intelligence, MITRE ATT&CK patterns, and IoC feeds.
    """

    def __init__(self, index: VectorIndex, cache_dir: str = "./data/raw/threat_intel") -> None:
        self.index = index
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_seed_threat_intel(self) -> int:
        """Loads curated MITRE ATT&CK and network attack signatures into the vector index."""
        logger.info(f"Loading {len(DEFAULT_NETWORK_INTEL_RECORDS)} threat intelligence patterns...")
        return self.index.index_raw_records(DEFAULT_NETWORK_INTEL_RECORDS)

    def ingest_cisa_kev_file(self, file_path: str | Path) -> int:
        """
        Parses a CISA Known Exploited Vulnerabilities (KEV) JSON catalog.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"CISA KEV file {path} not found.")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vulnerabilities = data.get("vulnerabilities", [])
        records = []
        for v in vulnerabilities:
            records.append({
                "cve_id": v.get("cveID"),
                "title": f"CISA KEV: {v.get('vulnerabilityName')}",
                "description": v.get("shortDescription", ""),
                "mitigation": v.get("requiredAction", ""),
                "severity": "CRITICAL",
                "cvss_score": 9.5,
                "source": "cisa_kev",
                "tags": ["cisa-kev", "actively-exploited"],
            })

        return self.index.index_raw_records(records)
