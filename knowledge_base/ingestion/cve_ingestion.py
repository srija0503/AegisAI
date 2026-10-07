"""
knowledge_base/ingestion/cve_ingestion.py
────────────────────────────────────────
Ingestion engine for Common Vulnerabilities and Exposures (CVE) feeds.
Parses official MITRE CVE records, local JSON databases, and provides
a curated seed corpus of network-exploitable CVEs for zero-day defense.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from knowledge_base.processing.metadata import DocumentMetadata, SeverityLevel, ThreatSource
from knowledge_base.vector_store.index import VectorIndex

logger = logging.getLogger(__name__)


# Curated high-impact network attack CVE seed dataset
DEFAULT_NETWORK_CVES: list[dict[str, Any]] = [
    {
        "cve_id": "CVE-2021-44228",
        "title": "Log4Shell Apache Log4j2 JNDI Remote Code Execution",
        "description": "Apache Log4j2 JNDI features used in configuration, log messages, and parameters do not protect against attacker controlled LDAP and other JNDI related endpoints. An attacker who can control log messages or log message parameters can execute arbitrary code loaded from LDAP servers when message lookup substitution is enabled.",
        "cvss_score": 10.0,
        "severity": "CRITICAL",
        "cwe_id": "CWE-502",
        "affected_protocols": ["TCP", "HTTP", "HTTPS"],
        "affected_ports": [80, 443, 8080, 8443, 389, 636],
        "attack_vector": "NETWORK",
        "mitigation": "Block outbound LDAP (port 389) and RMI (port 1099) from application tiers. Filter incoming HTTP request headers containing jndi:ldap strings. Apply firewall rate limiting.",
        "tags": ["zero-day", "rce", "jndi", "log4shell"],
    },
    {
        "cve_id": "CVE-2017-0144",
        "title": "EternalBlue Microsoft SMBv1 Remote Code Execution",
        "description": "A remote code execution vulnerability exists in Microsoft Server Message Block 1.0 (SMBv1) protocol when handling specially crafted packets, famously exploited by WannaCry and NotPetya malware to propagate laterally across networks.",
        "cvss_score": 9.8,
        "severity": "CRITICAL",
        "cwe_id": "CWE-119",
        "affected_protocols": ["TCP", "SMB"],
        "affected_ports": [445, 139],
        "attack_vector": "NETWORK",
        "mitigation": "Block all inbound and outbound SMB ports (139, 445) at network perimeter firewalls. Disable SMBv1 across all endpoints.",
        "tags": ["wannacry", "eternalblue", "smb", "worm"],
    },
    {
        "cve_id": "CVE-2014-0160",
        "title": "Heartbleed OpenSSL TLS Heartbeat Information Disclosure",
        "description": "The (1) TLS and (2) DTLS implementations in OpenSSL 1.0.1 before 1.0.1g do not properly handle Heartbeat Extension packets, allowing remote attackers to obtain sensitive information from process memory via malformed packets that trigger a buffer over-read.",
        "cvss_score": 7.5,
        "severity": "HIGH",
        "cwe_id": "CWE-125",
        "affected_protocols": ["TCP", "HTTPS", "TLS"],
        "affected_ports": [443, 8443],
        "attack_vector": "NETWORK",
        "mitigation": "Inspect TLS heartbeat lengths against payload length. Drop malformed TLS heartbeat requests. Terminate compromised sessions immediately.",
        "tags": ["heartbleed", "openssl", "memory-leak"],
    },
    {
        "cve_id": "CVE-2021-26855",
        "title": "ProxyLogon Microsoft Exchange Server SSRF",
        "description": "Microsoft Exchange Server Server-Side Request Forgery (SSRF) vulnerability allowing unauthenticated attackers to send arbitrary HTTP requests and authenticate as the Exchange server, paving the way for remote code execution.",
        "cvss_score": 9.8,
        "severity": "CRITICAL",
        "cwe_id": "CWE-918",
        "affected_protocols": ["TCP", "HTTP", "HTTPS"],
        "affected_ports": [443, 80],
        "attack_vector": "NETWORK",
        "mitigation": "Block unauthenticated external access to /owa/ and /ecp/ paths. Implement perimeter WAF rules restricting untrusted POST requests to Exchange frontend.",
        "tags": ["proxylogon", "exchange", "ssrf", "rce"],
    },
    {
        "cve_id": "CVE-2023-38831",
        "title": "WinRAR Spoofed Extension Remote Code Execution",
        "description": "Processing of crafted ZIP archives in RARLAB WinRAR before 6.23 allows remote attackers to execute arbitrary code when a user attempts to view a benign file within a specifically crafted archive.",
        "cvss_score": 7.8,
        "severity": "HIGH",
        "cwe_id": "CWE-20",
        "affected_protocols": ["TCP", "HTTP"],
        "affected_ports": [80, 443],
        "attack_vector": "NETWORK",
        "mitigation": "Block delivery of suspicious compressed archives containing spoofed file extension characters.",
        "tags": ["archive", "malware", "phishing"],
    },
    {
        "cve_id": "CVE-2023-48795",
        "title": "Terrapin Attack SSH Protocol Prefix Truncation",
        "description": "The SSH transport protocol with certain encryption modes allows a man-in-the-middle attacker to truncate extension negotiation messages without detection, breaking integrity guarantees.",
        "cvss_score": 5.9,
        "severity": "MEDIUM",
        "cwe_id": "CWE-345",
        "affected_protocols": ["TCP", "SSH"],
        "affected_ports": [22],
        "attack_vector": "NETWORK",
        "mitigation": "Disable ChaCha20-Poly1305 and CBC ciphers with ETM. Enforce strict key exchange extensions.",
        "tags": ["terrapin", "ssh", "mitm"],
    },
    {
        "cve_id": "CVE-2019-0708",
        "title": "BlueKeep Microsoft RDP Remote Code Execution",
        "description": "A remote code execution vulnerability exists in Remote Desktop Services (formerly Terminal Services) when an unauthenticated attacker connects to the target system using RDP and sends specially crafted requests.",
        "cvss_score": 9.8,
        "severity": "CRITICAL",
        "cwe_id": "CWE-416",
        "affected_protocols": ["TCP", "RDP"],
        "affected_ports": [3389],
        "attack_vector": "NETWORK",
        "mitigation": "Block TCP port 3389 at the enterprise edge. Enforce Network Level Authentication (NLA) on all internal RDP listeners.",
        "tags": ["bluekeep", "rdp", "wormable"],
    },
    {
        "cve_id": "CVE-2020-1472",
        "title": "Zerologon Netlogon Privilege Escalation",
        "description": "An elevation of privilege vulnerability exists when an attacker establishes a vulnerable Netlogon secure channel connection to a domain controller using the Netlogon Remote Protocol (MS-NRPC), allowing complete domain takeover in seconds.",
        "cvss_score": 10.0,
        "severity": "CRITICAL",
        "cwe_id": "CWE-330",
        "affected_protocols": ["TCP", "RPC"],
        "affected_ports": [135, 445],
        "attack_vector": "NETWORK",
        "mitigation": "Enforce secure RPC communication with Netlogon. Restrict MS-NRPC traffic to authorized domain members only.",
        "tags": ["zerologon", "active-directory", "privilege-escalation"],
    },
]


class CVEIngestionEngine:
    """
    Ingests and parses CVE feeds into the vector database.
    """

    def __init__(self, index: VectorIndex, raw_cve_dir: str = "./data/raw/cve") -> None:
        self.index = index
        self.raw_cve_dir = Path(raw_cve_dir)
        self.raw_cve_dir.mkdir(parents=True, exist_ok=True)

    def load_seed_cves(self) -> int:
        """Loads curated high-priority network attack CVE records into the index."""
        logger.info(f"Loading {len(DEFAULT_NETWORK_CVES)} seed CVE records into vector index...")
        return self.index.index_raw_records(DEFAULT_NETWORK_CVES)

    def ingest_from_json_file(self, file_path: str | Path) -> int:
        """
        Parses a local JSON file containing CVE records.
        Supports both raw MITRE format and simplified array format.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"CVE file {path} not found.")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records: list[dict[str, Any]] = []

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Check if MITRE 5.0 container format
            if "containers" in data:
                cve_meta = data.get("cveMetadata", {})
                cve_id = cve_meta.get("cveId", "CVE-UNKNOWN")
                cna = data["containers"].get("cna", {})
                title = cna.get("title", f"Vulnerability {cve_id}")
                descriptions = cna.get("descriptions", [])
                desc_text = descriptions[0].get("value", "") if descriptions else ""
                records.append({
                    "cve_id": cve_id,
                    "title": title,
                    "description": desc_text,
                    "source": "cve",
                })
            elif "CVE_Items" in data:  # Legacy format
                for item in data["CVE_Items"]:
                    cve_meta = item.get("cve", {})
                    cve_id = cve_meta.get("CVE_data_meta", {}).get("ID", "")
                    desc_data = cve_meta.get("description", {}).get("description_data", [])
                    desc = desc_data[0].get("value", "") if desc_data else ""
                    records.append({
                        "cve_id": cve_id,
                        "title": f"Advisory {cve_id}",
                        "description": desc,
                        "source": "cve",
                    })

        return self.index.index_raw_records(records)
