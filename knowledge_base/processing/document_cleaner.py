"""
knowledge_base/processing/document_cleaner.py
─────────────────────────────────────────────
Cleans, normalizes, and sanitizes security advisory texts, CVE disclosures,
and threat intelligence feeds. Extracts indicators of compromise (IoCs)
and standardizes terminology for optimal vector embedding.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


class DocumentCleaner:
    """
    Text sanitizer and feature extractor for threat intelligence documents.
    """

    # Regex patterns for security entities
    CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
    CWE_PATTERN = re.compile(r"\bCWE-\d{1,5}\b", re.IGNORECASE)
    IPV4_PATTERN = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
    PORT_PATTERN = re.compile(r"\b(?:port\s+|dst_port\s*[:=]\s*)(\d{1,5})\b", re.IGNORECASE)
    PROTOCOL_PATTERN = re.compile(r"\b(TCP|UDP|ICMP|HTTP|HTTPS|DNS|SSH|FTP|TELNET|SMB|RDP)\b", re.IGNORECASE)

    # HTML and artifact stripping
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    MULTIPLE_SPACES = re.compile(r"[ \t]+")
    MULTIPLE_NEWLINES = re.compile(r"\n{3,}")

    def __init__(self, lowercase_for_embedding: bool = False) -> None:
        self.lowercase = lowercase_for_embedding

    def clean_text(self, text: str) -> str:
        """
        Main text cleaning pipeline:
        1. Strips HTML tags and entities.
        2. Normalizes line endings and redundant whitespace.
        3. Standardizes CVE and CWE IDs to uppercase.
        4. Removes non-printable / control characters.
        """
        if not text:
            return ""

        # Remove HTML
        cleaned = self.HTML_TAG_PATTERN.sub(" ", text)

        # Standardize carriage returns
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Strip non-printable chars (preserve standard ASCII and unicode text)
        cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch in "\n\t")

        # Uppercase CVE and CWE patterns for consistency
        cleaned = self.CVE_PATTERN.sub(lambda m: m.group(0).upper(), cleaned)
        cleaned = self.CWE_PATTERN.sub(lambda m: m.group(0).upper(), cleaned)

        # Collapse whitespace
        cleaned = self.MULTIPLE_SPACES.sub(" ", cleaned)
        cleaned = self.MULTIPLE_NEWLINES.sub("\n\n", cleaned)
        cleaned = cleaned.strip()

        if self.lowercase:
            cleaned = cleaned.lower()

        return cleaned

    def extract_entities(self, text: str) -> dict[str, list[Any]]:
        """
        Extract known security identifiers from the text:
        CVE IDs, CWE IDs, IP addresses, Ports, Protocols.
        """
        cves = sorted(list(set(m.upper() for m in self.CVE_PATTERN.findall(text))))
        cwes = sorted(list(set(m.upper() for m in self.CWE_PATTERN.findall(text))))
        ips = sorted(list(set(self.IPV4_PATTERN.findall(text))))
        
        # Ports
        raw_ports = self.PORT_PATTERN.findall(text)
        ports = sorted(list(set(int(p) for p in raw_ports if 0 <= int(p) <= 65535)))

        # Protocols
        raw_protos = self.PROTOCOL_PATTERN.findall(text)
        protocols = sorted(list(set(p.upper() for p in raw_protos)))

        return {
            "cve_ids": cves,
            "cwe_ids": cwes,
            "ip_addresses": ips,
            "ports": ports,
            "protocols": protocols,
        }

    def format_threat_document(
        self,
        title: str,
        description: str,
        cve_id: str | None = None,
        cvss_score: float | None = None,
        severity: str | None = None,
        mitigation: str | None = None,
    ) -> str:
        """
        Creates a structured, high-signal canonical text representation
        optimized for semantic embedding and retrieval.
        """
        parts = []
        if cve_id:
            parts.append(f"VULNERABILITY ID: {cve_id.upper()}")
        parts.append(f"TITLE: {self.clean_text(title)}")
        if severity:
            parts.append(f"SEVERITY: {severity.upper()} (CVSS: {cvss_score or 'N/A'})")
        parts.append(f"SUMMARY: {self.clean_text(description)}")
        if mitigation:
            parts.append(f"MITIGATION: {self.clean_text(mitigation)}")

        return "\n".join(parts)
