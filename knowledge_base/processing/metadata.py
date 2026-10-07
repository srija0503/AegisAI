"""
knowledge_base/processing/metadata.py
─────────────────────────────────────
Standardized metadata schemas and validation for ingested security documents,
CVEs, and threat intelligence feeds. Ensures compliance with Vector DB storage
and enables multi-dimensional filtering (protocol, port, CVSS, severity).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class SeverityLevel(str, Enum):
    """CVSS Severity classifications."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_cvss(cls, score: Optional[float]) -> "SeverityLevel":
        """Compute severity level according to CVSS v3.1 standard thresholds."""
        if score is None:
            return cls.UNKNOWN
        if score >= 9.0:
            return cls.CRITICAL
        elif score >= 7.0:
            return cls.HIGH
        elif score >= 4.0:
            return cls.MEDIUM
        elif score > 0.0:
            return cls.LOW
        return cls.INFO


class ThreatSource(str, Enum):
    """Source repositories of security intelligence."""
    CVE = "cve"
    NVD = "nvd"
    MITRE_ATTACK = "mitre_attack"
    ALIENVAULT_OTX = "alienvault_otx"
    CISA_KEV = "cisa_kev"
    LOCAL_INTEL = "local_intel"
    NETWORK_TELEMETRY = "network_telemetry"


@dataclass
class DocumentMetadata:
    """
    Metadata representation accompanying every stored chunk or document.
    Enables rich retrieval filtering on edge nodes.
    """
    doc_id: str
    title: str
    source: ThreatSource = ThreatSource.LOCAL_INTEL
    cve_id: Optional[str] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    severity: SeverityLevel = SeverityLevel.UNKNOWN
    attack_vector: Optional[str] = None         # e.g., "NETWORK", "ADJACENT", "LOCAL"
    affected_protocols: list[str] = field(default_factory=list)  # e.g., ["TCP", "UDP"]
    affected_ports: list[int] = field(default_factory=list)        # e.g., [80, 443, 22]
    tags: list[str] = field(default_factory=list)
    published_date: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.source, str):
            try:
                self.source = ThreatSource(self.source.lower())
            except ValueError:
                self.source = ThreatSource.LOCAL_INTEL

        # Normalize severity
        if isinstance(self.severity, str):
            try:
                self.severity = SeverityLevel(self.severity.upper())
            except ValueError:
                self.severity = SeverityLevel.UNKNOWN

        # Auto-compute severity if unknown and CVSS score provided
        if self.severity == SeverityLevel.UNKNOWN and self.cvss_score is not None:
            self.severity = SeverityLevel.from_cvss(self.cvss_score)

        # Normalize protocols
        self.affected_protocols = [p.upper() for p in self.affected_protocols]

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to plain python dictionary."""
        d = asdict(self)
        d["source"] = self.source.value
        d["severity"] = self.severity.value
        return d

    def to_chroma_metadata(self) -> dict[str, Union[str, int, float, bool]]:
        """
        ChromaDB only supports primitive metadata types (str, int, float, bool).
        Flattens lists and complex nested structures into comma-separated strings or JSON.
        """
        flat: dict[str, Union[str, int, float, bool]] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source.value,
            "severity": self.severity.value,
            "created_at": self.created_at,
        }
        if self.cve_id:
            flat["cve_id"] = self.cve_id
        if self.cwe_id:
            flat["cwe_id"] = self.cwe_id
        if self.cvss_score is not None:
            flat["cvss_score"] = float(self.cvss_score)
        if self.attack_vector:
            flat["attack_vector"] = self.attack_vector
        if self.published_date:
            flat["published_date"] = self.published_date
        if self.affected_protocols:
            flat["protocols"] = ",".join(self.affected_protocols)
        if self.affected_ports:
            flat["ports"] = ",".join(str(p) for p in self.affected_ports)
        if self.tags:
            flat["tags"] = ",".join(self.tags)
        if self.extra:
            flat["extra_json"] = json.dumps(self.extra)

        return flat

    @classmethod
    def from_chroma_metadata(cls, flat: dict[str, Any]) -> "DocumentMetadata":
        """Reconstruct DocumentMetadata from ChromaDB primitive dictionary."""
        protocols = [p.strip() for p in str(flat.get("protocols", "")).split(",") if p.strip()]
        ports_str = [p.strip() for p in str(flat.get("ports", "")).split(",") if p.strip()]
        ports = [int(p) for p in ports_str if p.isdigit()]
        tags = [t.strip() for t in str(flat.get("tags", "")).split(",") if t.strip()]

        extra = {}
        if "extra_json" in flat and flat["extra_json"]:
            try:
                extra = json.loads(flat["extra_json"])
            except Exception:
                pass

        cvss = flat.get("cvss_score")
        if cvss is not None:
            try:
                cvss = float(cvss)
            except (ValueError, TypeError):
                cvss = None

        return cls(
            doc_id=str(flat.get("doc_id", "")),
            title=str(flat.get("title", "")),
            source=ThreatSource(flat.get("source", "local_intel")),
            cve_id=flat.get("cve_id"),
            cwe_id=flat.get("cwe_id"),
            cvss_score=cvss,
            severity=SeverityLevel(flat.get("severity", "UNKNOWN")),
            attack_vector=flat.get("attack_vector"),
            affected_protocols=protocols,
            affected_ports=ports,
            tags=tags,
            published_date=flat.get("published_date"),
            created_at=str(flat.get("created_at", "")),
            extra=extra,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMetadata":
        """Build from standard dictionary."""
        source_val = data.get("source", "local_intel")
        sev_val = data.get("severity", "UNKNOWN")
        return cls(
            doc_id=data.get("doc_id", ""),
            title=data.get("title", ""),
            source=ThreatSource(source_val) if isinstance(source_val, str) else source_val,
            cve_id=data.get("cve_id"),
            cwe_id=data.get("cwe_id"),
            cvss_score=data.get("cvss_score"),
            severity=SeverityLevel(sev_val) if isinstance(sev_val, str) else sev_val,
            attack_vector=data.get("attack_vector"),
            affected_protocols=data.get("affected_protocols", []),
            affected_ports=data.get("affected_ports", []),
            tags=data.get("tags", []),
            published_date=data.get("published_date"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            extra=data.get("extra", {}),
        )
