"""
knowledge_base/ingestion/nvd_ingestion.py
────────────────────────────────────────
Ingestion engine for the National Vulnerability Database (NVD) API v2.0
and NVD JSON feeds. Extracts CVSS v3.1 metrics, vector strings, CWE classifications,
and network exploitability indicators for vector indexing.
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


class NVDIngestionEngine:
    """
    Ingests and normalizes vulnerability records from NIST NVD feeds.
    """

    NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def __init__(
        self,
        index: VectorIndex,
        api_key: Optional[str] = None,
        cache_dir: str = "./data/raw/nvd",
    ) -> None:
        self.index = index
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def parse_nvd_cve_item(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Parses a single vulnerability object from NVD API 2.0 schema:
        item["cve"]: id, descriptions, metrics, weaknesses, configurations
        """
        cve_data = item.get("cve", item)
        cve_id = cve_data.get("id", "")
        if not cve_id:
            return None

        # Description
        descriptions = cve_data.get("descriptions", [])
        desc_text = ""
        for d in descriptions:
            if d.get("lang") == "en":
                desc_text = d.get("value", "")
                break
        if not desc_text and descriptions:
            desc_text = descriptions[0].get("value", "")

        # CVSS Metrics
        metrics = cve_data.get("metrics", {})
        cvss_score = None
        severity = "UNKNOWN"
        vector_str = ""
        attack_vector = None

        if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
            cvss_obj = metrics["cvssMetricV31"][0].get("cvssData", {})
            cvss_score = cvss_obj.get("baseScore")
            severity = cvss_obj.get("baseSeverity", "UNKNOWN")
            vector_str = cvss_obj.get("vectorString", "")
            attack_vector = cvss_obj.get("attackVector")
        elif "cvssMetricV30" in metrics and metrics["cvssMetricV30"]:
            cvss_obj = metrics["cvssMetricV30"][0].get("cvssData", {})
            cvss_score = cvss_obj.get("baseScore")
            severity = cvss_obj.get("baseSeverity", "UNKNOWN")
            vector_str = cvss_obj.get("vectorString", "")
            attack_vector = cvss_obj.get("attackVector")
        elif "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
            cvss_obj = metrics["cvssMetricV2"][0].get("cvssData", {})
            cvss_score = cvss_obj.get("baseScore")
            severity = metrics["cvssMetricV2"][0].get("baseSeverity", "UNKNOWN")
            vector_str = cvss_obj.get("vectorString", "")

        # CWE
        cwe_id = None
        weaknesses = cve_data.get("weaknesses", [])
        if weaknesses:
            desc_list = weaknesses[0].get("description", [])
            for w in desc_list:
                val = w.get("value", "")
                if val.startswith("CWE-"):
                    cwe_id = val
                    break

        # Protocol / ports heuristic based on description or vector
        affected_protocols: list[str] = []
        affected_ports: list[int] = []

        desc_lower = desc_text.lower()
        if "http" in desc_lower or "web" in desc_lower:
            affected_protocols.append("HTTP")
            affected_ports.extend([80, 443, 8080])
        if "ssh" in desc_lower:
            affected_protocols.append("SSH")
            affected_ports.append(22)
        if "dns" in desc_lower:
            affected_protocols.append("DNS")
            affected_ports.append(53)
        if "smb" in desc_lower:
            affected_protocols.append("SMB")
            affected_ports.extend([139, 445])
        if "rdp" in desc_lower:
            affected_protocols.append("RDP")
            affected_ports.append(3389)

        return {
            "cve_id": cve_id,
            "title": f"NVD Vulnerability {cve_id}",
            "description": desc_text,
            "cvss_score": cvss_score,
            "severity": severity,
            "cwe_id": cwe_id,
            "attack_vector": attack_vector,
            "affected_protocols": list(set(affected_protocols)),
            "affected_ports": list(set(affected_ports)),
            "source": "nvd",
            "extra": {"cvss_vector": vector_str},
        }

    def ingest_nvd_json_file(self, file_path: str | Path) -> int:
        """Parses an exported NVD JSON feed or local cache."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"NVD JSON file {path} not found.")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vulnerabilities = data.get("vulnerabilities", [])
        if not vulnerabilities and isinstance(data, list):
            vulnerabilities = data

        parsed_records: list[dict[str, Any]] = []
        for item in vulnerabilities:
            parsed = self.parse_nvd_cve_item(item)
            if parsed:
                parsed_records.append(parsed)

        return self.index.index_raw_records(parsed_records)

    def fetch_recent_cves(self, limit: int = 20, keyword: Optional[str] = "firewall") -> int:
        """Fetches recent matching CVEs from NVD API if internet access is available."""
        params: dict[str, Any] = {"resultsPerPage": limit}
        if keyword:
            params["keywordSearch"] = keyword

        headers = {}
        if self.api_key:
            headers["apiKey"] = self.api_key

        try:
            logger.info(f"Querying NVD API: {params}...")
            resp = requests.get(self.NVD_API_BASE, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                vulns = data.get("vulnerabilities", [])
                parsed = [self.parse_nvd_cve_item(v) for v in vulns]
                valid = [p for p in parsed if p is not None]
                return self.index.index_raw_records(valid)
            else:
                logger.warning(f"NVD API returned HTTP {resp.status_code}.")
        except Exception as e:
            logger.warning(f"Could not reach NVD API ({e}). Ingestion skipped.")
        return 0
