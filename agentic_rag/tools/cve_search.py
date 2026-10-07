"""
agentic_rag/tools/cve_search.py
───────────────────────────────
RAG Tool: Performs semantic and direct vulnerability searches across
the indexed CVE knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from knowledge_base.vector_store.search import SearchResult, VectorSearchEngine

logger = logging.getLogger(__name__)


class CVESearchTool:
    """
    Tool allowing the Agentic RAG Reasoner to query CVE vulnerabilities.
    """

    def __init__(self, search_engine: Optional[VectorSearchEngine] = None) -> None:
        self.search_engine = search_engine or VectorSearchEngine()

    def search_by_cve_id(self, cve_id: str) -> Optional[dict[str, Any]]:
        """Lookup vulnerability details by exact CVE identifier."""
        res = self.search_engine.search_by_cve(cve_id)
        if res:
            return res.to_dict()
        return None

    def search_vulnerabilities(
        self,
        query: str,
        protocol: Optional[str] = None,
        port: Optional[int] = None,
        top_k: int = 3,
        min_cvss: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant CVEs matching traffic characteristics or keywords.
        """
        results = self.search_engine.search(
            query=query,
            protocol=protocol,
            port=port,
            top_k=top_k,
            min_cvss=min_cvss,
        )
        return [r.to_dict() for r in results]

    def get_tool_metadata(self) -> dict[str, Any]:
        return {
            "name": "cve_search",
            "description": "Searches Common Vulnerabilities and Exposures (CVEs) by ID, keyword, port, or protocol.",
            "parameters": {
                "query": "string (search query or vulnerability keywords)",
                "protocol": "optional string (e.g. TCP, UDP, HTTP)",
                "port": "optional integer (e.g. 445, 80, 22)",
                "top_k": "integer (number of results, default 3)",
            },
        }
