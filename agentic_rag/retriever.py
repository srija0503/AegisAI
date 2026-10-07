"""
agentic_rag/retriever.py
────────────────────────
Contextual multi-angle retriever for the Agentic RAG system.
Queries the vector database using flow telemetry, extracted attack signatures,
and targeted port/protocol filters to find matching threat intelligence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agentic_rag.threat_analyzer import ThreatAssessment
from agentic_rag.tools.cve_search import CVESearchTool
from agentic_rag.tools.threat_intel import ThreatIntelTool
from knowledge_base.vector_store.search import SearchResult, VectorSearchEngine

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Coordinates multi-strategy retrieval for incoming network anomalies.
    """

    def __init__(
        self,
        search_engine: Optional[VectorSearchEngine] = None,
        cve_tool: Optional[CVESearchTool] = None,
        threat_intel_tool: Optional[ThreatIntelTool] = None,
    ) -> None:
        self.search_engine = search_engine or VectorSearchEngine()
        self.cve_tool = cve_tool or CVESearchTool(self.search_engine)
        self.threat_intel_tool = threat_intel_tool or ThreatIntelTool(self.search_engine)

    def retrieve_context(
        self,
        flow: dict[str, Any],
        assessment: ThreatAssessment,
        top_k: int = 4,
    ) -> list[SearchResult]:
        """
        Executes multi-angle retrieval:
        1. Feature-based search (protocol, port, service, flags)
        2. Threat hypothesis search (based on behavioral assessment category/indicators)
        3. Combines and deduplicates results
        """
        seen_ids: set[str] = set()
        candidates: list[SearchResult] = []

        # Angle 1: Flow feature retrieval
        flow_results = self.search_engine.search_by_traffic_features(flow, top_k=top_k)
        for r in flow_results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                candidates.append(r)

        # Angle 2: Behavioral assessment hypothesis retrieval
        query_str = f"{assessment.threat_name} {assessment.category} {' '.join(assessment.key_indicators)}"
        hypothesis_results = self.search_engine.search(
            query=query_str,
            protocol=assessment.protocol,
            port=assessment.dst_port,
            top_k=top_k,
        )
        for r in hypothesis_results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                candidates.append(r)

        logger.info(f"Retrieved {len(candidates)} candidate documents for flow analysis.")
        return candidates
