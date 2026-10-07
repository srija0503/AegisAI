"""
agentic_rag/reranker.py
───────────────────────
Contextual reranker for retrieved threat intelligence.
Applies cross-feature scoring, vulnerability severity weighting, and
port/protocol alignment to select the highest-precision evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agentic_rag.threat_analyzer import ThreatAssessment
from knowledge_base.processing.metadata import SeverityLevel
from knowledge_base.vector_store.search import SearchResult

logger = logging.getLogger(__name__)


class RAGReranker:
    """
    Reranks candidate search results against empirical flow telemetry.
    """

    def rerank(
        self,
        candidates: list[SearchResult],
        flow: dict[str, Any],
        assessment: ThreatAssessment,
        top_n: int = 3,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        flow_proto = str(flow.get("protocol", "")).upper()
        flow_port = flow.get("dst_port")

        scored_candidates: list[tuple[float, SearchResult]] = []

        for cand in candidates:
            # Base semantic score (0.0 to 1.0)
            score = cand.score
            meta = cand.metadata

            # 1. Port alignment boost (+0.20)
            if flow_port is not None and meta.affected_ports:
                if flow_port in meta.affected_ports:
                    score += 0.20
            elif flow_port is not None and str(flow_port) in cand.text:
                score += 0.10

            # 2. Protocol alignment boost (+0.15)
            if flow_proto and meta.affected_protocols:
                if flow_proto in meta.affected_protocols:
                    score += 0.15

            # 3. Severity weighting boost
            if meta.severity == SeverityLevel.CRITICAL:
                score += 0.15
            elif meta.severity == SeverityLevel.HIGH:
                score += 0.10

            # 4. Keyword indicator overlap
            for ind in assessment.key_indicators:
                for token in ind.split():
                    if len(token) > 3 and token.lower() in cand.text.lower():
                        score += 0.05

            scored_candidates.append((score, cand))

        # Sort descending by adjusted score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        reranked = [c for _, c in scored_candidates[:top_n]]
        logger.info(f"Reranked {len(candidates)} candidates down to top {len(reranked)} evidence items.")
        return reranked
