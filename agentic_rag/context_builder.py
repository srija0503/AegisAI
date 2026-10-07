"""
agentic_rag/context_builder.py
──────────────────────────────
Context synthesis engine for Agentic RAG. Formats reranked intelligence,
network flow telemetry, and memory history into structured LLM prompts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_rag.threat_analyzer import ThreatAssessment
from knowledge_base.vector_store.search import SearchResult


class ContextBuilder:
    """
    Constructs coherent contextual representations for threat reasoning.
    """

    def build_context(
        self,
        flow: dict[str, Any],
        assessment: ThreatAssessment,
        evidence: list[SearchResult],
        recent_history: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        sections: list[str] = []

        # 1. Behavioral Assessment Summary
        sections.append("BEHAVIORAL ASSESSMENT:")
        sections.append(f"- Hypothesized Threat: {assessment.threat_name}")
        sections.append(f"- Category: {assessment.category} | Severity: {assessment.severity} | Confidence: {assessment.confidence:.2f}")
        sections.append("- Key Indicators:")
        for ind in assessment.key_indicators:
            sections.append(f"  * {ind}")

        # 2. Retrieved Threat Intelligence Evidence
        sections.append("\nCORRELATED THREAT INTELLIGENCE & CVEs:")
        if evidence:
            for i, ev in enumerate(evidence, 1):
                meta = ev.metadata
                sections.append(f"[{i}] {meta.title} (Source: {meta.source.value.upper()})")
                if meta.cve_id:
                    sections.append(f"    CVE ID: {meta.cve_id} | CVSS: {meta.cvss_score} | Severity: {meta.severity.value}")
                if meta.affected_ports:
                    sections.append(f"    Affected Ports: {meta.affected_ports}")
                # Excerpt
                clean_lines = [line.strip() for line in ev.text.split("\n") if line.strip()]
                sections.append(f"    Details: {' '.join(clean_lines[:3])}")
        else:
            sections.append("  (No direct matching CVEs or signatures found; zero-day / novel behavior suspected.)")

        # 3. Recent Historical Memory Context
        if recent_history:
            sections.append("\nRECENT RECURRING INCIDENTS:")
            for item in recent_history[-3:]:
                sections.append(
                    f"  * [{item.get('timestamp')}] {item.get('src_ip')} -> {item.get('threat_name')} ({item.get('action')})"
                )

        return "\n".join(sections)
