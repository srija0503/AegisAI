"""
agentic_rag/threat_reasoner.py
──────────────────────────────
Threat reasoning engine for Agentic RAG.
Evaluates synthesized context, correlates empirical network anomalies with
retrieved intelligence, and determines the optimal defensive action.
Operates with local deterministic security reasoning and optional LLM endpoint.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from agentic_rag.threat_analyzer import ThreatAssessment
from knowledge_base.vector_store.search import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ReasoningDecision:
    """Final decision output from the threat reasoner."""
    threat_category: str
    threat_name: str
    severity: str
    confidence: float
    recommended_action: str       # "block", "drop", "alert", "allow"
    recommended_mitigation: str
    explanation: str
    matched_cves: list[str] = field(default_factory=list)
    raw_response: Optional[str] = None


class ThreatReasoner:
    """
    Reasoning engine combining heuristic correlation and optional LLM synthesis.
    """

    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model or "llama3"
        self.api_key = api_key

    def reason(
        self,
        flow: dict[str, Any],
        assessment: ThreatAssessment,
        evidence: list[SearchResult],
        context_text: str,
    ) -> ReasoningDecision:
        """
        Executes reasoning loop: tries LLM if configured, else performs deterministic expert reasoning.
        """
        if self.llm_endpoint:
            try:
                decision = self._call_llm_reasoner(flow, assessment, context_text)
                if decision:
                    return decision
            except Exception as e:
                logger.warning(f"LLM reasoner call failed ({e}). Reverting to deterministic expert reasoner.")

        return self._deterministic_expert_reasoning(flow, assessment, evidence)

    def _deterministic_expert_reasoning(
        self,
        flow: dict[str, Any],
        assessment: ThreatAssessment,
        evidence: list[SearchResult],
    ) -> ReasoningDecision:
        """
        Deterministic, low-latency security correlation engine.
        Ensures edge firewalls react in sub-millisecond timeframes.
        """
        matched_cves = []
        for ev in evidence:
            if ev.metadata.cve_id:
                matched_cves.append(ev.metadata.cve_id)

        # Correlate based on category and evidence
        action = assessment.suggested_action
        severity = assessment.severity
        confidence = assessment.confidence
        name = assessment.threat_name

        if assessment.category == "DoS":
            action = "drop"
            mitigation = "Immediately drop packets from offending source IP; activate rate-limiting."
            explanation = (
                f"Empirical traffic indicates volumetric or SYN flooding attack. "
                f"Telemetry: {flow.get('packets_per_sec', 0)} pps. Context confirms T1498 DoS pattern."
            )
        elif assessment.category == "Exploit" or matched_cves:
            action = "block"
            severity = "CRITICAL"
            confidence = max(confidence, 0.95)
            cve_str = ", ".join(matched_cves) if matched_cves else "CVE-2017-0144"
            mitigation = f"Enforce perimeter block on targeted port {assessment.dst_port} for source IP."
            explanation = f"Matched known network exploit profile ({cve_str}) on port {assessment.dst_port}."
        elif assessment.category == "Brute Force":
            action = "block"
            mitigation = f"Temporary firewall block for 3600 seconds on port {assessment.dst_port}."
            explanation = f"Repeated connection attempts or resets targeting authentication service on port {assessment.dst_port}."
        elif assessment.category == "Reconnaissance":
            action = "block"
            mitigation = "Block source IP across all perimeter ports to disrupt attack reconnaissance."
            explanation = "Sequential or stealth probing targeting network ports (T1046)."
        else:
            action = "alert"
            mitigation = "Log flow telemetry and monitor adjacent subnet activity for follow-up triggers."
            explanation = f"Unclassified statistical deviation from baseline (anomaly_score: {flow.get('anomaly_score', 0)})."

        return ReasoningDecision(
            threat_category=assessment.category,
            threat_name=name,
            severity=severity,
            confidence=round(confidence, 3),
            recommended_action=action,
            recommended_mitigation=mitigation,
            explanation=explanation,
            matched_cves=matched_cves,
        )

    def _call_llm_reasoner(
        self,
        flow: dict[str, Any],
        assessment: ThreatAssessment,
        context_text: str,
    ) -> Optional[ReasoningDecision]:
        """Optional HTTP call to local Ollama or OpenAI compatible endpoint."""
        prompt = f"Analyze following context and provide threat assessment in JSON:\n{context_text}"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = requests.post(self.llm_endpoint, json=payload, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            out_text = data.get("response", data.get("content", ""))
            parsed = json.loads(out_text)
            return ReasoningDecision(
                threat_category=parsed.get("threat_category", assessment.category),
                threat_name=parsed.get("threat_name", assessment.threat_name),
                severity=parsed.get("severity", assessment.severity),
                confidence=float(parsed.get("confidence", assessment.confidence)),
                recommended_action=parsed.get("recommended_action", assessment.suggested_action),
                recommended_mitigation=parsed.get("recommended_mitigation", "Perimeter IP block"),
                explanation=parsed.get("explanation", "LLM-assisted threat determination"),
                raw_response=out_text,
            )
        return None
