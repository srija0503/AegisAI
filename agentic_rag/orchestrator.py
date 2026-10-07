"""
agentic_rag/orchestrator.py
───────────────────────────
Top-level orchestration hub for the Agentic RAG subsystem.
Coordinates anomaly evaluation, multi-angle knowledge base retrieval,
contextual reranking, threat reasoning, rule synthesis, and validation.
Provides the callback hook registered with FirewallController.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentic_rag.anomaly_trigger import AnomalyTrigger
from agentic_rag.context_builder import ContextBuilder
from agentic_rag.rag_memory import RAGMemory
from agentic_rag.reranker import RAGReranker
from agentic_rag.retriever import RAGRetriever
from agentic_rag.rule_generator import RuleGenerator
from agentic_rag.rule_validator import RuleValidator
from agentic_rag.threat_analyzer import ThreatAnalyzer
from agentic_rag.threat_reasoner import ReasoningDecision, ThreatReasoner
from knowledge_base.vector_store.search import VectorSearchEngine

logger = logging.getLogger(__name__)


class AgenticRAGOrchestrator:
    """
    Central Coordinator for the Agentic RAG zero-day defense loop.
    """

    def __init__(
        self,
        search_engine: Optional[VectorSearchEngine] = None,
        anomaly_threshold: float = 0.65,
        llm_endpoint: Optional[str] = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.search_engine = search_engine or VectorSearchEngine()

        # Subsystems
        self.trigger = AnomalyTrigger(threshold=anomaly_threshold)
        self.analyzer = ThreatAnalyzer()
        self.retriever = RAGRetriever(search_engine=self.search_engine)
        self.reranker = RAGReranker()
        self.context_builder = ContextBuilder()
        self.reasoner = ThreatReasoner(
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            api_key=api_key,
        )
        self.rule_generator = RuleGenerator()
        self.rule_validator = RuleValidator()
        self.memory = RAGMemory()

    def handle_firewall_trigger(
        self,
        flow_features: dict[str, Any],
        anomaly_score: float,
    ) -> dict[str, Any]:
        """
        The integration callback hook called by FirewallController:
        callback(flow_features: dict, anomaly_score: float) -> dict[str, Any]
        """
        # Step 1: Check if this flow warrants a full RAG investigation
        if not self.trigger.should_trigger(flow_features, anomaly_score):
            return {
                "triggered": False,
                "reason": "Anomaly score below threshold or cooldown active",
                "rules": [],
            }

        # Step 2: Execute full pipeline
        return self.investigate_flow(flow_features, anomaly_score)

    def investigate_flow(
        self,
        flow: dict[str, Any],
        anomaly_score: float,
    ) -> dict[str, Any]:
        """
        Executes end-to-end agentic investigation:
        1. Behavioral feature analysis
        2. Threat intelligence & CVE retrieval
        3. Contextual reranking
        4. Structured context synthesis
        5. Deep reasoning & hypothesis determination
        6. Rule synthesis & safety validation
        7. Memory recording
        """
        # 1. Behavioral analysis
        assessment = self.analyzer.analyze_flow(flow, anomaly_score)

        # 2. Multi-angle retrieval
        candidates = self.retriever.retrieve_context(flow, assessment)

        # 3. Contextual reranking
        evidence = self.reranker.rerank(candidates, flow, assessment, top_n=3)

        # 4. Context synthesis
        recent_history = self.memory.get_recent_incidents(limit=3)
        context_text = self.context_builder.build_context(
            flow=flow,
            assessment=assessment,
            evidence=evidence,
            recent_history=recent_history,
        )

        # 5. Reasoning loop
        decision = self.reasoner.reason(
            flow=flow,
            assessment=assessment,
            evidence=evidence,
            context_text=context_text,
        )

        # 6. Rule synthesis & validation
        raw_rules = self.rule_generator.generate_rules(flow, decision)
        valid_rules = self.rule_validator.filter_valid_rules(raw_rules)

        # 7. Record in episodic memory
        record = self.memory.record_incident(
            flow=flow,
            anomaly_score=anomaly_score,
            decision=decision,
            rules=valid_rules,
        )

        logger.info(
            f"Investigation complete for {record.incident_id}: "
            f"{decision.threat_name} -> Action: {decision.recommended_action.upper()} "
            f"({len(valid_rules)} rules generated)"
        )

        return {
            "triggered": True,
            "incident_id": record.incident_id,
            "decision": {
                "threat_name": decision.threat_name,
                "threat_category": decision.threat_category,
                "severity": decision.severity,
                "confidence": decision.confidence,
                "action": decision.recommended_action,
                "mitigation": decision.recommended_mitigation,
                "explanation": decision.explanation,
                "matched_cves": decision.matched_cves,
            },
            "rules": valid_rules,
            "evidence_count": len(evidence),
            "evidence": [e.to_dict() for e in evidence],
        }

    def get_stats(self) -> dict[str, Any]:
        """Returns runtime orchestrator metrics."""
        return {
            "total_incidents_recorded": self.memory.count(),
            "trigger_stats": {
                "evaluations": self.trigger.stats.total_evaluations,
                "triggers": self.trigger.stats.total_triggers,
                "suppressed_by_threshold": self.trigger.stats.suppressed_by_threshold,
                "suppressed_by_cooldown": self.trigger.stats.suppressed_by_cooldown,
            },
            "vector_store_backend": self.search_engine.db.backend,
        }
