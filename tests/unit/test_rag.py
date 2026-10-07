"""
tests/unit/test_rag.py
──────────────────────
Unit tests for Member 2 Phase 2: Agentic RAG Subsystem.
Validates AnomalyTrigger, ThreatAnalyzer, ContextBuilder, ThreatReasoner,
RuleGenerator, RuleValidator, RAGMemory, and full Orchestrator integration.
"""

import tempfile
import pytest

from agentic_rag.anomaly_trigger import AnomalyTrigger
from agentic_rag.context_builder import ContextBuilder
from agentic_rag.orchestrator import AgenticRAGOrchestrator
from agentic_rag.rag_memory import RAGMemory
from agentic_rag.reranker import RAGReranker
from agentic_rag.retriever import RAGRetriever
from agentic_rag.rule_generator import RuleGenerator
from agentic_rag.rule_validator import RuleValidator
from agentic_rag.threat_analyzer import ThreatAnalyzer
from agentic_rag.threat_reasoner import ThreatReasoner
from knowledge_base.ingestion.cve_ingestion import CVEIngestionEngine
from knowledge_base.vector_store.database import VectorDatabase
from knowledge_base.vector_store.embeddings import EmbeddingEngine
from knowledge_base.vector_store.index import VectorIndex
from knowledge_base.vector_store.search import VectorSearchEngine


def test_anomaly_trigger_threshold_and_cooldown():
    trigger = AnomalyTrigger(threshold=0.70, cooldown_seconds=5.0)

    flow = {"src_ip": "192.168.1.50", "dst_port": 445, "protocol": "TCP"}

    # Below threshold -> should NOT trigger
    assert not trigger.should_trigger(flow, anomaly_score=0.60)

    # Above threshold -> should trigger
    assert trigger.should_trigger(flow, anomaly_score=0.85)

    # Immediate second attempt within cooldown -> should NOT trigger
    assert not trigger.should_trigger(flow, anomaly_score=0.90)


def test_threat_analyzer_syn_flood_and_exploit():
    analyzer = ThreatAnalyzer()

    # SYN flood flow
    syn_flow = {
        "src_ip": "10.0.0.99",
        "dst_port": 80,
        "protocol": "TCP",
        "packets_per_sec": 1200.0,
        "tcp_flags": {"syn": 50, "ack": 0},
    }
    assessment = analyzer.analyze_flow(syn_flow, anomaly_score=0.92)
    assert assessment.category == "DoS"
    assert assessment.suggested_action == "drop"
    assert assessment.severity == "CRITICAL"

    # SMB EternalBlue flow
    smb_flow = {
        "src_ip": "10.0.0.88",
        "dst_port": 445,
        "protocol": "TCP",
        "packets_per_sec": 15.0,
        "tcp_flags": {"syn": 1, "ack": 1},
    }
    smb_assessment = analyzer.analyze_flow(smb_flow, anomaly_score=0.80)
    assert smb_assessment.category == "Exploit"
    assert smb_assessment.suggested_action == "block"


def test_rule_generator_and_validator():
    gen = RuleGenerator(default_priority=20)
    val = RuleValidator()

    # Create dummy decision
    reasoner = ThreatReasoner()
    flow = {"src_ip": "198.51.100.22", "dst_port": 445, "protocol": "TCP"}
    analyzer = ThreatAnalyzer()
    assessment = analyzer.analyze_flow(flow, anomaly_score=0.88)

    decision = reasoner.reason(flow, assessment, evidence=[], context_text="SMB exploit threat")
    rules = gen.generate_rules(flow, decision)

    assert len(rules) == 1
    rule = rules[0]
    assert rule["src_ip"] == "198.51.100.22"
    assert rule["action"] == "block"
    assert rule["dst_port"] == 445

    # Validate
    is_valid, msg = val.validate_rule(rule)
    assert is_valid, msg


def test_rule_validator_rejects_dangerous_rules():
    val = RuleValidator()

    # Blocking 0.0.0.0/0 must be rejected
    dangerous_rule = {
        "rule_id": "bad_rule",
        "priority": 10,
        "action": "block",
        "src_ip": "0.0.0.0/0",
    }
    ok, _ = val.validate_rule(dangerous_rule)
    assert not ok

    # Blocking localhost must be rejected
    localhost_rule = {
        "rule_id": "bad_rule_2",
        "priority": 10,
        "action": "block",
        "src_ip": "127.0.0.1",
    }
    ok2, _ = val.validate_rule(localhost_rule)
    assert not ok2


def test_end_to_end_orchestrator():
    tmpdir = tempfile.mkdtemp()
    try:
        emb_engine = EmbeddingEngine(dimension=384, force_fallback=True)
        db = VectorDatabase(persist_directory=tmpdir, collection_name="orchestrator_kb", force_local=True)
        index = VectorIndex(db=db, embedding_engine=emb_engine)

        cve_engine = CVEIngestionEngine(index=index)
        cve_engine.load_seed_cves()

        search_engine = VectorSearchEngine(db=db, embedding_engine=emb_engine)
        orchestrator = AgenticRAGOrchestrator(
            search_engine=search_engine,
            anomaly_threshold=0.70,
        )

        flow_data = {
            "src_ip": "203.0.113.15",
            "dst_ip": "192.168.1.10",
            "dst_port": 445,
            "src_port": 49152,
            "protocol": "TCP",
            "service": "smb",
            "packets_per_sec": 45.0,
            "duration": 0.8,
            "kdd_flag": "SF",
            "tcp_flags": {"syn": 1, "ack": 1, "rst": 0},
        }

        result = orchestrator.handle_firewall_trigger(flow_data, anomaly_score=0.88)
        assert result["triggered"] is True
        assert "decision" in result
        assert result["decision"]["action"] in ("block", "drop")
        assert len(result["rules"]) > 0
        assert result["rules"][0]["src_ip"] == "203.0.113.15"

        db.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
