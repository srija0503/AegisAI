"""
tests/unit/test_knowledge_base.py
─────────────────────────────────
Unit tests for Member 2 Phase 1: Knowledge Base System.
Validates document cleaning, chunking, metadata flattening, embedding engine,
vector database persistence, CVE/Threat Intel ingestion, and semantic search.
"""

import tempfile
import pytest

from knowledge_base.processing.metadata import (
    DocumentMetadata,
    SeverityLevel,
    ThreatSource,
)
from knowledge_base.processing.document_cleaner import DocumentCleaner
from knowledge_base.processing.chunker import DocumentChunker
from knowledge_base.vector_store.embeddings import EmbeddingEngine
from knowledge_base.vector_store.database import VectorDatabase
from knowledge_base.vector_store.index import VectorIndex
from knowledge_base.vector_store.search import VectorSearchEngine
from knowledge_base.ingestion.cve_ingestion import CVEIngestionEngine
from knowledge_base.ingestion.threat_intel_ingestion import ThreatIntelIngestionEngine


def test_metadata_creation_and_chroma_flattening():
    meta = DocumentMetadata(
        doc_id="CVE-2021-44228",
        title="Log4Shell",
        source=ThreatSource.CVE,
        cve_id="CVE-2021-44228",
        cwe_id="CWE-502",
        cvss_score=10.0,
        affected_protocols=["TCP", "HTTP"],
        affected_ports=[80, 443, 389],
        tags=["rce", "log4j"],
    )

    assert meta.severity == SeverityLevel.CRITICAL
    flat = meta.to_chroma_metadata()
    assert flat["cve_id"] == "CVE-2021-44228"
    assert flat["cvss_score"] == 10.0
    assert "80,443,389" in flat["ports"]
    assert "TCP,HTTP" in flat["protocols"]

    restored = DocumentMetadata.from_chroma_metadata(flat)
    assert restored.cve_id == "CVE-2021-44228"
    assert restored.cvss_score == 10.0
    assert 389 in restored.affected_ports
    assert "HTTP" in restored.affected_protocols


def test_document_cleaner_entity_extraction():
    cleaner = DocumentCleaner()
    raw = (
        "<p>Alert for <b>cve-2021-44228</b> targeting host 192.168.1.100 on dst_port: 8080. "
        "Protocol used was TCP with CWE-502 execution.</p>"
    )
    cleaned = cleaner.clean_text(raw)
    assert "<p>" not in cleaned
    assert "CVE-2021-44228" in cleaned

    entities = cleaner.extract_entities(cleaned)
    assert "CVE-2021-44228" in entities["cve_ids"]
    assert "CWE-502" in entities["cwe_ids"]
    assert "192.168.1.100" in entities["ip_addresses"]
    assert 8080 in entities["ports"]
    assert "TCP" in entities["protocols"]


def test_document_chunker():
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30, preserve_header=True)
    text = (
        "Apache Log4j2 JNDI features do not protect against attacker controlled LDAP endpoints. "
        "An attacker can execute arbitrary remote code. "
        "Perimeter firewalls must block outbound port 389 immediately."
    )
    meta = DocumentMetadata(doc_id="test_doc", title="Log4j Advisory", cve_id="CVE-2021-44228")
    chunks = chunker.chunk_document(text, meta)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.doc_id == "test_doc"
        assert "CVE-2021-44228" in c.text


def test_embedding_engine_and_vector_database():
    tmpdir = tempfile.mkdtemp()
    try:
        emb_engine = EmbeddingEngine(dimension=384, force_fallback=True)
        vecs = emb_engine.encode(["Log4Shell exploit payload", "Benign HTTP GET request"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384

        db = VectorDatabase(persist_directory=tmpdir, collection_name="test_col", force_local=True)
        db.upsert(
            ids=["doc_1", "doc_2"],
            documents=["Log4Shell exploit payload", "Benign HTTP GET request"],
            metadatas=[{"title": "Doc 1"}, {"title": "Doc 2"}],
            embeddings=vecs,
        )
        assert db.count() == 2

        # Query
        q_emb = emb_engine.encode(["Log4j LDAP vulnerability"])
        results = db.query(q_emb, n_results=1)
        assert len(results["ids"][0]) == 1
        db.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_end_to_end_knowledge_base_indexing_and_search():
    tmpdir = tempfile.mkdtemp()
    try:
        emb_engine = EmbeddingEngine(dimension=384, force_fallback=True)
        db = VectorDatabase(persist_directory=tmpdir, collection_name="threat_kb", force_local=True)
        index = VectorIndex(db=db, embedding_engine=emb_engine)

        # Ingest Seed CVEs & Threat Intel
        cve_engine = CVEIngestionEngine(index=index)
        n_cves = cve_engine.load_seed_cves()
        assert n_cves > 0

        intel_engine = ThreatIntelIngestionEngine(index=index)
        n_intel = intel_engine.load_seed_threat_intel()
        assert n_intel > 0

        search_engine = VectorSearchEngine(
            db=db,
            embedding_engine=emb_engine,
            default_top_k=3,
            min_similarity_score=0.10,
        )

        # Search for SMB / WannaCry
        res_smb = search_engine.search("EternalBlue SMB remote execution", top_k=2)
        assert len(res_smb) > 0
        top_res = res_smb[0]
        assert "CVE-2017-0144" in top_res.text or "SMB" in top_res.text

        # Search by network flow features
        traffic_features = {
            "protocol": "TCP",
            "dst_port": 445,
            "service": "smb",
            "kdd_flag": "SF",
            "packets_per_sec": 120,
        }
        flow_results = search_engine.search_by_traffic_features(traffic_features, top_k=2)
        assert len(flow_results) > 0
        db.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
