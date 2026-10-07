"""
knowledge_base package
──────────────────────
Threat intelligence ingestion, semantic document processing, and vector storage
for the Quantum-Secure Federated Firewall.
"""

from knowledge_base.ingestion.cve_ingestion import CVEIngestionEngine
from knowledge_base.ingestion.nvd_ingestion import NVDIngestionEngine
from knowledge_base.ingestion.threat_intel_ingestion import ThreatIntelIngestionEngine
from knowledge_base.processing.chunker import DocumentChunk, DocumentChunker
from knowledge_base.processing.document_cleaner import DocumentCleaner
from knowledge_base.processing.metadata import (
    DocumentMetadata,
    SeverityLevel,
    ThreatSource,
)
from knowledge_base.vector_store.database import LocalVectorStore, VectorDatabase
from knowledge_base.vector_store.embeddings import (
    EmbeddingEngine,
    FallbackHashingEmbedding,
)
from knowledge_base.vector_store.index import VectorIndex
from knowledge_base.vector_store.search import SearchResult, VectorSearchEngine

__all__ = [
    "CVEIngestionEngine",
    "DocumentChunk",
    "DocumentChunker",
    "DocumentCleaner",
    "DocumentMetadata",
    "EmbeddingEngine",
    "FallbackHashingEmbedding",
    "LocalVectorStore",
    "NVDIngestionEngine",
    "SearchResult",
    "SeverityLevel",
    "ThreatIntelIngestionEngine",
    "ThreatSource",
    "VectorDatabase",
    "VectorIndex",
    "VectorSearchEngine",
]
