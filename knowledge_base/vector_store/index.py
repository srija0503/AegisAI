"""
knowledge_base/vector_store/index.py
────────────────────────────────────
High-level index management for threat intelligence.
Orchestrates cleaning, chunking, embedding generation, and indexing into the Vector DB.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from knowledge_base.processing.chunker import DocumentChunk, DocumentChunker
from knowledge_base.processing.document_cleaner import DocumentCleaner
from knowledge_base.processing.metadata import DocumentMetadata
from knowledge_base.vector_store.database import VectorDatabase
from knowledge_base.vector_store.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    Manages building and updating the semantic search index for threat intelligence.
    """

    def __init__(
        self,
        db: Optional[VectorDatabase] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        chunker: Optional[DocumentChunker] = None,
        cleaner: Optional[DocumentCleaner] = None,
        persist_dir: str = "./data/vector_db",
        collection_name: str = "threat_intelligence",
    ) -> None:
        self.db = db or VectorDatabase(persist_directory=persist_dir, collection_name=collection_name)
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.chunker = chunker or DocumentChunker()
        self.cleaner = cleaner or DocumentCleaner()

    def index_document(
        self,
        title: str,
        text: str,
        metadata: Optional[DocumentMetadata] = None,
    ) -> int:
        """
        Cleans, chunks, embeds, and indexes a single document.
        Returns the number of indexed chunks.
        """
        cleaned_text = self.cleaner.clean_text(text)
        if not cleaned_text:
            return 0

        if metadata is None:
            metadata = DocumentMetadata(
                doc_id=f"doc_{hash(cleaned_text[:50])}",
                title=title,
            )

        chunks = self.chunker.chunk_document(cleaned_text, metadata)
        return self.index_chunks(chunks)

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Embeds and stores a batch of DocumentChunks into the vector database.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata.to_chroma_metadata() for c in chunks]

        # Compute embeddings
        embeddings = self.embedding_engine.encode(texts)

        # Upsert to database
        self.db.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logger.info(f"Indexed {len(chunks)} chunks into collection '{self.db.collection_name}'.")
        return len(chunks)

    def index_raw_records(self, records: list[dict[str, Any]]) -> int:
        """
        Convenience method to index raw dictionaries containing:
        - title
        - description
        - cve_id (optional)
        - cvss_score (optional)
        - severity (optional)
        - mitigation (optional)
        - affected_protocols (optional)
        - affected_ports (optional)
        """
        total_chunks = 0
        for rec in records:
            title = rec.get("title", "Threat Advisory")
            desc = rec.get("description", "")
            cve_id = rec.get("cve_id")
            cvss = rec.get("cvss_score")
            sev = rec.get("severity")
            mitigation = rec.get("mitigation")

            canonical_text = self.cleaner.format_threat_document(
                title=title,
                description=desc,
                cve_id=cve_id,
                cvss_score=cvss,
                severity=sev,
                mitigation=mitigation,
            )

            meta = DocumentMetadata(
                doc_id=cve_id or f"intel_{abs(hash(title))}",
                title=title,
                cve_id=cve_id,
                cwe_id=rec.get("cwe_id"),
                cvss_score=cvss,
                affected_protocols=rec.get("affected_protocols", []),
                affected_ports=rec.get("affected_ports", []),
                tags=rec.get("tags", []),
                extra=rec.get("extra", {}),
            )

            total_chunks += self.index_document(title, canonical_text, meta)

        return total_chunks

    def count(self) -> int:
        return self.db.count()

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_records": self.db.count(),
            "backend": self.db.backend,
            "embedding_model": self.embedding_engine.model_name,
            "embedding_dimension": self.embedding_engine.dimension,
            "embedding_backend": self.embedding_engine.backend,
            "persist_dir": self.db.persist_directory,
        }
