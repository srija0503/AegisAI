"""
knowledge_base/vector_store/search.py
─────────────────────────────────────
Vector and hybrid search engine for threat intelligence retrieval.
Provides threshold scoring, metadata filtering, and result ranking
optimized for feeding contextual evidence into Agentic RAG reasoners.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from knowledge_base.processing.metadata import DocumentMetadata, SeverityLevel
from knowledge_base.vector_store.database import VectorDatabase
from knowledge_base.vector_store.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Retrieved vulnerability or intelligence item."""
    chunk_id: str
    text: str
    score: float          # Similarity score in [0.0, 1.0], 1.0 = exact match
    distance: float       # Cosine distance
    metadata: DocumentMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "distance": round(self.distance, 4),
            "metadata": self.metadata.to_dict(),
        }


class VectorSearchEngine:
    """
    Search interface supporting semantic queries, multi-criteria filtering,
    and hybrid keyword-dense scoring.
    """

    def __init__(
        self,
        db: Optional[VectorDatabase] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        persist_dir: str = "./data/vector_db",
        collection_name: str = "threat_intelligence",
        default_top_k: int = 5,
        min_similarity_score: float = 0.30,
    ) -> None:
        self.db = db or VectorDatabase(persist_directory=persist_dir, collection_name=collection_name)
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.default_top_k = default_top_k
        self.min_similarity_score = min_similarity_score

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        protocol: Optional[str] = None,
        port: Optional[int] = None,
        cve_id: Optional[str] = None,
        min_cvss: Optional[float] = None,
        where_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Executes hybrid semantic search against vector store with keyword boosting and metadata post-filtering.
        """
        k = top_k or self.default_top_k
        cutoff = min_score if min_score is not None else self.min_similarity_score

        # Query embedding
        q_emb = self.embedding_engine.encode_query(query)

        # Build where clause if specific metadata supported
        filter_dict = dict(where_filter) if where_filter else {}
        if cve_id:
            filter_dict["cve_id"] = cve_id.upper()

        # Over-fetch to allow post-filtering and reranking
        fetch_k = max(k * 4, 20)
        raw_res = self.db.query(
            query_embeddings=[q_emb],
            n_results=fetch_k,
            where=filter_dict if filter_dict else None,
        )

        ids = raw_res.get("ids", [[]])[0]
        docs = raw_res.get("documents", [[]])[0]
        metas = raw_res.get("metadatas", [[]])[0]
        distances = raw_res.get("distances", [[]])[0]

        candidates: list[SearchResult] = []
        query_tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-]+\b", query) if len(t) > 2]

        for r_id, doc, meta_dict, dist in zip(ids, docs, metas, distances):
            # Base semantic score in [0, 1]
            vec_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            # Keyword lexical overlap boost
            doc_lower = doc.lower()
            keyword_score = 0.0
            if query_tokens:
                matches = sum(1 for t in query_tokens if t in doc_lower)
                keyword_score = matches / len(query_tokens)

            # Combined hybrid score (70% semantic, 30% keyword overlap)
            combined_score = (vec_score * 0.70) + (keyword_score * 0.30)

            if combined_score < cutoff:
                continue

            metadata = DocumentMetadata.from_chroma_metadata(meta_dict)

            # Protocol post-filter
            if protocol and metadata.affected_protocols:
                if protocol.upper() not in metadata.affected_protocols:
                    if protocol.lower() not in doc_lower:
                        continue

            # Port post-filter
            if port is not None and metadata.affected_ports:
                if port not in metadata.affected_ports:
                    if str(port) not in doc:
                        continue

            # CVSS post-filter
            if min_cvss is not None and metadata.cvss_score is not None:
                if metadata.cvss_score < min_cvss:
                    continue

            candidates.append(
                SearchResult(
                    chunk_id=r_id,
                    text=doc,
                    score=combined_score,
                    distance=dist,
                    metadata=metadata,
                )
            )

        # Sort descending by hybrid score
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:k]

    def search_by_cve(self, cve_id: str) -> Optional[SearchResult]:
        """Direct lookup for a specific CVE identifier."""
        res = self.search(query=cve_id, cve_id=cve_id, top_k=1, min_score=0.1)
        return res[0] if res else None

    def search_by_traffic_features(
        self,
        features: dict[str, Any],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """
        Specialized search interface converting network flow features into a search prompt.
        """
        proto = str(features.get("protocol", "TCP")).upper()
        dst_port = features.get("dst_port")
        service = features.get("service", "unknown")
        kdd_flag = features.get("kdd_flag", "")

        query_terms = [
            f"Protocol {proto}",
            f"port {dst_port}" if dst_port else "",
            f"service {service}" if service != "unknown" else "",
            f"flag {kdd_flag}" if kdd_flag else "",
            "intrusion attack vulnerability exploit",
        ]
        query_str = " ".join(t for t in query_terms if t)

        return self.search(
            query=query_str,
            top_k=top_k,
            protocol=proto,
            port=int(dst_port) if dst_port else None,
        )
