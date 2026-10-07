"""
knowledge_base/vector_store/database.py
───────────────────────────────────────
Vector Database storage manager for threat intelligence and CVE records.
Supports:
1. Native ChromaDB PersistentClient (production backend)
2. Lightweight persistent JSON/SQLite local vector engine (air-gapped edge fallback)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class LocalVectorStore:
    """
    Self-contained SQLite and memory-backed vector database.
    Zero external dependencies, supports persistence, metadata filtering,
    and cosine-distance nearest neighbor search.
    """

    def __init__(self, persist_path: str, collection_name: str = "threat_intel") -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.db_file = self.persist_path / f"{collection_name}.sqlite"
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_file), timeout=10.0)

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS records (
                        id TEXT PRIMARY KEY,
                        document TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        embedding BLOB NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                for doc_id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
                    emb_bytes = np.array(emb, dtype=np.float32).tobytes()
                    meta_str = json.dumps(meta)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO records (id, document, metadata, embedding)
                        VALUES (?, ?, ?, ?)
                        """,
                        (doc_id, doc, meta_str, emb_bytes),
                    )
                conn.commit()
            finally:
                conn.close()

    def count(self) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM records")
                row = cur.fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[list[Any]]]:
        """
        Executes exact cosine similarity search across stored vectors with optional metadata filter.
        """
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict[str, Any]] = []
        all_embs: list[np.ndarray] = []

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT id, document, metadata, embedding FROM records")
                for r_id, r_doc, r_meta, r_emb in cur.fetchall():
                    meta_dict = json.loads(r_meta)
                    # Metadata filtering if requested
                    if where:
                        match = True
                        for k, v in where.items():
                            if str(meta_dict.get(k)) != str(v):
                                match = False
                                break
                        if not match:
                            continue

                    emb_arr = np.frombuffer(r_emb, dtype=np.float32)
                    all_ids.append(r_id)
                    all_docs.append(r_doc)
                    all_metas.append(meta_dict)
                    all_embs.append(emb_arr)
            finally:
                conn.close()

        res_ids: list[list[str]] = []
        res_docs: list[list[str]] = []
        res_metas: list[list[dict[str, Any]]] = []
        res_distances: list[list[float]] = []

        if not all_embs:
            for _ in query_embeddings:
                res_ids.append([])
                res_docs.append([])
                res_metas.append([])
                res_distances.append([])
            return {
                "ids": res_ids,
                "documents": res_docs,
                "metadatas": res_metas,
                "distances": res_distances,
            }

        matrix = np.stack(all_embs)  # [N, D]
        # Normalize stored matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1e-9
        norm_matrix = matrix / norms

        for q in query_embeddings:
            q_arr = np.array(q, dtype=np.float32)
            q_norm = np.linalg.norm(q_arr)
            if q_norm < 1e-9:
                q_norm = 1e-9
            q_arr = q_arr / q_norm

            sims = np.dot(norm_matrix, q_arr)  # Cosine similarities in [-1, 1]
            distances = 1.0 - sims             # Cosine distance in [0, 2]

            # Top-k indices
            top_k = min(n_results, len(distances))
            sorted_indices = np.argsort(distances)[:top_k]

            res_ids.append([all_ids[i] for i in sorted_indices])
            res_docs.append([all_docs[i] for i in sorted_indices])
            res_metas.append([all_metas[i] for i in sorted_indices])
            res_distances.append([float(distances[i]) for i in sorted_indices])

        return {
            "ids": res_ids,
            "documents": res_docs,
            "metadatas": res_metas,
            "distances": res_distances,
        }

    def delete(self, ids: list[str]) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                for doc_id in ids:
                    conn.execute("DELETE FROM records WHERE id = ?", (doc_id,))
                conn.commit()
            finally:
                conn.close()

    def clear(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM records")
                conn.commit()
            finally:
                conn.close()

    def close(self) -> None:
        """Explicit cleanup."""
        pass


class VectorDatabase:
    """
    Unified Vector DB interface supporting ChromaDB or LocalVectorStore.
    """

    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "threat_intelligence",
        force_local: bool = False,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.backend = "local"
        self._chroma_client = None
        self._chroma_collection = None
        self._local_store = None

        if not force_local:
            self._try_init_chroma()

        if self.backend == "local":
            self._local_store = LocalVectorStore(
                persist_path=self.persist_directory,
                collection_name=self.collection_name,
            )

    def _try_init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
            logger.info(f"Initializing ChromaDB PersistentClient at {self.persist_directory}...")
            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "chromadb"
            logger.info("ChromaDB vector store successfully initialized.")
        except Exception as e:
            logger.warning(
                f"ChromaDB not available or failed to load ({e}). Using persistent LocalVectorStore."
            )
            self.backend = "local"

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if not ids:
            return

        if self.backend == "chromadb" and self._chroma_collection:
            try:
                self._chroma_collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
                return
            except Exception as e:
                logger.error(f"Error upserting to ChromaDB ({e}). Delegating to local store.")

        if self._local_store is None:
            self._local_store = LocalVectorStore(self.persist_directory, self.collection_name)
        self._local_store.add(ids, documents, metadatas, embeddings)

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[list[Any]]]:
        if self.backend == "chromadb" and self._chroma_collection:
            try:
                res = self._chroma_collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where,
                )
                return {
                    "ids": res.get("ids", [[]]),
                    "documents": res.get("documents", [[]]),
                    "metadatas": res.get("metadatas", [[]]),
                    "distances": res.get("distances", [[]]),
                }
            except Exception as e:
                logger.error(f"Error querying ChromaDB ({e}). Delegating to local store.")

        if self._local_store is None:
            self._local_store = LocalVectorStore(self.persist_directory, self.collection_name)
        return self._local_store.query(query_embeddings, n_results=n_results, where=where)

    def count(self) -> int:
        if self.backend == "chromadb" and self._chroma_collection:
            try:
                return self._chroma_collection.count()
            except Exception:
                pass
        if self._local_store:
            return self._local_store.count()
        return 0

    def delete(self, ids: list[str]) -> None:
        if self.backend == "chromadb" and self._chroma_collection:
            try:
                self._chroma_collection.delete(ids=ids)
            except Exception:
                pass
        if self._local_store:
            self._local_store.delete(ids)

    def close(self) -> None:
        if self._local_store:
            self._local_store.close()
