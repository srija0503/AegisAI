"""
agentic_rag/tools/local_database.py
───────────────────────────────────
RAG Tool: Interacts with the local SQLite / Vector database to inspect
indexed vulnerability counts, past incident logs, and locally active rules.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from knowledge_base.vector_store.database import VectorDatabase

logger = logging.getLogger(__name__)


class LocalDatabaseTool:
    """
    Tool allowing RAG components to query local database metrics and collections.
    """

    def __init__(self, db: Optional[VectorDatabase] = None) -> None:
        self.db = db or VectorDatabase()

    def get_collection_statistics(self) -> dict[str, Any]:
        """Returns statistics on vector database records."""
        return {
            "collection_name": self.db.collection_name,
            "total_indexed_records": self.db.count(),
            "backend": self.db.backend,
            "storage_path": self.db.persist_directory,
        }

    def get_tool_metadata(self) -> dict[str, Any]:
        return {
            "name": "local_database",
            "description": "Inspects vector database status, collections, and record counts.",
            "parameters": {},
        }
