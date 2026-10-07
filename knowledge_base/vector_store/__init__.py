"""
knowledge_base/vector_store package
"""

from knowledge_base.vector_store.database import LocalVectorStore, VectorDatabase
from knowledge_base.vector_store.embeddings import (
    EmbeddingEngine,
    FallbackHashingEmbedding,
)
from knowledge_base.vector_store.index import VectorIndex
from knowledge_base.vector_store.search import SearchResult, VectorSearchEngine

__all__ = [
    "EmbeddingEngine",
    "FallbackHashingEmbedding",
    "LocalVectorStore",
    "SearchResult",
    "VectorDatabase",
    "VectorIndex",
    "VectorSearchEngine",
]
