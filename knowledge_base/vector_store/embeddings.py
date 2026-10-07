"""
knowledge_base/vector_store/embeddings.py
──────────────────────────────────────────
Embedding generation engine for threat intelligence retrieval.
Provides native support for SentenceTransformers (all-MiniLM-L6-v2)
with an automatic, resilient PyTorch/NumPy semantic hashing fallback
so the edge firewall operates reliably in air-gapped or lightweight environments.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


class FallbackHashingEmbedding:
    """
    Resilient, deterministic dense semantic embedding engine using subword
    tokenization, term-frequency weighting, and random projection (Murmur/SHA256).
    Produces normalized 384-dimensional dense vectors identical in shape to MiniLM.
    Requires only standard Python and NumPy / PyTorch.
    """

    def __init__(self, dimension: int = 384, seed: int = 42) -> None:
        self.dimension = dimension
        self.seed = seed
        rng = np.random.RandomState(seed)
        # Random orthogonal projection matrix for dimensional reduction
        self._projection = rng.randn(2048, self.dimension)
        self._projection /= np.linalg.norm(self._projection, axis=1, keepdims=True)

    def _tokenize(self, text: str) -> list[str]:
        # Extract word tokens and alphanumeric security tokens (e.g., CVE-2021-44228)
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
        subwords = []
        for t in tokens:
            subwords.append(t)
            # Character n-grams for typo resilience and substring matching
            if len(t) > 3:
                for i in range(len(t) - 2):
                    subwords.append(t[i : i + 3])
        return subwords

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)

        # Hash tokens into 2048-dim bag-of-words
        bow = np.zeros(2048, dtype=np.float32)
        for t in tokens:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % 2048
            bow[h] += 1.0

        # Sublinear term frequency scaling: 1 + log(tf) safely
        pos_indices = np.where(bow > 0)[0]
        bow[pos_indices] = 1.0 + np.log(bow[pos_indices])

        # Project to target dimension
        vec = np.dot(bow, self._projection)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t).tolist() for t in texts]


class EmbeddingEngine:
    """
    Main embedding gateway.
    Automatically loads SentenceTransformer if available, or seamlessly uses FallbackHashingEmbedding.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        device: Optional[str] = None,
        force_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.device = device
        self._backend = "fallback"
        self._st_model = None
        self._fallback_model = FallbackHashingEmbedding(dimension=dimension)

        if not force_fallback:
            self._try_load_sentence_transformer()

    def _try_load_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
            self._st_model = SentenceTransformer(self.model_name, device=self.device)
            self._backend = "sentence_transformers"
            self.dimension = self._st_model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer successfully initialized ({self._backend}, dim={self.dimension}).")
        except Exception as e:
            logger.warning(
                f"SentenceTransformer unavailable ({e}). Using resilient high-speed hashing fallback."
            )
            self._backend = "fallback"

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, texts: Union[str, list[str]], normalize: bool = True) -> list[list[float]]:
        """
        Generates dense vector embeddings for input text(s).
        Always returns a list of float lists.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        if self._backend == "sentence_transformers" and self._st_model is not None:
            try:
                embeddings = self._st_model.encode(
                    texts,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.error(f"Error in SentenceTransformer encoding: {e}. Falling back to internal engine.")

        # Fallback path
        return self._fallback_model.embed_batch(texts)

    def encode_query(self, query: str) -> list[float]:
        """Convenience method for a single search query."""
        results = self.encode([query])
        return results[0] if results else [0.0] * self.dimension

    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Computes cosine similarity between two float vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
