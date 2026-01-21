"""
Embeddings module for PlatformX: provides embedding backends for text-to-vector conversion, essential for RAG retrieval.

Backends:
- TFIDFBackend: Simple, deterministic, no dependencies (good for testing)
- SentenceTransformerBackend: High-quality, requires sentence-transformers
- CachedEmbeddingBackend: Caching wrapper for reproducibility and speed

Example usage:
    backend = create_embedding_backend("tfidf")
    vec = backend.embed_text("Aspirin is a drug.")

    st_backend = create_embedding_backend("sentence-transformer", model_name="all-MiniLM-L6-v2")
    vecs = st_backend.embed_batch(["Aspirin", "Ibuprofen"])

    cached = CachedEmbeddingBackend(st_backend)
    cached.embed_text("Aspirin")
    print(cached.cache_stats())
"""
import logging
import math
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("platformx.retrieval.embeddings")

__all__ = [
    "EmbeddingBackend",
    "TFIDFBackend",
    "SentenceTransformerBackend",
    "CachedEmbeddingBackend",
    "create_embedding_backend",
]

class EmbeddingBackend(ABC):
    """
    Abstract base class for embedding backends.
    """
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert a single text to an embedding vector."""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert a batch of texts to embedding vectors (default: loop)."""
        return [self.embed_text(t) for t in texts]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension size."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for reproducibility."""
        pass

class TFIDFBackend(EmbeddingBackend):
    """
    Simple TF-IDF embedding backend (deterministic, no dependencies).
    Good for testing and fallback.
    """
    def __init__(self, max_features: int = 1000):
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._dimension: int = max_features
        self._fitted: bool = False

    def fit(self, documents: List[str]) -> None:
        """Build vocabulary and IDF from documents."""
        from collections import Counter, defaultdict
        doc_count = len(documents)
        term_doc_freq = defaultdict(int)
        for doc in documents:
            tokens = set(doc.lower().split())
            for token in tokens:
                term_doc_freq[token] += 1
        most_common = sorted(term_doc_freq.items(), key=lambda x: -x[1])[:self._dimension]
        self._vocabulary = {w: i for i, (w, _) in enumerate(most_common)}
        self._idf = {w: math.log((1 + doc_count) / (1 + freq)) + 1 for w, freq in most_common}
        self._fitted = True

    def embed_text(self, text: str) -> List[float]:
        if not self._fitted:
            self.fit([text])
        tokens = text.lower().split()
        vec = [0.0] * self._dimension
        for token in tokens:
            idx = self._vocabulary.get(token)
            if idx is not None:
                tf = tokens.count(token) / len(tokens)
                vec[idx] = tf * self._idf.get(token, 1.0)
        # L2 normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "tfidf-local"

class SentenceTransformerBackend(EmbeddingBackend):
    """
    Embedding backend using sentence-transformers (requires optional dependency).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dimension: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("sentence-transformers is required for SentenceTransformerBackend. Install with 'pip install sentence-transformers'.")
            self._model = SentenceTransformer(self._model_name, device=self._device)
            # Infer dimension
            emb = self._model.encode(["test"])
            self._dimension = len(emb[0])
            logger.info(f"Loaded sentence-transformer model: {self._model_name} (dim={self._dimension})")

    def embed_text(self, text: str) -> List[float]:
        self._ensure_loaded()
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        return [v.tolist() for v in self._model.encode(texts)]

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

class CachedEmbeddingBackend(EmbeddingBackend):
    """
    Caching wrapper for any EmbeddingBackend. Caches embeddings for reproducibility and speed.
    """
    def __init__(self, backend: EmbeddingBackend):
        self._backend = backend
        self._cache: Dict[str, List[float]] = {}
        self._hits = 0
        self._misses = 0

    def embed_text(self, text: str) -> List[float]:
        if text in self._cache:
            self._hits += 1
            return self._cache[text]
        vec = self._backend.embed_text(text)
        self._cache[text] = vec
        self._misses += 1
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        results = []
        uncached = []
        uncached_indices = []
        for i, t in enumerate(texts):
            if t in self._cache:
                results.append(self._cache[t])
                self._hits += 1
            else:
                results.append(None)
                uncached.append(t)
                uncached_indices.append(i)
        if uncached:
            new_vecs = self._backend.embed_batch(uncached)
            for idx, vec in zip(uncached_indices, new_vecs):
                self._cache[texts[idx]] = vec
                results[idx] = vec
                self._misses += 1
        return results

    def cache_stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "cached_items": len(self._cache)}

    def clear_cache(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def save_cache(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    def load_cache(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self._cache = json.load(f)

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    @property
    def model_name(self) -> str:
        return f"cached-{self._backend.model_name}"

def create_embedding_backend(backend_type: str = "tfidf", **kwargs) -> EmbeddingBackend:
    """
    Factory for embedding backends.
    backend_type: "tfidf", "sentence-transformer"/"st"
    """
    if backend_type == "tfidf":
        return TFIDFBackend(**kwargs)
    elif backend_type in ("sentence-transformer", "st"):
        return SentenceTransformerBackend(**kwargs)
    else:
        raise ValueError(f"Unknown embedding backend: {backend_type}")
