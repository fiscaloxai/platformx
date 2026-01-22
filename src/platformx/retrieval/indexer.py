
from typing import Protocol, List, Dict, Any, Iterable, Tuple, Optional
from dataclasses import dataclass
from math import sqrt
from collections import Counter
from .embeddings import EmbeddingBackend, TFIDFBackend, CachedEmbeddingBackend
import json
from pathlib import Path
import logging
import os

from ..data.schema import DatasetSchema, IntendedUse

logger = logging.getLogger("platformx.retrieval.indexer")

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        """Fallback if tqdm not available"""
        return iterable


class VectorBackend(Protocol):
    """Abstract vector backend interface.

    Implementations must be deterministic and in-memory for this phase.
    """

    def index(self, items: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
        ...

    def query(self, vector: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
        ...


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    dataset_id: str
    source: str
    version: str
    text: str


class LocalVectorStore:
    """Simple deterministic local vector store using TF or embedding backends and cosine similarity.
    This is intentionally simple, explainable, and requires no external services.
    Now supports pluggable embedding backends.
    """
    def __init__(self, embedding_backend: Optional[EmbeddingBackend] = None) -> None:
        self._embedding_backend = embedding_backend or TFIDFBackend()
        self._vectors: Dict[str, Dict[str, float]] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def _text_to_vector(self, text: str) -> Dict[str, float]:
        if self._embedding_backend:
            # Use embedding backend (returns list of floats)
            emb = self._embedding_backend.embed_text(text)
            return {str(i): v for i, v in enumerate(emb)}
        # fallback: legacy TF vector
        tokens = [t.lower() for t in text.split() if t]
        counts = Counter(tokens)
        vec = dict(counts)
        norm = sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def index(self, items: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
        for chunk_id, payload in items:
            text = payload["text"]
            vec = self._text_to_vector(text)
            self._vectors[chunk_id] = vec
            self._meta[chunk_id] = dict(payload)

    def query(self, vector: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for cid, vec in self._vectors.items():
            score = 0.0
            for k, v in vector.items():
                score += v * vec.get(k, 0.0)
            if score > 0:
                results.append((cid, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def metadata(self, chunk_id: str) -> Dict[str, Any]:
        return self._meta[chunk_id]

    def save_index(self, path: str) -> None:
        """Save vectors and meta to a JSON file."""
        data = {"vectors": self._vectors, "meta": self._meta}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"LocalVectorStore index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load vectors and meta from a JSON file. If file does not exist, set empty index."""
        if not os.path.exists(path):
            logger.warning(f"Index file {path} not found. Initializing empty index.")
            self._vectors = {}
            self._meta = {}
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._vectors = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in data["vectors"].items()}
        self._meta = data["meta"]
        logger.info(f"LocalVectorStore index loaded from {path} ({len(self._vectors)} chunks)")


class Indexer:
    """
    Builds retrieval indexes from approved datasets only.

    - Only datasets with `intended_use` including `retrieval` are accepted.
    - Uses a pluggable VectorBackend or embedding backend.
    - Supports overlapping chunks for improved retrieval.
    - Index persistence supported via save_index/load_index.
    """

    @classmethod
    def load(cls, directory: str) -> "Indexer":
        """Load an Indexer from a directory."""
        indexer = cls()
        indexer.load_index(directory)
        return indexer

    def __init__(self, backend: VectorBackend = None, chunk_size: int = 200, chunk_overlap: int = 50, embedding_backend: EmbeddingBackend = None, show_progress: bool = True) -> None:
        if backend is None and embedding_backend is not None:
            self.backend = LocalVectorStore(embedding_backend=embedding_backend)
        elif backend is None:
            self.backend = LocalVectorStore()
        else:
            self.backend = backend
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.show_progress = show_progress and TQDM_AVAILABLE
        self._chunks: Dict[str, IndexedChunk] = {}

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        if self.chunk_overlap <= 0:
            return [" ".join(words[i : i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
        # Overlapping chunks
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk))
            if len(chunk) < self.chunk_size:
                break
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def _chunk_text_with_metadata(self, text: str) -> List[Dict[str, Any]]:
        words = text.split()
        if not words:
            return []
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i : i + self.chunk_size]
            start_char = len(" ".join(words[:i]))
            end_char = start_char + len(" ".join(chunk))
            chunks.append({"text": " ".join(chunk), "start_char": start_char, "end_char": end_char})
            if len(chunk) < self.chunk_size:
                break
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def index_dataset(self, dataset: DatasetSchema) -> List[str]:
        if dataset.intended_use != IntendedUse.RETRIEVAL:
            raise ValueError("Dataset intended_use must be 'RETRIEVAL' to be indexed")

        if not dataset.raw_text:
            raise ValueError("Dataset has no raw_text available; text extraction required before indexing")

        chunks = self._chunk_text(dataset.raw_text)
        to_index = []
        ids = []
        
        # Use progress bar if available
        chunk_iter = tqdm(enumerate(chunks), total=len(chunks), desc=f"Indexing {dataset.dataset_id}", disable=not self.show_progress)
        
        for i, c in chunk_iter:
            chunk_id = f"{dataset.dataset_id}::v{dataset.version}::chunk{i}"
            meta = {
                "text": c,
                "dataset_id": dataset.dataset_id,
                "source": dataset.provenance.source_uri or "",
                "version": dataset.version,
                "chunk_index": i,
            }
            to_index.append((chunk_id, meta))
            self._chunks[chunk_id] = IndexedChunk(chunk_id=chunk_id, dataset_id=dataset.dataset_id, source=meta["source"], version=dataset.version, text=c)
            ids.append(chunk_id)

        self.backend.index(to_index)
        logger.info(f"Indexed dataset {dataset.dataset_id} into {len(ids)} chunks.")
        return ids

    def index_datasets(self, datasets: Iterable[DatasetSchema]) -> List[str]:
        ids = []
        for ds in datasets:
            ids.extend(self.index_dataset(ds))
        return ids

    def retrieve(self, query_text: str, top_k: int = 10, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Query the index for relevant chunks. Optionally filter by min_score.
        Alias for query() method for better readability.
        """
        return self.query(query_text, top_k, min_score)

    def query(self, query_text: str, top_k: int = 10, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Query the index for relevant chunks. Optionally filter by min_score.
        """
        if not hasattr(self.backend, "_text_to_vector"):
            tokens = [t.lower() for t in query_text.split() if t]
            counts = Counter(tokens)
            norm = sqrt(sum(v * v for v in counts.values())) or 1.0
            qvec = {k: v / norm for k, v in counts.items()}
        else:
            qvec = self.backend._text_to_vector(query_text)

        raw_results = self.backend.query(qvec, top_k)
        structured = []
        for cid, score in raw_results:
            if score < min_score:
                continue
            meta = self.backend.metadata(cid)
            structured.append({
                "chunk_id": cid,
                "text": meta["text"],
                "dataset_id": meta["dataset_id"],
                "source": meta.get("source"),
                "version": meta.get("version"),
                "score": float(score),
            })
        logger.info(f"Query '{query_text}' returned {len(structured)} results (min_score={min_score})")
        return structured

    def save_index(self, directory: str) -> None:
        """Save backend index and chunk registry to directory."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        if isinstance(self.backend, LocalVectorStore):
            self.backend.save_index(str(Path(directory) / "vectors.json"))
        with open(str(Path(directory) / "chunks.json"), "w", encoding="utf-8") as f:
            json.dump({k: v.__dict__ for k, v in self._chunks.items()}, f)
        meta = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "backend": str(type(self.backend)),
        }
        with open(str(Path(directory) / "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        logger.info(f"Index saved to {directory}")

    def load_index(self, directory: str) -> None:
        """Load backend index and chunk registry from directory. If files do not exist, set empty index."""
        if isinstance(self.backend, LocalVectorStore):
            self.backend.load_index(str(Path(directory) / "vectors.json"))
        chunks_path = str(Path(directory) / "chunks.json")
        meta_path = str(Path(directory) / "meta.json")
        if not os.path.exists(chunks_path):
            logger.warning(f"Chunks file {chunks_path} not found. Initializing empty chunk registry.")
            self._chunks = {}
        else:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            self._chunks = {k: IndexedChunk(**v) for k, v in chunk_data.items()}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                logger.info(f"Index loaded from {directory} ({len(self._chunks)} chunks)")
        else:
            logger.warning(f"Meta file {meta_path} not found. Initializing with defaults.")

    def get_chunk(self, chunk_id: str) -> Optional[IndexedChunk]:
        """Return chunk by ID, or None if not found."""
        return self._chunks.get(chunk_id)

    def chunk_count(self) -> int:
        """Return total number of indexed chunks."""
        return len(self._chunks)

    def list_datasets(self) -> List[str]:
        """Return list of unique dataset IDs in the index."""
        dataset_ids = set()
        for chunk in self._chunks.values():
            dataset_ids.add(chunk.dataset_id)
        return sorted(list(dataset_ids))


# Public API
__all__ = [
    "VectorBackend",
    "IndexedChunk",
    "LocalVectorStore",
    "Indexer",
]
