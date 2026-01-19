from typing import Protocol, List, Dict, Any, Iterable, Tuple
from dataclasses import dataclass
from math import sqrt
from collections import Counter

from ..data.schema import DatasetSchema, IntendedUse


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
    """Simple deterministic local vector store using TF vectors and cosine similarity.

    This is intentionally simple, explainable, and requires no external services.
    """

    def __init__(self) -> None:
        # maps chunk_id -> vector (dict token->float)
        self._vectors: Dict[str, Dict[str, float]] = {}
        # maps chunk_id -> metadata
        self._meta: Dict[str, Dict[str, Any]] = {}

    def _text_to_vector(self, text: str) -> Dict[str, float]:
        tokens = [t.lower() for t in text.split() if t]
        counts = Counter(tokens)
        # convert to normalized float vector
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
            # compute dot product (since vectors are L2-normalized)
            score = 0.0
            for k, v in vector.items():
                score += v * vec.get(k, 0.0)
            if score > 0:
                results.append((cid, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def metadata(self, chunk_id: str) -> Dict[str, Any]:
        return self._meta[chunk_id]


class Indexer:
    """Builds retrieval indexes from approved datasets only.

    - Only datasets with `intended_use` including `retrieval` are accepted.
    - Uses a pluggable VectorBackend; defaults to LocalVectorStore.
    - Stores chunk text, dataset id, source reference, and version metadata.
    """

    def __init__(self, backend: VectorBackend = None, chunk_size_words: int = 200) -> None:
        self.backend = backend or LocalVectorStore()
        self.chunk_size_words = chunk_size_words
        # keep registry of chunks
        self._chunks: Dict[str, IndexedChunk] = {}

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = [" ".join(words[i : i + self.chunk_size_words]) for i in range(0, len(words), self.chunk_size_words)]
        return chunks

    def index_dataset(self, dataset: DatasetSchema) -> List[str]:
        if dataset.intended_use != IntendedUse.retrieval:
            raise ValueError("Dataset intended_use must be 'retrieval' to be indexed")

        if not dataset.raw_text:
            # We do not parse PDFs or binary content here.
            raise ValueError("Dataset has no raw_text available; text extraction required before indexing")

        chunks = self._chunk_text(dataset.raw_text)
        to_index = []
        ids = []
        for i, c in enumerate(chunks):
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

        # index with backend
        self.backend.index(to_index)
        return ids

    def index_datasets(self, datasets: Iterable[DatasetSchema]) -> List[str]:
        ids = []
        for ds in datasets:
            ids.extend(self.index_dataset(ds))
        return ids

    def query(self, query_text: str, top_k: int = 10):
        # Build query vector deterministically using same logic as backend
        if not hasattr(self.backend, "_text_to_vector"):
            # fallback: simple tokenization
            tokens = [t.lower() for t in query_text.split() if t]
            counts = Counter(tokens)
            norm = sqrt(sum(v * v for v in counts.values())) or 1.0
            qvec = {k: v / norm for k, v in counts.items()}
        else:
            qvec = self.backend._text_to_vector(query_text)

        raw_results = self.backend.query(qvec, top_k)
        # map to structured results
        structured = []
        for cid, score in raw_results:
            meta = self.backend.metadata(cid)
            structured.append({
                "chunk_id": cid,
                "text": meta["text"],
                "dataset_id": meta["dataset_id"],
                "source": meta.get("source"),
                "version": meta.get("version"),
                "score": float(score),
            })
        return structured
