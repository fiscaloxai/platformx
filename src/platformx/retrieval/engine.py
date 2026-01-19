from typing import List, Optional
import logging

from .indexer import Indexer
from .query import GroundedQuery


class RetrievalResult:
    def __init__(self, text: str, source: dict, score: float):
        self.text = text
        self.source = source
        self.score = score


class RetrievalEngine:
    """Retrieval engine that runs grounded queries against an index.

    - Enforces max_results and dataset-level access control
    - Returns structured evidence (text + source attribution + retrieval score)
    - Does not perform any interpretation or generation
    """

    def __init__(self, indexer: Indexer, logger: Optional[logging.Logger] = None):
        self.indexer = indexer
        self._logger = logger or logging.getLogger("platformx.retrieval")

    def retrieve(self, query: GroundedQuery, max_results: int = 10, allowed_dataset_ids: Optional[List[str]] = None) -> List[RetrievalResult]:
        if max_results < 1:
            raise ValueError("max_results must be >= 1")

        # Query validation already handled by GroundedQuery

        raw = self.indexer.query(query.text, top_k=max_results * 5)

        # Filter by allowed datasets if provided
        filtered = []
        for item in raw:
            if allowed_dataset_ids and item["dataset_id"] not in allowed_dataset_ids:
                continue
            filtered.append(item)

        # Limit to max_results and compute a normalized confidence in [0,1]
        results = filtered[:max_results]
        max_score = max((r["score"] for r in results), default=1.0)
        output = []
        for r in results:
            conf = r["score"] / max_score if max_score > 0 else 0.0
            source = {"dataset_id": r["dataset_id"], "source": r.get("source"), "version": r.get("version"), "chunk_id": r.get("chunk_id")}
            output.append(RetrievalResult(text=r["text"], source=source, score=float(conf)))

        self._logger.info("Retrieved %d items for query", len(output))
        return output
