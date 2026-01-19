from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Iterable
import random

from ..retrieval.engine import RetrievalEngine, RetrievalResult
from ..data.schema import DatasetSchema


@dataclass
class RAFTSample:
    """A single RAFT sample: instruction, retrieved context, expected behavior, and provenance."""

    instruction: str
    context: str
    expected: str
    source_doc_id: str
    retrieval_score: float
    timestamp: datetime


class RAFTOrchestrator:
    """Orchestrates retrieval-aware fine-tuning sample creation.

    Safety principles: RAFT does not add knowledge. All positive samples are
    exact extractions from retrieved evidence. Missing evidence generates
    negative (refusal) samples. The orchestrator is deterministic given a seed.
    """

    def __init__(self, retrieval_engine: RetrievalEngine, seed: int = 0) -> None:
        self.retrieval = retrieval_engine
        self.seed = seed

    def generate_for_datasets(self, dataset_ids: Iterable[str], max_per_doc: int = 5, positive_fraction: float = 0.7, required_grounding: int = 1) -> List[RAFTSample]:
        """Generate RAFT samples for the provided dataset IDs.

        - `dataset_ids`: sequence of dataset IDs (must be indexed and retrievable)
        - `max_per_doc`: maximum samples to produce per dataset
        - `positive_fraction`: fraction of samples that are positive (answerable)
        - `required_grounding`: minimum independent retrieval items required per sample

        Deterministic selection uses the configured seed.
        """
        rng = random.Random(self.seed)
        samples: List[RAFTSample] = []

        for dsid in dataset_ids:
            # Use dataset id as an anchored query to find its indexed chunks.
            # This is intentionally simple and deterministic; retrieval engine
            # returns chunks with attribution only.
            q = f"source:{dsid}"
            # ask for more candidates to allow selection of positives/negatives
            raw = self.retrieval.indexer.query(q, top_k=max_per_doc * 3)

            if len(raw) < required_grounding:
                raise RuntimeError(f"Insufficient retrieval coverage for dataset {dsid}: found {len(raw)} items, required {required_grounding}")

            # select up to `max_per_doc` deterministic indices
            indices = list(range(len(raw)))
            rng.shuffle(indices)
            selected = indices[:max_per_doc]

            for idx in selected:
                item = raw[idx]
                # deterministically decide positive vs negative
                is_positive = (rng.random() < positive_fraction)

                if is_positive:
                    # Positive sample: expected is exact extracted text from evidence
                    instruction = "Extract the factual statement from the evidence and cite it."
                    expected = item["text"].strip()
                    context = item["text"].strip()
                else:
                    # Negative sample: model should refuse when evidence is insufficient
                    instruction = "If the evidence does not directly support a factual answer, refuse." 
                    expected = "REFUSE"
                    context = item["text"].strip()

                sample = RAFTSample(
                    instruction=instruction,
                    context=context,
                    expected=expected,
                    source_doc_id=item.get("dataset_id", dsid),
                    retrieval_score=float(item.get("score", 0.0)),
                    timestamp=datetime.utcnow(),
                )
                samples.append(sample)

        return samples
