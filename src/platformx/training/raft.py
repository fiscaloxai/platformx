from __future__ import annotations


from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Iterable, Dict, Any, Callable
from enum import Enum
import random
import logging
import hashlib

from ..retrieval.engine import RetrievalEngine



logger = logging.getLogger("platformx.training.raft")

class SampleType(Enum):
    POSITIVE_EXTRACT = "positive_extract"
    POSITIVE_REASONING = "positive_reasoning"
    NEGATIVE_REFUSE = "negative_refuse"
    NEGATIVE_DISTRACTOR = "negative_distractor"

@dataclass
class RAFTSample:
    """
    A single RAFT sample: instruction, context, expected output, provenance, and sample type.
    """
    instruction: str
    context: str
    expected: str
    source_doc_id: str
    retrieval_score: float
    timestamp: datetime
    sample_type: SampleType
    distractor_doc_ids: List[str] = field(default_factory=list)
    reasoning_chain: Optional[str] = None
    sample_id: str = ""

    def __post_init__(self):
        if not self.sample_id:
            base = self.instruction + self.context + self.expected
            self.sample_id = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]

class SampleTemplate:
    """
    Templates for generating different types of RAFT samples.
    """
    EXTRACT_INSTRUCTIONS: List[str] = [
        "Based on the provided evidence, extract the relevant factual information.",
        "Using only the context provided, answer the following question.",
        "Extract and cite the specific information from the evidence that answers this query.",
        "From the given context, identify and state the factual answer.",
    ]
    REASONING_INSTRUCTIONS: List[str] = [
        "Using the provided evidence, reason step-by-step to answer the question. Show your reasoning, then provide the final answer.",
        "Analyze the context carefully. Explain your reasoning process, then give a conclusion based solely on the evidence.",
        "Think through this step-by-step using only the provided context. State your reasoning, then your answer.",
    ]
    REFUSE_INSTRUCTIONS: List[str] = [
        "If the provided context does not contain sufficient information to answer, respond with REFUSE and explain why.",
        "Answer only if the evidence directly supports a response. If not, refuse to answer.",
        "Based on the context, provide an answer only if well-supported. Otherwise, indicate the information is insufficient.",
    ]
    DISTRACTOR_INSTRUCTIONS: List[str] = [
        "Multiple documents are provided. Only use information from relevant sources. If no source answers the question, refuse.",
        "You are given several context passages. Identify which, if any, contain the answer. Refuse if none are relevant.",
        "Some provided contexts may be distractors. Use only genuinely relevant evidence, or refuse if none applies.",
    ]

    @classmethod
    def get_instruction(cls, sample_type: SampleType, rng: random.Random) -> str:
        if sample_type == SampleType.POSITIVE_EXTRACT:
            return rng.choice(cls.EXTRACT_INSTRUCTIONS)
        elif sample_type == SampleType.POSITIVE_REASONING:
            return rng.choice(cls.REASONING_INSTRUCTIONS)
        elif sample_type == SampleType.NEGATIVE_REFUSE:
            return rng.choice(cls.REFUSE_INSTRUCTIONS)
        elif sample_type == SampleType.NEGATIVE_DISTRACTOR:
            return rng.choice(cls.DISTRACTOR_INSTRUCTIONS)
        else:
            return "Answer the question based on the provided context."

class RAFTConfig:
    """
    Configuration for RAFT sample generation.
    """
    def __init__(self,
                 positive_fraction: float = 0.6,
                 reasoning_fraction: float = 0.3,
                 distractor_fraction: float = 0.2,
                 max_distractors: int = 3,
                 min_retrieval_score: float = 0.1,
                 required_grounding: int = 1,
                 seed: int = 42):
        self.positive_fraction = positive_fraction
        self.reasoning_fraction = reasoning_fraction
        self.distractor_fraction = distractor_fraction
        self.max_distractors = max_distractors
        self.min_retrieval_score = min_retrieval_score
        self.required_grounding = required_grounding
        self.seed = seed



class RAFTOrchestrator:
    """
    Orchestrates retrieval-aware fine-tuning sample creation for RAFT.

    RAFT trains models to use retrieval properly: positive samples are grounded in evidence, negatives require refusal, and distractors test robustness. Reasoning chains improve model behavior.
    """
    def __init__(self, retrieval_engine: RetrievalEngine, config: Optional[RAFTConfig] = None):
        self.retrieval = retrieval_engine
        self.config = config or RAFTConfig()
        self.seed = self.config.seed
        self._logger = logging.getLogger("platformx.training.raft")

    def _create_positive_sample(self, item: Dict, rng: random.Random, include_reasoning: bool = False) -> RAFTSample:
        sample_type = SampleType.POSITIVE_REASONING if include_reasoning else SampleType.POSITIVE_EXTRACT
        instruction = SampleTemplate.get_instruction(sample_type, rng)
        context = item["text"].strip()
        expected = item["text"].strip()
        reasoning_chain = None
        if include_reasoning:
            reasoning_chain = f"Reasoning: The answer is found in the evidence.\n\nAnswer: {expected}"
        return RAFTSample(
            instruction=instruction,
            context=context,
            expected=expected,
            source_doc_id=item.get("dataset_id", ""),
            retrieval_score=float(item.get("score", 0.0)),
            timestamp=datetime.utcnow(),
            sample_type=sample_type,
            reasoning_chain=reasoning_chain,
        )

    def _create_negative_sample(self, item: Dict, rng: random.Random, distractors: List[Dict] = None) -> RAFTSample:
        distractors = distractors or []
        if distractors:
            sample_type = SampleType.NEGATIVE_DISTRACTOR
            context = item["text"].strip() + "\n---\n" + "\n---\n".join(d["text"].strip() for d in distractors)
            distractor_doc_ids = [d.get("dataset_id", "") for d in distractors]
        else:
            sample_type = SampleType.NEGATIVE_REFUSE
            context = item["text"].strip()
            distractor_doc_ids = []
        instruction = SampleTemplate.get_instruction(sample_type, rng)
        expected = "REFUSE: Insufficient evidence to answer this question."
        return RAFTSample(
            instruction=instruction,
            context=context,
            expected=expected,
            source_doc_id=item.get("dataset_id", ""),
            retrieval_score=float(item.get("score", 0.0)),
            timestamp=datetime.utcnow(),
            sample_type=sample_type,
            distractor_doc_ids=distractor_doc_ids,
        )

    def _get_distractors(self, exclude_dataset_id: str, rng: random.Random, count: int) -> List[Dict]:
        # Query for random documents not matching exclude_dataset_id
        all_chunks = self.retrieval.indexer._chunks.values()
        candidates = [c for c in all_chunks if c.dataset_id != exclude_dataset_id]
        rng.shuffle(candidates)
        return [
            {"text": c.text, "dataset_id": c.dataset_id, "score": 0.0}
            for c in candidates[:count]
        ]

    def generate_for_datasets(self, dataset_ids: Iterable[str], max_per_doc: int = 5) -> List[RAFTSample]:
        """
        Generate RAFT samples for the provided dataset IDs using config.
        Returns a list of RAFTSample objects.
        """
        rng = random.Random(self.seed)
        samples: List[RAFTSample] = []
        cfg = self.config
        for dsid in dataset_ids:
            q = f"source:{dsid}"
            raw = self.retrieval.indexer.query(q, top_k=max_per_doc * 3)
            if len(raw) < cfg.required_grounding:
                raise RuntimeError(f"Insufficient retrieval coverage for dataset {dsid}: found {len(raw)} items, required {cfg.required_grounding}")
            indices = list(range(len(raw)))
            rng.shuffle(indices)
            selected = indices[:max_per_doc]
            for idx in selected:
                item = raw[idx]
                roll = rng.random()
                if roll < cfg.positive_fraction:
                    # Positive sample
                    reasoning = (rng.random() < cfg.reasoning_fraction)
                    sample = self._create_positive_sample(item, rng, include_reasoning=reasoning)
                else:
                    # Negative sample
                    distractors = []
                    if rng.random() < cfg.distractor_fraction:
                        distractors = self._get_distractors(dsid, rng, cfg.max_distractors)
                    sample = self._create_negative_sample(item, rng, distractors=distractors)
                samples.append(sample)
        # Log summary
        by_type = {t: 0 for t in SampleType}
        for s in samples:
            by_type[s.sample_type] += 1
        self._logger.info(f"Generated {len(samples)} RAFT samples: " + ", ".join(f"{t.name}={by_type[t]}" for t in SampleType))
        return samples

    def generate_balanced_set(self, dataset_ids: Iterable[str], samples_per_type: int = 10) -> List[RAFTSample]:
        """
        Generate exactly samples_per_type for each SampleType, shuffled deterministically.
        """
        rng = random.Random(self.seed)
        samples: List[RAFTSample] = []
        for sample_type in SampleType:
            for dsid in dataset_ids:
                q = f"source:{dsid}"
                raw = self.retrieval.indexer.query(q, top_k=samples_per_type * 3)
                if not raw:
                    continue
                indices = list(range(len(raw)))
                rng.shuffle(indices)
                selected = indices[:samples_per_type]
                for idx in selected:
                    item = raw[idx]
                    if sample_type == SampleType.POSITIVE_EXTRACT:
                        sample = self._create_positive_sample(item, rng, include_reasoning=False)
                    elif sample_type == SampleType.POSITIVE_REASONING:
                        sample = self._create_positive_sample(item, rng, include_reasoning=True)
                    elif sample_type == SampleType.NEGATIVE_REFUSE:
                        sample = self._create_negative_sample(item, rng, distractors=[])
                    elif sample_type == SampleType.NEGATIVE_DISTRACTOR:
                        distractors = self._get_distractors(dsid, rng, self.config.max_distractors)
                        sample = self._create_negative_sample(item, rng, distractors=distractors)
                    samples.append(sample)
        rng.shuffle(samples)
        return samples

    def validate_samples(self, samples: List[RAFTSample]) -> Dict[str, Any]:
        """
        Validate a set of RAFT samples and return statistics.
        """
        stats = {
            "total_count": len(samples),
            "counts_by_type": {t.name: 0 for t in SampleType},
            "avg_context_length": 0.0,
            "avg_retrieval_score": 0.0,
            "unique_source_docs": set(),
        }
        total_len = 0
        total_score = 0.0
        for s in samples:
            stats["counts_by_type"][s.sample_type.name] += 1
            total_len += len(s.context)
            total_score += s.retrieval_score
            stats["unique_source_docs"].add(s.source_doc_id)
        stats["avg_context_length"] = total_len / len(samples) if samples else 0.0
        stats["avg_retrieval_score"] = total_score / len(samples) if samples else 0.0
        stats["unique_source_docs"] = len(stats["unique_source_docs"])
        self._logger.info(f"Validated {len(samples)} samples: {stats}")
        return stats

# Public API
__all__ = [
    "RAFTSample",
    "RAFTOrchestrator",
    "RAFTConfig",
    "SampleType",
    "SampleTemplate",
]
