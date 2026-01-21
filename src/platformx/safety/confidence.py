"""
Confidence assessment for PlatformX: multiple scoring strategies, semantic similarity, and ensemble support.

Provides:
- ConfidenceScore, ConfidenceConfig, ConfidenceStrategy
- ConfidenceScorer, TokenOverlapScorer, SemanticSimilarityScorer, SourceAgreementScorer, RetrievalScoreAggregator, EnsembleConfidenceScorer
- ConfidenceAssessor: high-level interface
- create_confidence_assessor: factory for custom configuration

Scoring strategies:
- Token overlap: simple lexical agreement
- Semantic similarity: embedding-based
- Source agreement: independent source confirmation
- Weighted ensemble: combine multiple strategies

Thresholds and weights are configurable for different use cases.
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import math
from collections import Counter
logger = logging.getLogger("platformx.safety.confidence")


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceStrategy(str, Enum):
    """Strategy for computing confidence scores."""
    TOKEN_OVERLAP = "token_overlap"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SOURCE_AGREEMENT = "source_agreement"
    WEIGHTED_ENSEMBLE = "weighted_ensemble"


@dataclass
class ConfidenceScore:
    """Detailed confidence score with breakdown."""
    level: ConfidenceLevel
    score: float
    strategy_used: ConfidenceStrategy
    breakdown: Dict[str, float] = field(default_factory=dict)
    evidence_count: int = 0
    source_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "score": self.score,
            "strategy_used": self.strategy_used,
            "breakdown": self.breakdown,
            "evidence_count": self.evidence_count,
            "source_ids": self.source_ids,
            "warnings": self.warnings,
        }


@dataclass
class ConfidenceConfig:
    """Configuration for confidence assessment."""
    strategy: ConfidenceStrategy = ConfidenceStrategy.WEIGHTED_ENSEMBLE
    high_threshold: float = 0.75
    medium_threshold: float = 0.4
    min_sources: int = 1
    required_agreement: float = 0.5
    weights: Dict[str, float] = field(default_factory=lambda: {
        "retrieval_score": 0.3,
        "source_agreement": 0.3,
        "evidence_coverage": 0.2,
        "source_authority": 0.2
    })


class ConfidenceScorer(ABC):
    """Abstract base class for confidence scoring strategies."""
    @abstractmethod
    def compute(self, evidence: List[Dict[str, Any]], query: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Compute confidence score from evidence.
        Returns (score, details_dict).
        """
        pass

    @abstractmethod
    def scorer_id(self) -> str:
        """Return identifier for this scorer."""
        pass


class TokenOverlapScorer(ConfidenceScorer):
    """Score based on token overlap between evidence items."""
    def __init__(self):
        self._logger = logger

    def compute(self, evidence: List[Dict[str, Any]], query: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        texts = [item.get("text", "") for item in evidence if item.get("text")]
        n = len(texts)
        pairwise_overlap = _token_overlap(texts)
        query_overlap = 0.0
        unique_tokens = len(set(" ".join(texts).split())) if texts else 0
        if query:
            query_tokens = set(query.lower().split())
            evidence_tokens = set(" ".join(texts).lower().split())
            if evidence_tokens:
                query_overlap = len(query_tokens & evidence_tokens) / max(1, len(query_tokens | evidence_tokens))
        return pairwise_overlap, {"pairwise_overlap": pairwise_overlap, "query_overlap": query_overlap, "unique_tokens": unique_tokens}

    def scorer_id(self) -> str:
        return "token_overlap"


class SemanticSimilarityScorer(ConfidenceScorer):
    """Score based on semantic similarity using embeddings."""
    def __init__(self, embedding_backend: Optional[Any] = None):
        self.embedding_backend = embedding_backend
        self._logger = logger

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.embedding_backend:
            return self.embedding_backend.embed(texts)
        # fallback: simple bag-of-words tf-idf style vector
        vocab = list(set(word for t in texts for word in t.lower().split()))
        vectors = []
        for t in texts:
            tokens = t.lower().split()
            vec = [tokens.count(w) for w in vocab]
            vectors.append(vec)
        return vectors

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def compute(self, evidence: List[Dict[str, Any]], query: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        texts = [item.get("text", "") for item in evidence if item.get("text")]
        if not texts:
            return 0.0, {"pairwise_similarities": [], "query_similarities": [], "min_similarity": 0.0, "max_similarity": 0.0}
        vectors = self._get_embeddings(texts)
        pairwise_similarities = []
        n = len(vectors)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(vectors[i], vectors[j])
                pairwise_similarities.append(sim)
        avg_pairwise = sum(pairwise_similarities) / len(pairwise_similarities) if pairwise_similarities else 1.0
        query_similarities = []
        if query:
            q_vecs = self._get_embeddings([query])
            q_vec = q_vecs[0] if q_vecs else []
            for v in vectors:
                query_similarities.append(self._cosine_similarity(q_vec, v))
        min_sim = min(pairwise_similarities) if pairwise_similarities else 0.0
        max_sim = max(pairwise_similarities) if pairwise_similarities else 0.0
        return avg_pairwise, {"pairwise_similarities": pairwise_similarities, "query_similarities": query_similarities, "min_similarity": min_sim, "max_similarity": max_sim}

    def scorer_id(self) -> str:
        return "semantic_similarity"


class SourceAgreementScorer(ConfidenceScorer):
    """Score based on agreement between independent sources."""
    def __init__(self, require_different_sources: bool = True):
        self.require_different_sources = require_different_sources
        self._logger = logger

    def compute(self, evidence: List[Dict[str, Any]], query: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        source_ids = [item.get("dataset_id") or item.get("source") or "unknown" for item in evidence]
        unique_sources = set(source_ids)
        cross_source_agreement = 1.0 if len(unique_sources) > 1 else 0.5
        return cross_source_agreement, {"unique_sources": len(unique_sources), "source_ids": list(unique_sources), "cross_source_agreement": cross_source_agreement}

    def scorer_id(self) -> str:
        return "source_agreement"


class RetrievalScoreAggregator(ConfidenceScorer):
    """Score based on aggregated retrieval scores."""
    def __init__(self, aggregation: str = "mean"):
        self.aggregation = aggregation
        self._logger = logger

    def compute(self, evidence: List[Dict[str, Any]], query: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        scores = [float(item.get("score", 0.0)) for item in evidence]
        if not scores:
            return 0.0, {"scores": [], "aggregation": self.aggregation, "min": 0.0, "max": 0.0}
        if self.aggregation == "mean":
            agg = sum(scores) / len(scores)
        elif self.aggregation == "max":
            agg = max(scores)
        elif self.aggregation == "weighted_mean":
            # For demo, treat as mean
            agg = sum(scores) / len(scores)
        else:
            agg = sum(scores) / len(scores)
        return min(max(agg, 0.0), 1.0), {"scores": scores, "aggregation": self.aggregation, "min": min(scores), "max": max(scores)}

    def scorer_id(self) -> str:
        return f"retrieval_{self.aggregation}"


class EnsembleConfidenceScorer(ConfidenceScorer):
    """Combine multiple scorers with configurable weights."""
    def scorer_id(self) -> str:
        return "ensemble"
    def __init__(self):
        self._scorers: List[Tuple[ConfidenceScorer, float]] = []
        self._logger = logger

    def add_scorer(self, scorer: ConfidenceScorer, weight: float = 1.0) -> "EnsembleConfidenceScorer":
        self._scorers.append((scorer, weight))
        return self

    def compute(self, evidence: List[Dict[str, Any]], query: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        total_weight = sum(w for _, w in self._scorers) or 1.0
        scores = {}
        details = {}
        weighted_sum = 0.0
        for scorer, weight in self._scorers:
            score, breakdown = scorer.compute(evidence, query)
            scores[scorer.scorer_id()] = score
            details[scorer.scorer_id()] = breakdown
            weighted_sum += score * weight
        weighted_score = weighted_sum / total_weight
        return weighted_score, {"scorer_scores": scores, "scorer_details": details, "weights": {s.scorer_id(): w for s, w in self._scorers}}


class ConfidenceAssessor:
    """High-level confidence assessment with configurable strategies."""
    def __init__(self, config: Optional[ConfidenceConfig] = None, embedding_backend: Optional[Any] = None):
        self._config = config or ConfidenceConfig()
        self._logger = logger
        strat = self._config.strategy
        if strat == ConfidenceStrategy.TOKEN_OVERLAP:
            self._scorer = TokenOverlapScorer()
        elif strat == ConfidenceStrategy.SEMANTIC_SIMILARITY:
            self._scorer = SemanticSimilarityScorer(embedding_backend)
        elif strat == ConfidenceStrategy.SOURCE_AGREEMENT:
            self._scorer = SourceAgreementScorer()
        elif strat == ConfidenceStrategy.WEIGHTED_ENSEMBLE:
            ens = EnsembleConfidenceScorer()
            weights = self._config.weights
            ens.add_scorer(TokenOverlapScorer(), weights.get("token_overlap", 0.3))
            ens.add_scorer(SemanticSimilarityScorer(embedding_backend), weights.get("semantic_similarity", 0.3))
            ens.add_scorer(SourceAgreementScorer(), weights.get("source_agreement", 0.2))
            ens.add_scorer(RetrievalScoreAggregator(), weights.get("retrieval_score", 0.2))
            self._scorer = ens
        else:
            self._scorer = TokenOverlapScorer()

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        if score >= self._config.high_threshold:
            return ConfidenceLevel.HIGH
        elif score >= self._config.medium_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def assess(self, evidence: List[Dict[str, Any]], query: Optional[str] = None) -> ConfidenceScore:
        if not evidence:
            return ConfidenceScore(
                level=ConfidenceLevel.LOW,
                score=0.0,
                strategy_used=self._config.strategy,
                breakdown={},
                evidence_count=0,
                source_ids=[],
                warnings=["No evidence provided"],
            )
        score, breakdown = self._scorer.compute(evidence, query)
        level = self._score_to_level(score)
        source_ids = [item.get("dataset_id") or item.get("source") or "unknown" for item in evidence]
        warnings = []
        if len(set(source_ids)) < self._config.min_sources:
            warnings.append(f"Only {len(set(source_ids))} unique source(s), less than min_sources={self._config.min_sources}")
        return ConfidenceScore(
            level=level,
            score=score,
            strategy_used=self._config.strategy,
            breakdown=breakdown,
            evidence_count=len(evidence),
            source_ids=source_ids,
            warnings=warnings,
        )

    def assess_with_threshold(self, evidence: List[Dict[str, Any]], min_level: ConfidenceLevel, query: Optional[str] = None) -> Tuple[ConfidenceScore, bool]:
        score = self.assess(evidence, query)
        passed = (score.level == min_level or
                  (min_level == ConfidenceLevel.MEDIUM and score.level == ConfidenceLevel.HIGH) or
                  (min_level == ConfidenceLevel.LOW))
        return score, passed

def create_confidence_assessor(
    strategy: str = "ensemble",
    embedding_backend: Optional[Any] = None,
    **config_kwargs
) -> ConfidenceAssessor:
    """
    Factory for ConfidenceAssessor.
    strategy: "token_overlap", "semantic_similarity", "source_agreement", "ensemble"
    """
    strat_enum = {
        "token_overlap": ConfidenceStrategy.TOKEN_OVERLAP,
        "semantic_similarity": ConfidenceStrategy.SEMANTIC_SIMILARITY,
        "source_agreement": ConfidenceStrategy.SOURCE_AGREEMENT,
        "ensemble": ConfidenceStrategy.WEIGHTED_ENSEMBLE,
        "weighted_ensemble": ConfidenceStrategy.WEIGHTED_ENSEMBLE,
    }.get(strategy, ConfidenceStrategy.WEIGHTED_ENSEMBLE)
    config = ConfidenceConfig(strategy=strat_enum, **config_kwargs)
    return ConfidenceAssessor(config, embedding_backend)


def _token_overlap(texts: List[str]) -> float:
    """Compute a simple, deterministic agreement score between texts (0..1)."""
    token_sets = [set(t.lower().split()) for t in texts if t]
    if not token_sets:
        return 0.0
    # intersection over union across all pairs approximated by average pairwise overlap
    total = 0.0
    count = 0
    n = len(token_sets)
    for i in range(n):
        for j in range(i + 1, n):
            a = token_sets[i]
            b = token_sets[j]
            if not a or not b:
                overlap = 0.0
            else:
                overlap = len(a & b) / len(a | b)
            total += overlap
            count += 1
    return total / count if count else 1.0


def assess_confidence(retrieved_evidence: List[Dict[str, Any]], required_grounding: int = 1) -> Dict[str, Any]:
    """Assess system-derived confidence from retrieval coverage and agreement.

    Returns structured dict: { level: ConfidenceLevel, rationale: {...} }

    For backward compatibility, uses ConfidenceAssessor with default config.
    """
    assessor = ConfidenceAssessor(ConfidenceConfig(min_sources=required_grounding))
    score = assessor.assess(retrieved_evidence)
    return {"level": score.level, "rationale": score.breakdown}
    # ...existing code...
    def scorer_id(self) -> str:
        return "ensemble"

__all__ = [
    "ConfidenceLevel",
    "ConfidenceScore",
    "ConfidenceConfig",
    "ConfidenceStrategy",
    "ConfidenceScorer",
    "TokenOverlapScorer",
    "SemanticSimilarityScorer",
    "SourceAgreementScorer",
    "RetrievalScoreAggregator",
    "EnsembleConfidenceScorer",
    "ConfidenceAssessor",
    "create_confidence_assessor",
    "assess_confidence",
]
