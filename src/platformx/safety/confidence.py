from enum import Enum
from typing import List, Dict, Any
from collections import Counter


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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

    Deterministic, rule-based thresholds:
      - HIGH: >= required_grounding independent sources AND average overlap >= 0.75
      - MEDIUM: >= required_grounding AND average overlap >= 0.4
      - LOW: otherwise
    """
    texts = [item.get("text", "") for item in retrieved_evidence if item.get("text")]
    count = len(texts)
    avg_score = 0.0
    if retrieved_evidence:
        avg_score = sum(float(item.get("score", 0.0)) for item in retrieved_evidence) / max(1, len(retrieved_evidence))

    agreement = _token_overlap(texts)

    rationale = {"retrieved_count": count, "avg_score": avg_score, "agreement": agreement}

    if count >= required_grounding and agreement >= 0.75:
        level = ConfidenceLevel.HIGH
    elif count >= required_grounding and agreement >= 0.4:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    return {"level": level, "rationale": rationale}
