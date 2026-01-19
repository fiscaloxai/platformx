from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import Literal


class QueryIntent(str, Enum):
    informational = "informational"
    regulatory = "regulatory"
    safety = "safety"


class GroundedQuery(BaseModel):
    """Represents a query that must be grounded by retrieval only.

    Validation rules are intentionally strict to avoid speculative or generative-only queries.
    """

    text: str = Field(..., min_length=3, description="Raw query text")
    intent: QueryIntent = Field(..., description="High-level intent category")
    required_grounding_level: int = Field(1, ge=1, description="Minimum number of independent grounding sources")

    @validator("text")
    def no_generative_actions(cls, v: str) -> str:
        lowered = v.strip().lower()
        # Simple, deterministic heuristics to reject generative/speculative requests
        banned_phrases = ["generate", "create", "write", "compose", "fine-tune", "train", "synthesize"]
        for p in banned_phrases:
            if p in lowered:
                raise ValueError(f"Query appears generative/speculative (contains '{p}') and is not supported for retrieval-only grounding")
        if len(lowered) < 3:
            raise ValueError("Query text too short")
        return v

    @validator("required_grounding_level")
    def grounding_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("required_grounding_level must be >= 1")
        return v
