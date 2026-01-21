"""
Safety module for PlatformX.

This module provides safety filtering, confidence assessment, and refusal handling
for building production-quality LLM applications with responsible AI practices.

Safety Philosophy:
- Wrong answers can cause real harm
- Silence is safer than speculation
- Refusal preserves trust
- Traceability matters more than fluency
- Accuracy > coverage > speed

Core Components:
- SafetyFilterChain: Configurable chain of safety filters
- ConfidenceAssessor: Multi-strategy confidence scoring
- RefusalEngine: Professional refusal message generation

Filters:
- KeywordFilter: Block based on keyword matching
- RegexFilter: Block based on regex patterns
- PIIFilter: Detect personally identifiable information
- IntentFilter: Filter based on query intent

Confidence Scoring:
- TokenOverlapScorer: Simple token-based agreement
- SemanticSimilarityScorer: Embedding-based similarity
- SourceAgreementScorer: Cross-source verification
- EnsembleConfidenceScorer: Weighted combination of scorers

Example:
	from platformx.safety import (
		SafetyFilterChain,
		PIIFilter,
		ConfidenceAssessor,
		RefusalEngine,
		create_default_filter_chain
	)

	# Setup safety chain
	chain = create_default_filter_chain(domain="general")
	chain.add_filter(PIIFilter(), priority=10)  # High priority

	# Check query
	result = chain.check("What is my SSN 123-45-6789?")
	if result["decision"] == Decision.BLOCK:
		refusal = RefusalEngine().make_refusal(result["reason"].value)
		print(refusal.message)

	# Assess confidence in retrieved evidence
	assessor = ConfidenceAssessor()
	confidence = assessor.assess(evidence_list, query="What is X?")
	if confidence.level == ConfidenceLevel.LOW:
		print("Warning: Low confidence response")
"""

import logging
logger = logging.getLogger("platformx.safety")

# --- Filters ---
try:
	from .filters import (
		Decision,
		ReasonCode,
		FilterRule,
		SafetyFilter,
		KeywordFilter,
		RegexFilter,
		PIIFilter,
		IntentFilter,
		SafetyFilterChain,
		evaluate_safety,
		create_default_filter_chain,
	)
except ImportError as e:
	logger.warning(f"Could not import safety.filters: {e}")

# --- Confidence ---
try:
	from .confidence import (
		ConfidenceLevel,
		ConfidenceScore,
		ConfidenceConfig,
		ConfidenceStrategy,
		ConfidenceScorer,
		TokenOverlapScorer,
		SemanticSimilarityScorer,
		SourceAgreementScorer,
		RetrievalScoreAggregator,
		EnsembleConfidenceScorer,
		ConfidenceAssessor,
		assess_confidence,
		create_confidence_assessor,
	)
except ImportError as e:
	logger.warning(f"Could not import safety.confidence: {e}")

# --- Refusal ---
try:
	from .refusal import (
		Refusal,
		RefusalEngine,
	)
except ImportError as e:
	logger.warning(f"Could not import safety.refusal: {e}")

__all__ = [
	# Filters - Core
	"Decision",
	"ReasonCode",
	"FilterRule",
	"SafetyFilter",
	"SafetyFilterChain",
	"evaluate_safety",
	"create_default_filter_chain",
	# Filters - Implementations
	"KeywordFilter",
	"RegexFilter",
	"PIIFilter",
	"IntentFilter",
	# Confidence - Core
	"ConfidenceLevel",
	"ConfidenceScore",
	"ConfidenceConfig",
	"ConfidenceStrategy",
	"ConfidenceScorer",
	"ConfidenceAssessor",
	"assess_confidence",
	"create_confidence_assessor",
	# Confidence - Scorers
	"TokenOverlapScorer",
	"SemanticSimilarityScorer",
	"SourceAgreementScorer",
	"RetrievalScoreAggregator",
	"EnsembleConfidenceScorer",
	# Refusal
	"Refusal",
	"RefusalEngine",
]
"""Safety package for PlatformX.

Expose the high-level safety APIs: `evaluate_safety`, `RefusalEngine`, and
`assess_confidence`. Internals remain private to prevent accidental misuse.

Safety philosophy: Wrong answers can cause real harm. Silence is safer than
speculation. Refusal preserves trust. Traceability matters more than fluency.
Accuracy > coverage > speed.
"""

from .filters import evaluate_safety, Decision, ReasonCode
from .refusal import RefusalEngine
from .confidence import assess_confidence, ConfidenceLevel

__all__ = ["evaluate_safety", "Decision", "ReasonCode", "RefusalEngine", "assess_confidence", "ConfidenceLevel"]
