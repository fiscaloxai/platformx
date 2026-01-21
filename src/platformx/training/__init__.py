"""
Training module for PlatformX.

This module provides RAFT (Retrieval-Augmented Fine-Tuning) sample generation
and dataset preparation for training LLMs to effectively use retrieved context.

RAFT Philosophy:
- Models must learn to use retrieval, not just receive it
- Positive samples teach extraction from evidence
- Negative samples teach appropriate refusal
- Distractor samples teach relevance filtering
- Reasoning chains teach step-by-step thinking

Core Components:
- RAFTOrchestrator: Generate RAFT training samples
- RAFTDatasetBuilder: Convert samples to fine-tuning datasets
- RAFTConfig: Configuration for sample generation

Sample Types:
- POSITIVE_EXTRACT: Direct extraction from evidence
- POSITIVE_REASONING: Chain-of-thought with evidence
- NEGATIVE_REFUSE: Refuse when evidence insufficient
- NEGATIVE_DISTRACTOR: Refuse despite distractors present

Example:
	from platformx.training import RAFTOrchestrator, RAFTDatasetBuilder, RAFTConfig
	from platformx.retrieval import Indexer, RetrievalEngine

	# Configure RAFT generation
	config = RAFTConfig(
		positive_fraction=0.6,
		reasoning_fraction=0.3,
		distractor_fraction=0.2,
		seed=42
	)

	# Generate samples
	orchestrator = RAFTOrchestrator(retrieval_engine, config=config)
	samples = orchestrator.generate_for_datasets(["dataset-001", "dataset-002"])

	# Build fine-tuning dataset
	builder = RAFTDatasetBuilder()
	ft_records = builder.build(samples, "my_project", domain="general")

	# Validate distribution
	stats = orchestrator.validate_samples(samples)
	print(f"Generated {stats['total_count']} samples")
"""

import logging
logger = logging.getLogger("platformx.training")

# --- RAFT Core ---
try:
	from .raft import (
		RAFTSample,
		RAFTOrchestrator,
		RAFTConfig,
		SampleType,
		SampleTemplate,
	)
except ImportError as e:
	logger.warning(f"Could not import training.raft: {e}")

# --- Dataset Building ---
try:
	from .datasets import (
		RAFTDatasetBuilder,
		FTRecord,
	)
except ImportError as e:
	logger.warning(f"Could not import training.datasets: {e}")

__all__ = [
	# RAFT Core
	"RAFTOrchestrator",
	"RAFTConfig",
	"RAFTSample",
	"SampleType",
	"SampleTemplate",
	# Dataset Building
	"RAFTDatasetBuilder",
	"FTRecord",
]
"""RAFT training package for PlatformX.

Public API exposes `RAFTOrchestrator` and `RAFTDatasetBuilder` for generating
retrieval-aware fine-tuning datasets. Internals are intentionally minimal and
quietly deterministic. RAFT does not add knowledge; it enforces grounding.
"""

from .raft import RAFTOrchestrator
from .datasets import RAFTDatasetBuilder

__all__ = ["RAFTOrchestrator", "RAFTDatasetBuilder"]
