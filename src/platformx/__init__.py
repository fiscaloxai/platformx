"""
PlatformX - Production-Quality LLM Fine-Tuning, RAG, and RAFT Library.

PlatformX provides a complete toolkit for building production-quality LLM applications
with comprehensive support for fine-tuning, retrieval-augmented generation (RAG),
and retrieval-augmented fine-tuning (RAFT).

Key Features:
- Multi-format dataset loading with provenance tracking
- Flexible embedding backends (TF-IDF, Sentence Transformers, custom)
- Configurable retrieval with chunking and reranking
- RAFT sample generation for training retrieval-aware models
- LoRA/PEFT fine-tuning with full audit logging
- Multi-backend inference (local HuggingFace, OpenAI, Anthropic)
- Comprehensive safety filtering and confidence assessment
- Structured audit logging with correlation tracking

Quick Start:
	import platformx as pfx

	# Initialize platform
	config = pfx.PlatformConfig(project_name="my_project", data_dir="./data")
	platform = pfx.Platform(config)

	# Load and register dataset
	dataset = platform.register_dataset("docs.txt", {
		"dataset_id": "docs-001",
		"domain": "general",
		"intended_use": "retrieval"
	})

	# Index for retrieval
	platform.index_dataset("docs-001")

	# Query with safety checks
	from platformx.safety import create_default_filter_chain
	chain = create_default_filter_chain()
	result = chain.check("my query")

Modules:
- platformx.data: Dataset management and loading
- platformx.retrieval: Indexing and retrieval
- platformx.training: RAFT sample generation
- platformx.model: Fine-tuning and inference
- platformx.safety: Filtering and confidence
- platformx.audit: Logging and traceability

For more information, visit: https://github.com/your-org/platformx
"""

__version__ = "0.1.0"

import logging
logger = logging.getLogger("platformx")

# --- Core ---
try:
	from .config import PlatformConfig
	from .core import Platform
except ImportError as e:
	logger.warning(f"Could not import core components: {e}")
	PlatformConfig = None
	Platform = None

# --- Data ---
try:
	from .data import (
		DatasetSchema,
		DatasetRegistry,
		DataLoader,
		IntendedUse,
		SourceType,
		Domain,
	)
except ImportError as e:
	logger.warning(f"Could not import data components: {e}")
	DatasetSchema = DatasetRegistry = DataLoader = IntendedUse = SourceType = Domain = None

# --- Retrieval ---
try:
	from .retrieval import (
		Indexer,
		RetrievalEngine,
		create_embedding_backend,
	)
except ImportError as e:
	logger.warning(f"Could not import retrieval components: {e}")
	Indexer = RetrievalEngine = create_embedding_backend = None

# --- Training ---
try:
	from .training import (
		RAFTOrchestrator,
		RAFTConfig,
		RAFTDatasetBuilder,
	)
except ImportError as e:
	logger.warning(f"Could not import training components: {e}")
	RAFTOrchestrator = RAFTConfig = RAFTDatasetBuilder = None

# --- Model ---
try:
	from .model import (
		FineTuner,
		TrainingConfig,
		LoRAConfig,
		create_inference_pipeline,
	)
except ImportError as e:
	logger.warning(f"Could not import model components: {e}")
	FineTuner = TrainingConfig = LoRAConfig = create_inference_pipeline = None

# --- Safety ---
try:
	from .safety import (
		evaluate_safety,
		assess_confidence,
		create_default_filter_chain,
		RefusalEngine,
	)
except ImportError as e:
	logger.warning(f"Could not import safety components: {e}")
	evaluate_safety = assess_confidence = create_default_filter_chain = RefusalEngine = None

# --- Audit ---
try:
	from .audit import (
		AuditLogger,
		AuditEventType,
	)
except ImportError as e:
	logger.warning(f"Could not import audit components: {e}")
	AuditLogger = AuditEventType = None

__all__ = [
	# Version
	"__version__",
	# Core
	"PlatformConfig",
	"Platform",
	# Data
	"DatasetSchema",
	"DatasetRegistry",
	"DataLoader",
	"IntendedUse",
	"SourceType",
	"Domain",
	# Retrieval
	"Indexer",
	"RetrievalEngine",
	"create_embedding_backend",
	# Training
	"RAFTOrchestrator",
	"RAFTConfig",
	"RAFTDatasetBuilder",
	# Model
	"FineTuner",
	"TrainingConfig",
	"LoRAConfig",
	"create_inference_pipeline",
	# Safety
	"evaluate_safety",
	"assess_confidence",
	"create_default_filter_chain",
	"RefusalEngine",
	# Audit
	"AuditLogger",
	"AuditEventType",
]

def get_version() -> str:
	"""Return the current PlatformX version."""
	return __version__

def info() -> dict:
	"""Return information about PlatformX installation."""
	import sys
	return {
		"version": __version__,
		"python_version": sys.version,
		"modules": {
			"data": DatasetSchema is not None,
			"retrieval": Indexer is not None,
			"training": RAFTOrchestrator is not None,
			"model": FineTuner is not None,
			"safety": evaluate_safety is not None,
			"audit": AuditLogger is not None,
		}
	}
"""PlatformX

A pharma-focused AI platform

Built for accuracy, traceability, and safety

Not a generic chatbot library
"""

__version__ = "0.1.0"
