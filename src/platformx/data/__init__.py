
"""
Data management module for PlatformX.

This module provides dataset schema definitions, registry management, and file loading
capabilities with full provenance tracking for pharma/life-sciences AI workflows.

Core Components:
- DatasetSchema: Pydantic model defining dataset structure and metadata
- DatasetRegistry: Thread-safe registry for managing datasets
- DataLoader: Multi-format file loader with automatic text extraction

Enums:
- SourceType: Supported file formats (TEXT, PDF, CSV, JSON, etc.)
- IntendedUse: Dataset purpose (FINETUNING, RETRIEVAL, RAFT, EVALUATION)
- Domain: Subject domain (PHARMA, CLINICAL, REGULATORY, etc.)

Example:
	from platformx.data import DataLoader, DatasetRegistry, IntendedUse

	loader = DataLoader()
	registry = DatasetRegistry()

	dataset = loader.load("pharma_docs.txt", {
		"dataset_id": "pharma-001",
		"domain": "pharma",
		"intended_use": "retrieval"
	})
	registry.register(dataset)

	# Get all retrieval datasets
	retrieval_sets = registry.by_intended_use(IntendedUse.RETRIEVAL)
"""

# --- Schema ---
from .schema import (
	DatasetSchema,
	Provenance,
	SourceType,
	IntendedUse,
	Domain,
)

# --- Registry ---
from .registry import (
	DatasetRegistry,
	DatasetNotFoundError,
	DuplicateDatasetError,
)

# --- Loader ---
from .loader import (
	DataLoader,
	LoaderError,
	UnsupportedFormatError,
)

__all__ = [
	"DatasetSchema",
	"Provenance",
	"SourceType",
	"IntendedUse",
	"Domain",
	"DatasetRegistry",
	"DatasetNotFoundError",
	"DuplicateDatasetError",
	"DataLoader",
	"LoaderError",
	"UnsupportedFormatError",
]
