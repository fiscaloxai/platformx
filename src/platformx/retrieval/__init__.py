"""PlatformX retrieval package.

"""
Retrieval module for PlatformX.

This module provides document indexing, embedding generation, and retrieval
capabilities for RAG (Retrieval-Augmented Generation) workflows in pharma/life-sciences.

Core Components:
- Indexer: Chunks and indexes documents for retrieval
- RetrievalEngine: Executes queries against indexed documents
- EmbeddingBackend: Abstract interface for text embeddings

Embedding Backends:
- TFIDFBackend: Simple TF-IDF based embeddings (no dependencies)
- SentenceTransformerBackend: High-quality neural embeddings (requires sentence-transformers)
- CachedEmbeddingBackend: Wrapper for caching embeddings

Query Types:
- GroundedQuery: Validated query with intent classification

Example:
	from platformx.retrieval import Indexer, RetrievalEngine, SentenceTransformerBackend
	from platformx.data import DataLoader, DatasetRegistry

	# Setup with neural embeddings
	embedding = SentenceTransformerBackend(model_name="all-MiniLM-L6-v2")
	indexer = Indexer(embedding_backend=embedding, chunk_size_words=200)

	# Load and index a dataset
	loader = DataLoader()
	dataset = loader.load("pharma_docs.txt", {
		"dataset_id": "pharma-001",
		"domain": "pharma",
		"intended_use": "retrieval"
	})
	indexer.index_dataset(dataset)

	# Query
	engine = RetrievalEngine(indexer)
	results = engine.retrieve(query, max_results=5)
"""

import logging
logger = logging.getLogger("platformx.retrieval")

# --- Embeddings ---
try:
	from .embeddings import (
		EmbeddingBackend,
		TFIDFBackend,
		SentenceTransformerBackend,
		CachedEmbeddingBackend,
		create_embedding_backend,
	)
except Exception as e:
	logger.warning(f"Failed to import embeddings submodule: {e}")

# --- Indexer ---
try:
	from .indexer import (
		Indexer,
		IndexedChunk,
		LocalVectorStore,
		VectorBackend,
	)
except Exception as e:
	logger.warning(f"Failed to import indexer submodule: {e}")

# --- Engine ---
try:
	from .engine import (
		RetrievalEngine,
		RetrievalResult,
	)
except Exception as e:
	logger.warning(f"Failed to import engine submodule: {e}")

# --- Query ---
try:
	from .query import (
		GroundedQuery,
		QueryIntent,
	)
except Exception as e:
	logger.warning(f"Failed to import query submodule: {e}")

__all__ = [
	# Embeddings
	"EmbeddingBackend",
	"TFIDFBackend",
	"SentenceTransformerBackend",
	"CachedEmbeddingBackend",
	"create_embedding_backend",
	# Indexer
	"Indexer",
	"IndexedChunk",
	"LocalVectorStore",
	"VectorBackend",
	# Engine
	"RetrievalEngine",
	"RetrievalResult",
	# Query
	"GroundedQuery",
	"QueryIntent",
]
Public API exposes Indexer, RetrievalEngine, and GroundedQuery.
"""

from .indexer import Indexer
from .engine import RetrievalEngine
from .query import GroundedQuery, QueryIntent

__all__ = ["Indexer", "RetrievalEngine", "GroundedQuery", "QueryIntent"]
