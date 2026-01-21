"""Tests for the retrieval module."""

import pytest


class TestEmbeddingBackends:
    """Tests for embedding backends."""

    def test_tfidf_backend(self):
        from platformx.retrieval import TFIDFBackend

        backend = TFIDFBackend(max_features=100)
        embedding = backend.embed_text("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_tfidf_batch_embedding(self):
        from platformx.retrieval import TFIDFBackend

        backend = TFIDFBackend()
        texts = ["Hello world", "Goodbye world", "Test text"]
        embeddings = backend.embed_batch(texts)

        assert len(embeddings) == 3

    def test_cached_backend(self):
        from platformx.retrieval import TFIDFBackend, CachedEmbeddingBackend

        base = TFIDFBackend()
        cached = CachedEmbeddingBackend(base)

        # First call - cache miss
        emb1 = cached.embed_text("Hello world")
        stats1 = cached.cache_stats()
        assert stats1["misses"] == 1
        assert stats1["hits"] == 0

        # Second call - cache hit
        emb2 = cached.embed_text("Hello world")
        stats2 = cached.cache_stats()
        assert stats2["hits"] == 1

        # Embeddings should be identical
        assert emb1 == emb2

    def test_create_embedding_backend(self):
        from platformx.retrieval import create_embedding_backend

        backend = create_embedding_backend("tfidf")
        assert backend is not None
        assert backend.model_name == "tfidf-local"


class TestIndexer:
    """Tests for Indexer class."""

    def test_indexer_creation(self):
        from platformx.retrieval import Indexer

        indexer = Indexer(chunk_size=100)
        assert indexer.chunk_size == 100

    def test_index_dataset(self, sample_dataset):
        from platformx.retrieval import Indexer

        indexer = Indexer(chunk_size=50)
        chunk_ids = indexer.index_dataset(sample_dataset)

        assert len(chunk_ids) > 0
        assert indexer.chunk_count() > 0

    def test_query_indexed_content(self, sample_dataset):
        from platformx.retrieval import Indexer

        indexer = Indexer(chunk_size=50)
        indexer.index_dataset(sample_dataset)

        results = indexer.query("test content", top_k=5)
        assert len(results) > 0
        assert "text" in results[0]
        assert "score" in results[0]

    def test_index_wrong_intended_use(self, sample_dataset_for_finetuning):
        from platformx.retrieval import Indexer

        indexer = Indexer()
        with pytest.raises(ValueError):
            indexer.index_dataset(sample_dataset_for_finetuning)


class TestRetrievalEngine:
    """Tests for RetrievalEngine class."""

    def test_engine_creation(self):
        from platformx.retrieval import Indexer, RetrievalEngine

        indexer = Indexer()
        engine = RetrievalEngine(indexer)
        assert engine is not None

    def test_retrieve(self, sample_dataset):
        from platformx.retrieval import Indexer, RetrievalEngine, GroundedQuery, QueryIntent

        indexer = Indexer(chunk_size=50)
        indexer.index_dataset(sample_dataset)
        engine = RetrievalEngine(indexer)

        query = GroundedQuery(text="test content", intent=QueryIntent.informational)
        results = engine.retrieve(query, max_results=3)

        assert len(results) > 0
        assert hasattr(results[0], "text")
        assert hasattr(results[0], "score")
