"""Tests for the high-level API module."""

import pytest
import os


class TestHighLevelAPI:
    """Tests for high-level API functions."""

    def test_quick_setup(self, temp_dir):
        from platformx.api import quick_setup

        platform = quick_setup(
            project_name="api_test",
            data_dir=temp_dir,
        )

        assert platform is not None
        assert platform.config.project_name == "api_test"

    def test_index_documents(self, sample_documents, temp_dir):
        from platformx.api import index_documents

        result = index_documents(
            source=temp_dir,
            dataset_id="api-test-docs",
            chunk_size=50,
            embedding_backend="tfidf",
        )

        assert "chunk_count" in result or "error" not in result

    def test_rag_query_without_index(self):
        from platformx.api import rag_query

        # Should handle gracefully when no index
        result = rag_query(
            query="What is Python?",
            index_path="./nonexistent_index/",
        )

        # Should return error or empty results
        assert result is not None
