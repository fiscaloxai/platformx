"""Tests for the training module."""

import pytest
from datetime import datetime, timezone


class TestRAFTConfig:
    """Tests for RAFTConfig."""

    def test_default_config(self):
        from platformx.training import RAFTConfig

        config = RAFTConfig()
        assert config.positive_fraction == 0.6
        assert config.seed == 42

    def test_custom_config(self):
        from platformx.training import RAFTConfig

        config = RAFTConfig(
            positive_fraction=0.7,
            reasoning_fraction=0.4,
            seed=123,
        )
        assert config.positive_fraction == 0.7
        assert config.seed == 123


class TestRAFTSample:
    """Tests for RAFTSample."""

    def test_sample_creation(self):
        from platformx.training import RAFTSample, SampleType

        sample = RAFTSample(
            instruction="Extract information",
            context="Python is a language.",
            expected="Python is a language.",
            source_doc_id="doc-001",
            retrieval_score=0.85,
            timestamp=datetime.now(timezone.utc),
            sample_type=SampleType.POSITIVE_EXTRACT,
        )

        assert sample.instruction == "Extract information"
        assert sample.sample_type == SampleType.POSITIVE_EXTRACT
        assert sample.sample_id != ""  # Auto-generated

    def test_sample_types(self):
        from platformx.training import SampleType

        assert SampleType.POSITIVE_EXTRACT.value == "positive_extract"
        assert SampleType.NEGATIVE_REFUSE.value == "negative_refuse"


class TestRAFTDatasetBuilder:
    """Tests for RAFTDatasetBuilder."""

    def test_builder_creation(self):
        from platformx.training import RAFTDatasetBuilder

        builder = RAFTDatasetBuilder()
        assert builder is not None

    def test_build_from_samples(self):
        from platformx.training import RAFTDatasetBuilder, RAFTSample, SampleType

        samples = [
            RAFTSample(
                instruction="Extract information",
                context="Test context",
                expected="Test expected",
                source_doc_id="doc-001",
                retrieval_score=0.85,
                timestamp=datetime.now(timezone.utc),
                sample_type=SampleType.POSITIVE_EXTRACT,
            ),
        ]

        builder = RAFTDatasetBuilder()
        records = builder.build(samples, "test-prefix", domain="general")

        assert len(records) == 1
        assert records[0].dataset is not None
