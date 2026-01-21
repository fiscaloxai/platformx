"""Tests for the data schema module."""

import pytest
from datetime import datetime, timezone


class TestProvenance:
    """Tests for Provenance model."""

    def test_provenance_creation(self):
        from platformx.data import Provenance

        prov = Provenance(
            source_uri="file://test.txt",
            ingested_by="tester",
            ingested_at=datetime.now(timezone.utc),
        )
        assert prov.source_uri == "file://test.txt"
        assert prov.ingested_by == "tester"
        assert prov.checksum is None

    def test_provenance_with_checksum(self):
        from platformx.data import Provenance

        prov = Provenance(
            source_uri="file://test.txt",
            ingested_by="tester",
            ingested_at=datetime.now(timezone.utc),
            checksum="sha256:abc123",
        )
        assert prov.checksum == "sha256:abc123"

    def test_provenance_transformation_history(self):
        from platformx.data import Provenance

        prov = Provenance(
            source_uri="file://test.txt",
            ingested_by="tester",
            ingested_at=datetime.now(timezone.utc),
            transformation_history=["cleaned", "tokenized"],
        )
        assert len(prov.transformation_history) == 2


class TestDatasetSchema:
    """Tests for DatasetSchema model."""

    def test_dataset_creation(self, sample_provenance):
        from platformx.data import DatasetSchema, SourceType, IntendedUse, Domain

        ds = DatasetSchema(
            dataset_id="test-001",
            domain=Domain.GENERAL,
            source_type=SourceType.TEXT,
            intended_use=IntendedUse.RETRIEVAL,
            version="1.0.0",
            provenance=sample_provenance,
            raw_text="Test content",
        )
        assert ds.dataset_id == "test-001"
        assert ds.domain == Domain.GENERAL
        assert ds.source_type == SourceType.TEXT
        assert ds.intended_use == IntendedUse.RETRIEVAL

    def test_dataset_version_validation(self, sample_provenance):
        from platformx.data import DatasetSchema, SourceType, IntendedUse, Domain
        from pydantic import ValidationError

        # Valid semver
        ds = DatasetSchema(
            dataset_id="test-001",
            domain=Domain.GENERAL,
            source_type=SourceType.TEXT,
            intended_use=IntendedUse.RETRIEVAL,
            version="1.2.3",
            provenance=sample_provenance,
        )
        assert ds.version == "1.2.3"

    def test_dataset_fingerprint(self, sample_dataset):
        fingerprint = sample_dataset.compute_fingerprint()
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0

        # Same dataset should have same fingerprint
        fingerprint2 = sample_dataset.compute_fingerprint()
        assert fingerprint == fingerprint2

    def test_dataset_id_not_empty(self, sample_provenance):
        from platformx.data import DatasetSchema, SourceType, IntendedUse, Domain
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DatasetSchema(
                dataset_id="",
                domain=Domain.GENERAL,
                source_type=SourceType.TEXT,
                intended_use=IntendedUse.RETRIEVAL,
                version="1.0.0",
                provenance=sample_provenance,
            )


class TestEnums:
    """Tests for data enums."""

    def test_source_type_values(self):
        from platformx.data import SourceType

        assert SourceType.TEXT.value == "TEXT"
        assert SourceType.PDF.value == "PDF"
        assert SourceType.CSV.value == "CSV"

    def test_intended_use_values(self):
        from platformx.data import IntendedUse

        assert IntendedUse.FINETUNING.value == "FINETUNING"
        assert IntendedUse.RETRIEVAL.value == "RETRIEVAL"
        assert IntendedUse.RAFT.value == "RAFT"

    def test_domain_values(self):
        from platformx.data import Domain

        assert Domain.GENERAL.value == "GENERAL"
        assert Domain.PHARMA.value == "PHARMA"
