"""Tests for the data loader module."""

import pytest
import os


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_loader_creation(self):
        from platformx.data import DataLoader

        loader = DataLoader()
        assert loader is not None

    def test_load_text_file(self, sample_text_file):
        from platformx.data import DataLoader

        loader = DataLoader()
        dataset = loader.load(sample_text_file, {
            "dataset_id": "test-load-001",
            "domain": "general",
            "intended_use": "retrieval",
        })

        assert dataset.dataset_id == "test-load-001"
        assert dataset.raw_text is not None
        assert len(dataset.raw_text) > 0
        assert dataset.provenance.checksum is not None

    def test_load_missing_file_raises(self, temp_dir):
        from platformx.data import DataLoader, LoaderError

        loader = DataLoader()
        with pytest.raises(LoaderError):
            loader.load(os.path.join(temp_dir, "nonexistent.txt"), {
                "dataset_id": "test",
                "domain": "general",
                "intended_use": "retrieval",
            })

    def test_load_missing_required_metadata(self, sample_text_file):
        from platformx.data import DataLoader, LoaderError

        loader = DataLoader()
        with pytest.raises(LoaderError):
            loader.load(sample_text_file, {
                "domain": "general",
                # missing dataset_id and intended_use
            })

    def test_detect_source_type(self, temp_dir):
        from platformx.data import DataLoader, SourceType

        loader = DataLoader()

        # Create test files with different extensions
        txt_file = os.path.join(temp_dir, "test.txt")
        csv_file = os.path.join(temp_dir, "test.csv")
        json_file = os.path.join(temp_dir, "test.json")

        for f in [txt_file, csv_file, json_file]:
            with open(f, "w") as file:
                file.write("test content")

        assert loader._detect_source_type(txt_file) == SourceType.TEXT
        assert loader._detect_source_type(csv_file) == SourceType.CSV
        assert loader._detect_source_type(json_file) == SourceType.JSON

    def test_load_directory(self, sample_documents, temp_dir):
        from platformx.data import DataLoader

        loader = DataLoader()
        datasets = loader.load_directory(temp_dir, {
            "dataset_id_prefix": "batch",
            "domain": "general",
            "intended_use": "retrieval",
        }, pattern="*.txt")

        assert len(datasets) == 3
