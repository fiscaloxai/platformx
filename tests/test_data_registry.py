"""Tests for the data registry module."""

import pytest


class TestDatasetRegistry:
    """Tests for DatasetRegistry class."""

    def test_registry_creation(self):
        from platformx.data import DatasetRegistry

        registry = DatasetRegistry()
        assert len(registry) == 0

    def test_register_dataset(self, sample_dataset):
        from platformx.data import DatasetRegistry

        registry = DatasetRegistry()
        registry.register(sample_dataset)
        assert len(registry) == 1
        assert sample_dataset.dataset_id in registry

    def test_get_dataset(self, sample_dataset):
        from platformx.data import DatasetRegistry

        registry = DatasetRegistry()
        registry.register(sample_dataset)

        retrieved = registry.get(sample_dataset.dataset_id)
        assert retrieved.dataset_id == sample_dataset.dataset_id

    def test_get_nonexistent_raises(self):
        from platformx.data import DatasetRegistry, DatasetNotFoundError

        registry = DatasetRegistry()
        with pytest.raises(DatasetNotFoundError):
            registry.get("nonexistent-id")

    def test_duplicate_registration_raises(self, sample_dataset):
        from platformx.data import DatasetRegistry, DuplicateDatasetError

        registry = DatasetRegistry()
        registry.register(sample_dataset)

        with pytest.raises(DuplicateDatasetError):
            registry.register(sample_dataset)

    def test_by_intended_use(self, sample_dataset, sample_dataset_for_finetuning):
        from platformx.data import DatasetRegistry, IntendedUse

        registry = DatasetRegistry()
        registry.register(sample_dataset)
        registry.register(sample_dataset_for_finetuning)

        retrieval_datasets = registry.by_intended_use(IntendedUse.RETRIEVAL)
        assert len(retrieval_datasets) == 1
        assert retrieval_datasets[0].dataset_id == sample_dataset.dataset_id

        finetune_datasets = registry.by_intended_use(IntendedUse.FINETUNING)
        assert len(finetune_datasets) == 1

    def test_by_intended_use_string(self, sample_dataset):
        from platformx.data import DatasetRegistry

        registry = DatasetRegistry()
        registry.register(sample_dataset)

        # Test with string (case-insensitive)
        results = registry.by_intended_use("retrieval")
        assert len(results) == 1

    def test_list_all(self, sample_dataset, sample_dataset_for_finetuning):
        from platformx.data import DatasetRegistry

        registry = DatasetRegistry()
        registry.register(sample_dataset)
        registry.register(sample_dataset_for_finetuning)

        all_datasets = registry.list_all()
        assert len(all_datasets) == 2

    def test_remove_dataset(self, sample_dataset):
        from platformx.data import DatasetRegistry

        registry = DatasetRegistry()
        registry.register(sample_dataset)
        assert len(registry) == 1

        registry.remove(sample_dataset.dataset_id)
        assert len(registry) == 0

    def test_clear_registry(self, sample_dataset, sample_dataset_for_finetuning):
        from platformx.data import DatasetRegistry

        registry = DatasetRegistry()
        registry.register(sample_dataset)
        registry.register(sample_dataset_for_finetuning)

        registry.clear()
        assert len(registry) == 0

    def test_export_import_state(self, sample_dataset):
        from platformx.data import DatasetRegistry

        registry = DatasetRegistry()
        registry.register(sample_dataset)

        state = registry.export_state()
        assert "datasets" in state or len(state) > 0

        new_registry = DatasetRegistry()
        count = new_registry.import_state(state)
        assert count >= 1
