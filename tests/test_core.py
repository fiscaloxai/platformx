"""Tests for the core module."""

import pytest


class TestPlatformConfig:
    """Tests for PlatformConfig."""

    def test_config_creation(self, temp_dir):
        from platformx import PlatformConfig

        config = PlatformConfig(
            project_name="test",
            data_dir=temp_dir,
        )
        assert config.project_name == "test"
        assert config.logging_level == "INFO"

    def test_config_logging_level_validation(self, temp_dir):
        from platformx import PlatformConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PlatformConfig(
                project_name="test",
                data_dir=temp_dir,
                logging_level="INVALID",
            )

    def test_config_with_seed(self, temp_dir):
        from platformx import PlatformConfig

        config = PlatformConfig(
            project_name="test",
            data_dir=temp_dir,
            seed=42,
            reproducible=True,
        )
        assert config.seed == 42
        assert config.reproducible is True


class TestPlatform:
    """Tests for Platform class."""

    def test_platform_creation(self, platform_config):
        from platformx import Platform

        platform = Platform(platform_config)
        assert platform.config.project_name == "test_project"

    def test_platform_has_registry(self, platform_config):
        from platformx import Platform

        platform = Platform(platform_config)
        assert hasattr(platform, "registry")
        assert len(platform.registry) == 0

    def test_platform_has_loader(self, platform_config):
        from platformx import Platform

        platform = Platform(platform_config)
        assert hasattr(platform, "loader")

    def test_platform_has_audit(self, platform_config):
        from platformx import Platform

        platform = Platform(platform_config)
        assert hasattr(platform, "audit")

    def test_register_dataset(self, platform_config, sample_text_file):
        from platformx import Platform

        platform = Platform(platform_config)
        dataset = platform.register_dataset(sample_text_file, {
            "dataset_id": "test-001",
            "domain": "general",
            "intended_use": "retrieval",
        })

        assert dataset.dataset_id == "test-001"
        assert len(platform.registry) == 1

    def test_datasets_for_retrieval(self, platform_config, sample_text_file):
        from platformx import Platform

        platform = Platform(platform_config)
        platform.register_dataset(sample_text_file, {
            "dataset_id": "test-001",
            "domain": "general",
            "intended_use": "retrieval",
        })

        datasets = platform.datasets_for_retrieval()
        assert len(datasets) == 1

    def test_datasets_for_finetuning_empty(self, platform_config, sample_text_file):
        from platformx import Platform

        platform = Platform(platform_config)
        platform.register_dataset(sample_text_file, {
            "dataset_id": "test-001",
            "domain": "general",
            "intended_use": "retrieval",
        })

        datasets = platform.datasets_for_finetuning()
        assert len(datasets) == 0
