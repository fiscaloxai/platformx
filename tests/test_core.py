import pytest
from platformx.config import PlatformConfig
from platformx.core import Platform

def test_platform_init():
    cfg = PlatformConfig(project_name="test", data_dir="/tmp", logging_level="INFO")
    platform = Platform(cfg)
    assert platform.config.project_name == "test"
    assert hasattr(platform, "register_dataset")
    assert hasattr(platform, "datasets_for_finetuning")
