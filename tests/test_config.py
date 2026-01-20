from platformx.config import PlatformConfig
import pytest

def test_config_fields():
    cfg = PlatformConfig(project_name="demo", data_dir="/tmp/data", logging_level="DEBUG")
    assert cfg.project_name == "demo"
    assert cfg.data_dir == "/tmp/data"
    assert cfg.logging_level == "DEBUG"
    assert cfg.reproducible is True

    # Test invalid logging level
    with pytest.raises(ValueError):
        PlatformConfig(project_name="fail", data_dir="/tmp", logging_level="INVALID")
