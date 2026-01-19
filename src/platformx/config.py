from pydantic import BaseModel, Field, validator
from typing import Optional


class PlatformConfig(BaseModel):
    """Central configuration for PlatformX.

    This model intentionally avoids environment loading and external secrets.
    It provides a single source of truth for platform-wide settings.
    """

    project_name: str = Field(..., description="Logical project name")
    data_dir: str = Field(..., description="Base directory for client datasets")
    logging_level: str = Field("INFO", description="Logging level")
    reproducible: bool = Field(True, description="Determinism / reproducible mode")
    seed: Optional[int] = Field(None, description="Optional global RNG seed for reproducibility")

    @validator("logging_level")
    def validate_logging_level(cls, v: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if v not in allowed:
            raise ValueError(f"logging_level must be one of {sorted(allowed)}")
        return v
