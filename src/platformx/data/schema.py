
"""
PlatformX Data Schema Module
---------------------------
Defines core data structures for dataset management and provenance tracking in pharmaceutical AI workflows.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator
import re
import hashlib

__all__ = [
    "SourceType",
    "IntendedUse",
    "Domain",
    "Provenance",
    "DatasetSchema",
]

class SourceType(str, Enum):
    """
    Enum for supported dataset source types.
    """
    TEXT = "TEXT"
    PDF = "PDF"
    CSV = "CSV"
    JSON = "JSON"
    PARQUET = "PARQUET"
    HTML = "HTML"
    XML = "XML"

class IntendedUse(str, Enum):
    """
    Enum for intended use of the dataset (e.g., fine-tuning, retrieval, RAFT, evaluation).
    Includes a helper for case-insensitive string conversion.
    """
    FINETUNING = "FINETUNING"
    RETRIEVAL = "RETRIEVAL"
    RAFT = "RAFT"
    EVALUATION = "EVALUATION"

    @classmethod
    def from_string(cls, value: str) -> "IntendedUse":
        value = value.strip().lower()
        for member in cls:
            if member.value == value or member.name.lower() == value:
                return member
        raise ValueError(f"Unknown IntendedUse: {value}")

class Domain(str, Enum):
    """
    Enum for dataset domain (pharma, clinical, regulatory, etc).
    """
    PHARMA = "PHARMA"
    CLINICAL = "CLINICAL"
    REGULATORY = "REGULATORY"
    CHEMISTRY = "CHEMISTRY"
    BIOLOGICS = "BIOLOGICS"
    GENERAL = "GENERAL"
    TEST = "TEST"  # Allow 'test' as a valid domain for testing

class Provenance(BaseModel):
    """
    Tracks the origin and transformation history of a dataset for full auditability.
    """
    source_uri: str = Field(..., description="Original file path or URL")
    ingested_by: str = Field(..., description="System or user that ingested the data")
    ingested_at: datetime = Field(..., description="Timestamp of ingestion (UTC)")
    checksum: Optional[str] = Field(None, description="SHA256 hash of source content")
    transformation_history: List[str] = Field(default_factory=list, description="List of transformations applied")

class DatasetSchema(BaseModel):
    """
    Main schema for a dataset in PlatformX, with full provenance and metadata.
    """
    dataset_id: str = Field(..., description="Unique dataset identifier")
    domain: Union[Domain, str] = Field(..., description="Domain or use-case")
    source_type: SourceType = Field(..., description="Source type (text, pdf, etc)")
    intended_use: IntendedUse = Field(..., description="Intended use (finetuning, retrieval, raft, evaluation)")
    version: str = Field(..., description="Dataset version (semver, e.g. 1.0.0)")
    provenance: Provenance = Field(..., description="Provenance metadata")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    raw_text: Optional[str] = Field(None, description="Extracted text content")
    record_count: Optional[int] = Field(None, description="Number of records if tabular")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp (UTC)")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise ValueError("version must be in semver format X.Y.Z")
        return v

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v):
        if not v or not v.strip():
            raise ValueError("dataset_id cannot be empty or whitespace only")
        return v.strip()

    @field_validator("domain", mode="before")
    @classmethod
    def validate_domain(cls, v):
        if isinstance(v, Domain):
            return v
        v_str = str(v).strip().upper()
        for member in Domain:
            if member.value == v_str or member.name.upper() == v_str:
                return member
        # Allow 'test' as a valid domain for testing
        if v_str == "TEST":
            return Domain.TEST
        raise ValueError(f"Unknown domain: {v}")

    def compute_fingerprint(self) -> str:
        """
        Returns a SHA256 hash fingerprint of key dataset fields for reproducibility.
        """
        base = f"{self.dataset_id}{self.version}{self.provenance.source_uri}"
        if self.provenance.checksum:
            base += self.provenance.checksum
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    model_config = dict(populate_by_name=True, extra="allow")
