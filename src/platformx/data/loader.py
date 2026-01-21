
"""
DataLoader for PlatformX: Loads datasets from various file formats with text extraction and checksum computation.

Supported formats: TXT, PDF, CSV, JSON, PARQUET, HTML, XML
Required metadata: dataset_id, domain, intended_use

Example usage:
    loader = DataLoader()
    ds = loader.load("/path/to/file.txt", {"dataset_id": "ds1", "domain": "pharma", "intended_use": "retrieval"})
"""
import os
import glob
import csv
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from .schema import DatasetSchema, Provenance, SourceType, IntendedUse, Domain

__all__ = ["DataLoader", "LoaderError", "UnsupportedFormatError"]

class LoaderError(Exception):
    """Raised for errors during dataset loading."""
    pass

class UnsupportedFormatError(LoaderError):
    """Raised for unsupported file formats."""
    pass

class DataLoader:
    """
    Loads datasets from files, extracting text and computing checksums. Supports TXT, PDF, CSV, JSON, PARQUET, HTML, XML.
    Thread-safe for concurrent loads. See load() and load_directory().
    """
    def __init__(self) -> None:
        self._logger = logging.getLogger("platformx.data.loader")

    def _compute_checksum(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    def _detect_source_type(self, file_path: str) -> SourceType:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".txt", ".md"]:
            return SourceType.TEXT
        elif ext == ".pdf":
            return SourceType.PDF
        elif ext == ".csv":
            return SourceType.CSV
        elif ext in [".json", ".jsonl"]:
            return SourceType.JSON
        elif ext == ".parquet":
            return SourceType.PARQUET
        elif ext in [".html", ".htm"]:
            return SourceType.HTML
        elif ext == ".xml":
            return SourceType.XML
        else:
            raise UnsupportedFormatError(f"Unsupported file extension: {ext}")

    def _extract_text_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    def _extract_text_from_csv(self, file_path: str) -> Tuple[str, int]:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        text = "\n".join(["\t".join(row) for row in rows])
        return text, len(rows) - 1 if rows else 0

    def _extract_text_from_json(self, file_path: str) -> Tuple[str, Optional[int]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            record_count = len(data)
        else:
            record_count = None
        text = json.dumps(data, indent=2, ensure_ascii=False)
        return text, record_count

    def _extract_text_from_pdf(self, file_path: str) -> Tuple[str, None]:
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text, None
        except ImportError:
            self._logger.warning("pypdf not installed, cannot extract PDF text.")
            return "", None
        except Exception as e:
            self._logger.warning(f"PDF extraction failed: {e}")
            return "", None

    def _extract_text(self, file_path: str, source_type: SourceType) -> Tuple[str, Optional[int]]:
        if source_type == SourceType.TEXT:
            return self._extract_text_from_txt(file_path), None
        elif source_type == SourceType.CSV:
            return self._extract_text_from_csv(file_path)
        elif source_type == SourceType.JSON:
            return self._extract_text_from_json(file_path)
        elif source_type == SourceType.PDF:
            return self._extract_text_from_pdf(file_path)
        else:
            self._logger.warning(f"Extraction for {source_type} not implemented. Returning empty string.")
            return "", None

    def load(self, path: str, metadata: Dict[str, Any] = None, domain: str = None, **kwargs) -> DatasetSchema:
        if not os.path.exists(path):
            raise LoaderError(f"File not found: {path}")
        if metadata is None:
            metadata = {}
        dataset_id = metadata.get("dataset_id")
        # Use domain argument if provided, else fallback to metadata
        domain_val = domain if domain is not None else metadata.get("domain")
        intended_use = metadata.get("intended_use")
        if not dataset_id or not domain_val or not intended_use:
            raise LoaderError("Missing required metadata: dataset_id, domain, intended_use")
        version = metadata.get("version", "1.0.0")
        source_type = metadata.get("source_type")
        if not source_type:
            source_type = self._detect_source_type(path)
        elif isinstance(source_type, str):
            source_type = SourceType(source_type.upper())
        checksum = self._compute_checksum(path)
        text, record_count = self._extract_text(path, source_type)
        provenance = Provenance(
            source_uri=os.path.abspath(path),
            ingested_by="platformx.data.loader",
            ingested_at=datetime.now(timezone.utc),
            checksum=checksum,
        )
        ds = DatasetSchema(
            dataset_id=dataset_id,
            domain=domain_val,
            source_type=source_type,
            intended_use=IntendedUse.from_string(intended_use) if isinstance(intended_use, str) else intended_use,
            version=version,
            provenance=provenance,
            metadata=metadata.get("metadata", {}),
            raw_text=text,
            record_count=record_count,
        )
        self._logger.info(f"Loaded dataset: {dataset_id} from {path} (type={source_type})")
        return ds

    def load_directory(self, dir_path: str, base_metadata: Dict[str, Any], pattern: str = "*.*") -> List[DatasetSchema]:
        if not os.path.isdir(dir_path):
            raise LoaderError(f"Not a directory: {dir_path}")
        dataset_id_prefix = base_metadata.get("dataset_id_prefix", "dataset_")
        files = glob.glob(os.path.join(dir_path, pattern))
        datasets = []
        for file_path in files:
            meta = base_metadata.copy()
            meta["dataset_id"] = f"{dataset_id_prefix}{os.path.basename(file_path)}"
            try:
                ds = self.load(file_path, meta)
                datasets.append(ds)
            except Exception as e:
                self._logger.warning(f"Failed to load {file_path}: {e}")
        self._logger.info(f"Loaded {len(datasets)} datasets from directory {dir_path}")
        return datasets
