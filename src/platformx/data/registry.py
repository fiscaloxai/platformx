
"""
Thread-safe dataset registry for PlatformX, supporting registration, retrieval, and filtering by intended use or domain.
"""
import threading
import logging
from typing import Dict, List, Any, Union
from .schema import DatasetSchema, IntendedUse, Domain

__all__ = ["DatasetRegistry", "DatasetNotFoundError", "DuplicateDatasetError"]

logger = logging.getLogger("platformx.data.registry")

class DatasetNotFoundError(Exception):
    """Raised when a dataset is not found in the registry."""
    def __init__(self, dataset_id: str):
        super().__init__(f"Dataset not found: {dataset_id}")
        self.dataset_id = dataset_id

class DuplicateDatasetError(Exception):
    """Raised when a duplicate dataset is registered."""
    def __init__(self, dataset_id: str):
        super().__init__(f"Duplicate dataset: {dataset_id}")
        self.dataset_id = dataset_id

class DatasetRegistry:
    """
    Thread-safe registry for managing PlatformX datasets.
    Guarantees atomic registration, removal, and lookup operations.
    Usage:
        registry = DatasetRegistry()
        registry.register(ds)
        ds = registry.get("my_id")
    """
    def __init__(self):
        self._datasets: Dict[str, DatasetSchema] = {}
        self._lock = threading.RLock()

    def register(self, dataset: DatasetSchema) -> None:
        """Register a new dataset. Allows version upgrades, blocks exact duplicates."""
        with self._lock:
            existing = self._datasets.get(dataset.dataset_id)
            if existing:
                if existing.version == dataset.version:
                    logger.error(f"Duplicate registration for dataset_id={dataset.dataset_id} version={dataset.version}")
                    raise DuplicateDatasetError(dataset.dataset_id)
                else:
                    logger.warning(f"Version upgrade for dataset_id={dataset.dataset_id}: {existing.version} -> {dataset.version}")
            self._datasets[dataset.dataset_id] = dataset
            logger.info(f"Registered dataset: {dataset.dataset_id} (version={dataset.version})")

    def get(self, dataset_id: str) -> DatasetSchema:
        """Retrieve a dataset by ID."""
        with self._lock:
            ds = self._datasets.get(dataset_id)
            if not ds:
                logger.error(f"Dataset not found: {dataset_id}")
                raise DatasetNotFoundError(dataset_id)
            return ds

    def remove(self, dataset_id: str) -> None:
        """Remove a dataset from the registry."""
        with self._lock:
            if dataset_id not in self._datasets:
                logger.error(f"Attempted to remove missing dataset: {dataset_id}")
                raise DatasetNotFoundError(dataset_id)
            del self._datasets[dataset_id]
            logger.info(f"Removed dataset: {dataset_id}")

    def by_intended_use(self, intended_use: Union[IntendedUse, str]) -> List[DatasetSchema]:
        """Return all datasets matching the intended use (case-insensitive, enum or str)."""
        if isinstance(intended_use, str):
            intended_use = IntendedUse.from_string(intended_use)
        with self._lock:
            return [d for d in self._datasets.values() if d.intended_use == intended_use]

    def by_domain(self, domain: Union[Domain, str]) -> List[DatasetSchema]:
        """Return all datasets matching the domain (case-insensitive, enum or str)."""
        if isinstance(domain, str):
            domain = Domain(domain.lower()) if domain.lower() in [d.value for d in Domain] else Domain(domain.upper())
        with self._lock:
            return [d for d in self._datasets.values() if d.domain == domain]

    def list_all(self) -> List[DatasetSchema]:
        """Return a copy of all registered datasets."""
        with self._lock:
            return list(self._datasets.values())

    def list_ids(self) -> List[str]:
        """Return a list of all dataset IDs."""
        with self._lock:
            return list(self._datasets.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._datasets)

    def __contains__(self, dataset_id: str) -> bool:
        with self._lock:
            return dataset_id in self._datasets

    def clear(self) -> None:
        """Remove all datasets (useful for testing)."""
        with self._lock:
            self._datasets.clear()
            logger.warning("All datasets cleared from registry.")

    def export_state(self) -> Dict[str, Any]:
        """Export registry state as a JSON-serializable dict."""
        with self._lock:
            return {k: v.model_dump() for k, v in self._datasets.items()}

    def import_state(self, state: Dict[str, Any]) -> int:
        """Import datasets from exported state. Skips duplicates, returns count imported."""
        count = 0
        with self._lock:
            for k, v in state.items():
                if k in self._datasets:
                    logger.warning(f"Skipped duplicate dataset during import: {k}")
                    continue
                ds = DatasetSchema(**v)
                self._datasets[k] = ds
                count += 1
            logger.info(f"Imported {count} datasets into registry.")
        return count
