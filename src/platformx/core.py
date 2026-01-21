from typing import Iterable, List, Optional
import os
import logging


from .config import PlatformConfig
from .data import DatasetRegistry, DataLoader, DatasetSchema, IntendedUse
# Future integration imports
from .audit import AuditLogger
from .retrieval import Indexer, RetrievalEngine



__all__ = ["Platform"]

class Platform:
    """
    Main orchestration entrypoint for PlatformX.

    Responsibilities:
    - Hold configuration
    - Provide dataset registration helpers with audit logging
    - Expose queries for datasets by intended use
    - Integrate with retrieval, indexing, and audit subsystems
    - Serve as the main hub for pharma/life-sciences AI workflows
    """

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config
        self.registry = DatasetRegistry()
        self.loader = DataLoader()
        self.audit = AuditLogger()
        self._indexer: Optional[Indexer] = None
        self._retrieval_engine: Optional[RetrievalEngine] = None

        logging.basicConfig(level=getattr(logging, config.logging_level))
        self._logger = logging.getLogger("platformx.core")
        self._logger.info(f"Platform initialized for project: {getattr(config, 'project_name', 'PlatformX')}")

    @property
    def indexer(self) -> Indexer:
        """Lazily initialize and return the Indexer instance."""
        if self._indexer is None:
            self._indexer = Indexer()
            self._logger.debug("Indexer initialized.")
        return self._indexer

    @property
    def retrieval_engine(self) -> RetrievalEngine:
        """Lazily initialize and return the RetrievalEngine instance."""
        if self._retrieval_engine is None:
            self._retrieval_engine = RetrievalEngine(self.indexer, self._logger)
            self._logger.debug("RetrievalEngine initialized.")
        return self._retrieval_engine

    def register_dataset(self, path: str, metadata: dict) -> DatasetSchema:
        """Load a dataset from `path` and register it using provided `metadata`.

        `path` may be absolute or relative to `config.data_dir`.
        `metadata` must contain the fields required by `DatasetSchema`.
        Logs registration event to audit log.
        """
        if not os.path.isabs(path):
            path = os.path.join(self.config.data_dir, path)

        self._logger.debug("Registering dataset from path=%s", path)
        dataset = self.loader.load(path, metadata)
        self.registry.register(dataset)
        self.audit.log("dataset_registered", {
            "dataset_id": dataset.dataset_id,
            "domain": str(dataset.domain),
            "intended_use": str(dataset.intended_use)
        })
        self._logger.info("Registered dataset %s", dataset.dataset_id)
        return dataset

    def register_bulk(self, entries: Iterable[tuple]) -> List[DatasetSchema]:
        """Register multiple datasets. Logs bulk registration to audit log.

        entries: iterable of (path, metadata) tuples.
        Returns list of registered DatasetSchema objects.
        """
        registered = []
        for path, metadata in entries:
            registered.append(self.register_dataset(path, metadata))
        self.audit.log("bulk_registration", {
            "count": len(registered),
            "dataset_ids": [d.dataset_id for d in registered]
        })
        return registered

    def index_dataset(self, dataset_id: str) -> List[str]:
        """
        Index a dataset for retrieval. Only allowed for datasets with intended_use=RETRIEVAL.
        Returns list of chunk IDs.
        """
        dataset = self.registry.get(dataset_id)
        if dataset.intended_use != IntendedUse.RETRIEVAL:
            raise ValueError(f"Dataset {dataset_id} is not marked for retrieval.")
        chunk_ids = self.indexer.index_dataset(dataset)
        self.audit.log("dataset_indexed", {
            "dataset_id": dataset_id,
            "chunk_count": len(chunk_ids)
        })
        self._logger.info(f"Indexed dataset {dataset_id} into {len(chunk_ids)} chunks.")
        return chunk_ids

    def index_retrieval_datasets(self) -> List[str]:
        """
        Index all datasets marked for retrieval. Returns all chunk IDs.
        """
        datasets = self.datasets_for_retrieval()
        all_chunk_ids = []
        for ds in datasets:
            all_chunk_ids.extend(self.index_dataset(ds.dataset_id))
        self.audit.log("retrieval_datasets_indexed", {
            "count": len(datasets),
            "total_chunks": len(all_chunk_ids)
        })
        self._logger.info(f"Indexed {len(datasets)} retrieval datasets, total {len(all_chunk_ids)} chunks.")
        return all_chunk_ids


    def datasets_for_finetuning(self):
        """Return all datasets marked for fine-tuning."""
        return self.registry.by_intended_use(IntendedUse.FINETUNING)

    def datasets_for_retrieval(self):
        """Return all datasets marked for retrieval."""
        return self.registry.by_intended_use(IntendedUse.RETRIEVAL)

    def datasets_for_raft(self):
        """Return all datasets marked for RAFT."""
        return self.registry.by_intended_use(IntendedUse.RAFT)

    def get_dataset(self, dataset_id: str) -> DatasetSchema:
        """Get a dataset by ID."""
        return self.registry.get(dataset_id)

    def list_datasets(self) -> List[DatasetSchema]:
        """List all registered datasets."""
        return self.registry.list_all()

    def export_audit_log(self, path: str) -> None:
        """Export the audit log to a JSON file."""
        self.audit.export_json(path)
        self._logger.info(f"Audit log exported to {path}")
