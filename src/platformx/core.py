from typing import Iterable, List
import os
import logging

from .config import PlatformConfig
from .data.registry import DatasetRegistry
from .data.loader import DataLoader
from .data.schema import DatasetSchema


class Platform:
    """High-level orchestration entrypoint for PlatformX.

    Responsibilities:
    - Hold configuration
    - Provide dataset registration helpers that validate and forward to the registry
    - Expose simple queries for datasets by intended use

    This class performs orchestration only and contains no ML logic.
    """

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config
        self.registry = DatasetRegistry()
        self.loader = DataLoader()

        logging.basicConfig(level=getattr(logging, config.logging_level))
        self._logger = logging.getLogger("platformx.core")

    def register_dataset(self, path: str, metadata: dict) -> DatasetSchema:
        """Load a dataset from `path` and register it using provided `metadata`.

        `path` may be absolute or relative to `config.data_dir`.
        `metadata` must contain the fields required by `DatasetSchema`.
        """
        if not os.path.isabs(path):
            path = os.path.join(self.config.data_dir, path)

        self._logger.debug("Registering dataset from path=%s", path)
        dataset = self.loader.load(path, metadata)
        self.registry.register(dataset)
        self._logger.info("Registered dataset %s", dataset.dataset_id)
        return dataset

    def register_bulk(self, entries: Iterable[tuple]) -> List[DatasetSchema]:
        """Register multiple datasets.

        entries: iterable of (path, metadata) tuples.
        Returns list of registered DatasetSchema objects.
        """
        registered = []
        for path, metadata in entries:
            registered.append(self.register_dataset(path, metadata))
        return registered

    def datasets_for_finetuning(self):
        return self.registry.by_intended_use("finetuning")

    def datasets_for_retrieval(self):
        return self.registry.by_intended_use("retrieval")

    def datasets_for_raft(self):
        return self.registry.by_intended_use("raft")
