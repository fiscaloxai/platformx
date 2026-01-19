from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List
import hashlib

from ..data.schema import DatasetSchema


@dataclass
class AdapterArtifact:
    adapter_id: str
    domain: str
    created_at: datetime
    model_fingerprint: str
    training_datasets: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Adapter:
    """Represents a fine-tuned adapter (behavioral identity).

    Responsibilities:
    - Hold provenance and training metadata
    - Validate compatibility with a backend via fingerprint matching rules
    - Prevent silent swapping by exposing stable `adapter_id`
    """

    def __init__(self, artifact: AdapterArtifact, compatible_base_fingerprints: List[str]):
        self.artifact = artifact
        self.adapter_id = artifact.adapter_id
        self._compatible = set(compatible_base_fingerprints)

    def compatible_with(self, backend: Any) -> bool:
        # backend is expected to expose `fingerprint` attribute
        return getattr(backend, "fingerprint", None) in self._compatible

    @property
    def domain(self) -> str:
        return self.artifact.domain


class AdapterRegistry:
    """Registry for adapters per domain. Prevents silent adapter swapping.

    Supports multiple adapters per domain and explicit attach/detach workflow.
    """

    def __init__(self) -> None:
        self._by_domain: Dict[str, List[Adapter]] = {}

    def register(self, adapter: Adapter) -> None:
        self._by_domain.setdefault(adapter.domain, []).append(adapter)

    def list_for_domain(self, domain: str) -> List[Adapter]:
        return list(self._by_domain.get(domain, []))

    def find(self, adapter_id: str) -> Adapter:
        for adapters in self._by_domain.values():
            for a in adapters:
                if a.adapter_id == adapter_id:
                    return a
        raise KeyError("Adapter not found: %s" % adapter_id)
