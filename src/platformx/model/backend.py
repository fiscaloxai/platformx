from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Any


@dataclass
class ModelMetadata:
    model_path: str
    model_version: Optional[str] = None
    fingerprint: Optional[str] = None
    loaded_at: Optional[datetime] = None


class BaseModelBackend:
    """Abstract, read-only model backend representing base (pre-finetuning) weights.

    Design notes:
    - Base weights are treated as immutable and read-only.
    - This backend refuses to perform inference unless one or more adapters
      that represent fine-tuned behavior are attached.
    - A checksum / fingerprint is computed to provide traceability.

    Fine-tuning philosophy (visible here):
    Base models are chemically illiterate. Domain behavior must be earned through
    fine-tuning. Retrieval changes what the model sees. Both are required.
    """

    def __init__(self, model_path: str, model_version: Optional[str] = None) -> None:
        self._meta = ModelMetadata(model_path=model_path, model_version=model_version)
        self._attached_adapters: List[Any] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        # Compute fingerprint from file contents if available, otherwise from path string
        path = self._meta.model_path
        if os.path.isfile(path):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            self._meta.fingerprint = h.hexdigest()
        else:
            # deterministic fingerprint from path + version
            base = f"{path}|{self._meta.model_version or ''}"
            self._meta.fingerprint = hashlib.sha256(base.encode("utf-8")).hexdigest()
        self._meta.loaded_at = datetime.utcnow()

    @property
    def model_version(self) -> Optional[str]:
        return self._meta.model_version

    @property
    def fingerprint(self) -> str:
        return self._meta.fingerprint  # type: ignore[return-value]

    @property
    def model_path(self) -> str:
        return self._meta.model_path

    def attach_adapter(self, adapter: Any) -> None:
        """Attach an adapter to this backend.

        The adapter must implement a `compatible_with(backend)` method.
        Silent swapping is prevented; attaching an adapter with the same
        identity twice will raise.
        """
        # validate adapter compatibility
        if not hasattr(adapter, "compatible_with"):
            raise TypeError("Adapter missing required method 'compatible_with'")
        if not adapter.compatible_with(self):
            raise ValueError("Adapter not compatible with this backend")

        # prevent duplicate attachment
        for a in self._attached_adapters:
            if getattr(a, "adapter_id", None) == getattr(adapter, "adapter_id", None):
                raise RuntimeError("Adapter with same ID already attached; explicit detach required to replace")

        self._attached_adapters.append(adapter)

    def detach_adapter(self, adapter_id: str) -> None:
        self._attached_adapters = [a for a in self._attached_adapters if getattr(a, "adapter_id", None) != adapter_id]

    def attached_adapters(self) -> List[Any]:
        return list(self._attached_adapters)

    def is_ready_for_inference(self) -> bool:
        return len(self._attached_adapters) > 0

    def infer(self, *args, **kwargs):
        """Refuse inference at the backend level; adapters represent behavioral identity.

        Inference is not supported here to avoid accidental use of untuned base models.
        """
        raise RuntimeError("Inference is not supported on BaseModelBackend; attach adapters and use a controlled inference pipeline")
