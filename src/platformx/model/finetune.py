from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

from ..data.schema import DatasetSchema, IntendedUse
from .adapters import AdapterArtifact


@dataclass
class FinetuneReport:
    adapter_id: str
    created_at: datetime
    model_fingerprint: str
    training_dataset_ids: List[str]
    seed: Optional[int]
    metadata: Dict[str, Any]


class FineTuner:
    """Deterministic, auditable fine-tuning interface.

    - Only accepts datasets intended for `finetuning`.
    - Enforces provenance checks on datasets.
    - Produces adapter artifacts (logical artifacts) with deterministic fingerprints
      derived from training inputs and options. This file contains orchestration
      and tracking logic only — it does not perform model weight updates.

    Fine-tuning philosophy (visible here):
    Base models are chemically illiterate. Domain behavior must be earned through data.
    Fine-tuning changes what the model knows; retrieval changes what the model sees.
    Both are required — neither is optional.
    """

    def __init__(self) -> None:
        pass

    def validate_datasets(self, datasets: List[DatasetSchema]) -> None:
        if not datasets:
            raise ValueError("At least one dataset is required for fine-tuning")
        for d in datasets:
            if d.intended_use != IntendedUse.finetuning:
                raise ValueError(f"Dataset {d.dataset_id} not intended for finetuning")
            # Ensure provenance contains minimal required fields
            if not d.provenance or not d.provenance.source_uri or not d.provenance.ingested_at:
                raise ValueError(f"Dataset {d.dataset_id} missing required provenance information")

    def run(self, base_model_fingerprint: str, datasets: List[DatasetSchema], seed: Optional[int] = None, training_options: Optional[Dict[str, Any]] = None) -> FinetuneReport:
        """Execute a deterministic fine-tuning run that returns a training report.

        This method verifies datasets and produces a deterministic adapter identifier
        and metadata that downstream systems can use to attach adapters to backends.

        Note: no actual weight updates or external calls are made here.
        """
        training_options = training_options or {}
        self.validate_datasets(datasets)

        # Deterministic artifact id computed from model fingerprint + dataset ids + seed + options
        ds_ids = ",".join(sorted(d.dataset_id for d in datasets))
        payload = f"{base_model_fingerprint}|{ds_ids}|{seed or ''}|{sorted(training_options.items())}"
        adapter_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        artifact = AdapterArtifact(
            adapter_id=adapter_id,
            domain=datasets[0].domain.value if datasets else "unknown",
            created_at=datetime.utcnow(),
            model_fingerprint=base_model_fingerprint,
            training_datasets=[d.dataset_id for d in datasets],
            metadata={"training_options": training_options, "seed": seed},
        )

        report = FinetuneReport(
            adapter_id=artifact.adapter_id,
            created_at=artifact.created_at,
            model_fingerprint=artifact.model_fingerprint,
            training_dataset_ids=artifact.training_datasets,
            seed=seed,
            metadata=artifact.metadata,
        )

        return report
