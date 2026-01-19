from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from ..data.schema import DatasetSchema, Provenance, SourceType, IntendedUse
from .raft import RAFTSample


@dataclass
class FTRecord:
    dataset: DatasetSchema
    sample_index: int


class RAFTDatasetBuilder:
    """Converts RAFT samples into fine-tuning-ready DatasetSchema instances.

    Design notes:
    - Each RAFT sample becomes a small text dataset (intended_use=finetuning).
    - `raw_text` contains a structured representation with instruction, context,
      and expected behavior. No paraphrasing beyond evidence scope is applied.
    - Every produced dataset is fully attributed via `provenance` and timestamped.
    """

    def build(self, samples: List[RAFTSample], dataset_id_prefix: str, domain: str, version: str = "1.0.0") -> List[FTRecord]:
        records: List[FTRecord] = []
        for i, s in enumerate(samples):
            did = f"{dataset_id_prefix}::{i}"
            # Compose canonical raw_text: instruction + separator + context + separator + expected
            raw_text = "\n---INSTRUCTION---\n" + s.instruction + "\n---CONTEXT---\n" + s.context + "\n---EXPECTED---\n" + s.expected

            prov = Provenance(source_uri=s.source_doc_id, ingested_by="raft-generator", ingested_at=s.timestamp)

            ds = DatasetSchema(
                dataset_id=did,
                domain=domain,
                source_type=SourceType.TEXT,
                intended_use=IntendedUse.finetuning,
                version=version,
                provenance=prov,
                metadata={"retrieval_score": s.retrieval_score, "generated_at": s.timestamp.isoformat()},
                raw_text=raw_text,
            )

            records.append(FTRecord(dataset=ds, sample_index=i))

        return records
