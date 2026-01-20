from platformx.model.finetune import FineTuner
from platformx.data.schema import DatasetSchema, Provenance, IntendedUse, SourceType
from datetime import datetime
import pytest

def make_dataset(intended_use=IntendedUse.FINETUNING):
    prov = Provenance(source_uri="file.txt", ingested_by="tester", ingested_at=datetime.utcnow())
    return DatasetSchema(
        dataset_id="ds1",
        domain="test",
        source_type=SourceType.TEXT,
        intended_use=intended_use,
        version="1.0.0",
        provenance=prov,
        metadata={},
        raw_text="sample"
    )

def test_validate_datasets():
    ft = FineTuner()
    ds = make_dataset()
    ft.validate_datasets([ds])
    with pytest.raises(ValueError):
        ft.validate_datasets([])
    ds2 = make_dataset(intended_use=IntendedUse.RETRIEVAL)
    with pytest.raises(ValueError):
        ft.validate_datasets([ds2])
