from platformx.data.schema import DatasetSchema, Provenance, SourceType, IntendedUse
from datetime import datetime

def test_dataset_schema():
    prov = Provenance(source_uri="file.txt", ingested_by="tester", ingested_at=datetime.utcnow())
    ds = DatasetSchema(
        dataset_id="ds1",
        domain="test",
        source_type=SourceType.TEXT,
        intended_use=IntendedUse.RETRIEVAL,
        version="1.0.0",
        provenance=prov,
        metadata={"foo": "bar"},
        raw_text="hello world"
    )
    assert ds.dataset_id == "ds1"
    assert ds.source_type == SourceType.TEXT
    assert ds.intended_use == IntendedUse.RETRIEVAL
    assert ds.provenance.source_uri == "file.txt"
