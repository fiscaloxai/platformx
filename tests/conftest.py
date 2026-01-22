"""Pytest configuration and shared fixtures for PlatformX tests."""

import pytest
import os
import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dir_path = tempfile.mkdtemp(prefix="platformx_test_")
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    file_path = os.path.join(temp_dir, "sample.txt")
    content = "This is a sample document for testing. It contains multiple sentences. Machine learning is useful."
    with open(file_path, "w") as f:
        f.write(content)
    return file_path


@pytest.fixture
def sample_documents(temp_dir):
    """Create multiple sample documents for testing."""
    docs = {
        "doc1.txt": "Python is a programming language. It is widely used for data science.",
        "doc2.txt": "Machine learning uses algorithms to learn from data. Deep learning is a subset.",
        "doc3.txt": "Natural language processing enables computers to understand text.",
    }
    paths = []
    for filename, content in docs.items():
        path = os.path.join(temp_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        paths.append(path)
    return paths


@pytest.fixture
def sample_provenance():
    """Create a sample Provenance object."""
    from platformx.data import Provenance
    return Provenance(
        source_uri="file://test.txt",
        ingested_by="test_suite",
        ingested_at=datetime.now(timezone.utc),
        checksum="abc123",
    )


@pytest.fixture
def sample_dataset(sample_provenance):
    """Create a sample DatasetSchema object."""
    from platformx.data import DatasetSchema, SourceType, IntendedUse, Domain
    return DatasetSchema(
        dataset_id="test-dataset-001",
        domain=Domain.GENERAL,
        source_type=SourceType.TEXT,
        intended_use=IntendedUse.RETRIEVAL,
        version="1.0.0",
        provenance=sample_provenance,
        raw_text="This is test content for the dataset.",
    )


@pytest.fixture
def sample_dataset_for_finetuning(sample_provenance):
    """Create a sample DatasetSchema for fine-tuning."""
    from platformx.data import DatasetSchema, SourceType, IntendedUse, Domain
    return DatasetSchema(
        dataset_id="test-finetune-001",
        domain=Domain.GENERAL,
        source_type=SourceType.TEXT,
        intended_use=IntendedUse.FINETUNING,
        version="1.0.0",
        provenance=sample_provenance,
        raw_text="Sample training content.",
    )


@pytest.fixture
def platform_config(temp_dir):
    """Create a PlatformConfig for testing."""
    from platformx import PlatformConfig
    return PlatformConfig(
        project_name="test_project",
        data_dir=temp_dir,
        logging_level="DEBUG",
        seed=42,
    )


@pytest.fixture
def sample_evidence():
    """Create sample evidence for confidence testing."""
    return [
        {"text": "Python is a programming language.", "score": 0.9, "dataset_id": "doc1"},
        {"text": "Python is widely used for programming.", "score": 0.85, "dataset_id": "doc2"},
        {"text": "Programming with Python is popular.", "score": 0.8, "dataset_id": "doc3"},
    ]


@pytest.fixture
def sample_html_file(temp_dir):
    """Create a sample HTML file for testing."""
    file_path = os.path.join(temp_dir, "sample.html")
    content = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Document</title></head>
    <body>
        <h1>Clinical Trial Results</h1>
        <p>This is a test document with <strong>important</strong> information.</p>
        <script>console.log('test');</script>
    </body>
    </html>
    """
    with open(file_path, "w") as f:
        f.write(content)
    return file_path


@pytest.fixture
def sample_xml_file(temp_dir):
    """Create a sample XML file for testing."""
    file_path = os.path.join(temp_dir, "sample.xml")
    content = """<?xml version="1.0"?>
    <clinical_trial>
        <title>Test Trial</title>
        <description>This is a test clinical trial description.</description>
        <results>Positive outcomes observed.</results>
    </clinical_trial>
    """
    with open(file_path, "w") as f:
        f.write(content)
    return file_path


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    file_path = os.path.join(temp_dir, "sample.csv")
    content = "name,age,condition\nJohn,30,diabetes\nJane,25,hypertension\n"
    with open(file_path, "w") as f:
        f.write(content)
    return file_path


@pytest.fixture
def sample_json_file(temp_dir):
    """Create a sample JSON file for testing."""
    import json
    file_path = os.path.join(temp_dir, "sample.json")
    data = {
        "trial_id": "NCT001",
        "title": "Test Trial",
        "participants": 100,
        "results": ["outcome1", "outcome2"]
    }
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def indexer_with_data(sample_dataset):
    """Create an indexer with pre-loaded data."""
    from platformx.retrieval import Indexer
    indexer = Indexer(chunk_size=50, chunk_overlap=10)
    indexer.index_dataset(sample_dataset)
    return indexer


@pytest.fixture
def audit_logger():
    """Create an AuditLogger for testing."""
    from platformx.audit import AuditLogger
    return AuditLogger(component="test")


@pytest.fixture
def safety_filter_chain():
    """Create a safety filter chain for testing."""
    from platformx.safety import SafetyFilterChain
    return SafetyFilterChain()

