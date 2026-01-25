# PlatformX

<div align="center">
        <div style="display: flex; align-items: center; justify-content: center; gap: 12px;">
                <img src="https://github.com/fiscaloxai/platformx/raw/main/PlatformX.png" alt="PlatformX Logo" width="100"/>
        </div>
        <p><strong>Enterprise-Grade AI Library for Pharmaceutical & Life Sciences</strong></p>
        <p>
                <a href="#features">Features</a> •
                <a href="#installation">Installation</a> •
                <a href="#quick-start">Quick Start</a> •
                <a href="#documentation">Documentation</a> •
                <a href="#examples">Examples</a>
        </p>
</div>

---

## Overview

**PlatformX** is a production-ready Python library specifically designed for building **accurate, auditable, and safety-conscious AI applications** in the pharmaceutical and life sciences domains. 

Whether you're building RAG systems for clinical trial data, fine-tuning models on regulatory documents, or generating training data with RAFT, PlatformX provides the tools you need with built-in compliance and traceability.

### Why PlatformX?

**Pharma-Focused**: Built specifically for regulated industries  
**Audit-First**: Complete provenance tracking and structured logging  
**Safety-Built-In**: PII detection, content filtering, confidence scoring  
**Production-Ready**: Type-safe, tested, and documented  
**Flexible**: Modular architecture with pluggable components  
**Compliant**: Designed for regulatory review and validation  

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [Documentation](#documentation)
- [Design Principles](#design-principles)
- [License](#license)

---

## Features

### Retrieval-Augmented Generation (RAG)
- **Multi-format document support**: PDF, DOCX, HTML, XML, CSV, JSON, Parquet
- **Flexible embeddings**: TF-IDF, Sentence Transformers, or custom backends
- **Smart chunking**: Configurable overlap for better context retention
- **Semantic search**: Fast, deterministic retrieval with scoring

### Model Fine-Tuning
- **LoRA/PEFT**: Parameter-efficient fine-tuning for large models
- **HuggingFace integration**: Seamless model loading and training
- **Quantization support**: 8-bit and 4-bit training for memory efficiency
- **Audit logging**: Complete training lineage for compliance

### RAFT Sample Generation
- **Automated dataset creation**: Generate training samples from retrieved context
- **Configurable ratios**: Control positive/negative sample distribution
- **Reasoning chains**: Include step-by-step reasoning in samples
- **Distractor injection**: Add hard negatives for robust training

### Safety & Compliance
- **PII detection**: Automatic detection of emails, phones, SSN, credit cards
- **Content filtering**: Keyword and regex-based safety filters
- **Intent classification**: Block out-of-scope queries
- **Confidence scoring**: Multi-factor confidence assessment
- **Audit trails**: Structured logging for regulatory review

### Data Management
- **Dataset registry**: Centralized tracking with provenance
- **Version control**: Semantic versioning for datasets and models
- **Checksums**: SHA256 hashing for data integrity
- **Metadata tracking**: Rich metadata for discovery and governance

---

## Installation

### Basic Installation

```bash
pip install platformx
```

### With All Features

```bash
pip install platformx[retrieval,training,documents,openai,anthropic]
```

### From Source

```bash
git clone https://github.com/your-org/platformx.git
cd platformx
pip install -e ".[dev]"
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

---

## Architecture Overview

PlatformX is organized into seven core modules:

```
platformx/
├── data/          # Dataset loading, schema, registry
├── retrieval/     # Indexing, embeddings, query engine
├── model/         # Fine-tuning, adapters, inference
├── training/      # RAFT generation, dataset builders
├── safety/        # Filters, confidence, refusal logic
├── audit/         # Structured logging, compliance
└── api/           # High-level user-friendly API
```

### Module Details

- **`data`**: Load datasets from various formats with automatic text extraction and provenance tracking
- **`retrieval`**: Index documents and perform semantic search with configurable backends
- **`model`**: Fine-tune models using LoRA/PEFT with full audit logging
- **`training`**: Generate RAFT samples for retrieval-aware model training
- **`safety`**: Filter content, detect PII, assess confidence, generate refusals
- **`audit`**: Log all operations with correlation IDs for traceability
- **`api`**: Simple one-liner functions for common workflows

For detailed API reference, see [docs/api.md](docs/api.md).

---

## Quick Start

### 1. Index Clinical Trial Documents

```python
import platformx.api as pfx

# Index a directory of clinical trial documents
result = pfx.index_documents(
    source="./clinical_trials/",
    dataset_id="trials-2024-q1",
    index_path="./index/trials/",
    chunk_size=200,
    embedding_backend="tfidf",
    domain="clinical"
)

print(f"Indexed {result['chunk_count']} chunks")
```

### 2. Run RAG Query with Safety

```python
# Query with automatic safety filtering
response = pfx.rag_query(
    query="What are the adverse events in pediatric trials?",
    index_path="./index/trials/",
    top_k=5,
    safety_check=True,
    min_confidence="medium"
)

# Check results
if response['safety_result']['decision'] == 'allow':
    for i, result in enumerate(response['results'], 1):
        print(f"{i}. [{result['score']:.3f}] {result['text'][:100]}...")
else:
    print(f"Query blocked: {response['safety_result']['reason']}")
```

### 3. Generate RAFT Training Samples

```python
# Generate training samples from indexed data
samples = pfx.generate_raft_samples(
    dataset_ids=["trials-2024-q1", "trials-2024-q2"],
    index_path="./index/trials/",
    samples_per_dataset=100,
    positive_fraction=0.6,
    include_reasoning=True,
    output_path="./training_data/raft_samples.json"
)

print(f"Generated {len(samples)} RAFT samples")
```

### 4. Fine-Tune with Compliance Logging

```python
# Fine-tune a model with full audit trail
report = pfx.finetune(
    base_model="meta-llama/Llama-2-7b-hf",
    dataset_path="./training_data/raft_samples.json",
    output_dir="./models/pharma-qa-v1",
    num_epochs=3,
    learning_rate=2e-4,
    lora_r=16,
    seed=42
)

print(f"Model fine-tuned: {report['adapter_id']}")
print(f"Training datasets: {report['training_dataset_ids']}")
```

### 5. Full Platform Setup

```python
import platformx as pfx

# Initialize platform with configuration
config = pfx.PlatformConfig(
    project_name="pharma_qa_system",
    data_dir="./data",
    logging_level="INFO",
    reproducible=True,
    seed=42
)

platform = pfx.Platform(config)

# Register a dataset
dataset = platform.register_dataset(
    "clinical_protocols.pdf",
    {
        "dataset_id": "protocols-001",
        "domain": "clinical",
        "intended_use": "retrieval"
    }
)

# Index for retrieval
chunk_ids = platform.index_dataset("protocols-001")

print(f"Registered and indexed {len(chunk_ids)} chunks")
```

---

## Use Cases

### 1. Clinical Trial Q&A System

```python
# Build a Q&A system over clinical trial documents
import platformx.api as pfx

# Step 1: Index trial documents
pfx.index_documents(
    source="./trials/",
    dataset_id="clinical-trials-2024",
    domain="clinical"
)

# Step 2: Query with safety
result = pfx.rag_query(
    "What is the efficacy rate in Phase 3 trials?",
    index_path="./index/",
    safety_check=True
)

# Step 3: Generate response with confidence
if result['confidence']['level'] == 'high':
    print(f"Answer: {result['results'][0]['text']}")
else:
    print("Low confidence - review required")
```

### 2. Regulatory Document Analysis

```python
# Analyze FDA submissions and guidance documents
from platformx import Platform, PlatformConfig
from platformx.safety import create_default_filter_chain

config = PlatformConfig(
    project_name="regulatory_analysis",
    data_dir="./fda_docs"
)
platform = Platform(config)

# Load regulatory documents
platform.register_dataset("fda_guidance.pdf", {
    "dataset_id": "fda-guidance-001",
    "domain": "regulatory",
    "intended_use": "retrieval"
})

# Index with pharma-specific safety filters
platform.index_dataset("fda-guidance-001")

# Query with domain-specific filters
chain = create_default_filter_chain("pharma")
query_result = chain.check("What are the requirements?", {})
```

### 3. Fine-Tune Domain-Specific Models

```python
# Train a model specifically for pharma Q&A
import platformx.api as pfx

# Generate RAFT samples from your documents
samples = pfx.generate_raft_samples(
    dataset_ids=["protocols", "trials", "guidance"],
    index_path="./index/",
    samples_per_dataset=200
)

# Fine-tune with audit logging
pfx.finetune(
    base_model="microsoft/phi-2",
    datasets=samples,
    output_dir="./models/pharma-phi-2",
    num_epochs=5
)
```

---

## Documentation

Comprehensive documentation is available:

- **[Getting Started Guide](docs/getting_started.md)** - Step-by-step tutorial
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Configuration](docs/configuration.md)** - Configuration options
- **[Strategy & Compliance](docs/strategy.md)** - Design principles
- **[Module Overview](docs/modules/)** - Deep dive into each module
- **[Installation Guide](INSTALL.md)** - Detailed setup instructions

### Examples

Explore the [examples/](examples/) directory:

1. **[01_basic_indexing.py](examples/01_basic_indexing.py)** - Document indexing basics
2. **[02_rag_pipeline.py](examples/02_rag_pipeline.py)** - Complete RAG workflow
3. **[03_raft_generation.py](examples/03_raft_generation.py)** - RAFT sample generation
4. **[04_safety_filtering.py](examples/04_safety_filtering.py)** - Safety configuration
5. **[05_quick_start.py](examples/05_quick_start.py)** - Quick start demo

---

## Design Principles

### Reproducibility
- Deterministic workflows with seed control
- Dataset and model fingerprinting
- Version tracking for all artifacts

### Transparency
- Structured audit logs for all operations
- Complete provenance tracking
- Traceable model and dataset lineage

### Extensibility
- Plugin architecture for adapters and backends
- Custom policy injection points
- Flexible compliance controls

### Safety
- Built-in PII detection and content filtering
- Confidence scoring and refusal logic
- Domain-specific safety policies

---

## Performance & Benchmarks

PlatformX is designed for production use:

- **Indexing**: ~1000 documents/minute (TF-IDF backend)
- **Retrieval**: <100ms for top-10 queries on 10K documents
- **Fine-tuning**: Supports models up to 70B parameters with quantization
- **Memory**: <2GB RAM for indexing 10K documents

See [benchmarks/](benchmarks/) for detailed performance metrics.

### Quick Start for Contributors

```bash
# Clone and setup
git clone https://github.com/your-org/platformx.git
cd platformx
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
mypy src/

# Format code
black src/
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
