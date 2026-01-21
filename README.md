# PlatformX

<div align="center">
	<img src="PlatformX.png" alt="PlatformX Logo" width="80"/>
</div>

PlatformX is an enterprise-grade Python library developed by Fiscal Ox for building accurate, auditable, and safety-conscious AI applications in the pharmaceutical and life sciences domains. PlatformX provides modular, production-ready components for controlled fine-tuning, retrieval-augmented generation (RAG), RAFT sample generation, and regulatory-grade model auditing.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Design Principles](#design-principles)
- [License](#license)

## Features

- Modular architecture: data loading, registry, retrieval, model adapters, safety, training, and audit modules
- Controlled, auditable fine-tuning workflows for regulatory compliance
- Retrieval-augmented generation (RAG) with fast, configurable document indexing and semantic search
- Automated RAFT sample generation for robust model evaluation
- Built-in safety filters, confidence scoring, and refusal logic
- Structured audit logging for traceability and governance
- Extensible plugin points for custom adapters, policies, and domain logic

## Installation

**Requirements:** Python 3.10 or later

Install from source:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install .
```

For details and virtual environment recommendations, see [docs/installation.md](docs/installation.md).

## Architecture Overview

PlatformX is organized into the following modules:

- `data`: Data loading, schema definitions, and registry
- `retrieval`: Indexing, semantic search, and query engine
- `model`: Model adapters, backend, and fine-tuning
- `training`: RAFT sample generation and dataset builder
- `safety`: Safety filters, confidence scoring, and refusal logic
- `audit`: Structured logging for compliance and traceability
- `api`: High-level, user-friendly API functions

For a detailed API reference, see [docs/api.md](docs/api.md). For compliance and strategy, see [docs/strategy.md](docs/strategy.md).

## Quick Start

```python
import platformx.api as pfx

# Index documents for retrieval
pfx.index_documents("./docs/", dataset_id="my-docs")

# Run a RAG query
result = pfx.rag_query("What is machine learning?", index_path="./index/")

# Generate RAFT training samples
samples = pfx.generate_raft_samples(["dataset-001"], retrieval_index="./index/")

# Fine-tune a model
pfx.finetune("meta-llama/Llama-2-7b-hf", dataset_path="./training_data/")

# Run inference
response = pfx.generate("Explain quantum computing", model="gpt-4")
```

For more usage examples, see [docs/getting_started.md](docs/getting_started.md).

## Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api.md)
- [Configuration](docs/configuration.md)
- [Project Strategy & Compliance](docs/strategy.md)
- [Modules Overview](docs/modules/)

## Design Principles

- **Reproducibility:** Deterministic workflows, dataset and adapter fingerprinting
- **Transparency:** Structured audit logs and traceable model/dataset lineage
- **Extensibility:** Plugin points for new adapters, policies, and compliance controls
- **Safety:** Built-in filters, confidence scoring, and refusal logic
- **Collaboration:** Clear changelogs, contribution guidelines, and open development

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
