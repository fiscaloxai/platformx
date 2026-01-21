

<p align="center">
	<img src="PlatformX.png" alt="PlatformX Logo" width="120"/>
</p>

# PlatformX

**PlatformX** is a robust, production-ready AI library for pharmaceutical and life sciences, focused on accuracy, auditability, and safety. It provides modular tools for controlled fine-tuning, retrieval-augmented generation (RAG), RAFT sample generation, and regulatory-grade model auditing.

---

## 🚀 Features

- **Modular Architecture:** Data loading, registry, retrieval, model adapters, safety, training, and audit modules.
- **Controlled Fine-Tuning:** Deterministic, auditable workflows for regulatory compliance.
- **Retrieval-Augmented Generation (RAG):** Fast, configurable document indexing and semantic search.
- **RAFT Sample Generation:** Automated creation of training samples for robust model evaluation.
- **Safety & Confidence:** Built-in safety filters, confidence scoring, and refusal logic.
- **Audit Logging:** Structured logs for traceability and governance.
- **Extensible & Typed:** Plugin points for custom adapters, policies, and domain logic.

---

## 📦 Installation

Requirements: Python 3.10+

```bash
python -m pip install --upgrade pip setuptools wheel
pip install .
```

See [docs/installation.md](docs/installation.md) for details and virtual environment tips.

---

## 🏗️ High-Level Architecture

PlatformX is organized into:

- `data`: Loading, schema, registry
- `retrieval`: Indexing, semantic search, query engine
- `model`: Adapters, backend, fine-tuning
- `training`: RAFT sample generation, dataset builder
- `safety`: Filters, confidence, refusal
- `audit`: Structured logging
- `api`: High-level one-liner functions

See [docs/api.md](docs/api.md) for API reference and [docs/strategy.md](docs/strategy.md) for compliance notes.

---

## 🧑‍💻 Quick Start

```python
import platformx.api as pfx

# Index documents for retrieval
pfx.index_documents("./docs/", dataset_id="my-docs")

# RAG query
result = pfx.rag_query("What is machine learning?", index_path="./index/")

# Generate RAFT training samples
samples = pfx.generate_raft_samples(["dataset-001"], retrieval_index="./index/")

# Fine-tune a model
pfx.finetune("meta-llama/Llama-2-7b-hf", dataset_path="./training_data/")

# Run inference
response = pfx.generate("Explain quantum computing", model="gpt-4")
```

See [docs/getting_started.md](docs/getting_started.md) for more examples.

---

## 📚 Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api.md)
- [Configuration](docs/configuration.md)
- [Project Strategy & Compliance](docs/strategy.md)
- [Modules Overview](docs/modules/)

---

## 🏛️ Design Philosophy

- **Reproducibility:** Deterministic workflows, dataset and adapter fingerprints.
- **Transparency:** Structured audit logs, traceable model and dataset lineage.
- **Extensibility:** Plugin points for new adapters, policies, and compliance controls.
- **Safety:** Built-in filters, confidence scoring, refusal logic.
- **Community:** Clear changelogs, contribution guidelines, and open collaboration.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features. For major changes, discuss in an issue first. See [CHANGELOG.md](CHANGELOG.md) for release notes.

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
