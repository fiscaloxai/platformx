# Getting Started

Example: basic import and configuration

```python
from platformx.config import PlatformConfig
from platformx.core import Platform
from platformx.training import RAFTOrchestrator, RAFTDatasetBuilder
from platformx.model.finetune import FineTuner

# Initialize
cfg = PlatformConfig(project_name="example", reproducible=True)
platform = Platform(cfg)

# RAFT generation + dataset build (conceptual)
orchestrator = RAFTOrchestrator(platform.retrieval_engine, seed=42)
samples = orchestrator.generate_for_datasets(["doc1", "doc2"])

builder = RAFTDatasetBuilder()
dataset = builder.build(samples, "pharma", "life_science")

# Fine-tune validation and run (conceptual)
finetuner = FineTuner()
finetuner.validate_datasets([dataset])
report = finetuner.run("/path/to/base_model_fp", [dataset])
```

This example demonstrates the intended high-level flow: generate RAFT samples, build a finetuning dataset, validate, and run a controlled fine-tune flow. Implementations are intentionally deterministic and auditable.
# Getting Started

Example: basic import and configuration

```python
from platformx.config import PlatformConfig
from platformx.core import Platform

## Getting Started

This guide walks you through the basics of using PlatformX.

## 1. Initialize a Project
```python
from platformx import PlatformConfig, Platform

config = PlatformConfig(
	project_name="my_first_project",
	data_dir="./data",
	logging_level="INFO",
	seed=42,  # For reproducibility
)

platform = Platform(config)
```

## 2. Load and Register Data
```python
# Register a single document
dataset = platform.register_dataset("./data/document.txt", {
	"dataset_id": "doc-001",
	"domain": "general",
	"intended_use": "retrieval",
	"version": "1.0.0",
})

print(f"Registered: {dataset.dataset_id}")
print(f"Checksum: {dataset.provenance.checksum}")
```

## 3. Index for Retrieval
```python
# Index the dataset
chunk_ids = platform.index_dataset("doc-001")
print(f"Created {len(chunk_ids)} chunks")
```

## 4. Query the Index
```python
from platformx.retrieval import GroundedQuery, QueryIntent

query = GroundedQuery(
	text="What are the main features?",
	intent=QueryIntent.informational,
)

results = platform.retrieval_engine.retrieve(query, max_results=5)

for result in results:
	print(f"Score: {result.score:.3f}")
	print(f"Text: {result.text[:200]}...")
	print(f"Source: {result.source}")
	print("---")
```

## 5. Add Safety Checks
```python
from platformx.safety import create_default_filter_chain, ConfidenceAssessor

# Setup safety
safety_chain = create_default_filter_chain()
assessor = ConfidenceAssessor()

# Check query safety
query_text = "What are the main features?"
safety_result = safety_chain.check(query_text)

if safety_result["decision"].value == "block":
	print(f"Query blocked: {safety_result['reason']}")
else:
	# Proceed with retrieval
	results = platform.retrieval_engine.retrieve(query, max_results=5)
    
	# Assess confidence
	evidence = [{"text": r.text, "score": r.score} for r in results]
	confidence = assessor.assess(evidence, query=query_text)
    
	print(f"Confidence: {confidence.level.value}")
```

## 6. Export Audit Log
```python
# Export audit trail
platform.export_audit_log("./audit_log.json")
```

## Next Steps

- [Configuration Guide](configuration.md) - Customize PlatformX behavior
- [RAG Pipeline](modules/core.md) - Build complete retrieval pipelines
- [RAFT Training](modules/training_raft.md) - Generate fine-tuning samples
- [Safety Guide](modules/safety.md) - Configure safety filters
print(platform)
```

Core concepts

Next steps
