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

cfg = PlatformConfig(project_name="example", reproducible=True)
platform = Platform(cfg)

print(platform)
```

Core concepts
- `PlatformConfig` — typed config model
- `Platform` — high-level orchestrator to register datasets and components
- `DatasetSchema` — pydantic schema for dataset metadata and provenance

Next steps
- See `docs/api.md` for how to generate API docs.
