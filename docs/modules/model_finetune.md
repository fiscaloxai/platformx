# `platformx.model.finetune`

Purpose
- Interfaces and lightweight tooling to perform deterministic, auditable fine-tune workflows.

Key classes
- `FineTuner`: validate datasets, orchestrate adapter creation, and produce a `FinetuneReport`.

Safety features
- FineTuner performs pre-flight dataset validation and emits structured reports for audit.
