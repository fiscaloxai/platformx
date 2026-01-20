# `platformx.training.raft`

Purpose
- RAFTOrchestrator and RAFT dataset builder for evidence-focused synthetic sample generation.

Key classes
- `RAFTOrchestrator`: generates labeled, provenance-linked RAFT samples.
- `RAFTDatasetBuilder`: converts RAFT samples into a `DatasetSchema` suitable for controlled fine-tuning.

Safety features
- Only evidence-backed items are used for dataset generation; every sample includes provenance.
