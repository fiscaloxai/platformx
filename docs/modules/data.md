# `platformx.data`

Purpose
- Data schemas, loaders, and registry for dataset provenance and validation.

Key classes
- `DatasetSchema` (pydantic): domain, source_type, provenance, and metadata.
- `DataLoader`: basic file loaders with explicit content validation.
- `DatasetRegistry`: in-memory registry with duplicate detection and provenance tracking.

Safety features
- Loaders validate source types and provenance; registry enforces unique identifiers.
