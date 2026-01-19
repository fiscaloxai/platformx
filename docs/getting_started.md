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
