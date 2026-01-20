# Configuration & Environment

Environment setup
- Use Python 3.10+ and a virtual environment (`venv` or `conda`).

Dependencies
- The project includes optional heavy ML deps (transformers, peft, datasets). Install only what you need.

PYTHONPATH
- When building docs locally with mkdocstrings, set `PYTHONPATH=./src` so `platformx` is importable.

Custom adapters/backends
- Implement the `Adapter` interface in `platformx.model.adapters` to integrate new model backends.
- Register adapters with the `AdapterRegistry` and attach them to `BaseModelBackend` instances.

Notes on compatibility
- Prefer pinned versions for production deployments and record dependency manifests for audits.
