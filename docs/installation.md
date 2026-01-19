# Installation

Requirements
- Python 3.10 or later

Install from source (recommended for development):

```bash
python -m pip install --upgrade pip setuptools wheel
pip install .
```

Virtual environments
- We recommend using `venv` or `conda` to isolate dependencies.

Notes
- The project declares optional heavy ML dependencies (transformers, peft, datasets) in `pyproject.toml`. Install only what you need for your use-case.
