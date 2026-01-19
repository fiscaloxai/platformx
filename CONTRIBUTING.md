# Contributing to PlatformX

Thanks for contributing! Please follow these guidelines to keep the project high-quality and auditable.

1. Fork the repository and create a feature branch.
2. Keep changes small and focused; include tests where appropriate.
3. Follow black formatting and add type hints for public interfaces.
4. Open a pull request and include a clear description and rationale.

Code style
- Use `black` for formatting and `ruff`/`mypy` for linting and typing checks.

Security and data
- Do not commit secrets or protected data. Use environment variables or secret stores in CI.
