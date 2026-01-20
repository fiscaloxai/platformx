# `platformx.config`

Purpose
- Typed configuration model (Pydantic) for reproducible runs and environment settings.

Key classes
- `PlatformConfig`: fields include `project_name`, `data_dir`, `logging_level`, and `reproducible`.

Usage
- Instantiate `PlatformConfig` and pass to `Platform` to keep settings centralized.
