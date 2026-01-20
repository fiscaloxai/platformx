# `platformx.safety`

Purpose
- Safety filters, refusal engine, and confidence assessment utilities to control model outputs.

Key components
- `evaluate_safety`: simple deterministic checks.
- `RefusalEngine`: rules-based refusals for sensitive or disallowed content.
- `assess_confidence`: heuristic confidence estimator for pipeline decisions.

Safety features
- Deterministic, auditable decisions and hooks to integrate into inference pipelines.
