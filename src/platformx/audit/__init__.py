"""Audit trail package for PlatformX.

Provides a deterministic, structured, in-memory audit logger suitable for
regulatory inspection. Do not expose internals by default.
"""

from .logger import AuditLogger

__all__ = ["AuditLogger"]
