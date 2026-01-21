"""
Audit module for PlatformX.

This module provides comprehensive audit logging with structured events,
correlation tracking, and multiple export formats for compliance and debugging.

Design Principles:
- Every significant action is logged
- Events are correlated across a request lifecycle
- Logs are structured for machine parsing
- Thread-safe for concurrent operations
- Export to multiple formats (JSON, CSV, callbacks)

Core Components:
- AuditLogger: Main logging interface
- AuditEntry: Structured log entry
- AuditContext: Correlation context for related events
- AuditEventType: Categorized event types

Exporters:
- JSONExporter: Export to JSON files
- CSVExporter: Export to CSV files
- CallbackExporter: Stream to custom handlers

Example:
	from platformx.audit import AuditLogger, AuditContext, AuditEventType

	# Create logger with auto-export
	audit = AuditLogger(component="my_app", auto_export_threshold=1000)
	audit.add_exporter(JSONExporter("./logs/audit.json"))

	# Set context for request correlation
	ctx = AuditContext(session_id="user-123")
	with audit.context_manager(ctx):
		audit.log(AuditEventType.QUERY_RECEIVED, "query", {"text": "What is X?"})
		audit.log(AuditEventType.RETRIEVAL_EXECUTED, "retrieval", {"results": 5})
		audit.log(AuditEventType.INFERENCE_RESPONSE, "response", {"tokens": 150})

	# Export and get summary
	audit.flush()
	print(audit.get_summary())
"""

import logging
logger = logging.getLogger("platformx.audit")

try:
	from .logger import (
		AuditEntry,
		AuditEventType,
		AuditContext,
		AuditExporter,
		JSONExporter,
		CSVExporter,
		CallbackExporter,
		AuditLogger,
	)
except ImportError as e:
	logger.warning(f"Could not import audit.logger: {e}")

__all__ = [
	# Core
	"AuditLogger",
	"AuditEntry",
	"AuditEventType",
	"AuditContext",
	# Exporters
	"AuditExporter",
	"JSONExporter",
	"CSVExporter",
	"CallbackExporter",
]
"""Audit trail package for PlatformX.

Provides a deterministic, structured, in-memory audit logger suitable for
regulatory inspection. Do not expose internals by default.
"""

from .logger import AuditLogger

__all__ = ["AuditLogger"]
