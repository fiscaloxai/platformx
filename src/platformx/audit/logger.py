
"""
Audit logger for PlatformX: structured events, correlation tracking, and multiple export formats.

Features:
- Structured AuditEntry with event types and correlation
- AuditContext for request/session correlation
- Multiple exporters: JSON, CSV, callback
- Thread safety and auto-export
- Custom exporter support
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import uuid
import hashlib
import logging
import json
import csv
import threading
from contextlib import contextmanager

logger = logging.getLogger("platformx.audit.logger")

class AuditEventType(str, Enum):
    """Types of audit events for categorization."""
    # Data events
    DATASET_REGISTERED = "dataset_registered"
    DATASET_LOADED = "dataset_loaded"
    DATASET_INDEXED = "dataset_indexed"
    # Query events
    QUERY_RECEIVED = "query_received"
    QUERY_PROCESSED = "query_processed"
    # Retrieval events
    RETRIEVAL_EXECUTED = "retrieval_executed"
    RETRIEVAL_RESULT = "retrieval_result"
    # Safety events
    SAFETY_CHECK = "safety_check"
    SAFETY_BLOCKED = "safety_blocked"
    SAFETY_PASSED = "safety_passed"
    # Confidence events
    CONFIDENCE_ASSESSED = "confidence_assessed"
    # Inference events
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    # Training events
    TRAINING_STARTED = "training_started"
    TRAINING_STEP = "training_step"
    TRAINING_COMPLETED = "training_completed"
    # Refusal events
    REFUSAL_GENERATED = "refusal_generated"
    # System events
    SYSTEM_ERROR = "system_error"
    CUSTOM = "custom"

@dataclass
class AuditEntry:
    """A single audit log entry with full traceability."""
    entry_id: str
    timestamp: str
    event_type: AuditEventType
    event: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    component: str = "platformx"
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

@dataclass
class AuditContext:
    """Context that flows through a request for correlation."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def new_correlated(self) -> "AuditContext":
        return AuditContext(session_id=self.session_id, user_id=self.user_id, metadata=self.metadata)

class AuditExporter:
    """Abstract base for audit log exporters."""
    def export(self, entries: List[AuditEntry]) -> None:
        raise NotImplementedError
    def exporter_id(self) -> str:
        raise NotImplementedError

class JSONExporter(AuditExporter):
    """Export audit logs to JSON file."""
    def __init__(self, path: str, append: bool = True, pretty: bool = True):
        self.path = path
        self.append = append
        self.pretty = pretty

    def export(self, entries: List[AuditEntry]) -> None:
        path = Path(self.path)
        all_entries = []
        if self.append and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    all_entries = json.load(f)
            except Exception:
                all_entries = []
        all_entries += [e.to_dict() for e in entries]
        with open(path, "w", encoding="utf-8") as f:
            if self.pretty:
                json.dump(all_entries, f, indent=2, ensure_ascii=False)
            else:
                json.dump(all_entries, f, ensure_ascii=False)
        logger.info(f"Exported {len(entries)} audit entries to {path}")

    def exporter_id(self) -> str:
        return "json_file"

class CSVExporter(AuditExporter):
    """Export audit logs to CSV file."""
    def __init__(self, path: str, append: bool = True):
        self.path = path
        self.append = append

    def export(self, entries: List[AuditEntry]) -> None:
        path = Path(self.path)
        fieldnames = [
            "entry_id", "timestamp", "event_type", "event", "correlation_id", "session_id", "user_id", "component", "version"
        ]
        # Flatten payload keys
        payload_keys = set()
        for e in entries:
            payload_keys.update(e.payload.keys())
        fieldnames += sorted(payload_keys)
        mode = "a" if self.append and path.exists() else "w"
        with open(path, mode, newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            for e in entries:
                row = {k: getattr(e, k, None) for k in fieldnames}
                for pk in payload_keys:
                    val = e.payload.get(pk)
                    if isinstance(val, (dict, list)):
                        row[pk] = json.dumps(val, ensure_ascii=False)
                    else:
                        row[pk] = val
                writer.writerow(row)
        logger.info(f"Exported {len(entries)} audit entries to {path}")

    def exporter_id(self) -> str:
        return "csv_file"

class CallbackExporter(AuditExporter):
    """Export audit logs via callback function (for streaming/webhooks)."""
    def __init__(self, callback: Callable[[AuditEntry], None]):
        self.callback = callback

    def export(self, entries: List[AuditEntry]) -> None:
        for e in entries:
            self.callback(e)

    def exporter_id(self) -> str:
        return "callback"

class AuditLogger:
    """Structured, deterministic, thread-safe audit logger with correlation and export.

    - Correlation IDs link related events
    - Thread safety via RLock
    - Auto-export when entry count reaches threshold
    - Supports custom exporters (JSON, CSV, callback, etc)
    """
    def __init__(self, component: str = "platformx", auto_export_threshold: int = 0):
        self._entries: List[AuditEntry] = []
        self._lock = threading.RLock()
        self._exporters: List[AuditExporter] = []
        self._context: Optional[AuditContext] = None
        self._auto_export_threshold = auto_export_threshold
        self._component = component
        self._logger = logger

    def set_context(self, context: AuditContext) -> None:
        self._context = context

    def clear_context(self) -> None:
        self._context = None

    @contextmanager
    def context_manager(self, context: AuditContext):
        prev = self._context
        self.set_context(context)
        try:
            yield
        finally:
            self._context = prev

    def add_exporter(self, exporter: AuditExporter) -> "AuditLogger":
        self._exporters.append(exporter)
        return self

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _generate_entry_id(self) -> str:
        return str(uuid.uuid4())

    def log(self, event_type: AuditEventType, event: str, payload: Optional[Dict[str, Any]] = None, **extra) -> AuditEntry:
        with self._lock:
            payload = dict(payload or {})
            payload.update(extra)
            entry = AuditEntry(
                entry_id=self._generate_entry_id(),
                timestamp=self._now(),
                event_type=event_type,
                event=event,
                payload=payload,
                correlation_id=self._context.correlation_id if self._context else None,
                session_id=self._context.session_id if self._context else None,
                user_id=self._context.user_id if self._context else None,
                component=self._component,
                version="1.0",
            )
            self._entries.append(entry)
            if self._auto_export_threshold > 0 and len(self._entries) >= self._auto_export_threshold:
                self.flush()
            return entry

    def log_query(self, query_text: str, intent: str) -> AuditEntry:
        return self.log(AuditEventType.QUERY_RECEIVED, "query", {"text": query_text, "intent": intent})

    def log_retrieval(self, results: List[Dict[str, Any]]) -> AuditEntry:
        items = [
            {"dataset_id": r.get("dataset_id"), "chunk_id": r.get("chunk_id"), "score": float(r.get("score", 0.0))}
            for r in results
        ]
        return self.log(AuditEventType.RETRIEVAL_RESULT, "retrieval", {"results": items})

    def log_safety_decision(self, decision: str, reason: str, details: Optional[Dict[str, Any]] = None) -> AuditEntry:
        return self.log(AuditEventType.SAFETY_CHECK, "safety", {"decision": decision, "reason": reason, "details": details or {}})

    def log_refusal(self, refusal_message: str, reason: str) -> AuditEntry:
        return self.log(AuditEventType.REFUSAL_GENERATED, "refusal", {"message": refusal_message, "reason": reason})

    def log_confidence(self, level: str, rationale: Dict[str, Any]) -> AuditEntry:
        return self.log(AuditEventType.CONFIDENCE_ASSESSED, "confidence", {"level": level, "rationale": rationale})

    def log_inference(self, prompt: str, response: str, model_id: str, latency_ms: float, **kwargs) -> AuditEntry:
        return self.log(AuditEventType.INFERENCE_RESPONSE, "inference_response", {"prompt": prompt, "response": response, "model_id": model_id, "latency_ms": latency_ms, **kwargs})

    def log_training_step(self, step: int, loss: float, metrics: Dict[str, Any] = None) -> AuditEntry:
        return self.log(AuditEventType.TRAINING_STEP, "training_step", {"step": step, "loss": loss, "metrics": metrics or {}})

    def log_error(self, error: str, exception: Optional[Exception] = None, **context) -> AuditEntry:
        payload = {"error": error, **context}
        if exception:
            payload["exception_type"] = type(exception).__name__
            payload["exception_message"] = str(exception)
        return self.log(AuditEventType.SYSTEM_ERROR, "system_error", payload)

    def get_entries(self, event_type: Optional[AuditEventType] = None, correlation_id: Optional[str] = None, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        with self._lock:
            result = self._entries
            if event_type:
                result = [e for e in result if e.event_type == event_type]
            if correlation_id:
                result = [e for e in result if e.correlation_id == correlation_id]
            if since:
                result = [e for e in result if datetime.fromisoformat(e.timestamp) >= since]
            return [e.to_dict() for e in result]

    def flush(self) -> int:
        with self._lock:
            count = len(self._entries)
            for exporter in self._exporters:
                exporter.export(self._entries)
            self._entries.clear()
            return count

    def export_json(self, path: str) -> None:
        JSONExporter(path).export(self._entries)

    def export_csv(self, path: str) -> None:
        CSVExporter(path).export(self._entries)

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._entries)
            by_type = {}
            for e in self._entries:
                by_type[e.event_type.value] = by_type.get(e.event_type.value, 0) + 1
            times = [e.timestamp for e in self._entries]
            time_range = {"first": min(times) if times else None, "last": max(times) if times else None}
            unique_corr = len(set(e.correlation_id for e in self._entries if e.correlation_id))
            return {
                "total_entries": total,
                "entries_by_type": by_type,
                "time_range": time_range,
                "unique_correlation_ids": unique_corr,
            }

__all__ = [
    "AuditEntry",
    "AuditEventType",
    "AuditContext",
    "AuditExporter",
    "JSONExporter",
    "CSVExporter",
    "CallbackExporter",
    "AuditLogger",
]
