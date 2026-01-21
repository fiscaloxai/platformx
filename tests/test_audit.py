"""Tests for the audit module."""

import pytest
import os
import json


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_audit_logger_creation(self):
        from platformx.audit import AuditLogger

        logger = AuditLogger()
        assert logger.count() == 0

    def test_log_event(self):
        from platformx.audit import AuditLogger, AuditEventType

        logger = AuditLogger()
        entry = logger.log(AuditEventType.QUERY_RECEIVED, "query", {"text": "test query"})

        assert entry is not None
        assert entry.event_type == AuditEventType.QUERY_RECEIVED
        assert logger.count() == 1

    def test_log_query(self):
        from platformx.audit import AuditLogger

        logger = AuditLogger()
        entry = logger.log_query("What is Python?", "informational")

        assert entry is not None
        assert "text" in entry.payload

    def test_log_with_context(self):
        from platformx.audit import AuditLogger, AuditContext, AuditEventType

        logger = AuditLogger()
        ctx = AuditContext(session_id="test-session", user_id="user-123")

        logger.set_context(ctx)
        entry = logger.log(AuditEventType.QUERY_RECEIVED, "query", {"text": "test"})

        assert entry.session_id == "test-session"
        assert entry.correlation_id is not None

        logger.clear_context()

    def test_get_entries_filtered(self):
        from platformx.audit import AuditLogger, AuditEventType

        logger = AuditLogger()
        logger.log(AuditEventType.QUERY_RECEIVED, "query", {})
        logger.log(AuditEventType.RETRIEVAL_EXECUTED, "retrieval", {})
        logger.log(AuditEventType.QUERY_RECEIVED, "query", {})

        entries = logger.get_entries(event_type=AuditEventType.QUERY_RECEIVED)
        assert len(entries) == 2

    def test_export_json(self, temp_dir):
        from platformx.audit import AuditLogger, AuditEventType

        logger = AuditLogger()
        logger.log(AuditEventType.QUERY_RECEIVED, "query", {"text": "test"})

        output_path = os.path.join(temp_dir, "audit.json")
        logger.export_json(output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_export_csv(self, temp_dir):
        from platformx.audit import AuditLogger, AuditEventType

        logger = AuditLogger()
        logger.log(AuditEventType.QUERY_RECEIVED, "query", {"text": "test"})

        output_path = os.path.join(temp_dir, "audit.csv")
        logger.export_csv(output_path)

        assert os.path.exists(output_path)

    def test_get_summary(self):
        from platformx.audit import AuditLogger, AuditEventType

        logger = AuditLogger()
        logger.log(AuditEventType.QUERY_RECEIVED, "query", {})
        logger.log(AuditEventType.RETRIEVAL_EXECUTED, "retrieval", {})

        summary = logger.get_summary()
        assert summary["total_entries"] == 2
        assert "entries_by_type" in summary

    def test_flush_with_exporter(self, temp_dir):
        from platformx.audit import AuditLogger, AuditEventType, JSONExporter

        output_path = os.path.join(temp_dir, "flushed.json")
        logger = AuditLogger()
        logger.add_exporter(JSONExporter(output_path))

        logger.log(AuditEventType.QUERY_RECEIVED, "query", {})
        count = logger.flush()

        assert count == 1
        assert logger.count() == 0
        assert os.path.exists(output_path)
