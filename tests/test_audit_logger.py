from platformx.audit.logger import AuditLogger

def test_audit_logger_basic():
    logger = AuditLogger()
    logger.log_query("test query", "informational")
    logger.log_refusal("refused", "no_evidence")
    entries = logger.get_entries()
    assert any(e["event"] == "query" for e in entries)
    assert any(e["event"] == "refusal" for e in entries)
