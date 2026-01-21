"""Tests for the safety module."""

import pytest


class TestSafetyFilters:
    """Tests for safety filters."""

    def test_evaluate_safety_allow(self):
        from platformx.safety import evaluate_safety, Decision

        result = evaluate_safety("informational", [{"text": "evidence"}], "What is Python?")
        assert result["decision"] == Decision.ALLOW

    def test_evaluate_safety_block_no_evidence(self):
        from platformx.safety import evaluate_safety, Decision, ReasonCode

        result = evaluate_safety("informational", [], "What is Python?")
        assert result["decision"] == Decision.BLOCK
        assert result["reason"] == ReasonCode.NO_EVIDENCE

    def test_keyword_filter(self):
        from platformx.safety import KeywordFilter, FilterRule, Decision, ReasonCode

        rule = FilterRule(
            rule_id="test-keywords",
            name="Test Filter",
            description="Test keyword filter",
            rule_type="keyword",
            patterns=["hack", "exploit"],
            action=Decision.BLOCK,
            reason_code=ReasonCode.CUSTOM_RULE,
        )
        filter = KeywordFilter(rule)

        # Should block
        result = filter.check("How to hack a website", {})
        assert result is not None
        assert result["decision"] == Decision.BLOCK

        # Should pass
        result = filter.check("How to learn Python", {})
        assert result is None

    def test_pii_filter(self):
        from platformx.safety import PIIFilter

        filter = PIIFilter()

        # Email detection
        result = filter.check("My email is test@example.com", {})
        assert result is not None

        # Phone detection
        result = filter.check("Call me at 555-123-4567", {})
        assert result is not None

        # No PII
        result = filter.check("Hello world", {})
        assert result is None

    def test_safety_filter_chain(self):
        from platformx.safety import SafetyFilterChain, PIIFilter, Decision

        chain = SafetyFilterChain()
        chain.add_filter(PIIFilter(), priority=10)

        # Should block PII
        result = chain.check("My SSN is 123-45-6789")
        assert result["decision"] == Decision.BLOCK

        # Should allow clean text
        result = chain.check("What is machine learning?")
        assert result["decision"] == Decision.ALLOW

    def test_create_default_filter_chain(self):
        from platformx.safety import create_default_filter_chain

        chain = create_default_filter_chain(domain="general")
        assert chain is not None

        result = chain.check("Normal question about Python")
        assert result["decision"].value == "allow"


class TestConfidenceAssessment:
    """Tests for confidence assessment."""

    def test_assess_confidence_basic(self, sample_evidence):
        from platformx.safety import assess_confidence

        result = assess_confidence(sample_evidence)
        assert "level" in result
        assert "rationale" in result

    def test_confidence_assessor(self, sample_evidence):
        from platformx.safety import ConfidenceAssessor, ConfidenceLevel

        assessor = ConfidenceAssessor()
        result = assessor.assess(sample_evidence, query="What is Python?")

        assert isinstance(result.level, ConfidenceLevel)
        assert 0 <= result.score <= 1
        assert result.evidence_count == len(sample_evidence)

    def test_confidence_empty_evidence(self):
        from platformx.safety import ConfidenceAssessor, ConfidenceLevel

        assessor = ConfidenceAssessor()
        result = assessor.assess([], query="What is Python?")

        assert result.level == ConfidenceLevel.LOW
        assert len(result.warnings) > 0

    def test_confidence_with_threshold(self, sample_evidence):
        from platformx.safety import ConfidenceAssessor, ConfidenceLevel

        assessor = ConfidenceAssessor()
        score, passed = assessor.assess_with_threshold(
            sample_evidence,
            min_level=ConfidenceLevel.LOW,
            query="What is Python?"
        )

        assert isinstance(passed, bool)


class TestRefusalEngine:
    """Tests for refusal engine."""

    def test_refusal_engine(self):
        from platformx.safety import RefusalEngine

        engine = RefusalEngine()
        refusal = engine.make_refusal("no_evidence")

        assert refusal.message is not None
        assert "insufficient" in refusal.message.lower() or "cannot" in refusal.message.lower()
        assert refusal.reason == "no_evidence"
