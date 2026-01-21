"""Example 04: Safety Filtering and Confidence Assessment

This example demonstrates how to configure and use safety filters
and confidence assessment for production applications.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from platformx.safety import (
    SafetyFilterChain,
    FilterRule,
    KeywordFilter,
    RegexFilter,
    PIIFilter,
    IntentFilter,
    Decision,
    ReasonCode,
    create_default_filter_chain,
    ConfidenceAssessor,
    ConfidenceConfig,
    ConfidenceStrategy,
    ConfidenceLevel,
)

def main():
    print("Safety Filtering Example")
    print("=" * 50)

    # Create custom safety chain
    print("\n1. Creating custom safety filter chain...")
    chain = SafetyFilterChain()

    # Add PII filter (high priority)
    chain.add_filter(PIIFilter(
        detect_emails=True,
        detect_phones=True,
        detect_ssn=True,
        detect_credit_cards=True,
    ), priority=10)

    # Add custom keyword filter
    harmful_keywords_rule = FilterRule(
        rule_id="harmful-content",
        name="Harmful Content Filter",
        description="Block requests for harmful content",
        rule_type="keyword",
        patterns=["how to hack", "make a bomb", "steal password"],
        action=Decision.BLOCK,
        reason_code=ReasonCode.CUSTOM_RULE,
        case_sensitive=False,
    )
    chain.add_filter(KeywordFilter(harmful_keywords_rule), priority=20)

    # Add regex filter for sensitive patterns
    sensitive_regex_rule = FilterRule(
        rule_id="sensitive-data",
        name="Sensitive Data Pattern",
        description="Block requests containing API keys or tokens",
        rule_type="regex",
        patterns=[
            r"sk-[a-zA-Z0-9]{32,}",  # OpenAI API key pattern
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub token pattern
        ],
        action=Decision.BLOCK,
        reason_code=ReasonCode.PII_DETECTED,
    )
    chain.add_filter(RegexFilter(sensitive_regex_rule), priority=15)

    print("  Added filters: PII, Harmful Keywords, Sensitive Data Patterns")

    # Test queries
    test_queries = [
        "What is machine learning?",  # Safe
        "My email is john@example.com, can you help?",  # PII
        "How to hack into a website?",  # Harmful
        "My API key is sk-abc123def456ghi789jkl012mno345pqr678",  # Sensitive
        "Call me at 555-123-4567",  # Phone number
        "My SSN is 123-45-6789",  # SSN
        "What's the weather like?",  # Safe
    ]

    print("\n2. Testing safety filters...")
    print("-" * 50)

    for query in test_queries:
        result = chain.check(query)
        decision = result["decision"].value
        status = "✓ ALLOWED" if decision == "allow" else "✗ BLOCKED"
        reason = result.get("reason", "").value if hasattr(result.get("reason", ""), "value") else str(result.get("reason", ""))

        print(f"\nQuery: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
        print(f"  {status}")
        if decision == "block":
            print(f"  Reason: {reason}")
            if "details" in result and result["details"]:
                print(f"  Details: {result['details']}")

    # Confidence Assessment
    print("\n" + "=" * 50)
    print("Confidence Assessment Example")
    print("=" * 50)

    # Create confidence assessor
    config = ConfidenceConfig(
        strategy=ConfidenceStrategy.WEIGHTED_ENSEMBLE,
        high_threshold=0.75,
        medium_threshold=0.4,
        min_sources=2,
    )
    assessor = ConfidenceAssessor(config=config)

    # Test evidence sets
    evidence_sets = [
        {
            "name": "High confidence (multiple agreeing sources)",
            "evidence": [
                {"text": "Python is a programming language created by Guido van Rossum.", "score": 0.92, "dataset_id": "wiki-001"},
                {"text": "Python was created by Guido van Rossum in 1991.", "score": 0.88, "dataset_id": "docs-002"},
                {"text": "Guido van Rossum is the creator of Python programming language.", "score": 0.85, "dataset_id": "books-003"},
            ],
        },
        {
            "name": "Medium confidence (some agreement)",
            "evidence": [
                {"text": "Python is used for web development.", "score": 0.75, "dataset_id": "wiki-001"},
                {"text": "Python supports multiple paradigms.", "score": 0.65, "dataset_id": "docs-002"},
            ],
        },
        {
            "name": "Low confidence (single weak source)",
            "evidence": [
                {"text": "Python might be useful for some tasks.", "score": 0.35, "dataset_id": "forum-001"},
            ],
        },
        {
            "name": "No evidence",
            "evidence": [],
        },
    ]

    print("\n3. Testing confidence assessment...")
    print("-" * 50)

    for evidence_set in evidence_sets:
        result = assessor.assess(evidence_set["evidence"], query="What is Python?")

        print(f"\n{evidence_set['name']}:")
        print(f"  Level: {result.level.value.upper()}")
        print(f"  Score: {result.score:.2f}")
        print(f"  Sources: {result.evidence_count}")

        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")

    # Using default filter chain
    print("\n" + "=" * 50)
    print("Using Default Filter Chain")
    print("=" * 50)

    default_chain = create_default_filter_chain(domain="general")
    result = default_chain.check("How do I write a for loop in Python?")
    print(f"\nQuery: 'How do I write a for loop in Python?'")
    print(f"Result: {result['decision'].value.upper()}")

if __name__ == "__main__":
    main()
