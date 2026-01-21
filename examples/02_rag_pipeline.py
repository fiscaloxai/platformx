"""Example 02: Complete RAG Pipeline

This example demonstrates a full RAG pipeline with retrieval, safety checks,
and confidence assessment.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from platformx.retrieval import Indexer, RetrievalEngine, create_embedding_backend, GroundedQuery, QueryIntent
from platformx.safety import SafetyFilterChain, PIIFilter, create_default_filter_chain, ConfidenceAssessor
from platformx.audit import AuditLogger, AuditEventType, AuditContext

def main():
    # Setup audit logging
    audit = AuditLogger(component="rag_example")
    ctx = AuditContext(session_id="example-session")

    # Create sample knowledge base
    knowledge_base = [
        {"id": "kb-001", "text": "Python is a high-level programming language known for its simplicity and readability."},
        {"id": "kb-002", "text": "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming."},
        {"id": "kb-003", "text": "Popular Python frameworks include Django for web development and PyTorch for machine learning."},
        {"id": "kb-004", "text": "Python's package manager pip allows easy installation of third-party libraries from PyPI."},
        {"id": "kb-005", "text": "Python was created by Guido van Rossum and first released in 1991."},
    ]

    # Create and populate indexer
    print("Setting up indexer...")
    embedding = create_embedding_backend("tfidf")
    indexer = Indexer(embedding_backend=embedding, chunk_size_words=100)

    # Index knowledge base directly
    from platformx.data import DatasetSchema, Provenance, SourceType, IntendedUse, Domain
    from datetime import datetime, timezone

    for item in knowledge_base:
        prov = Provenance(source_uri=f"memory://{item['id']}", ingested_by="example", ingested_at=datetime.now(timezone.utc))
        dataset = DatasetSchema(
            dataset_id=item["id"],
            domain=Domain.GENERAL,
            source_type=SourceType.TEXT,
            intended_use=IntendedUse.RETRIEVAL,
            version="1.0.0",
            provenance=prov,
            raw_text=item["text"],
        )
        indexer.index_dataset(dataset)

    print(f"Indexed {len(knowledge_base)} documents\n")

    # Create retrieval engine
    engine = RetrievalEngine(indexer)

    # Setup safety chain
    safety_chain = create_default_filter_chain(domain="general")
    safety_chain.add_filter(PIIFilter(), priority=10)

    # Setup confidence assessor
    confidence_assessor = ConfidenceAssessor()

    # Example queries
    queries = [
        "What programming paradigms does Python support?",
        "Who created Python?",
        "What is Django used for?",
    ]

    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)

    for query_text in queries:
        print(f"\nQuery: {query_text}")
        print("-" * 40)

        # Log query
        audit.log(AuditEventType.QUERY_RECEIVED, "query", {"text": query_text})

        # Safety check
        safety_result = safety_chain.check(query_text)
        audit.log(AuditEventType.SAFETY_CHECK, "safety", {"decision": str(safety_result["decision"])})

        if safety_result["decision"].value == "block":
            print(f"  BLOCKED: {safety_result['reason']}")
            continue

        # Retrieve
        query = GroundedQuery(text=query_text, intent=QueryIntent.informational)
        results = engine.retrieve(query, max_results=3)

        # Log retrieval
        audit.log(AuditEventType.RETRIEVAL_EXECUTED, "retrieval", {"result_count": len(results)})

        # Assess confidence
        evidence = [{"text": r.text, "score": r.score, "dataset_id": r.source["dataset_id"]} for r in results]
        confidence = confidence_assessor.assess(evidence, query=query_text)

        print(f"  Confidence: {confidence.level.value} (score: {confidence.score:.2f})")
        print(f"  Sources: {len(results)}")

        for i, result in enumerate(results, 1):
            print(f"    {i}. [{result.score:.2f}] {result.text[:60]}...")

    # Print audit summary
    print("\n" + "=" * 60)
    print("Audit Summary")
    print("=" * 60)
    summary = audit.get_summary()
    print(f"Total events: {summary['total_entries']}")
    for event_type, count in summary.get("entries_by_type", {}).items():
        print(f"  - {event_type}: {count}")

if __name__ == "__main__":
    main()
