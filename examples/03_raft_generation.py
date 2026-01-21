"""Example 03: RAFT Sample Generation

This example demonstrates how to generate RAFT (Retrieval-Augmented Fine-Tuning)
training samples for teaching models to use retrieved context effectively.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from platformx.retrieval import Indexer, RetrievalEngine, create_embedding_backend
from platformx.training import RAFTOrchestrator, RAFTConfig, RAFTDatasetBuilder, SampleType
from platformx.data import DatasetSchema, Provenance, SourceType, IntendedUse, Domain
from datetime import datetime, timezone

def main():
    print("RAFT Sample Generation Example")
    print("=" * 50)

    # Create sample documents
    documents = [
        {
            "id": "doc-001",
            "text": """
            Transfer learning is a machine learning technique where a model trained on one task
            is reused as the starting point for a model on a different task. This approach is
            particularly useful when the target task has limited training data. Pre-trained
            models like BERT and GPT have revolutionized NLP by enabling effective transfer learning.
            """
        },
        {
            "id": "doc-002",
            "text": """
            Fine-tuning is the process of taking a pre-trained model and further training it on
            a specific dataset for a particular task. During fine-tuning, the model's weights
            are adjusted to better fit the new task while retaining the knowledge learned during
            pre-training. This typically requires much less data than training from scratch.
            """
        },
        {
            "id": "doc-003",
            "text": """
            RAFT (Retrieval-Augmented Fine-Tuning) is a technique that trains models to effectively
            use retrieved context when generating responses. Unlike standard fine-tuning, RAFT
            explicitly teaches models when to use retrieved information and when to refuse
            answering if the context doesn't support a response.
            """
        },
    ]

    # Setup indexer
    print("\nIndexing documents...")
    embedding = create_embedding_backend("tfidf")
    indexer = Indexer(embedding_backend=embedding, chunk_size_words=100)

    dataset_ids = []
    for doc in documents:
        prov = Provenance(source_uri=f"example://{doc['id']}", ingested_by="example", ingested_at=datetime.now(timezone.utc))
        dataset = DatasetSchema(
            dataset_id=doc["id"],
            domain=Domain.GENERAL,
            source_type=SourceType.TEXT,
            intended_use=IntendedUse.RETRIEVAL,
            version="1.0.0",
            provenance=prov,
            raw_text=doc["text"].strip(),
        )
        indexer.index_dataset(dataset)
        dataset_ids.append(doc["id"])
        print(f"  Indexed: {doc['id']}")

    # Create retrieval engine
    engine = RetrievalEngine(indexer)

    # Configure RAFT generation
    config = RAFTConfig(
        positive_fraction=0.6,
        reasoning_fraction=0.3,
        distractor_fraction=0.2,
        seed=42,
    )

    print(f"\nRAFT Configuration:")
    print(f"  - Positive fraction: {config.positive_fraction}")
    print(f"  - Reasoning fraction: {config.reasoning_fraction}")
    print(f"  - Distractor fraction: {config.distractor_fraction}")

    # Generate samples
    print("\nGenerating RAFT samples...")
    orchestrator = RAFTOrchestrator(engine, config=config)

    try:
        samples = orchestrator.generate_for_datasets(dataset_ids, max_per_doc=3)
    except RuntimeError as e:
        print(f"Note: {e}")
        print("Using simplified sample generation for demo...")
        samples = []

    # If no samples generated, create demo samples manually
    if not samples:
        from platformx.training import RAFTSample
        samples = [
            RAFTSample(
                instruction="Based on the provided evidence, extract the relevant factual information.",
                context="Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a different task.",
                expected="Transfer learning reuses a model trained on one task as the starting point for a different task.",
                source_doc_id="doc-001",
                retrieval_score=0.85,
                timestamp=datetime.now(timezone.utc),
                sample_type=SampleType.POSITIVE_EXTRACT,
            ),
            RAFTSample(
                instruction="Using the provided evidence, reason step-by-step to answer the question. Show your reasoning, then provide the final answer.",
                context="Fine-tuning is the process of taking a pre-trained model and further training it on a specific dataset.",
                expected="Reasoning: The context explains that fine-tuning involves additional training on specific data.\n\nAnswer: Fine-tuning adapts pre-trained models to specific tasks through additional training.",
                source_doc_id="doc-002",
                retrieval_score=0.82,
                timestamp=datetime.now(timezone.utc),
                sample_type=SampleType.POSITIVE_REASONING,
            ),
            RAFTSample(
                instruction="If the provided context does not contain sufficient information to answer, respond with REFUSE and explain why.",
                context="RAFT is a technique that trains models to use retrieved context.",
                expected="REFUSE: Insufficient evidence to answer this question.",
                source_doc_id="doc-003",
                retrieval_score=0.45,
                timestamp=datetime.now(timezone.utc),
                sample_type=SampleType.NEGATIVE_REFUSE,
            ),
        ]

    # Display generated samples
    print(f"\nGenerated {len(samples)} samples:")
    print("-" * 50)

    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Type: {sample.sample_type.value}")
        print(f"  Source: {sample.source_doc_id}")
        print(f"  Score: {sample.retrieval_score:.2f}")
        print(f"  Instruction: {sample.instruction[:60]}...")
        print(f"  Expected: {sample.expected[:60]}...")

    # Validate sample distribution
    print("\n" + "-" * 50)
    print("Sample Distribution:")
    type_counts = {}
    for sample in samples:
        t = sample.sample_type.value
        type_counts[t] = type_counts.get(t, 0) + 1
    for sample_type, count in type_counts.items():
        print(f"  - {sample_type}: {count}")

    # Build fine-tuning dataset
    print("\n" + "-" * 50)
    print("Building fine-tuning dataset...")
    builder = RAFTDatasetBuilder()
    ft_records = builder.build(samples, dataset_id_prefix="raft_example", domain="general")
    print(f"Created {len(ft_records)} fine-tuning records")

    # Save samples to file
    output_path = "./raft_samples.json"
    samples_data = [
        {
            "instruction": s.instruction,
            "context": s.context,
            "expected": s.expected,
            "sample_type": s.sample_type.value,
            "source": s.source_doc_id,
        }
        for s in samples
    ]
    with open(output_path, "w") as f:
        json.dump(samples_data, f, indent=2)
    print(f"Saved samples to: {output_path}")

if __name__ == "__main__":
    main()
