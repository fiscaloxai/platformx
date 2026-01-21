"""Example 01: Basic Document Indexing

This example demonstrates how to load and index documents for retrieval.
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from platformx import PlatformConfig, Platform
from platformx.data import DataLoader, IntendedUse
from platformx.retrieval import Indexer, create_embedding_backend

def main():
    # Create sample documents (in real usage, these would be files)
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with many layers to model complex patterns in data.",
        "Natural language processing allows computers to understand and generate human language.",
        "Transformers are a type of neural network architecture that excels at sequence-to-sequence tasks.",
        "Fine-tuning adapts a pre-trained model to a specific task using task-specific data.",
    ]

    # Save sample docs to temp files
    os.makedirs("./temp_docs", exist_ok=True)
    for i, doc in enumerate(sample_docs):
        with open(f"./temp_docs/doc_{i}.txt", "w") as f:
            f.write(doc)

    # Initialize platform
    config = PlatformConfig(project_name="indexing_example", data_dir="./temp_docs")
    platform = Platform(config)

    # Load and register documents
    loader = DataLoader()
    for i in range(len(sample_docs)):
        dataset = loader.load(f"./temp_docs/doc_{i}.txt", {
            "dataset_id": f"doc-{i:03d}",
            "domain": "general",
            "intended_use": "retrieval",
        })
        platform.registry.register(dataset)
        print(f"Registered: {dataset.dataset_id}")

    # Create indexer with TF-IDF embeddings (no dependencies required)
    embedding = create_embedding_backend("tfidf")
    indexer = Indexer(embedding_backend=embedding, chunk_size_words=50)

    # Index all retrieval datasets
    for dataset in platform.datasets_for_retrieval():
        chunk_ids = indexer.index_dataset(dataset)
        print(f"Indexed {dataset.dataset_id}: {len(chunk_ids)} chunks")

    # Test query
    results = indexer.query("What is deep learning?", top_k=3)
    print("\nQuery: 'What is deep learning?'")
    print("Results:")
    for r in results:
        print(f"  - Score: {r['score']:.3f} | {r['text'][:80]}...")

    # Cleanup
    import shutil
    shutil.rmtree("./temp_docs")

    print("\nDone!")

if __name__ == "__main__":
    main()
