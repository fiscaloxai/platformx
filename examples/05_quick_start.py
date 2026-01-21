"""Example 05: Quick Start with High-Level API

This example demonstrates the simplest way to use PlatformX
using the high-level API functions.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Use the high-level API
import platformx.api as pfx

def main():
    print("PlatformX Quick Start")
    print("=" * 50)

    # Create sample documents
    print("\n1. Creating sample documents...")
    os.makedirs("./quick_start_docs", exist_ok=True)

    docs = {
        "python_basics.txt": "Python is an interpreted, high-level programming language. It emphasizes code readability with significant indentation. Python supports multiple programming paradigms.",
        "ml_intro.txt": "Machine learning is a branch of artificial intelligence. It uses algorithms to learn from data and make predictions. Common techniques include supervised and unsupervised learning.",
        "deep_learning.txt": "Deep learning is a subset of machine learning using neural networks. It excels at processing unstructured data like images and text. Popular frameworks include TensorFlow and PyTorch.",
    }

    for filename, content in docs.items():
        with open(f"./quick_start_docs/{filename}", "w") as f:
            f.write(content)
        print(f"  Created: {filename}")

    # Quick setup
    print("\n2. Quick platform setup...")
    platform = pfx.quick_setup(
        project_name="quick_start_demo",
        data_dir="./quick_start_docs",
        index_dir="./quick_start_index",
    )
    print("  Platform initialized!")

    # Index documents
    print("\n3. Indexing documents...")
    result = pfx.index_documents(
        source="./quick_start_docs/",
        dataset_id="quickstart-docs",
        index_path="./quick_start_index/",
        chunk_size=100,
        embedding_backend="tfidf",
    )
    print(f"  Indexed {result.get('chunk_count', 'N/A')} chunks")

    # Run queries
    print("\n4. Running RAG queries...")
    print("-" * 50)

    queries = [
        "What is Python?",
        "How does machine learning work?",
        "What frameworks are used for deep learning?",
    ]

    for query in queries:
        print(f"\nQ: {query}")
        try:
            result = pfx.rag_query(
                query=query,
                index_path="./quick_start_index/",
                top_k=2,
                safety_check=True,
            )

            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Confidence: {result.get('confidence', {}).get('level', 'N/A')}")
                for i, r in enumerate(result.get("results", [])[:2], 1):
                    text = r.get("text", "")[:80]
                    score = r.get("score", 0)
                    print(f"  {i}. [{score:.2f}] {text}...")
        except Exception as e:
            print(f"  Error: {e}")

    # Cleanup
    print("\n5. Cleaning up...")
    import shutil
    shutil.rmtree("./quick_start_docs", ignore_errors=True)
    shutil.rmtree("./quick_start_index", ignore_errors=True)
    print("  Done!")

    print("\n" + "=" * 50)
    print("Quick start complete! Check out other examples for more features.")

if __name__ == "__main__":
    main()
