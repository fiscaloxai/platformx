"""
High-level API for PlatformX.

This module provides simple one-liner functions for common tasks.
For advanced usage, import from specific submodules.

Quick Examples:
    import platformx.api as pfx

    # Load and index documents
    pfx.index_documents("./docs/", dataset_id="my-docs")

    # RAG query
    result = pfx.rag_query("What is machine learning?", index_path="./index/")

    # Generate RAFT training samples
    samples = pfx.generate_raft_samples(["dataset-001"], retrieval_index="./index/")

    # Fine-tune a model
    pfx.finetune("meta-llama/Llama-2-7b-hf", dataset_path="./training_data/")

    # Run inference
    response = pfx.generate("Explain quantum computing", model="gpt-4")
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger("platformx.api")

__all__ = [
    "index_documents",
    "rag_query",
    "generate_raft_samples",
    "finetune",
    "generate",
    "evaluate_retrieval",
    "quick_setup",
]

def index_documents(
    source: Union[str, List[str]],
    dataset_id: str,
    index_path: Optional[str] = None,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    embedding_backend: str = "tfidf",
    domain: str = "general",
    **kwargs
) -> Dict[str, Any]:
    """
    Load and index documents for retrieval.
    ...existing docstring...
    """
    try:
        from platformx.data import DataLoader
        from platformx.retrieval import Indexer, create_embedding_backend
    except ImportError as e:
        logger.error(f"Required modules missing: {e}")
        raise
    loader = DataLoader()
    import os
    if isinstance(source, str) and os.path.isdir(source):
        # Directory: load all files with required metadata
        base_metadata = {
            "dataset_id_prefix": dataset_id + "_",
            "domain": domain,
            "intended_use": "RETRIEVAL",
        }
        docs = loader.load_directory(source, base_metadata)
    else:
        # Single file: require all metadata
        metadata = {
            "dataset_id": dataset_id,
            "domain": domain,
            "intended_use": "RETRIEVAL",
        }
        docs = loader.load(source, metadata)
    embedding = create_embedding_backend(embedding_backend)
    indexer = Indexer(embedding_backend=embedding, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # If docs is a list, index all; if single, wrap in list
    if isinstance(docs, list):
        chunk_count = 0
        for ds in docs:
            chunk_count += len(indexer.index_dataset(ds))
    else:
        chunk_count = len(indexer.index_dataset(docs))
    if index_path:
        Path(index_path).mkdir(parents=True, exist_ok=True)
        indexer.save(index_path)
    metadata = {"domain": domain, "embedding_backend": embedding_backend}
    return {
        "dataset_id": dataset_id,
        "chunk_count": chunk_count,
        "index_path": index_path,
        "metadata": metadata,
    }

def rag_query(
    query: str,
    index_path: Optional[str] = None,
    indexer: Optional[Any] = None,
    top_k: int = 5,
    min_confidence: str = "low",
    safety_check: bool = True,
    return_sources: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a RAG query against indexed documents.
    ...existing docstring...
    """
    try:
        from platformx.retrieval import Indexer
        from platformx.safety import evaluate_safety, assess_confidence
    except ImportError as e:
        logger.error(f"Required modules missing: {e}")
        raise
    if not indexer and index_path:
        indexer = Indexer.load(index_path)
    if not indexer:
        raise ValueError("Indexer or index_path required")
    safety_result = None
    if safety_check:
        safety_result = evaluate_safety(query, [], query)
        if safety_result["decision"] != "allow":
            return {"query": query, "results": [], "confidence": None, "sources": [], "safety_result": safety_result}
    results = indexer.retrieve(query, top_k=top_k, **kwargs)
    confidence = assess_confidence(results)
    sources = [r.get("text") for r in results] if return_sources else []
    return {
        "query": query,
        "results": results,
        "confidence": confidence,
        "sources": sources,
        "safety_result": safety_result,
    }

def generate_raft_samples(
    dataset_ids: List[str],
    index_path: Optional[str] = None,
    indexer: Optional[Any] = None,
    output_path: Optional[str] = None,
    samples_per_dataset: int = 10,
    positive_fraction: float = 0.6,
    include_reasoning: bool = True,
    include_distractors: bool = True,
    seed: int = 42,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Generate RAFT training samples from indexed datasets.
    ...existing docstring...
    """
    try:
        from platformx.training import RAFTConfig, RAFTOrchestrator
        from platformx.retrieval import Indexer
    except ImportError as e:
        logger.error(f"Required modules missing: {e}")
        raise
    if not indexer and index_path:
        indexer = Indexer.load(index_path)
    if not indexer:
        raise ValueError("Indexer or index_path required")
    config = RAFTConfig(
        positive_fraction=positive_fraction,
        reasoning_fraction=0.3 if include_reasoning else 0.0,
        distractor_fraction=0.2 if include_distractors else 0.0,
        seed=seed,
        **kwargs
    )
    orchestrator = RAFTOrchestrator(indexer, config=config)
    samples = orchestrator.generate_for_datasets(dataset_ids, samples_per_dataset=samples_per_dataset)
    if output_path:
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    return samples

def finetune(
    base_model: str,
    dataset_path: Optional[str] = None,
    datasets: Optional[List[Any]] = None,
    output_dir: str = "./output",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    seed: int = 42,
    dry_run: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Fine-tune a model using LoRA/PEFT.
    ...existing docstring...
    """
    try:
        from platformx.model import FineTuner, TrainingConfig, LoRAConfig
        from platformx.data import DataLoader
    except ImportError as e:
        logger.error(f"Required modules missing: {e}")
        raise
    if datasets is None and dataset_path:
        loader = DataLoader()
        datasets = loader.load(dataset_path)
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        dry_run=dry_run,
        **kwargs
    )
    lora_config = LoRAConfig(r=lora_r, lora_alpha=lora_alpha)
    finetuner = FineTuner(training_config=training_config, lora_config=lora_config)
    report = finetuner.run(base_model, datasets, output_dir=output_dir)
    return report if isinstance(report, dict) else report.__dict__

def generate(
    prompt: str,
    model: Optional[str] = None,
    adapter_path: Optional[str] = None,
    backend: str = "local",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    context: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    safety_check: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate text using a model with optional RAG context.
    ...existing docstring...
    """
    try:
        from platformx.model import create_inference_pipeline, GenerationConfig
        from platformx.safety import create_default_filter_chain
    except ImportError as e:
        logger.error(f"Required modules missing: {e}")
        raise
    if safety_check:
        chain = create_default_filter_chain()
        safety_result = chain.check(prompt)
        if safety_result["decision"] != "allow":
            return {"response": None, "model_id": model, "latency_ms": None, "tokens": None, "safety_result": safety_result}
    pipeline = create_inference_pipeline(
        backend_type=backend,
        model_path=model,
        adapter_path=adapter_path,
        provider=provider,
        api_key=api_key,
        **kwargs
    )
    config = GenerationConfig(max_new_tokens=max_tokens, temperature=temperature)
    if context:
        result = pipeline.run_with_context(prompt, context, config)
    else:
        result = pipeline.run(prompt, config)
    return result.to_dict() if hasattr(result, "to_dict") else result

def evaluate_retrieval(
    queries: List[str],
    expected_docs: List[List[str]],
    index_path: Optional[str] = None,
    indexer: Optional[Any] = None,
    top_k: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality with standard metrics.
    ...existing docstring...
    """
    try:
        from platformx.retrieval import Indexer
    except ImportError as e:
        logger.error(f"Required modules missing: {e}")
        raise
    if not indexer and index_path:
        indexer = Indexer.load(index_path)
    if not indexer:
        raise ValueError("Indexer or index_path required")
    per_query_results = []
    total_prec, total_rec, total_mrr, total_map = 0, 0, 0, 0
    for q, expected in zip(queries, expected_docs):
        results = indexer.retrieve(q, top_k=top_k, **kwargs)
        retrieved_ids = [r.get("doc_id") or r.get("dataset_id") for r in results]
        hits = [1 if doc in retrieved_ids else 0 for doc in expected]
        prec = sum(hits) / max(1, len(retrieved_ids))
        rec = sum(hits) / max(1, len(expected))
        ranks = [retrieved_ids.index(doc) + 1 for doc in expected if doc in retrieved_ids]
        mrr = 1.0 / min(ranks) if ranks else 0.0
        ap = sum([sum(hits[:i+1])/(i+1) for i in range(len(hits))]) / max(1, len(hits))
        per_query_results.append({"query": q, "precision": prec, "recall": rec, "mrr": mrr, "map": ap})
        total_prec += prec
        total_rec += rec
        total_mrr += mrr
        total_map += ap
    n = len(queries)
    return {
        "precision": total_prec / n if n else 0.0,
        "recall": total_rec / n if n else 0.0,
        "mrr": total_mrr / n if n else 0.0,
        "map": total_map / n if n else 0.0,
        "per_query_results": per_query_results,
    }

def quick_setup(
    project_name: str,
    data_dir: str = "./data",
    index_dir: str = "./index",
    output_dir: str = "./output",
    log_level: str = "INFO"
) -> Any:
    """
    Quick setup for a new PlatformX project.
    ...existing docstring...
    """
    try:
        from platformx import PlatformConfig, Platform
    except ImportError as e:
        logger.error(f"Required modules missing: {e}")
        raise
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    config = PlatformConfig(project_name=project_name, data_dir=data_dir)
    platform = Platform(config)
    return platform
