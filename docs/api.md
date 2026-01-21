
# API Reference

The API reference below is generated automatically from the `platformx` package using `mkdocstrings`.

::: platformx

# API Reference

## High-Level API

The `platformx.api` module provides simple functions for common tasks.

### index_documents
```python
def index_documents(
	source: Union[str, List[str]],
	dataset_id: str,
	index_path: Optional[str] = None,
	chunk_size: int = 200,
	chunk_overlap: int = 50,
	embedding_backend: str = "tfidf",
	domain: str = "general",
) -> Dict[str, Any]
```

Index documents for retrieval.

### rag_query
```python
def rag_query(
	query: str,
	index_path: Optional[str] = None,
	top_k: int = 5,
	min_confidence: str = "low",
	safety_check: bool = True,
) -> Dict[str, Any]
```

Execute a RAG query with safety and confidence.

### generate_raft_samples
```python
def generate_raft_samples(
	dataset_ids: List[str],
	index_path: Optional[str] = None,
	samples_per_dataset: int = 10,
	positive_fraction: float = 0.6,
	seed: int = 42,
) -> List[Dict[str, Any]]
```

Generate RAFT training samples.

### finetune
```python
def finetune(
	base_model: str,
	dataset_path: Optional[str] = None,
	output_dir: str = "./output",
	num_epochs: int = 3,
	learning_rate: float = 2e-4,
	lora_r: int = 16,
	dry_run: bool = False,
) -> Dict[str, Any]
```

Fine-tune a model using LoRA.

### generate
```python
def generate(
	prompt: str,
	model: Optional[str] = None,
	backend: str = "local",
	provider: Optional[str] = None,
	max_tokens: int = 256,
	temperature: float = 0.7,
) -> Dict[str, Any]
```

Generate text with a model.

## Core Classes

See module documentation for detailed API reference:

- [Data Module](modules/data.md)
- [Retrieval Module](modules/core.md)
- [Training Module](modules/training_raft.md)
- [Model Module](modules/model_finetune.md)
- [Safety Module](modules/safety.md)
