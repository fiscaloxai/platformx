# Configuration & Environment

Environment setup

Dependencies

PYTHONPATH

Custom adapters/backends

Notes on compatibility

# Configuration

PlatformX uses a centralized configuration system for consistent behavior.

## PlatformConfig

The main configuration class:
```python
from platformx import PlatformConfig

config = PlatformConfig(
	project_name="my_project",      # Required: Project identifier
	data_dir="./data",              # Required: Base data directory
	logging_level="INFO",           # DEBUG, INFO, WARNING, ERROR
	reproducible=True,              # Enable deterministic mode
	seed=42,                        # Random seed for reproducibility
)
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `project_name` | str | Required | Unique project identifier |
| `data_dir` | str | Required | Base directory for datasets |
| `logging_level` | str | "INFO" | Logging verbosity |
| `reproducible` | bool | True | Enable deterministic operations |
| `seed` | int | None | Global random seed |

## Module-Specific Configuration

### Retrieval Configuration
```python
from platformx.retrieval import Indexer, create_embedding_backend

# Embedding backend
embedding = create_embedding_backend(
	"sentence-transformer",
	model_name="all-MiniLM-L6-v2",
)

# Indexer with chunking options
indexer = Indexer(
	embedding_backend=embedding,
	chunk_size_words=200,
	chunk_overlap_words=50,
)
```

### RAFT Configuration
```python
from platformx.training import RAFTConfig

config = RAFTConfig(
	positive_fraction=0.6,      # 60% positive samples
	reasoning_fraction=0.3,     # 30% include reasoning
	distractor_fraction=0.2,    # 20% negative with distractors
	max_distractors=3,
	seed=42,
)
```

### Training Configuration
```python
from platformx.model import TrainingConfig, LoRAConfig

training = TrainingConfig(
	output_dir="./output",
	num_epochs=3,
	per_device_batch_size=4,
	learning_rate=2e-4,
	seed=42,
)

lora = LoRAConfig(
	r=16,
	lora_alpha=32,
	lora_dropout=0.05,
	target_modules=["q_proj", "v_proj"],
)
```

### Safety Configuration
```python
from platformx.safety import ConfidenceConfig, ConfidenceStrategy

config = ConfidenceConfig(
	strategy=ConfidenceStrategy.WEIGHTED_ENSEMBLE,
	high_threshold=0.75,
	medium_threshold=0.4,
	min_sources=2,
)
```

## Environment Variables

PlatformX respects these environment variables:

| Variable | Description |
|----------|-------------|
| `PLATFORMX_LOG_LEVEL` | Override logging level |
| `OPENAI_API_KEY` | OpenAI API key for inference |
| `ANTHROPIC_API_KEY` | Anthropic API key for inference |
