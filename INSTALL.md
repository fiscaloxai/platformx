# PlatformX Installation Guide

## Requirements

- **Python**: 3.10 or later
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB+ recommended for fine-tuning)
- **GPU**: Optional, but recommended for fine-tuning operations

## Basic Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/your-org/platformx.git
cd platformx

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with core dependencies
pip install -e .
```

### From PyPI (When Published)

```bash
pip install platformx
```

## Feature-Specific Installation

PlatformX uses optional dependencies for different features. Install only what you need:

### For Retrieval (RAG)

```bash
# Embeddings and vector search
pip install platformx[retrieval]
```

Includes:
- `sentence-transformers`: Neural embedding models
- `chromadb`: Vector database for semantic search

### For Fine-Tuning

```bash
# Model training with LoRA/PEFT
pip install platformx[training]
```

Includes:
- `transformers`: HuggingFace transformers library
- `torch`: PyTorch deep learning framework
- `datasets`: Dataset management
- `peft`: Parameter-efficient fine-tuning
- `accelerate`: Distributed training
- `bitsandbytes`: Quantization support

### For Inference

```bash
# Model inference capabilities
pip install platformx[inference]
```

Includes:
- `transformers`: Model inference
- `torch`: PyTorch runtime

### For API Providers

#### OpenAI

```bash
pip install platformx[openai]
```

#### Anthropic (Claude)

```bash
pip install platformx[anthropic]
```

### For Document Processing

```bash
# PDF, HTML, XML, Parquet support
pip install platformx[documents]
```

Includes:
- `pypdf`: PDF text extraction
- `python-docx`: Word document support
- `beautifulsoup4`: HTML parsing
- `lxml`: XML processing
- `pyarrow`: Parquet file support

### For Development

```bash
# Testing, linting, formatting
pip install platformx[dev]
```

Includes:
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `mypy`: Type checking
- `ruff`: Fast linting
- `black`: Code formatting
- `pre-commit`: Git hooks

## Complete Installation (All Features)

```bash
# Install everything
pip install platformx[retrieval,training,inference,openai,anthropic,documents,dev]
```

## Verify Installation

```python
import platformx as pfx

# Check version
print(f"PlatformX version: {pfx.__version__}")

# Check installed modules
info = pfx.info()
print("\nInstalled modules:")
for module, available in info["modules"].items():
    status = "✓" if available else "✗"
    print(f"  {status} {module}")
```

## Quick Start

```python
import platformx.api as pfx

# Set up a new project
platform = pfx.quick_setup(
    project_name="my_pharma_project",
    data_dir="./data",
    index_dir="./index"
)

# Index documents for RAG
result = pfx.index_documents(
    source="./my_documents/",
    dataset_id="clinical-trials",
    index_path="./index/"
)

# Run a RAG query
response = pfx.rag_query(
    query="What are the side effects?",
    index_path="./index/",
    top_k=5
)

print(f"Found {len(response['results'])} relevant documents")
```

## GPU Support

For GPU-accelerated training and inference:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# MPS (Apple Silicon)
pip install torch  # MPS support included by default
```

## Troubleshooting

### ImportError: No module named 'transformers'

**Solution**: Install training or inference dependencies:
```bash
pip install platformx[training]
```

### Memory Error during fine-tuning

**Solution**: Enable gradient checkpointing and use smaller batch sizes:
```python
from platformx import TrainingConfig

config = TrainingConfig(
    per_device_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True  # Use mixed precision
)
```

### ChromaDB installation fails

**Solution**: Ensure you have build tools installed:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install

# Windows: Install Visual Studio Build Tools
```

## Environment Variables

```bash
# Optional: Set cache directories
export HF_HOME=/path/to/huggingface/cache
export TRANSFORMERS_CACHE=/path/to/transformers/cache

# Optional: Offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Docker Installation

```bash
# Build Docker image
docker build -t platformx:latest .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  platformx:latest
```

## Next Steps

- Read the [Getting Started Guide](docs/getting_started.md)
- Explore [Examples](examples/)
- Check the [API Reference](docs/api.md)
- Review [Configuration Options](docs/configuration.md)

## Support

- **Issues**: https://github.com/your-org/platformx/issues
- **Documentation**: https://platformx.readthedocs.io
- **Discussions**: https://github.com/your-org/platformx/discussions
