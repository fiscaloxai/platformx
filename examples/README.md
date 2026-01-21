# PlatformX Examples

This directory contains example scripts demonstrating various PlatformX features.

## Examples

| File | Description |
|------|-------------|
| `01_basic_indexing.py` | Basic document loading and indexing |
| `02_rag_pipeline.py` | Complete RAG pipeline with safety and confidence |
| `03_raft_generation.py` | RAFT training sample generation |
| `04_safety_filtering.py` | Safety filters and confidence assessment |
| `05_quick_start.py` | Quick start using high-level API |

## Running Examples

### Prerequisites

Install PlatformX with development dependencies:
```bash
pip install -e ".[dev]"
```

For examples using neural embeddings:
```bash
pip install -e ".[retrieval]"
```

For fine-tuning examples:
```bash
pip install -e ".[training]"
```

### Run an Example
```bash
cd examples
python 01_basic_indexing.py
```

## Example Progression

1. **Start with `05_quick_start.py`** - Simplest introduction using high-level API
2. **Try `01_basic_indexing.py`** - Understand document indexing
3. **Explore `02_rag_pipeline.py`** - See a complete RAG workflow
4. **Learn `04_safety_filtering.py`** - Configure safety for production
5. **Advanced: `03_raft_generation.py`** - Generate training data for fine-tuning

## Notes

- Examples create temporary files/directories and clean up after themselves
- Some examples may require additional dependencies (see comments in each file)
- For GPU-based examples, ensure CUDA is available
