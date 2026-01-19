"""RAFT training package for PlatformX.

Public API exposes `RAFTOrchestrator` and `RAFTDatasetBuilder` for generating
retrieval-aware fine-tuning datasets. Internals are intentionally minimal and
quietly deterministic. RAFT does not add knowledge; it enforces grounding.
"""

from .raft import RAFTOrchestrator
from .datasets import RAFTDatasetBuilder

__all__ = ["RAFTOrchestrator", "RAFTDatasetBuilder"]
