"""
Model module for PlatformX.

This module provides model management, fine-tuning, and inference capabilities
for building production-quality LLM applications with full traceability.

Core Components:
- FineTuner: LoRA/PEFT-based fine-tuning with audit logging
- InferencePipeline: Controlled inference with pre/post processing
- Adapter: Fine-tuned adapter management with compatibility checks
- BaseModelBackend: Model loading with fingerprinting

Fine-Tuning:
- TrainingConfig: Training hyperparameters
- LoRAConfig: LoRA adapter configuration
- TrainingCallback: Hook into training events

Inference:
- LocalHFBackend: Local HuggingFace model inference
- APIBackend: API-based inference (OpenAI, Anthropic, etc.)
- GenerationConfig: Text generation parameters
- InferenceResult: Traced inference output

Example:
    from platformx.model import FineTuner, TrainingConfig, LoRAConfig
    from platformx.data import DataLoader

    # Configure fine-tuning
    training_cfg = TrainingConfig(num_epochs=3, learning_rate=2e-4)
    lora_cfg = LoRAConfig(r=16, lora_alpha=32)

    # Fine-tune
    finetuner = FineTuner(training_config=training_cfg, lora_config=lora_cfg)
    report = finetuner.run("meta-llama/Llama-2-7b-hf", datasets)

    # Inference with adapter
    from platformx.model import create_inference_pipeline
    pipeline = create_inference_pipeline(
        backend_type="local",
        model_path="meta-llama/Llama-2-7b-hf",
        adapter_path="./output/adapter"
    )
    result = pipeline.run("What is the capital of France?")
"""

import logging
logger = logging.getLogger("platformx.model")

# --- Fine-tuning ---
try:
    from .finetune import (
        FineTuner,
        FinetuneReport,
        TrainingConfig,
        LoRAConfig,
        TrainingCallback,
        AuditTrainingCallback,
    )
except ImportError as e:
    logger.warning(f"Could not import fine-tuning components: {e}")

# --- Inference ---
try:
    from .inference import (
        InferenceResult,
        GenerationConfig,
        InferenceBackend,
        LocalHFBackend,
        APIBackend,
        InferencePipeline,
        create_inference_pipeline,
    )
except ImportError as e:
    logger.warning(f"Could not import inference components: {e}")

# --- Adapters ---
try:
    from .adapters import (
        Adapter,
        AdapterArtifact,
        AdapterRegistry,
    )
except ImportError as e:
    logger.warning(f"Could not import adapter components: {e}")

# --- Backend ---
try:
    from .backend import (
        BaseModelBackend,
        ModelMetadata,
    )
except ImportError as e:
    logger.warning(f"Could not import backend components: {e}")

__all__ = [
    # Fine-tuning
    "FineTuner",
    "FinetuneReport",
    "TrainingConfig",
    "LoRAConfig",
    "TrainingCallback",
    "AuditTrainingCallback",
    # Inference
    "InferenceResult",
    "GenerationConfig",
    "InferenceBackend",
    "LocalHFBackend",
    "APIBackend",
    "InferencePipeline",
    "create_inference_pipeline",
    # Adapters
    "Adapter",
    "AdapterArtifact",
    "AdapterRegistry",
    # Backend
    "BaseModelBackend",
    "ModelMetadata",
]"""Model backend and fine-tuning interfaces for PlatformX.

Expose a limited, safe public API for model backends and adapters.
Do not expose raw inference or internal weight handling here.
"""

from .backend import BaseModelBackend
from .adapters import Adapter, AdapterArtifact, AdapterRegistry
from .finetune import FineTuner, FinetuneReport

__all__ = ["BaseModelBackend", "Adapter", "AdapterArtifact", "AdapterRegistry", "FineTuner", "FinetuneReport"]
