"""
Inference module for PlatformX: controlled, auditable inference pipelines for LLMs with adapter and validation support.

Supports local HuggingFace models (with LoRA/PEFT) and API-based backends (OpenAI, Anthropic, etc).

Example usage:
    pipeline = create_inference_pipeline(backend_type="local", model_path="./llama", adapter_path="./adapter")
    result = pipeline.run("What is aspirin?", config=GenerationConfig())
"""
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import logging
import hashlib

logger = logging.getLogger("platformx.model.inference")

__all__ = [
    "InferenceResult",
    "GenerationConfig",
    "InferenceBackend",
    "LocalHFBackend",
    "APIBackend",
    "InferencePipeline",
    "create_inference_pipeline",
]

@dataclass
class InferenceResult:
    """Result from a single inference call with full traceability."""
    response_text: str
    model_id: str
    adapter_id: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def __post_init__(self):
        if not self.request_id:
            base = self.response_text + str(self.timestamp)
            self.request_id = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None

class InferenceBackend(ABC):
    """Abstract interface for model inference backends."""
    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> InferenceResult:
        pass
    @abstractmethod
    def generate_batch(self, prompts: List[str], config: GenerationConfig) -> List[InferenceResult]:
        pass
    @abstractmethod
    def model_id(self) -> str:
        pass
    @abstractmethod
    def is_ready(self) -> bool:
        pass

class LocalHFBackend(InferenceBackend):
    """Inference backend using local HuggingFace models with optional LoRA adapters."""
    def __init__(self, model_path: str, adapter_path: Optional[str] = None, device: str = "auto", load_immediately: bool = False):
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._logger = logging.getLogger("platformx.model.inference")
        if load_immediately:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            if self._adapter_path:
                from peft import PeftModel
        except ImportError:
            raise ImportError("transformers and peft are required for local inference. Install with 'pip install transformers peft'.")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if self._adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
        self._loaded = True
        self._logger.info(f"Loaded model {self._model_path} with adapter {self._adapter_path} on device {self._device}")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_model()

    def generate(self, prompt: str, config: GenerationConfig) -> InferenceResult:
        self._ensure_loaded()
        import time
        import torch
        start = time.time()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device if self._device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        gen_args = dict(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        if config.seed is not None:
            torch.manual_seed(config.seed)
        output = self._model.generate(**inputs, **gen_args)
        response = self._tokenizer.decode(output[0], skip_special_tokens=True)
        latency = (time.time() - start) * 1000
        return InferenceResult(
            response_text=response,
            model_id=self.model_id(),
            adapter_id=self.adapter_id() if hasattr(self, "adapter_id") else None,
            prompt_tokens=len(inputs["input_ids"][0]),
            completion_tokens=len(output[0]) - len(inputs["input_ids"][0]),
            latency_ms=latency,
            metadata={},
        )

    def generate_batch(self, prompts: List[str], config: GenerationConfig) -> List[InferenceResult]:
        self._ensure_loaded()
        return [self.generate(p, config) for p in prompts]

    def model_id(self) -> str:
        return self._model_path

    def adapter_id(self) -> Optional[str]:
        return self._adapter_path

    def is_ready(self) -> bool:
        return self._loaded

class APIBackend(InferenceBackend):
    """Inference backend for API-based models (OpenAI, Anthropic, etc.)."""
    def __init__(self, provider: str, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._provider = provider
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("API_KEY")
        self._base_url = base_url
        self._client = None
        self._logger = logging.getLogger("platformx.model.inference")

    def _get_client(self) -> Any:
        if self._client:
            return self._client
        if self._provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("openai is required for OpenAI API inference. Install with 'pip install openai'.")
            openai.api_key = self._api_key
            self._client = openai
        elif self._provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic is required for Anthropic API inference. Install with 'pip install anthropic'.")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")
        return self._client

    def generate(self, prompt: str, config: GenerationConfig) -> InferenceResult:
        import time
        start = time.time()
        client = self._get_client()
        if self._provider == "openai":
            response = client.Completion.create(
                model=self._model_name,
                prompt=prompt,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                n=1,
                stop=config.stop_sequences or None,
            )
            text = response.choices[0].text.strip()
            prompt_tokens = response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0
            completion_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
        elif self._provider == "anthropic":
            response = client.completions.create(
                model=self._model_name,
                prompt=prompt,
                max_tokens_to_sample=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop_sequences=config.stop_sequences or None,
            )
            text = response.completion.strip()
            prompt_tokens = 0
            completion_tokens = 0
        else:
            raise ValueError(f"Unknown provider: {self._provider}")
        latency = (time.time() - start) * 1000
        return InferenceResult(
            response_text=text,
            model_id=self.model_id(),
            adapter_id=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency,
            metadata={},
        )

    def generate_batch(self, prompts: List[str], config: GenerationConfig) -> List[InferenceResult]:
        return [self.generate(p, config) for p in prompts]

    def model_id(self) -> str:
        return f"{self._provider}/{self._model_name}"

    def is_ready(self) -> bool:
        return True

class InferencePipeline:
    """High-level inference pipeline with validation and hooks."""
    def __init__(self, backend: InferenceBackend, default_config: Optional[GenerationConfig] = None):
        self._backend = backend
        self._pre_processors: List[Callable[[str], str]] = []
        self._post_processors: List[Callable[[str], str]] = []
        self._validators: List[Callable[[str], bool]] = []
        self._default_config = default_config or GenerationConfig()
        self._logger = logging.getLogger("platformx.model.inference")

    def add_pre_processor(self, fn: Callable[[str], str]) -> "InferencePipeline":
        self._pre_processors.append(fn)
        return self

    def add_post_processor(self, fn: Callable[[str], str]) -> "InferencePipeline":
        self._post_processors.append(fn)
        return self

    def add_validator(self, fn: Callable[[str], bool]) -> "InferencePipeline":
        self._validators.append(fn)
        return self

    def _apply_pre_processors(self, prompt: str) -> str:
        for fn in self._pre_processors:
            prompt = fn(prompt)
        return prompt

    def _apply_post_processors(self, response: str) -> str:
        for fn in self._post_processors:
            response = fn(response)
        return response

    def _validate_response(self, response: str) -> bool:
        for fn in self._validators:
            if not fn(response):
                self._logger.warning(f"Validation failed for response: {response}")
                return False
        return True

    def run(self, prompt: str, config: Optional[GenerationConfig] = None) -> InferenceResult:
        prompt_proc = self._apply_pre_processors(prompt)
        result = self._backend.generate(prompt_proc, config or self._default_config)
        result.response_text = self._apply_post_processors(result.response_text)
        valid = self._validate_response(result.response_text)
        if not valid:
            self._logger.warning(f"Response failed validation: {result.response_text}")
        return result

    def run_batch(self, prompts: List[str], config: Optional[GenerationConfig] = None) -> List[InferenceResult]:
        return [self.run(p, config) for p in prompts]

    def run_with_context(self, prompt: str, context: str, config: Optional[GenerationConfig] = None) -> InferenceResult:
        full_prompt = f"{context}\n\n---\n\nQuestion: {prompt}\n\nAnswer:"
        return self.run(full_prompt, config)

def create_inference_pipeline(
    backend_type: str = "local",
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> InferencePipeline:
    """
    Factory for creating an InferencePipeline.
    backend_type: "local" or "api"
    """
    if backend_type == "local":
        backend = LocalHFBackend(model_path=model_path, adapter_path=adapter_path, **kwargs)
    elif backend_type == "api":
        backend = APIBackend(provider=provider, model_name=model_name, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown backend_type: {backend_type}")
    return InferencePipeline(backend)
