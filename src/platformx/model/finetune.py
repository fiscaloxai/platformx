from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
import json
import logging
import os
import hashlib

from ..data.schema import DatasetSchema, IntendedUse
from .adapters import AdapterArtifact

logger = logging.getLogger("platformx.model.finetune")
from dataclasses import field

@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter training."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    output_dir: str = "./output"
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    max_seq_length: int = 512
    fp16: bool = False
    bf16: bool = False
    seed: int = 42
    report_to: str = "none"  # options: none, wandb, tensorboard

class TrainingCallback:
    """Callback interface for training events."""
    def on_train_begin(self, args: Dict[str, Any]) -> None: pass
    def on_train_end(self, metrics: Dict[str, Any]) -> None: pass
    def on_epoch_begin(self, epoch: int) -> None: pass
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None: pass
    def on_step(self, step: int, loss: float) -> None: pass
    def on_save(self, checkpoint_path: str) -> None: pass

class AuditTrainingCallback(TrainingCallback):
    """Callback that logs training events to AuditLogger."""
    def __init__(self, audit_logger: Any) -> None:
        self.audit_logger = audit_logger
    def on_train_begin(self, args: Dict[str, Any]) -> None:
        self.audit_logger.log("train_begin", args)
    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        self.audit_logger.log("train_end", metrics)
    def on_epoch_begin(self, epoch: int) -> None:
        self.audit_logger.log("epoch_begin", {"epoch": epoch})
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        self.audit_logger.log("epoch_end", {"epoch": epoch, **metrics})
    def on_step(self, step: int, loss: float) -> None:
        self.audit_logger.log("step", {"step": step, "loss": loss})
    def on_save(self, checkpoint_path: str) -> None:
        self.audit_logger.log("save", {"checkpoint_path": checkpoint_path})


@dataclass
class FinetuneReport:
    adapter_id: str
    created_at: datetime
    model_fingerprint: str
    training_dataset_ids: List[str]
    seed: Optional[int]
    metadata: Dict[str, Any]



class FineTuner:
    """
    Fine-tuning interface using HuggingFace and PEFT/LoRA for efficient adapter training.
    Supports dry_run for pipeline testing, callback system for monitoring, and full auditability.
    """
    def __init__(self, training_config: Optional[TrainingConfig] = None, lora_config: Optional[LoRAConfig] = None):
        self.training_config = training_config or TrainingConfig()
        self.lora_config = lora_config or LoRAConfig()
        self._callbacks: List[TrainingCallback] = []
        self._logger = logging.getLogger("platformx.model.finetune")

    def add_callback(self, callback: TrainingCallback) -> None:
        self._callbacks.append(callback)

    def validate_datasets(self, datasets: List[DatasetSchema]) -> None:
        if not datasets:
            raise ValueError("At least one dataset is required for fine-tuning")
        for d in datasets:
            if d.intended_use != IntendedUse.finetuning:
                raise ValueError(f"Dataset {d.dataset_id} not intended for finetuning")
            # Ensure provenance contains minimal required fields
            if not d.provenance or not d.provenance.source_uri or not d.provenance.ingested_at:
                raise ValueError(f"Dataset {d.dataset_id} missing required provenance information")


    def _prepare_training_data(self, datasets: List[DatasetSchema]) -> List[Dict[str, str]]:
        """
        Convert DatasetSchema list to training format. Handles RAFT and plain text.
        """
        examples = []
        for ds in datasets:
            if ds.raw_text and "---INSTRUCTION---" in ds.raw_text:
                # Parse RAFT-style
                parts = ds.raw_text.split("---INSTRUCTION---")
                for part in parts:
                    if not part.strip():
                        continue
                    instr = self._extract_between(part, "", "---CONTEXT---")
                    ctx = self._extract_between(part, "---CONTEXT---", "---EXPECTED---")
                    resp = self._extract_between(part, "---EXPECTED---", None)
                    if instr and ctx and resp:
                        examples.append({"instruction": instr.strip(), "context": ctx.strip(), "response": resp.strip()})
            elif ds.raw_text:
                examples.append({"text": ds.raw_text.strip()})
        self._logger.info(f"Prepared {len(examples)} training examples.")
        return examples

    def _extract_between(self, text: str, start: str, end: Optional[str]) -> str:
        s = text.find(start) + len(start) if start else 0
        e = text.find(end, s) if end and end in text[s:] else None
        return text[s:e].strip() if e else text[s:].strip()

    def _format_for_training(self, examples: List[Dict]) -> List[str]:
        formatted = []
        for ex in examples:
            if "instruction" in ex and "context" in ex and "response" in ex:
                formatted.append(f"### Instruction:\n{ex['instruction']}\n\n### Context:\n{ex['context']}\n\n### Response:\n{ex['response']}")
            elif "text" in ex:
                formatted.append(ex["text"])
        return formatted

    def _setup_model_and_tokenizer(self, base_model_path: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model
            import torch
        except ImportError:
            raise ImportError("transformers and peft are required. Install with 'pip install transformers peft'.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16 if self.training_config.fp16 else torch.float32)
        lora_cfg = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type,
        )
        model = get_peft_model(model, lora_cfg)
        self._logger.info(f"Loaded model {base_model_path} with LoRA config: {self.lora_config}")
        return model, tokenizer

    def _create_dataset(self, formatted_texts: List[str], tokenizer: Any):
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets is required. Install with 'pip install datasets'.")
        ds = Dataset.from_dict({"text": formatted_texts})
        def tokenize_fn(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=self.training_config.max_seq_length)
        return ds.map(tokenize_fn, batched=True)

    def run(self, base_model_path: str, datasets: List[DatasetSchema], seed: Optional[int] = None, training_options: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> FinetuneReport:
        """
        Execute a fine-tuning run using HuggingFace and PEFT/LoRA. Supports dry_run for pipeline testing.
        """
        training_options = training_options or {}
        self.validate_datasets(datasets)
        seed = seed or self.training_config.seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Prepare data
        examples = self._prepare_training_data(datasets)
        formatted = self._format_for_training(examples)
        if dry_run:
            self._logger.info("Dry run: would train on %d examples with model %s", len(formatted), base_model_path)
            return FinetuneReport(
                adapter_id="dryrun",
                created_at=datetime.utcnow(),
                model_fingerprint=base_model_path,
                training_dataset_ids=[d.dataset_id for d in datasets],
                seed=seed,
                metadata={"dry_run": True, "training_options": training_options},
            )
        try:
            model, tokenizer = self._setup_model_and_tokenizer(base_model_path)
            train_dataset = self._create_dataset(formatted, tokenizer)
            from transformers import TrainingArguments, Trainer
            args = TrainingArguments(
                output_dir=self.training_config.output_dir,
                num_train_epochs=self.training_config.num_epochs,
                per_device_train_batch_size=self.training_config.per_device_batch_size,
                gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
                learning_rate=self.training_config.learning_rate,
                warmup_ratio=self.training_config.warmup_ratio,
                weight_decay=self.training_config.weight_decay,
                logging_steps=self.training_config.logging_steps,
                save_steps=self.training_config.save_steps,
                eval_steps=self.training_config.eval_steps,
                max_seq_length=self.training_config.max_seq_length,
                fp16=self.training_config.fp16,
                bf16=self.training_config.bf16,
                seed=seed,
                report_to=self.training_config.report_to,
            )
            for cb in self._callbacks:
                cb.on_train_begin(vars(args))
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
            )
            trainer.train()
            for cb in self._callbacks:
                cb.on_train_end({})
            adapter_id = self.save_adapter(self.training_config.output_dir, model)
            return FinetuneReport(
                adapter_id=adapter_id,
                created_at=datetime.utcnow(),
                model_fingerprint=base_model_path,
                training_dataset_ids=[d.dataset_id for d in datasets],
                seed=seed,
                metadata={"output_dir": self.training_config.output_dir, "training_options": training_options},
            )
        except Exception as e:
            self._logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")

    def save_adapter(self, output_path: str, model: Any) -> str:
        model.save_pretrained(output_path)
        config_path = os.path.join(output_path, "lora_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.lora_config.__dict__, f)
        adapter_id = hashlib.sha256((output_path + json.dumps(self.lora_config.__dict__)).encode("utf-8")).hexdigest()
        self._logger.info(f"Adapter saved to {output_path} with id {adapter_id}")
        return adapter_id

    def load_adapter(self, adapter_path: str, base_model_path: str) -> Any:
        try:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
        except ImportError:
            raise ImportError("transformers and peft are required. Install with 'pip install transformers peft'.")
        model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        self._logger.info(f"Loaded adapter from {adapter_path} onto base model {base_model_path}")
        return model

# Public API
__all__ = [
    "FineTuner",
    "FinetuneReport",
    "TrainingConfig",
    "LoRAConfig",
    "TrainingCallback",
    "AuditTrainingCallback",
]
