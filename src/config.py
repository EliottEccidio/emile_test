"""Configuration typee, chargee depuis config.yaml.

Un seul objet `Config` est partage par l'entrainement et l'evaluation : c'est la
source de verite unique des hyperparametres et des chemins.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    precision: str = "auto"  # auto | fp32 | bf16 | fp16
    max_seq_length: int = 1024


@dataclass
class DataConfig:
    dataset: str = "openai/gsm8k"
    subset: str = "main"
    train_split: str = "train"
    eval_split: str = "test"
    train_limit: Optional[int] = None
    eval_limit: Optional[int] = 200


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    output_dir: str = "outputs/qwen2.5-1.5b-gsm8k-lora"
    epochs: float = 1.0
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    gradient_checkpointing: bool = True


@dataclass
class EvalConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    batch_size: int = 8


@dataclass
class MLflowConfig:
    tracking_uri: str = "file:./mlruns"
    experiment: str = "qwen2.5-1.5b-gsm8k-lora"


@dataclass
class Config:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


def _build(cls, data: Optional[dict]):
    """Instancie une sous-config en ignorant les cles inconnues du YAML."""
    if not data:
        return cls()
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        seed=raw.get("seed", 42),
        model=_build(ModelConfig, raw.get("model")),
        data=_build(DataConfig, raw.get("data")),
        lora=_build(LoRAConfig, raw.get("lora")),
        train=_build(TrainConfig, raw.get("train")),
        eval=_build(EvalConfig, raw.get("eval")),
        mlflow=_build(MLflowConfig, raw.get("mlflow")),
    )
