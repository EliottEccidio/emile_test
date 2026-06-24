"""Chargement modele/tokenizer, choix device + dtype, config LoRA.

Isole tout ce qui touche au hardware et au SDK (transformers/peft) du reste du
code metier, pour que train.py et evaluate.py restent lisibles.
"""
from __future__ import annotations

from typing import Optional

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    """cuda si dispo, sinon mps (Apple Silicon), sinon cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(precision: str, device: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    # auto : bf16 seulement sur GPU qui le supporte, fp32 ailleurs (dont MPS).
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(name: str, dtype: torch.dtype):
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


def load_for_eval(name: str, dtype: torch.dtype, device: str,
                  adapter_path: Optional[str] = None):
    """Modele pret pour la generation, avec adaptateur LoRA fusionne si fourni."""
    model = load_base_model(name, dtype)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model.to(device).eval()


def build_lora_config(cfg) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
