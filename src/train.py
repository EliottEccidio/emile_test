"""Fine-tuning LoRA (SFT) de Qwen2.5-1.5B-Instruct sur GSM8K, trace par MLflow.

Le code est ecrit pour resister aux differences d'API entre versions de TRL :
  - les champs de SFTConfig sont filtres dynamiquement ;
  - le nom du parametre tokenizer/processing_class de SFTTrainer est detecte.

Utilisation directe :
    python -m src.train
    python -m src.train --train-limit 100 --epochs 1   # smoke test
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect

import mlflow
import torch
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from .config import Config, load_config
from .data import build_formatting_func, load_split
from .model import (build_lora_config, load_base_model, load_tokenizer,
                    pick_device, resolve_dtype)
from .tracking import setup_mlflow


class MLflowLossCallback(TrainerCallback):
    """Logge les scalaires d'entrainement (loss, lr, grad_norm...) dans le run actif."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        metrics = {k: float(v) for k, v in logs.items()
                   if isinstance(v, (int, float))}
        if metrics:
            mlflow.log_metrics(metrics, step=state.global_step)


def _make_sft_config(cfg: Config, device: str) -> SFTConfig:
    """Construit SFTConfig en ne gardant que les champs supportes par la version installee."""
    valid = {f.name for f in dataclasses.fields(SFTConfig)}
    dtype = resolve_dtype(cfg.model.precision, device)

    kwargs = dict(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        logging_steps=cfg.train.logging_steps,
        save_strategy=cfg.train.save_strategy,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        packing=False,
        report_to="none",  # MLflow gere par notre callback, pas par le Trainer
        seed=cfg.seed,
    )
    # Le nom du champ de longueur max a change selon les versions de TRL.
    seq_key = "max_seq_length" if "max_seq_length" in valid else "max_length"
    kwargs[seq_key] = cfg.model.max_seq_length

    return SFTConfig(**{k: v for k, v in kwargs.items() if k in valid})


def train(cfg: Config, nested: bool = False) -> str:
    """Entraine l'adaptateur LoRA, le sauvegarde et renvoie son repertoire."""
    device = pick_device()
    dtype = resolve_dtype(cfg.model.precision, device)
    tokenizer = load_tokenizer(cfg.model.name)
    model = load_base_model(cfg.model.name, dtype)
    model.config.use_cache = False  # incompatible avec gradient_checkpointing

    dataset = load_split(cfg, cfg.data.train_split, cfg.data.train_limit)
    sft_config = _make_sft_config(cfg, device)
    lora_config = build_lora_config(cfg)

    # tokenizer= (ancien) vs processing_class= (TRL recent)
    init_params = inspect.signature(SFTTrainer.__init__).parameters
    tok_key = "processing_class" if "processing_class" in init_params else "tokenizer"

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        formatting_func=build_formatting_func(tokenizer),
        peft_config=lora_config,
        callbacks=[MLflowLossCallback()],
        **{tok_key: tokenizer},
    )

    with mlflow.start_run(run_name="train", nested=nested):
        mlflow.log_params({
            "model": cfg.model.name,
            "device": device,
            "precision": str(dtype).replace("torch.", ""),
            "epochs": cfg.train.epochs,
            "learning_rate": cfg.train.learning_rate,
            "per_device_batch_size": cfg.train.per_device_batch_size,
            "grad_accum": cfg.train.gradient_accumulation_steps,
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.alpha,
            "lora_dropout": cfg.lora.dropout,
            "train_n": len(dataset),
            "max_seq_length": cfg.model.max_seq_length,
        })
        trainer.train()
        trainer.save_model(cfg.train.output_dir)
        tokenizer.save_pretrained(cfg.train.output_dir)
        final = next((h["train_loss"] for h in reversed(trainer.state.log_history)
                      if "train_loss" in h), None)
        if final is not None:
            mlflow.log_metric("final_train_loss", final)

    return cfg.train.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA GSM8K")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Surcharge data.train_limit (smoke test)")
    parser.add_argument("--epochs", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.train_limit is not None:
        cfg.data.train_limit = args.train_limit
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    setup_mlflow(cfg)
    output_dir = train(cfg)
    print(f"Adaptateur LoRA sauvegarde dans : {output_dir}")


if __name__ == "__main__":
    main()
