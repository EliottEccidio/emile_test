"""Chargement de GSM8K et mise au format chat Qwen.

Le meme system prompt sert a l'entrainement et a l'eval, pour que le modele
apprenne (puis reproduise) la convention de reponse finale `#### <nombre>`.
"""
from __future__ import annotations

from typing import Optional

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then state the final answer on its own line as '#### <number>'."
)


def load_split(cfg, split: str, limit: Optional[int] = None):
    """Charge un split GSM8K, eventuellement tronque a `limit` exemples."""
    dataset = load_dataset(cfg.data.dataset, cfg.data.subset, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def _messages(question: str, answer: Optional[str] = None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def build_formatting_func(tokenizer):
    """Renvoie la formatting_func attendue par TRL : un exemple -> une string.

    Chaque exemple devient une conversation complete (system/user/assistant)
    serialisee via le chat template de Qwen.
    """
    def format_example(example):
        return tokenizer.apply_chat_template(
            _messages(example["question"], example["answer"]), tokenize=False
        )

    return format_example


def build_eval_prompt(tokenizer, question: str) -> str:
    """Prompt d'inference : system + user + amorce de reponse de l'assistant."""
    return tokenizer.apply_chat_template(
        _messages(question), tokenize=False, add_generation_prompt=True
    )
