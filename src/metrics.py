"""Extraction de la reponse finale GSM8K et scoring exact-match.

Logique pure, sans IO ni dependance modele : c'est le coeur testable de l'eval.

Convention GSM8K : la reponse de reference se termine par une ligne `#### <nombre>`.
On demande au modele (via le system prompt) de respecter la meme convention, puis
on compare numeriquement.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

# Capture un entier/decimal eventuellement signe et avec separateurs de milliers.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize(token: str) -> str:
    return token.replace(",", "").rstrip(".")


def extract_gold_answer(answer: str) -> str:
    """Reponse de reference : le nombre apres le marqueur '####'."""
    tail = answer.split("####")[-1]
    numbers = _NUMBER.findall(tail)
    return _normalize(numbers[0]) if numbers else _normalize(tail.strip())


def extract_pred_answer(generation: str) -> Optional[str]:
    """Reponse predite : le nombre apres '####' si present, sinon le dernier nombre."""
    if "####" in generation:
        tail = generation.split("####")[-1]
        numbers = _NUMBER.findall(tail)
        if numbers:
            return _normalize(numbers[0])
    numbers = _NUMBER.findall(generation)
    return _normalize(numbers[-1]) if numbers else None


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_correct(pred: Optional[str], gold: str) -> bool:
    """Egalite numerique (tolerance 1e-4), avec repli sur l'egalite de chaines."""
    p, g = _as_float(pred), _as_float(gold)
    if p is not None and g is not None:
        return abs(p - g) < 1e-4
    return pred is not None and pred == gold


def accuracy(predictions: Sequence[Optional[str]], golds: Sequence[str]) -> float:
    if not golds:
        return 0.0
    hits = sum(is_correct(p, g) for p, g in zip(predictions, golds))
    return hits / len(golds)
