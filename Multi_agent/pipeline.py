"""Pipeline d'evaluation sur GSM8K : SLM seul vs systeme multi-agent.

Trois mesures sur les MEMES 100 questions :
  1. Baseline       : Qwen repond directement (batch).
  2. Agentique 1    : single-shot (extraction -> calcul Python).
  3. Agentique 2    : iteratif (boucle SLM<->Python).
Puis comparaison des precisions.

Usage : python pipeline.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

from agent import CalculusExtractor, PythonCalculator
from core import solve_iterative, solve_single_shot
from slm import SLMClient
from src.metrics import accuracy, extract_gold_answer, extract_pred_answer, is_correct

SEED = 42
N = 100
MAX_STEPS = 6          # garde de terminaison de la strategie iterative
DATASET = "openai/gsm8k"

_BASELINE_PROMPT = (
    "Solve the following math problem step by step. "
    "At the very end write '#### ' followed by the final numeric answer.\n\n"
    "{question}"
)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_benchmark(seed: int, n: int) -> tuple[list[str], list[str]]:
    ds = load_dataset(DATASET, "main", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)
    subset = ds.select(indices)
    questions = [row["question"] for row in subset]
    golds = [extract_gold_answer(row["answer"]) for row in subset]
    print(f"[data] {n} questions chargees depuis {DATASET} (seed={seed})")
    return questions, golds


def run_baseline(slm: SLMClient, questions: list[str]) -> list[str | None]:
    print(f"[baseline] inference directe Qwen sur {len(questions)} questions ...")
    prompts = [_BASELINE_PROMPT.format(question=q) for q in questions]
    outputs = slm.complete_batch(prompts)
    return [extract_pred_answer(o) for o in outputs]


def _rate(preds: list[str | None]) -> float:
    return sum(1 for p in preds if p is not None) / len(preds)


def report(
    golds: list[str],
    baseline: list[str | None],
    single_shot: list[str | None],
    iterative: list[str | None],
) -> None:
    n = len(golds)
    base_acc = accuracy(baseline, golds)
    ss_acc = accuracy(single_shot, golds)
    it_acc = accuracy(iterative, golds)

    print("\n" + "=" * 64)
    print(f"  N questions               : {n}")
    print(f"  Baseline Qwen (seul)      : {base_acc:.1%}")
    print(f"  Agentique single-shot     : {ss_acc:.1%}   "
          f"(delta {ss_acc - base_acc:+.1%}, resolu {_rate(single_shot):.0%})")
    print(f"  Agentique iteratif        : {it_acc:.1%}   "
          f"(delta {it_acc - base_acc:+.1%}, resolu {_rate(iterative):.0%})")
    print("=" * 64)

    print("\n8 premiers exemples (gold | baseline | single-shot | iteratif) :")
    for i in range(min(8, n)):
        def tag(p: str | None) -> str:
            return "OK" if is_correct(p, golds[i]) else "  "
        print(
            f"  [{i + 1:02d}] gold={golds[i]:>8} | "
            f"{tag(baseline[i])} {str(baseline[i]):>10} | "
            f"{tag(single_shot[i])} {str(single_shot[i]):>10} | "
            f"{tag(iterative[i])} {str(iterative[i]):>10}"
        )


def main() -> None:
    set_seed(SEED)
    questions, golds = load_benchmark(SEED, N)
    slm = SLMClient(max_new_tokens=512, batch_size=8)
    extractor = CalculusExtractor(slm)
    calculator = PythonCalculator()

    baseline = run_baseline(slm, questions)

    print(f"[agentic] single-shot sur {len(questions)} questions ...")
    single_shot = solve_single_shot(extractor, calculator, questions)

    print(f"[agentic] iteratif (max_steps={MAX_STEPS}) sur {len(questions)} questions ...")
    iterative = solve_iterative(extractor, calculator, questions, max_steps=MAX_STEPS)

    report(golds, baseline, single_shot, iterative)


if __name__ == "__main__":
    main()
