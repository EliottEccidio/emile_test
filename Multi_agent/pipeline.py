"""Pipeline d'evaluation sur GSM8K : SLM seul vs systeme multi-agent (PoT).

Deux mesures sur les MEMES 100 questions :
  1. Baseline : Qwen repond directement en chain-of-thought (batch).
  2. Agentique : Program-of-Thoughts -> ProgramWriter (SLM ecrit du Python)
                 puis PythonCalculator (execution sure).
Puis comparaison des precisions.

Usage : python pipeline.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

from agent import ProgramWriter, PythonCalculator
from core import solve_pot
from slm import SLMClient
from src.metrics import accuracy, extract_gold_answer, extract_pred_answer, is_correct

SEED = 42
N = 100
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
    agentic: list[str | None],
) -> None:
    n = len(golds)
    base_acc = accuracy(baseline, golds)
    agt_acc = accuracy(agentic, golds)

    print("\n" + "=" * 60)
    print(f"  N questions             : {n}")
    print(f"  Baseline Qwen (seul)    : {base_acc:.1%}")
    print(f"  Agentique PoT           : {agt_acc:.1%}   "
          f"(delta {agt_acc - base_acc:+.1%}, programmes valides {_rate(agentic):.0%})")
    print("=" * 60)

    print("\n10 premiers exemples (gold | baseline | agentique PoT) :")
    for i in range(min(10, n)):
        def tag(p: str | None) -> str:
            return "OK" if is_correct(p, golds[i]) else "  "
        print(
            f"  [{i + 1:02d}] gold={golds[i]:>8} | "
            f"{tag(baseline[i])} {str(baseline[i]):>12} | "
            f"{tag(agentic[i])} {str(agentic[i]):>12}"
        )


def main() -> None:
    set_seed(SEED)
    questions, golds = load_benchmark(SEED, N)
    slm = SLMClient(max_new_tokens=512, batch_size=8)
    writer = ProgramWriter(slm)
    calculator = PythonCalculator()

    baseline = run_baseline(slm, questions)

    print(f"[agentic] Program-of-Thoughts sur {len(questions)} questions ...")
    agentic = solve_pot(writer, calculator, questions)

    report(golds, baseline, agentic)


if __name__ == "__main__":
    main()
