"""Pipeline d'evaluation agentique sur GSM8K.

Trois etapes :
  1. Baseline   : Qwen repond directement a chaque question (batch).
  2. Agentique  : CalculusExtractor extrait l'expression (batch //),
                  PythonCalculator evalue (pur Python, instantane).
  3. Comparaison: accuracy agentique vs baseline vs gold.

Usage : python pipeline.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

from agent import CalculusExtractor, PythonCalculator
from memory import WorldState
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


def run_agentic(
    extractor: CalculusExtractor,
    calculator: PythonCalculator,
    questions: list[str],
) -> list[str | None]:
    # Etape 1 : extraction en batch (un seul appel SLM batche pour tout le lot)
    print(f"[agentic] extraction en batch sur {len(questions)} questions ...")
    extraction_obs = extractor.execute_batch(questions)
    n_extracted = sum(1 for o in extraction_obs if o.ok)
    print(f"[agentic] {n_extracted}/{len(questions)} expressions extraites")

    # Etape 2 : calcul Python (pur, instantane — PythonCalculator.propose valide le scope)
    results: list[str | None] = []
    for obs in extraction_obs:
        if not obs.ok or obs.result is None:
            results.append(None)
            continue
        state = WorldState(goal=obs.result)
        action = calculator.propose(state, None)  # type: ignore[arg-type]
        if action is None:
            results.append(None)
            continue
        calc_obs = calculator.execute(action, state)
        results.append(str(calc_obs.result) if calc_obs.ok else None)
    return results


def report(
    golds: list[str],
    baseline: list[str | None],
    agentic: list[str | None],
) -> None:
    n = len(golds)
    base_acc = accuracy(baseline, golds)
    agt_acc = accuracy(agentic, golds)
    extraction_rate = sum(1 for a in agentic if a is not None) / n

    print("\n" + "=" * 52)
    print(f"  N questions          : {n}")
    print(f"  Baseline Qwen        : {base_acc:.1%}")
    print(f"  Pipeline agentique   : {agt_acc:.1%}")
    print(f"  Taux d'extraction    : {extraction_rate:.1%}")
    print(f"  Delta                : {agt_acc - base_acc:+.1%}")
    print("=" * 52)

    print("\n5 premiers exemples :")
    for i in range(min(5, n)):
        b = "OK" if is_correct(baseline[i], golds[i]) else "KO"
        a = "OK" if is_correct(agentic[i], golds[i]) else "KO"
        print(
            f"  [{i+1:02d}] gold={golds[i]:>8} | "
            f"baseline[{b}]={str(baseline[i]):>10} | "
            f"agentic[{a}]={str(agentic[i])}"
        )


def main() -> None:
    set_seed(SEED)
    questions, golds = load_benchmark(SEED, N)
    slm = SLMClient(max_new_tokens=512, batch_size=8)
    extractor = CalculusExtractor(slm)
    calculator = PythonCalculator()
    baseline_preds = run_baseline(slm, questions)
    agentic_preds = run_agentic(extractor, calculator, questions)
    report(golds, baseline_preds, agentic_preds)


if __name__ == "__main__":
    main()
