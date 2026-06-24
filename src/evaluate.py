"""Evaluation GSM8K (pre / post fine-tuning) avec logging MLflow.

Separation nette :
  - `run_evaluation` : logique pure (genere, extrait, score) ; le modele est injecte.
  - `evaluate_and_log` : couche infra (charge le modele, ouvre un run MLflow, logge).

Utilisation directe :
    python -m src.evaluate --run-name eval-pre
    python -m src.evaluate --adapter outputs/qwen2.5-1.5b-gsm8k-lora --run-name eval-post
    python -m src.evaluate --limit 20 --run-name smoke   # smoke test rapide
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlflow
import torch
from tqdm import tqdm

from .config import Config, load_config
from .data import build_eval_prompt, load_split
from .metrics import extract_gold_answer, extract_pred_answer, is_correct
from .model import load_for_eval, load_tokenizer, pick_device, resolve_dtype
from .tracking import setup_mlflow


@dataclass
class EvalResult:
    accuracy: float
    n_total: int
    n_correct: int
    samples: list


def _generate(model, tokenizer, prompts, eval_cfg, device) -> list[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(max_new_tokens=eval_cfg.max_new_tokens,
                      pad_token_id=tokenizer.pad_token_id)
    if eval_cfg.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=eval_cfg.temperature)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    completion = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(completion, skip_special_tokens=True)


def run_evaluation(model, tokenizer, dataset, eval_cfg, device,
                   n_samples: int = 8) -> EvalResult:
    """Genere une reponse par question et mesure l'exact-match sur la reponse finale."""
    tokenizer.padding_side = "left"  # requis pour une generation batchee correcte
    questions = list(dataset["question"])
    golds = [extract_gold_answer(a) for a in dataset["answer"]]

    preds: list[Optional[str]] = []
    generations: list[str] = []
    batch_size = max(1, eval_cfg.batch_size)
    for start in tqdm(range(0, len(questions), batch_size), desc="eval"):
        batch_q = questions[start:start + batch_size]
        prompts = [build_eval_prompt(tokenizer, q) for q in batch_q]
        gens = _generate(model, tokenizer, prompts, eval_cfg, device)
        generations.extend(gens)
        preds.extend(extract_pred_answer(g) for g in gens)

    correct = [is_correct(p, g) for p, g in zip(preds, golds)]
    n_correct = int(sum(correct))
    samples = [
        {
            "question": questions[i],
            "gold": golds[i],
            "pred": preds[i],
            "correct": bool(correct[i]),
            "generation": generations[i][:600],
        }
        for i in range(min(n_samples, len(questions)))
    ]
    return EvalResult(
        accuracy=n_correct / len(golds) if golds else 0.0,
        n_total=len(golds),
        n_correct=n_correct,
        samples=samples,
    )


def _save_and_log_report(result: EvalResult, run_name: str) -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{run_name}.json"
    payload = {
        "accuracy": result.accuracy,
        "n_total": result.n_total,
        "n_correct": result.n_correct,
        "samples": result.samples,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    mlflow.log_artifact(str(path), artifact_path="eval")


def evaluate_and_log(cfg: Config, adapter_path: Optional[str] = None,
                     run_name: str = "eval", nested: bool = False) -> EvalResult:
    device = pick_device()
    dtype = resolve_dtype(cfg.model.precision, device)
    tokenizer = load_tokenizer(cfg.model.name)
    model = load_for_eval(cfg.model.name, dtype, device, adapter_path)
    dataset = load_split(cfg, cfg.data.eval_split, cfg.data.eval_limit)

    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.log_params({
            "model": cfg.model.name,
            "adapter": adapter_path or "none",
            "eval_split": cfg.data.eval_split,
            "eval_n": len(dataset),
            "max_new_tokens": cfg.eval.max_new_tokens,
            "temperature": cfg.eval.temperature,
            "device": device,
        })
        result = run_evaluation(model, tokenizer, dataset, cfg.eval, device)
        mlflow.log_metric("accuracy", result.accuracy)
        mlflow.log_metric("n_correct", result.n_correct)
        mlflow.log_metric("n_total", result.n_total)
        _save_and_log_report(result, run_name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation GSM8K avec MLflow")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter", default=None,
                        help="Chemin de l'adaptateur LoRA (omis = modele de base)")
    parser.add_argument("--run-name", default="eval")
    parser.add_argument("--limit", type=int, default=None,
                        help="Surcharge data.eval_limit (utile pour un smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.data.eval_limit = args.limit

    setup_mlflow(cfg)
    result = evaluate_and_log(cfg, args.adapter, args.run_name)
    print(f"accuracy={result.accuracy:.4f} "
          f"({result.n_correct}/{result.n_total})")


if __name__ == "__main__":
    main()
