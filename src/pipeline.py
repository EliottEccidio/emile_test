"""Pipeline complet pre -> train -> post, sous un unique run parent MLflow.

Produit dans l'UI MLflow un run "pre-vs-post" avec trois runs imbriques
(eval-pre, train, eval-post) et, au niveau parent, les metriques de synthese
pre_accuracy / post_accuracy / accuracy_delta.

    python -m src.pipeline
    python -m src.pipeline --config config.yaml
"""
from __future__ import annotations

import argparse

import mlflow

from .config import load_config
from .evaluate import evaluate_and_log
from .tracking import setup_mlflow
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline pre/post fine-tuning")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_mlflow(cfg)

    with mlflow.start_run(run_name="pre-vs-post"):
        print(">>> Evaluation PRE (modele de base)")
        pre = evaluate_and_log(cfg, adapter_path=None,
                               run_name="eval-pre", nested=True)

        print(">>> Fine-tuning LoRA")
        adapter = train(cfg, nested=True)

        print(">>> Evaluation POST (modele fine-tune)")
        post = evaluate_and_log(cfg, adapter_path=adapter,
                                run_name="eval-post", nested=True)

        delta = post.accuracy - pre.accuracy
        mlflow.log_metrics({
            "pre_accuracy": pre.accuracy,
            "post_accuracy": post.accuracy,
            "accuracy_delta": delta,
        })
        print(f"\n=== Resultat ===")
        print(f"pre  = {pre.accuracy:.4f} ({pre.n_correct}/{pre.n_total})")
        print(f"post = {post.accuracy:.4f} ({post.n_correct}/{post.n_total})")
        print(f"delta = {delta:+.4f}")


if __name__ == "__main__":
    main()
