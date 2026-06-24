PYTHON ?= python

.PHONY: setup test eval-pre train eval-post pipeline mlflow-ui clean

## Cree un venv et installe les dependances
setup:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

## Lance les tests unitaires (metrics)
test:
	pytest -q

## Eval baseline (modele de base, sans adaptateur)
eval-pre:
	$(PYTHON) -m src.evaluate --run-name eval-pre

## Fine-tuning LoRA
train:
	$(PYTHON) -m src.train

## Eval post-finetuning (avec l'adaptateur LoRA)
eval-post:
	$(PYTHON) -m src.evaluate --adapter outputs/qwen2.5-1.5b-gsm8k-lora --run-name eval-post

## Pipeline complet : pre -> train -> post (un seul run parent MLflow)
pipeline:
	$(PYTHON) -m src.pipeline

## Ouvre l'UI MLflow sur http://localhost:5000
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

## Supprime artefacts et caches
clean:
	rm -rf outputs mlruns mlartifacts .pytest_cache __pycache__ src/__pycache__
