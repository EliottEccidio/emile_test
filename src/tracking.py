"""Configuration centralisee de MLflow.

MLflow 3.x a mis le backend "file store" (./mlruns) en maintenance mode ; on
utilise donc un backend SQLite local (sqlite:///mlflow.db) et un dossier
d'artefacts explicite (./mlartifacts).
"""
from __future__ import annotations

from pathlib import Path

import mlflow


def setup_mlflow(cfg) -> None:
    """Pointe MLflow sur le backend configure et garantit l'experience cible."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    if mlflow.get_experiment_by_name(cfg.mlflow.experiment) is None:
        artifact_dir = Path("mlartifacts").absolute()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(cfg.mlflow.experiment,
                                 artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment)
