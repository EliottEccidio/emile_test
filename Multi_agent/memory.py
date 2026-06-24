"""Memoire partagee du systeme multi-agent : le Blackboard (WorldState).

Le WorldState porte l'artefact courant (`goal`) que les agents lisent dans propose
et transforment dans execute. Aucune logique metier ici : seulement l'etat et les
types de domaine echanges (Action, Observation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    """Intention d'un agent : QUOI faire (produit par propose)."""
    agent: str
    tool: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """Resultat d'une Action executee (produit par execute)."""
    action: Action
    ok: bool
    result: Any = None
    error: str | None = None


@dataclass
class WorldState:
    """Blackboard : l'artefact courant sur lequel les agents operent.

    Pour l'extraction : `goal` = l'enonce de la question.
    Pour le calcul    : `goal` = le programme Python a executer.
    """
    goal: str
