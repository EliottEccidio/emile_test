"""Memoire partagee du systeme multi-agent : le Blackboard (WorldState).

Le WorldState est l'etat partage que les agents lisent (goal) et alimentent
(history). Aucune logique metier ici : seulement l'etat et les types de domaine
echanges (Action, Observation).
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
    """Blackboard : objectif courant + memoire des etapes deja calculees.

    `history` sert de memoire a la strategie iterative : chaque Observation y
    conserve l'expression calculee et son resultat, reinjectes au SLM au tick
    suivant.
    """
    goal: str
    history: list[Observation] = field(default_factory=list)

    def record(self, obs: Observation) -> None:
        self.history.append(obs)
