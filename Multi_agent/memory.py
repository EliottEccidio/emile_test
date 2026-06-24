"""Memoire partagee du systeme multi-agent : le Blackboard (WorldState).

Le WorldState est l'unique source de verite sur laquelle le superviseur et les
agents agissent. Les agents y LISENT le contexte (dans propose) et le superviseur
y ECRIT les resultats apres execution. Aucune logique metier ici : seulement
l'etat et les types de domaine echanges (Action, Observation, Finding).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    """Intention d'un agent : QUOI faire, sans effet de bord (produit par propose).

    `key()` donne une signature canonique servant a la deduplication : on evite
    qu'un agent rejoue exactement la meme action.
    """
    agent: str
    tool: str
    payload: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        items = sorted(self.payload.items())
        return f"{self.agent}:{self.tool}:{items}"


@dataclass(frozen=True)
class Observation:
    """Resultat d'une Action executee (produit par execute)."""
    action: Action
    ok: bool
    result: Any = None
    error: str | None = None


@dataclass(frozen=True)
class Finding:
    """Fait utile a conserver sur le blackboard (resultat durable)."""
    source: str          # nom de l'agent qui l'a produit
    content: Any


@dataclass
class WorldState:
    """Blackboard : etat partage d'un run.

    budget + max_steps garantissent la terminaison (cf. Oracle). `done` peut etre
    leve quand l'objectif est atteint. history conserve la trace de chaque tick.
    """
    goal: str
    budget: int = 8
    max_steps: int = 16
    step: int = 0
    done: bool = False
    history: list[Observation] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)

    # --- mutations : le superviseur agit sur l'etat via ces helpers ---
    def seen(self, action: Action) -> bool:
        """Action deja jouee ? (dedup sur la cle canonique)."""
        return any(o.action.key() == action.key() for o in self.history)

    def record(self, obs: Observation) -> None:
        """Enregistre une observation et consomme un pas + une unite de budget."""
        self.history.append(obs)
        self.step += 1
        self.budget -= 1

    def add_finding(self, finding: Finding) -> None:
        self.findings.append(finding)
