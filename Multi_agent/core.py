"""Orchestration de la strategie agentique : PoT + self-consistency.

Pour chaque question, on echantillonne `k` programmes (reason-then-code,
temperature > 0), on execute chacun de facon sure, puis on VOTE la reponse
numerique majoritaire. Cela preserve le raisonnement du SLM (sa force) tout en
fiabilisant la reponse par l'ensemble.
"""
from __future__ import annotations

from collections import Counter

from agent import ProgramWriter, PythonCalculator
from memory import WorldState


def solve_pot(
    writer: ProgramWriter,
    calculator: PythonCalculator,
    questions: list[str],
    k: int = 5,
    temperature: float = 0.7,
) -> list[str | None]:
    """k programmes par question (echantillonnes), executes, puis vote majoritaire."""
    # chaque question est repetee k fois -> un seul appel SLM batche pour tout
    flat = [q for q in questions for _ in range(k)]
    programs = writer.write_program_batch(flat, temperature=temperature)
    answers = [_run_program(calculator, prog) for prog in programs]
    return [_majority_vote(answers[i:i + k]) for i in range(0, len(answers), k)]


def _run_program(calculator: PythonCalculator, program: str | None) -> float | None:
    """Execute un programme s'il est sur ; rend le resultat numerique ou None."""
    if not program:
        return None
    state = WorldState(goal=program)
    action = calculator.propose(state)  # valide le scope (programme sur)
    if action is None:
        return None
    obs = calculator.execute(action, state)
    return obs.result if obs.ok else None


def _majority_vote(values: list[float | None]) -> str | None:
    """Reponse la plus frequente parmi les echantillons valides (None ignores)."""
    votes = Counter(round(float(v), 4) for v in values if v is not None)
    if not votes:
        return None
    winner, _ = votes.most_common(1)[0]
    return _fmt(winner)


def _fmt(value: float) -> str:
    """Affichage propre : entier sans .0 quand c'est un entier."""
    return str(int(value)) if float(value).is_integer() else str(value)
