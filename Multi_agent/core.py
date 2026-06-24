"""Orchestration de la strategie agentique Program-of-Thoughts (pas de superviseur).

solve_pot combine les 2 agents : ProgramWriter (SLM ecrit un programme Python),
puis PythonCalculator (execution sure -> `answer`). Renvoie une prediction
str | None par question.
"""
from __future__ import annotations

from agent import ProgramWriter, PythonCalculator
from memory import WorldState


def solve_pot(
    writer: ProgramWriter,
    calculator: PythonCalculator,
    questions: list[str],
) -> list[str | None]:
    """Ecriture batchee des programmes, puis execution sure de chacun."""
    predictions: list[str | None] = []
    for program in writer.write_program_batch(questions):
        if not program:
            predictions.append(None)
            continue
        state = WorldState(goal=program)
        action = calculator.propose(state)  # valide le scope (programme sur)
        if action is None:
            predictions.append(None)
            continue
        obs = calculator.execute(action, state)
        predictions.append(str(obs.result) if obs.ok else None)
    return predictions
