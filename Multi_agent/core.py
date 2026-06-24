"""Orchestration : les deux strategies agentiques (pas de superviseur).

Chaque strategie combine les 2 agents (CalculusExtractor + PythonCalculator) pour
resoudre une question GSM8K, et renvoie une prediction str | None par question.

  - solve_single_shot : 1 appel SLM batche -> une expression composee -> 1 calcul.
  - solve_iterative   : boucle SLM<->Python jusqu'a STOP ou max_steps.
"""
from __future__ import annotations

from agent import CalculusExtractor, PythonCalculator
from memory import Action, WorldState


def solve_single_shot(
    extractor: CalculusExtractor,
    calculator: PythonCalculator,
    questions: list[str],
) -> list[str | None]:
    """Extraction batchee d'une expression par question, puis calcul Python."""
    predictions: list[str | None] = []
    for obs in extractor.execute_batch(questions):
        if not obs.ok or not obs.result:
            predictions.append(None)
            continue
        state = WorldState(goal=obs.result)
        action = calculator.propose(state)  # valide le scope arithmetique
        if action is None:
            predictions.append(None)
            continue
        calc = calculator.execute(action, state)
        predictions.append(str(calc.result) if calc.ok else None)
    return predictions


def solve_iterative(
    extractor: CalculusExtractor,
    calculator: PythonCalculator,
    questions: list[str],
    max_steps: int = 6,
) -> list[str | None]:
    """Boucle SLM<->Python par question (le SLM est rappele a chaque etape)."""
    return [_solve_one(extractor, calculator, q, max_steps) for q in questions]


def _solve_one(
    extractor: CalculusExtractor,
    calculator: PythonCalculator,
    question: str,
    max_steps: int,
) -> str | None:
    state = WorldState(goal=question)
    last_result: float | None = None
    for _ in range(max_steps):  # garde de terminaison : pas de boucle infinie
        expr = extractor.extract_next(state)  # appel SLM (rappel a chaque tick)
        if expr is None:  # STOP, rien d'extrait, ou expression deja calculee
            break
        action = Action(agent=calculator.name, tool="eval_arithmetic",
                        payload={"expression": expr})
        obs = calculator.execute(action, state)
        if not obs.ok:  # expression hors scope / erreur de calcul -> on s'arrete
            break
        state.record(obs)
        last_result = obs.result
    return str(last_result) if last_result is not None else None
