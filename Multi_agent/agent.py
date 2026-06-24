"""Roster d'agents : contrat commun + registre extensible + 2 agents concrets.

Contrat (point d'extension) :
  - propose(state) -> Action | None       : le PLAN, sans effet de bord.
  - execute(action, state) -> Observation  : l'ACT, seul endroit a effet de bord.

On ajoute un agent SANS toucher au reste : il suffit de le decorer avec @register.
Son attribut de CLASSE `name` sert de cle dans AGENT_REGISTRY.
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Protocol

from memory import Action, Observation, WorldState


class LLMClient(Protocol):
    """Interface du LLM (injectee) : rend les agents mockables."""
    def complete(self, prompt: str) -> str: ...


class Agent(Protocol):
    """Contrat commun a tous les agents."""
    name: str  # cle dans AGENT_REGISTRY (attribut de CLASSE)

    def propose(self, state: WorldState) -> Action | None: ...
    def execute(self, action: Action, state: WorldState) -> Observation: ...


AGENT_REGISTRY: dict[str, type[Agent]] = {}


def register(cls: type[Agent]) -> type[Agent]:
    """Enregistre un agent dans AGENT_REGISTRY sous son attribut de classe `name`."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} doit definir un attribut de classe `name`.")
    if name in AGENT_REGISTRY:
        raise ValueError(f"Agent deja enregistre : {name!r}")
    AGENT_REGISTRY[name] = cls
    return cls


# Operateurs autorises : c'est l'allow-list qui definit le scope du calculateur.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@register
class PythonCalculator:
    """Agent de calcul : evalue une expression arithmetique sure.

    Scope : uniquement de l'arithmetique (+, -, *, /, //, %, **, parentheses,
    nombres). Aucun appel de fonction, aucun nom, aucun acces attribut. Le scope
    structurel est verifie dans propose ; le calcul (qui peut echouer, ex. 1/0)
    a lieu dans execute.
    """
    name = "python_calculator"

    def propose(self, state: WorldState) -> Action | None:
        expr = state.goal
        if not self._is_arithmetic(expr):
            return None
        return Action(agent=self.name, tool="eval_arithmetic",
                      payload={"expression": expr})

    def execute(self, action: Action, state: WorldState) -> Observation:
        expr = action.payload["expression"]
        try:
            value = self._safe_eval(expr)
        except Exception as exc:  # ZeroDivisionError, overflow, expr invalide...
            return Observation(action=action, ok=False, error=str(exc))
        return Observation(action=action, ok=True, result=value)

    # --- coeur de domaine : pur et testable, sans IO ---
    @classmethod
    def _is_arithmetic(cls, expr: Any) -> bool:
        """Vrai si l'expression ne contient que de l'arithmetique autorisee."""
        if not isinstance(expr, str) or not expr.strip():
            return False
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return False
        return all(cls._node_allowed(node) for node in ast.walk(tree))

    @staticmethod
    def _node_allowed(node: ast.AST) -> bool:
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp)):
            return True
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        return isinstance(node, tuple(_OPS))  # noeud operateur autorise

    @classmethod
    def _safe_eval(cls, expr: str) -> float:
        return cls._eval_node(ast.parse(expr, mode="eval").body)

    @classmethod
    def _eval_node(cls, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](cls._eval_node(node.left),
                                       cls._eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](cls._eval_node(node.operand))
        raise ValueError(f"noeud non autorise : {ast.dump(node)}")


_EXTRACTION_PROMPT = (
    "Solve the math problem step by step. "
    "On the VERY LAST LINE write ONLY the arithmetic expression that gives the "
    "final answer (numbers and +, -, *, /, **, ( ) only — no text, no variable names).\n\n"
    "Problem: {question}"
)

_STEP_HEADER = (
    "You solve a math word problem ONE calculation at a time, using a calculator.\n"
    "Problem: {question}\n"
)
_STEP_FOOTER = (
    "Write ONLY the next arithmetic expression to evaluate (numbers and + - * / ( ) ), "
    "using the results above as plain numbers. "
    "If the last result is already the final answer, reply exactly: STOP\n"
    "Next:"
)


def _parse_expression(raw: str) -> str | None:
    """Extrait la derniere ligne non vide du output SLM et nettoie les prefixes."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    for prefix in ("expression:", "answer:", "next:", "####", "=", "result:"):
        if candidate.lower().startswith(prefix.lower()):
            candidate = candidate[len(prefix):].strip()
    if " = " in candidate:
        candidate = candidate.split(" = ")[0].strip()
    return candidate or None


def _format_history(history: list[Observation]) -> str:
    """Rend les etapes deja calculees pour les reinjecter dans le prompt SLM."""
    if not history:
        return "No calculations done yet.\n"
    lines = [f"  {o.action.payload['expression']} = {o.result}" for o in history]
    return "Calculations so far:\n" + "\n".join(lines) + "\n"


@register
class CalculusExtractor:
    """Extrait l'expression arithmetique d'un enonce GSM8K via appel SLM.

    - propose/execute     : contrat de base, extraction single-shot d'une question.
    - execute_batch       : version batchee de l'extraction single-shot (perf).
    - extract_next        : etape de la strategie iterative (lit l'historique).
    """
    name = "calculus"

    def __init__(self, slm: LLMClient) -> None:
        self._slm = slm

    def propose(self, state: WorldState) -> Action | None:
        if not state.goal:
            return None
        return Action(agent=self.name, tool="extract_calculus",
                      payload={"question": state.goal})

    def execute(self, action: Action, state: WorldState) -> Observation:
        question = action.payload["question"]
        raw = self._slm.complete(_EXTRACTION_PROMPT.format(question=question))
        return self._observe(action, _parse_expression(raw))

    def execute_batch(self, questions: list[str]) -> list[Observation]:
        """Extraction single-shot batchee : un appel SLM batche pour tout le lot."""
        prompts = [_EXTRACTION_PROMPT.format(question=q) for q in questions]
        raws = self._slm.complete_batch(prompts)
        observations = []
        for q, raw in zip(questions, raws):
            action = Action(agent=self.name, tool="extract_calculus",
                            payload={"question": q})
            observations.append(self._observe(action, _parse_expression(raw)))
        return observations

    def extract_next(self, state: WorldState) -> str | None:
        """Strategie iterative : propose la prochaine expression a calculer.

        Renvoie None si le SLM signale STOP, ne produit rien, ou propose une
        expression deja calculee (anti-boucle).
        """
        prompt = (_STEP_HEADER.format(question=state.goal)
                  + _format_history(state.history)
                  + _STEP_FOOTER)
        raw = self._slm.complete(prompt, max_new_tokens=64)
        if raw.strip().upper().startswith("STOP"):
            return None
        expr = _parse_expression(raw)
        if expr is None:
            return None
        already_done = {o.action.payload["expression"] for o in state.history}
        return None if expr in already_done else expr

    @staticmethod
    def _observe(action: Action, expr: str | None) -> Observation:
        return Observation(action=action, ok=bool(expr), result=expr,
                           error=None if expr else "aucune expression valide")
