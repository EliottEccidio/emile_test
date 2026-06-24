"""Roster d'agents : contrat commun + registre extensible + 2 agents concrets.

Strategie Program-of-Thoughts (PoT) :
  - ProgramWriter (SLM)     : ecrit un petit programme Python qui resout l'enonce.
  - PythonCalculator (pur)  : execute ce programme de facon SURE et rend `answer`.

Le modele garde son raisonnement (il structure le calcul en code) ; Python fait
l'arithmetique exacte. C'est ce qui permet a l'agentique de battre le SLM seul.

Contrat (point d'extension) :
  - propose(state) -> Action | None       : le PLAN, sans effet de bord.
  - execute(action, state) -> Observation  : l'ACT, seul endroit a effet de bord.
On ajoute un agent SANS toucher au reste : @register le range dans AGENT_REGISTRY
sous son attribut de CLASSE `name`.
"""
from __future__ import annotations

import ast
import operator
import re
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


# Operateurs et fonctions autorises : c'est l'allow-list qui definit le scope.
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
_SAFE_FUNCS = {
    "min": min, "max": max, "abs": abs, "round": round,
    "int": int, "float": float, "sum": sum,
}


@register
class PythonCalculator:
    """Execute SUREMENT un petit programme Python arithmetique et rend `answer`.

    Scope (invariant verifie dans propose) : uniquement des affectations
    (`x = ...`, `x += ...`) dont les expressions combinent nombres, variables
    deja definies, operateurs arithmetiques et un petit jeu de fonctions sures
    (min/max/abs/round/int/float/sum). Aucun import, attribut, appel arbitraire :
    on interprete l'AST a la main (pas de exec/eval).
    """
    name = "python_calculator"

    def propose(self, state: WorldState) -> Action | None:
        program = state.goal
        if not self._is_safe_program(program):
            return None
        return Action(agent=self.name, tool="run_program",
                      payload={"program": program})

    def execute(self, action: Action, state: WorldState) -> Observation:
        program = action.payload["program"]
        try:
            value = self._run(program)
        except Exception as exc:  # variable indefinie, 1/0, overflow...
            return Observation(action=action, ok=False, error=str(exc))
        if value is None:
            return Observation(action=action, ok=False, error="aucune variable `answer`")
        return Observation(action=action, ok=True, result=value)

    # --- coeur de domaine : pur et testable, sans IO ni exec ---
    @classmethod
    def _is_safe_program(cls, program: Any) -> bool:
        if not isinstance(program, str) or not program.strip():
            return False
        try:
            tree = ast.parse(program, mode="exec")
        except SyntaxError:
            return False
        return all(cls._node_allowed(node) for node in ast.walk(tree))

    @staticmethod
    def _node_allowed(node: ast.AST) -> bool:
        if isinstance(node, (ast.Module, ast.Assign, ast.AugAssign, ast.Expr,
                             ast.Load, ast.Store, ast.BinOp, ast.UnaryOp,
                             ast.Name, ast.List, ast.Tuple)):
            return True
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        if isinstance(node, ast.Call):
            return (isinstance(node.func, ast.Name)
                    and node.func.id in _SAFE_FUNCS and not node.keywords)
        return isinstance(node, tuple(_OPS))  # noeud operateur autorise

    @classmethod
    def _run(cls, program: str) -> float | None:
        tree = ast.parse(program, mode="exec")
        env: dict[str, Any] = {}
        last: str | None = None
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                value = cls._eval(stmt.value, env)
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
                        last = target.id
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                env[name] = _OPS[type(stmt.op)](env[name], cls._eval(stmt.value, env))
                last = name
            # les Expr (expressions nues) sont ignorees : pas d'effet de bord
        if "answer" in env:
            return env["answer"]
        return env[last] if last is not None else None

    @classmethod
    def _eval(cls, node: ast.AST, env: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"variable non definie : {node.id}")
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [cls._eval(elt, env) for elt in node.elts]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](cls._eval(node.left, env),
                                       cls._eval(node.right, env))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](cls._eval(node.operand, env))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return _SAFE_FUNCS[node.func.id](*[cls._eval(a, env) for a in node.args])
        raise ValueError(f"noeud non autorise : {ast.dump(node)}")


_POT_PROMPT = (
    "You are given a math word problem. Write a short Python program that computes "
    "the answer.\n"
    "Rules:\n"
    "- Use only +, -, *, /, //, %, ** and the functions min, max, abs, round, int, sum.\n"
    "- One step per line with simple variables. No imports, no input(), no other "
    "functions.\n"
    "- Store the final numeric result in a variable named `answer`.\n"
    "- Output ONLY the program inside a ```python code block.\n\n"
    "Example\n"
    "Problem: Natalia sold clips to 48 friends in April, then sold half as many in "
    "May. How many clips did she sell altogether?\n"
    "```python\n"
    "april = 48\n"
    "may = april / 2\n"
    "answer = april + may\n"
    "```\n\n"
    "Problem: {question}\n"
)

_ASSIGN = re.compile(r"^\s*[A-Za-z_]\w*\s*(?://|\*\*|[-+*/%])?=(?!=)")


def _extract_code(raw: str) -> str | None:
    """Recupere le programme : contenu du bloc ```...``` sinon lignes d'affectation."""
    text = raw.strip()
    if "```" in text:
        block = text.split("```")[1]
        first, _, rest = block.partition("\n")
        if first.strip().lower() in ("python", "py"):
            block = rest
        text = block.split("```")[0].strip()
    # ne garder que les lignes qui sont des affectations (retire la prose)
    lines = [line for line in text.splitlines() if _ASSIGN.match(line)]
    code = "\n".join(lines).strip()
    return code or None


@register
class ProgramWriter:
    """Demande au SLM d'ecrire un programme Python resolvant l'enonce (PoT)."""
    name = "program_writer"

    def __init__(self, slm: LLMClient) -> None:
        self._slm = slm

    def propose(self, state: WorldState) -> Action | None:
        if not state.goal:
            return None
        return Action(agent=self.name, tool="write_program",
                      payload={"question": state.goal})

    def execute(self, action: Action, state: WorldState) -> Observation:
        code = self.write_program_batch([action.payload["question"]])[0]
        return Observation(action=action, ok=bool(code), result=code,
                           error=None if code else "aucun programme genere")

    def write_program_batch(self, questions: list[str]) -> list[str | None]:
        """Ecrit un programme par question (un seul appel SLM batche)."""
        prompts = [_POT_PROMPT.format(question=q) for q in questions]
        raws = self._slm.complete_batch(prompts)
        return [_extract_code(raw) for raw in raws]
