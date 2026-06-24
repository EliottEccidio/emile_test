"""Superviseur : la boucle de controle du systeme multi-agent.

Le superviseur AGIT sur le WorldState ; il ne porte pas de logique metier
(celle-ci vit dans les agents). A chaque tick :

    oracle.should_stop ? -> router (QUI) -> valider le scope -> dedup
    -> execute -> mettre a jour l'etat.

Le LLM est injecte derriere LLMClient : ici une version MOCKEE, remplacable par
un vrai client sans toucher a la boucle.
"""
from __future__ import annotations

from agent import AGENT_REGISTRY, Agent, LLMClient
from memory import Action, Finding, Observation, WorldState



class MockSupervisorLLM:
    """LLM superviseur mocke : indique un agent a privilegier pour l'objectif.

    A remplacer par un vrai client (meme interface LLMClient). Renvoie une chaine
    vide quand il n'a pas d'avis -> le superviseur retombe sur un routage
    deterministe via propose().
    """
    def __init__(self, hint: str = "") -> None:
        self._hint = hint

    def complete(self, prompt: str) -> str:
        return self._hint


class Oracle:
    """Oracle de terminaison : decide quand le run est termine.

    Definition minimale a etoffer (convergence / exhaustivite). La garantie de
    terminaison repose sur : objectif atteint OU budget epuise OU max_steps.
    """
    def should_stop(self, state: WorldState) -> bool:
        return state.done or state.budget <= 0 or state.step >= state.max_steps


class Supervisor:
    """Orchestrateur : charge les agents du registre et fait tourner la boucle."""

    def __init__(self, llm: LLMClient, oracle: Oracle | None = None) -> None:
        self.llm = llm
        self.oracle = oracle or Oracle()
        # un exemplaire de chaque agent enregistre (open/closed : zero couplage
        # au superviseur, il decouvre les agents via le registre).
        self.agents: dict[str, Agent] = {
            name: cls() for name, cls in AGENT_REGISTRY.items()
        }

    def run(self, state: WorldState) -> WorldState:
        while not self.oracle.should_stop(state):
            action = self._route(state)
            if action is None:
                break                                  # aucun agent ne propose
            if not self._in_scope(action):
                state.record(Observation(action, ok=False, error="hors scope"))
                break                                  # action invalide
            if state.seen(action):
                break                                  # deja jouee : aucun progres
            obs = self.agents[action.agent].execute(action, state)
            state.record(obs)
            if obs.ok:
                state.add_finding(Finding(source=action.agent, content=obs.result))
        return state

    # --- routage : QUI agit ---
    def _route(self, state: WorldState) -> Action | None:
        """Avis du LLM (mocke) puis repli deterministe : on rend la 1re action
        proposee par les agents, dans l'ordre suggere par le LLM."""
        hint = self.llm.complete(state.goal).strip()
        for agent in self._order_agents(hint):
            action = agent.propose(state, self.llm)
            if action is not None:
                return action
        return None

    def _order_agents(self, hint: str) -> list[Agent]:
        agents = list(self.agents.values())
        if hint in self.agents:
            preferred = self.agents[hint]
            agents = [preferred] + [a for a in agents if a is not preferred]
        return agents

    # --- scope : INVARIANT, jamais delegue au LLM ---
    def _in_scope(self, action: Action) -> bool:
        agent = self.agents.get(action.agent)
        return agent is not None and action.tool in agent.tools


if __name__ == "__main__":
    llm = MockSupervisorLLM(hint="python_calculator")
    state = WorldState(goal="2 + 3 * (4 - 1)")
    Supervisor(llm).run(state)
    for finding in state.findings:
        print(f"[{finding.source}] {state.goal} = {finding.content}")
    print(f"steps={state.step} budget_restant={state.budget} "
          f"findings={len(state.findings)}")
