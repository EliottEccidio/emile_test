---
name: agentic-systems
description: Use when building or extending agent loops, a supervisor/orchestrator, an agent roster, routing, or coordinating multiple agents. Provides the control-loop skeleton, the agent contract, failure-mode guardrails, and architecture-choice heuristics.
---
# Agentic systems — patterns & guardrails

## Control loop (supervisor)
Inject the LLM behind an interface (`LLMClient.complete`) so it's mockable. State lives in a plain `WorldState` dataclass; the `Supervisor` acts ON it (no business logic in the supervisor).
Each tick: `should_stop?` -> route -> validate(scope) -> dedup -> execute -> update (budget, step, history, findings). Always `return state`. Termination is guaranteed by budget + max_steps.

## Agent contract
`propose(state, llm) -> Action | None` = PLAN (no side-effects). `execute(action, state) -> Observation` = ACT (side-effects here only). Never mutate the agent inside `propose` (the router calls propose to select, then run calls it again).

## Roster extensibility
Common `Agent` Protocol + a registry (`@register` adds the class to `AGENT_REGISTRY` by its class attribute `name`). Add agents without touching the supervisor (open/closed). `name` must be a CLASS attribute.

## Routing (WHO) vs scheduling (WHEN / parallel)
Router matches ONE task -> best agent (capabilities / scoring / LLM-as-router; ALWAYS a deterministic fallback). Scheduler orders the WHOLE queue (priority + aging anti-starvation) and decides sequential (dependency) vs parallel (independent + budget). Push (routing) when control/global view needed; pull (events) when local/parallel/scalable. Push/pull is a design choice per phase, not a per-tick computation.

## Failure modes — always guard
infinite loop (budget+steps+oracle) · tool thrashing (dedup on canonical `action.key()`) · hallucinated plan / scope creep (validate allow-list + scope BEFORE execute — scope is an INVARIANT, never delegated to the LLM) · duplicate work (claims/locks) · deadlock (lock ordering + timeout + wait-for-graph cycle detection) · context explosion (summarize) · cascading failure (circuit breaker + sandbox) · reward hacking (multiple metrics).

## Checklist before "done"
[ ] LLM injected/mockable [ ] budget + stop oracle [ ] dedup [ ] scope validation [ ] tests for invariants (terminates, never out of scope, no duplicate).
