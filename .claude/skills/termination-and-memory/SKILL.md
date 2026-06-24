---
name: termination-and-memory
description: Use when deciding when an agentic run is done (termination oracle, stopping criteria, convergence, exhaustiveness scoring) or building a memory layer (dedup, retrieval, summarization, eviction) so long runs don't repeat work.
---
# Termination oracle & memory layer

## Termination oracle (composite, conservative)
Hard rules short-circuit (budget / max_steps / goal reached). Graded signals vote, weighted: no-progress (PATIENCE/hysteresis — don't stop on one flat step; re-arm on a new finding), coverage saturation (coverage >= threshold AND frontier empty), uncertainty (Chao1). Stop if aggregate >= HIGH threshold (~0.85) — conservative, because a false negative (missed vuln) costs more than over-testing. Each signal carries a `reason` (explainable -> audit PDF). Acts at levels: action / agent / branch / phase / global. "Done at a level" = that scope's frontier is empty (e.g. agent.propose -> None).

## Chao1 (estimate the unknown)
`S_est = S_obs + f1^2 / (2*f2)`; f1 = findings seen once, f2 = seen twice (the memory `seen` counter). Many singletons -> keep going; saturated -> candidate to stop. Combine with structural gap-detection (Chao1 is blind to classes you never test).

## Memory layer
working = `deque(maxlen=N)` ring buffer (bounded, auto-evicts oldest). episodic = append-only log (audit). semantic = deduped facts = the coverage graph (STRUCTURED -> KV/graph, NOT vectors; vectors only for free text). long-term = Postgres (persist + resume + merge).
Triggers: before a decision -> retrieve relevant + dedup-check ("already done?"); after an observation -> store + dedup (exact + near-dup -> `seen += 1`); on overflow -> summarize old, keep recent; at capacity -> evict (LRU / importance-decay); on checkpoint -> persist; at run end -> consolidate episodic->semantic + merge across pentests.
