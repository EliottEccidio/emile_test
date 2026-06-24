# CLAUDE.md

Behavioral guardrails to reduce common LLM coding mistakes. They bias toward **caution over speed** — for trivial tasks, use judgment. Merge with project-specific instructions.

## 0. Project facts (fill in, or discover them first)
- Run tests: `<cmd>`   ·   Lint/format: `<cmd>`   ·   Build/run: `<cmd>`
- Stack & conventions: `<e.g. Python 3.11, pytest, type hints, ruff>`
- **Always run the test command after a change and report pass/fail.**
- If these are unknown: first MAP the repo (entry points, how to run, how to test) and **read before editing**.

## 1. Think before coding
Don't assume. Don't hide confusion. Surface trade-offs.
- State assumptions explicitly; if uncertain, ask **one** focused question.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what's confusing, ask.

## 2. Simplicity first
Minimum code that solves the problem. Nothing speculative.
- No features, abstractions, configurability, or error handling beyond what's asked.
- Single use ⇒ no abstraction. If you wrote 200 lines and it could be 50, rewrite.
- Test: *"Would a senior engineer call this overcomplicated?"* If yes, simplify.

## 3. Understand before changing; change surgically
- Read the relevant code and conventions **before** editing. Match existing style, even if you'd do it differently.
- Touch only what the request requires. Don't "improve", refactor, or reformat what isn't broken.
- Remove only the imports/vars/functions **your** change orphaned. Mention pre-existing dead code — don't delete it.
- The test: every changed line traces directly to the user's request.

## 4. Code quality & design (clean code + pragmatic DDD)
- **Names carry intent** — use the **domain's vocabulary** (ubiquitous language): `coverage`, `frontier`, `finding`, `scope`, not `data`, `tmp`, `mgr`.
- **Small functions, single responsibility.** Separate pure logic from IO/side-effects.
- **Isolate the domain from infrastructure** (LLM, DB, network, time) behind interfaces → testable + swappable. **Inject** dependencies; don't hardcode a vendor SDK in the logic.
- **Model the domain with explicit types** (dataclasses / enums) over loose dicts and magic strings.
- **Pragmatic DDD only**: ubiquitous language + domain/infra separation. Do **NOT** add layers, repositories, or aggregates the task doesn't need — that violates §2.

## 5. Goal-driven execution (define success, loop until verified)
Turn tasks into verifiable goals:
- "Add validation" → write tests for invalid inputs, then make them pass.
- "Fix the bug" → write a failing test that reproduces it, then make it pass.
- "Refactor X" → ensure tests pass before and after.

For multi-step tasks, state a brief plan with a check per step:
```
1. [step] → verify: [check]
2. [step] → verify: [check]
```
Strong success criteria let you loop independently. Weak ones ("make it work") cause churn.

## 6. Communication
- Keep a running TODO: **DONE / NEXT / DEFERRED**. Flag risks, assumptions, and out-of-scope items explicitly.
- On a real trade-off: give options **A/B** in one line and **recommend** one.

---
**Working if:** smaller diffs, fewer rewrites from overcomplication, clarifying questions come **before** mistakes, and tests run after every change.
