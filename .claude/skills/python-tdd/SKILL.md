---
name: python-tdd
description: Use when writing Python with tests, especially test-first (TDD) and testing non-deterministic or agentic code. Covers pytest patterns, invariant tests, dependency injection and mocking.
---
# Python TDD patterns

## Test-first loop
1) Write a failing test (happy path + 1-2 edges). Run -> RED. 2) Minimum code -> GREEN. 3) Refactor, tests stay green. Run the relevant test after EACH small change, not all at the end (catch bugs early, faster).

## Testing agentic / non-deterministic code
Inject the LLM behind an interface; use a deterministic `FakeLLM` (scripted outputs). Test INVARIANTS, not exact transcripts: terminates (<= max_steps), budget respected, never out of scope, no duplicate actions, finding recorded when present.

## Hygiene
Small pure functions; dependency injection over hardcoded IO; dataclasses + type hints; `from __future__ import annotations`. READ the test to know what's actually required — don't over-engineer (e.g. if a test compares as a `set`, ordering doesn't matter, so no `sorted`).
