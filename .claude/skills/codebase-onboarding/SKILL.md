---
name: codebase-onboarding
description: Use at the START of working on an unfamiliar or existing codebase, before making changes. Rapidly map the project (entry points, run/test commands, module structure, where the relevant logic lives, conventions) and propose the smallest vertical slice to make progress. Do not modify code while onboarding.
---
# Codebase onboarding — map before you touch

When dropped into an unfamiliar repo (especially a time-boxed, larger-than-the-time task):

## 1. Map (read-only, ~5 min) — change NOTHING
Report, concisely:
- **Entry points** + how to run the app, and **how to run the tests** (the exact command).
- **Module structure**: a short tree of the main packages and how they connect.
- **Where the logic for the task lives**: the files + key functions to touch.
- **Conventions**: style, patterns, test layout, typing — so changes match them.

## 2. Restate understanding
Summarize the architecture in 3-4 lines and confirm with the user. Surface unknowns/assumptions.

## 3. Scope the unfinishable
Identify the **smallest vertical slice** that delivers value end-to-end. State: "I'll do [slice] first, then extend." List what you DEFER (with a one-line TODO each). Explicit prioritization > trying to finish everything.

## 4. Plan in small verifiable steps
Each step independently testable. Keep the system runnable at every step. Read a test before implementing to know what's actually required.

## Rules
- Read before edit; respect existing style.
- Touch only what the task requires.
- Run the test command after each small change.
- Narrate trade-offs out loud (A/B -> recommend one).
