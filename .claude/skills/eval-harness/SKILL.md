---
name: eval-harness
description: Use when building or discussing an evaluation/benchmark harness for an agent or ML system — ground-truth scenarios, recall/precision/cost metrics, regression detection per release, anti-cheat.
---
# Eval harness patterns

## Offline eval
Scenarios = fixtures with ground-truth (e.g. intentionally vulnerable targets with known findings). Run the system, compute: recall (|found & gt| / |gt|), precision (|found & gt| / |found|), steps, cost, safety violations (out-of-scope must be 0). MULTIPLE metrics, never one (Goodhart). Baseline vs random/heuristic. Several seeds (non-deterministic).

## Regression detection
Golden set + per-commit run; fail CI if recall drops or cost rises > X% vs last green ("did this commit regress us against the ground-truth set"). Golden traces for stable cases. LLM-as-judge for free-form outputs, calibrated against humans.

## Anti-cheat
Hidden/holdout targets so agents can't memorize ground truth; rotate scenarios; flag suspiciously fast/perfect runs; split train vs eval targets BY TARGET (not by step) to avoid leakage.
