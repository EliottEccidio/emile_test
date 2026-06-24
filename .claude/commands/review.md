Review the current diff (or "$ARGUMENTS") as a senior engineer. Be concise. Check:
- Correctness & edge cases (empty input, None, boundaries).
- Tests: is the new logic covered? What's missing?
- Simplicity: anything overcomplicated or speculative to cut?
- Domain/infra separation: is logic testable, dependencies injected?
- Scope: any change that doesn't trace to the request?
Output: a short bullet list of concrete issues, highest-impact first. Do NOT edit code.
