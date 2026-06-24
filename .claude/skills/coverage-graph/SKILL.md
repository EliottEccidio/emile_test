---
name: coverage-graph
description: Use when building or querying a graph that tracks coverage/exploration (endpoints, params, auth contexts, hosts, attack paths). Covers adjacency lists, BFS/DFS/Dijkstra, frontier/coverage, gap-detection, merge across runs, and the max-probability path trick.
---
# Coverage graph — patterns

## Structure
Sparse + dynamic -> adjacency list `dict[node, set]` (+ reverse adjacency `radj` for predecessors). `@dataclass` for Node/Edge. `enum.Enum` for states (DISCOVERED -> SCANNED -> TESTED -> EXPLOITED -> CLEARED). NOT an adjacency matrix.

## Core ops
- `frontier()` = nodes still DISCOVERED (what's left to do).
- `coverage()` = sum(value of nodes with state >= TESTED) / sum(all values), weighted; `0.0` if empty; must be monotonic. Compare enum via `.value`.
- `most_valuable_frontier(k)` = top-k by `value * (1 + len(radj[node]))` via `heapq.nlargest(k, frontier, key=...)`.
- `bfs(src)` = `collections.deque` (popleft O(1); list.pop(0) is O(n)).
- `merge(g1, g2)` = union nodes, MAX state, dedup findings; CANONICALIZE ids first (URL templating: /users/123 == /users/{id}).
- gap-detection = untested cells ranked by criticality.

## Shortest attack path
Dijkstra on a min-heap (`heapq`). `mode="cost"` -> sum of costs. `mode="proba"` -> MAXIMIZE product of probs = MINIMIZE sum of `-log(p)` (p<=1 so -log p >= 0); weight = `-math.log(max(p, 1e-9))`. Recover product = `exp(-dist)`. NEVER sum raw probabilities. Reconstruct path via `prev` pointers.

## Webapp framing
Nodes = endpoint / param / auth_context / finding. Coverage is multi-dimensional: (endpoint x param x auth_context x exploit_class). Exhaustiveness score (-> audit PDF) = weighted coverage + frontier-empty + Chao1 uncertainty. In prod: networkx; in an interview, hand-roll to show understanding and mention networkx.
