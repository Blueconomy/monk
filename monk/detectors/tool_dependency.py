"""
Detector 8 — Tool Dependency Graph

Requires span-level data (OTEL format).

Analyses the parent-child span tree to build a tool dependency DAG and detects:

  A) Cycles — tool A triggers tool B which eventually re-triggers A.
     In practice this manifests as an agent stuck in an infinite tool-call loop.

  B) Redundant fan-out — the same tool is called N≥3 times as siblings under
     the same parent with identical or near-identical arguments. The agent is
     doing parallel redundant work instead of calling once and caching.

  C) Deep call chains — tool depth > 5 levels is a signal of over-engineered
     orchestration that's hard to debug and accumulates latency at every level.

The dependency graph is also exported as a JSON artifact for visualisation.
"""
from __future__ import annotations

from collections import defaultdict, deque
from monk.parsers.otel import Span
from .base import BaseDetector, Finding

FANOUT_THRESHOLD = 3      # same tool N times as siblings = redundant
MAX_CHAIN_DEPTH = 5       # tool chain deeper than this = smell

# Generic orchestration span names that appear as structural wrappers —
# not real tools. Cycles through these are false positives.
_ORCHESTRATION_PATTERNS = {
    "toolcallingagent.run", "agent.run", "agentexecutor", "codecagent.run",
    "managedagent.run", "step", "run", "executor", "planner", "main",
    "workflow", "pipeline", "chain", "tool_calling_agent",
}


class ToolDependencyDetector(BaseDetector):
    name = "tool_dependency"
    requires_spans = True

    def run_spans(self, roots: list[Span]) -> list[Finding]:
        findings: list[Finding] = []
        for root in roots:
            findings.extend(self._analyse_tree(root))
        return findings

    def run(self, calls) -> list[Finding]:  # type: ignore[override]
        return []

    def _analyse_tree(self, root: Span) -> list[Finding]:
        findings: list[Finding] = []

        # Build adjacency: parent_tool → set of child_tools
        adj: dict[str, set[str]] = defaultdict(set)
        all_spans = [root] + root.all_descendants()

        by_id: dict[str, Span] = {s.span_id: s for s in all_spans}

        for span in all_spans:
            if span.kind not in ("tool", "llm", "agent", "chain"):
                continue
            if not span.parent_span_id:
                continue
            parent = by_id.get(span.parent_span_id)
            if not parent:
                continue
            parent_name = _span_label(parent)
            child_name = _span_label(span)
            # Skip orchestration wrappers — they're structural, not real tool deps
            if (_is_orchestration(parent_name) or _is_orchestration(child_name)):
                continue
            if parent_name != child_name:
                adj[parent_name].add(child_name)

        # A) Cycle detection via DFS
        cycle = _find_cycle(adj)
        if cycle:
            cycle_str = " → ".join(cycle)
            findings.append(Finding(
                detector=self.name,
                severity="high",
                title=f"Tool cycle detected: {cycle_str}",
                detail=(
                    f"The tool dependency graph contains a cycle: {cycle_str}. "
                    f"This means the agent can call itself into an infinite loop "
                    f"if the termination condition isn't met on every path."
                ),
                affected_sessions=[root.trace_id],
                estimated_waste_usd_per_day=0.0,
                fix=(
                    "Break the cycle by introducing a visited-set or step counter. "
                    "The last tool in the cycle should check whether it's already been "
                    "called with these arguments before re-invoking the first."
                ),
            ))

        # B) Redundant fan-out — siblings with same tool name
        def check_fanout(span: Span) -> None:
            tool_children: dict[str, list[Span]] = defaultdict(list)
            for child in span.children:
                if child.kind == "tool":
                    tool_children[child.tool_name or child.name].append(child)
            for tool_name, siblings in tool_children.items():
                if len(siblings) >= FANOUT_THRESHOLD:
                    # Check if args are similar (heuristic: same first 50 chars)
                    args_set = {s.tool_args[:50] for s in siblings}
                    if len(args_set) <= max(1, len(siblings) // 2):
                        findings.append(Finding(
                            detector=self.name,
                            severity="medium",
                            title=f"Redundant fan-out: '{tool_name}' called {len(siblings)}x with similar args",
                            detail=(
                                f"Tool '{tool_name}' was called {len(siblings)} times as parallel siblings "
                                f"under '{_span_label(span)}' with similar arguments. "
                                f"This suggests the agent is not caching results and is redoing the same work."
                            ),
                            affected_sessions=[span.session_id],
                            estimated_waste_usd_per_day=0.0,
                            fix=(
                                f"Cache the result of '{tool_name}' for the duration of the agent run. "
                                f"Use a simple dict: `cache[hash(args)] = result`. "
                                f"Only call the tool once per unique argument set."
                            ),
                        ))
            for child in span.children:
                check_fanout(child)

        check_fanout(root)

        # C) Deep call chains
        max_depth, deepest_path = _max_depth(root)
        if max_depth > MAX_CHAIN_DEPTH:
            path_str = " → ".join(deepest_path)
            findings.append(Finding(
                detector=self.name,
                severity="low",
                title=f"Deep tool chain: {max_depth} levels deep",
                detail=(
                    f"The deepest tool call chain is {max_depth} levels: {path_str}. "
                    f"Deep chains accumulate latency at every level and are hard to debug. "
                    f"Each level adds network round-trips, error surface, and context overhead."
                ),
                affected_sessions=[root.trace_id],
                estimated_waste_usd_per_day=0.0,
                fix=(
                    "Flatten your agent architecture. If you need >5 levels, consider whether "
                    "intermediate orchestration steps can be collapsed. Sub-agents calling "
                    "sub-agents calling tools is usually a sign of over-engineering."
                ),
            ))

        return findings


# ── Graph utilities ───────────────────────────────────────────────────────────

def _find_cycle(adj: dict[str, set[str]]) -> list[str] | None:
    """DFS cycle detection. Returns cycle path or None."""
    visited: set[str] = set()
    path: list[str] = []
    path_set: set[str] = set()

    def dfs(node: str) -> list[str] | None:
        if node in path_set:
            idx = path.index(node)
            return path[idx:] + [node]
        if node in visited:
            return None
        visited.add(node)
        path.append(node)
        path_set.add(node)
        for neighbour in adj.get(node, set()):
            result = dfs(neighbour)
            if result:
                return result
        path.pop()
        path_set.discard(node)
        return None

    for node in list(adj.keys()):
        if node not in visited:
            result = dfs(node)
            if result:
                return result
    return None


def _max_depth(root: Span, depth: int = 0) -> tuple[int, list[str]]:
    """Return (max_depth, path_to_deepest)."""
    label = _span_label(root)
    if not root.children:
        return depth, [label]
    best_depth = depth
    best_path = [label]
    for child in root.children:
        d, p = _max_depth(child, depth + 1)
        if d > best_depth:
            best_depth = d
            best_path = [label] + p
    return best_depth, best_path


def _span_label(s: Span) -> str:
    return s.tool_name or s.model or s.name or s.span_id[:8]


def _is_orchestration(label: str) -> bool:
    """Return True if label is a generic orchestration wrapper (not a real tool)."""
    clean = label.lower().replace(" ", "").replace("_", "").replace(".", "")
    # Match exact patterns or step-N patterns like "step1", "step2"
    import re
    if re.match(r'^step\d+$', clean):
        return True
    return clean in {p.replace(".", "").replace("_", "") for p in _ORCHESTRATION_PATTERNS}


def _collect_all_spans(roots: list[Span]) -> list[Span]:
    result: list[Span] = []
    def walk(s: Span) -> None:
        result.append(s)
        for child in s.children:
            walk(child)
    for root in roots:
        walk(root)
    return result
