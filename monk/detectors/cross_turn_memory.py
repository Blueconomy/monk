"""
Detector 9 — Cross-Turn Memory (Redundant Re-fetch)

Requires span-level data (OTEL format).

Detects when the agent calls the same tool with the same (or very similar)
arguments across multiple turns in the same session — a clear signal that
results are not being cached between turns.

Common in:
  - RAG pipelines that re-embed and re-retrieve on every turn
  - Voice AI agents that re-fetch user profile on every utterance
  - Customer support bots that re-query order status every reply

The waste is doubled: you pay for the tool call AND for the extra tokens
from passing the same retrieved context back into the LLM every turn.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from monk.parsers.otel import Span
from monk.pricing import cost_for_call
from .base import BaseDetector, Finding

# How similar do args need to be to count as "same call" (0-1, 1=identical)
SIMILARITY_THRESHOLD = 0.8
MIN_REDUNDANT_CALLS = 2     # same tool+args seen this many extra times = flag


class CrossTurnMemoryDetector(BaseDetector):
    name = "cross_turn_memory"
    requires_spans = True

    def run_spans(self, roots: list[Span]) -> list[Finding]:
        findings: list[Finding] = []

        for root in roots:
            # Collect all tool spans in this trace in temporal order
            tool_spans = _collect_tool_spans_ordered(root)
            if not tool_spans:
                continue

            # Group by tool name → list of (args_hash, args_preview, span)
            by_tool: dict[str, list[tuple[str, str, Span]]] = defaultdict(list)
            for span in tool_spans:
                name = span.tool_name or span.name
                args_hash = _hash_args(span.tool_args)
                args_preview = span.tool_args[:80] if span.tool_args else "(no args)"
                by_tool[name].append((args_hash, args_preview, span))

            for tool_name, calls in by_tool.items():
                if len(calls) < MIN_REDUNDANT_CALLS + 1:
                    continue

                # Count how many times each unique arg hash appears
                hash_counts: dict[str, list[tuple[str, Span]]] = defaultdict(list)
                for args_hash, args_preview, span in calls:
                    hash_counts[args_hash].append((args_preview, span))

                for args_hash, instances in hash_counts.items():
                    if len(instances) < MIN_REDUNDANT_CALLS + 1:
                        continue

                    # First call is legitimate; the rest are redundant
                    redundant = instances[1:]
                    args_preview = instances[0][0]
                    affected_spans = [s for _, s in instances]

                    # Estimate waste: tokens spent on redundant retrieval + re-injecting context
                    # Proxy: use input tokens of the LLM call following each redundant tool call
                    wasted_tokens = sum(
                        _tokens_of_next_llm(s, root) for _, s in redundant
                    )
                    model = _nearest_llm_model(root)
                    waste_usd = cost_for_call(model, wasted_tokens, 0) if wasted_tokens else 0.0

                    findings.append(Finding(
                        detector=self.name,
                        severity="high" if len(redundant) >= 3 else "medium",
                        title=(
                            f"'{tool_name}' re-fetched {len(redundant)}x with same args — "
                            f"no cross-turn caching"
                        ),
                        detail=(
                            f"Tool '{tool_name}' was called {len(instances)} times in trace "
                            f"'{root.trace_id[:12]}' with identical arguments "
                            f"(args: {args_preview!r}). "
                            f"The result from the first call was never reused. "
                            f"Each redundant call re-injects the same context into the LLM."
                        ),
                        affected_sessions=[root.trace_id],
                        estimated_waste_usd_per_day=waste_usd,
                        fix=(
                            f"Cache '{tool_name}' results keyed by (tool_name, args_hash) for the "
                            f"lifetime of the conversation. A simple in-memory dict is sufficient: "
                            f"`cache = {{}}; result = cache.setdefault(key, fetch(args))`. "
                            f"For multi-user services, use Redis with a TTL matching your data freshness needs."
                        ),
                    ))

        return findings

    def run(self, calls) -> list[Finding]:  # type: ignore[override]
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_tool_spans_ordered(root: Span) -> list[Span]:
    """Collect all tool spans sorted by start time."""
    result: list[Span] = []
    def walk(s: Span) -> None:
        if s.kind == "tool":
            result.append(s)
        for child in s.children:
            walk(child)
    walk(root)
    result.sort(key=lambda s: s.start_time_ns)
    return result


def _hash_args(args: str) -> str:
    """Normalise and hash tool arguments for comparison."""
    # Try to parse JSON and re-serialise for canonical form
    try:
        parsed = json.loads(args)
        normalised = json.dumps(parsed, sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        normalised = args.strip().lower()
    return hashlib.md5(normalised.encode()).hexdigest()  # noqa: S324 — not crypto


def _tokens_of_next_llm(tool_span: Span, root: Span) -> int:
    """Find the LLM span that immediately follows this tool span and return its input tokens."""
    all_spans: list[Span] = []
    def walk(s: Span) -> None:
        all_spans.append(s)
        for child in s.children:
            walk(child)
    walk(root)
    all_spans.sort(key=lambda s: s.start_time_ns)

    found = False
    for s in all_spans:
        if found and s.kind == "llm":
            return s.input_tokens
        if s.span_id == tool_span.span_id:
            found = True
    return 0


def _nearest_llm_model(root: Span) -> str:
    """Return the first LLM model name found in the tree."""
    def walk(s: Span) -> str:
        if s.kind == "llm" and s.model:
            return s.model
        for child in s.children:
            result = walk(child)
            if result:
                return result
        return ""
    return walk(root) or "gpt-4o-mini"
