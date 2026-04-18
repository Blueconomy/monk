"""
Detector 5 — Agent Loop

Detects when an agent cycles through the same sequence of steps repeatedly
without making progress — a sign the agent is stuck in a reasoning loop.

Heuristic: within a session, if the same tool-call sequence (A→B or A→B→C)
appears 3+ times, the agent is looping.
"""
from __future__ import annotations
from collections import defaultdict
from monk.parsers.auto import TraceCall
from monk.pricing import cost_for_call
from .base import BaseDetector, Finding

LOOP_THRESHOLD = 3       # pattern repeated this many times → loop
SEQUENCE_LENGTH = 2      # look for repeated pairs of consecutive tool calls


class AgentLoopDetector(BaseDetector):
    name = "agent_loop"

    def run(self, calls: list[TraceCall]) -> list[Finding]:
        sessions: dict[str, list[TraceCall]] = defaultdict(list)
        for c in calls:
            sessions[c.session_id].append(c)

        findings: list[Finding] = []

        for sid, scalls in sessions.items():
            loops = _detect_loops(scalls)
            for pattern, count, wasted_tokens, model in loops:
                waste_usd = cost_for_call(model, wasted_tokens, 0)
                pattern_str = " → ".join(pattern)
                findings.append(Finding(
                    detector=self.name,
                    severity="high",
                    title=f"Agent loop: [{pattern_str}] repeated {count}x",
                    detail=(
                        f"Session '{sid}': the tool sequence [{pattern_str}] "
                        f"repeated {count} times. The agent is not making progress — "
                        f"it's cycling through the same steps and burning tokens."
                    ),
                    affected_sessions=[sid],
                    estimated_waste_usd_per_day=waste_usd,
                    fix=(
                        f"Add a step-deduplication guard: track which (tool, args) pairs "
                        f"have been called this session and short-circuit if the same "
                        f"sequence appears again. Consider a max_steps limit at the "
                        f"orchestration level."
                    ),
                ))

        return findings


def _detect_loops(
    calls: list[TraceCall],
) -> list[tuple[tuple[str, ...], int, int, str]]:
    """Return list of (pattern_tuple, repeat_count, wasted_tokens, model)."""
    # Build flat sequence of tool names in call order
    tool_seq: list[tuple[str, int, str]] = []  # (tool_name, input_tokens, model)
    for call in calls:
        for tc in call.tool_calls:
            tool_seq.append((tc.name, call.input_tokens, call.model))

    if len(tool_seq) < SEQUENCE_LENGTH * LOOP_THRESHOLD:
        return []

    results = []
    names = [t[0] for t in tool_seq]

    # Check for repeated N-grams
    for n in range(1, SEQUENCE_LENGTH + 1):
        ngram_counts: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i in range(len(names) - n + 1):
            gram = tuple(names[i:i + n])
            ngram_counts[gram].append(i)

        for gram, positions in ngram_counts.items():
            if len(positions) >= LOOP_THRESHOLD:
                # Only count positions after the first occurrence as waste
                wasted_positions = positions[1:]
                wasted_tokens = sum(
                    sum(tool_seq[p + k][1] for k in range(n) if p + k < len(tool_seq))
                    for p in wasted_positions
                )
                model = tool_seq[positions[0]][2]
                results.append((gram, len(positions), wasted_tokens, model))

    # De-duplicate: if A→B is flagged, don't also flag A and B individually
    results.sort(key=lambda x: -len(x[0]))
    seen_positions: set[int] = set()
    deduped = []
    for item in results:
        if item not in deduped:
            deduped.append(item)

    return deduped
