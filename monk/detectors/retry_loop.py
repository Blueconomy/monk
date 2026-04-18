"""
Detector 1 — Retry Loop

Fires when the same tool is called 3+ consecutive times within a session
without meaningful state change. Classic sign of an agent stuck retrying
instead of handling failure gracefully.
"""
from __future__ import annotations
from collections import defaultdict
from monk.parsers.auto import TraceCall
from monk.pricing import cost_for_call
from .base import BaseDetector, Finding

CONSECUTIVE_THRESHOLD = 3


class RetryLoopDetector(BaseDetector):
    name = "retry_loop"

    def run(self, calls: list[TraceCall]) -> list[Finding]:
        # Group by session, preserve order
        sessions: dict[str, list[TraceCall]] = defaultdict(list)
        for c in calls:
            sessions[c.session_id].append(c)

        findings: list[Finding] = []

        for session_id, session_calls in sessions.items():
            loops = _find_retry_loops(session_calls)
            for tool_name, count, wasted_tokens, model in loops:
                waste_usd = cost_for_call(model, wasted_tokens, 0)
                findings.append(Finding(
                    detector=self.name,
                    severity="high",
                    title=f"Retry loop: '{tool_name}' called {count}x in a row",
                    detail=(
                        f"Session '{session_id}': tool '{tool_name}' was called "
                        f"{count} consecutive times with no state change. "
                        f"Each redundant call costs ~{waste_usd / count * 100:.3f}¢. "
                        f"At this volume that adds up fast."
                    ),
                    affected_sessions=[session_id],
                    estimated_waste_usd_per_day=waste_usd,
                    fix=(
                        f"Add a max-retries guard (e.g. `if retries >= 2: raise`) "
                        f"before calling '{tool_name}'. Log the failure and let the "
                        f"agent escalate instead of spinning."
                    ),
                ))

        return findings


def _find_retry_loops(
    calls: list[TraceCall],
) -> list[tuple[str, int, int, str]]:
    """Return list of (tool_name, count, wasted_input_tokens, model)."""
    results = []

    # Flatten tool calls in order, keeping parent call metadata
    sequence: list[tuple[str, int, str]] = []  # (tool_name, input_tokens, model)
    for call in calls:
        for tc in call.tool_calls:
            sequence.append((tc.name, call.input_tokens, call.model))

    if not sequence:
        return []

    i = 0
    while i < len(sequence):
        tool = sequence[i][0]
        j = i + 1
        while j < len(sequence) and sequence[j][0] == tool:
            j += 1
        run_len = j - i
        if run_len >= CONSECUTIVE_THRESHOLD:
            # First call is legitimate; the rest are waste
            wasted = sum(tok for _, tok, _ in sequence[i + 1:j])
            model = sequence[i][2]
            results.append((tool, run_len, wasted, model))
        i = j

    return results
