"""
Detector 2 — Empty Return Retry

Fires when a tool frequently returns empty/null results and the agent
calls the LLM again anyway — burning tokens on a context that will
never improve without fixing the tool.
"""
from __future__ import annotations
from collections import defaultdict
from monk.parsers.auto import TraceCall
from monk.pricing import cost_for_call
from .base import BaseDetector, Finding

EMPTY_RATE_THRESHOLD = 0.30   # >30% empty returns = problem
MIN_CALLS = 3                  # ignore tools with very few calls


_EMPTY_PATTERNS = {"", "none", "null", "[]", "{}", "n/a", "error", "undefined"}


def _is_empty(result: str) -> bool:
    return result.strip().lower() in _EMPTY_PATTERNS or len(result.strip()) < 3


class EmptyReturnDetector(BaseDetector):
    name = "empty_return"

    def run(self, calls: list[TraceCall]) -> list[Finding]:
        # tool_name -> (total_calls, empty_calls, total_wasted_tokens, model)
        stats: dict[str, list] = defaultdict(lambda: [0, 0, 0, "unknown"])

        for call in calls:
            for tc in call.tool_calls:
                stats[tc.name][0] += 1
                if _is_empty(tc.result):
                    stats[tc.name][1] += 1
                    stats[tc.name][2] += call.input_tokens
                stats[tc.name][3] = call.model

        findings: list[Finding] = []
        affected_sessions = list({c.session_id for c in calls})

        for tool_name, (total, empty, wasted_tokens, model) in stats.items():
            if total < MIN_CALLS:
                continue
            rate = empty / total
            if rate < EMPTY_RATE_THRESHOLD:
                continue

            waste_usd = cost_for_call(model, wasted_tokens, 0)
            findings.append(Finding(
                detector=self.name,
                severity="high" if rate > 0.5 else "medium",
                title=f"'{tool_name}' returns empty {rate:.0%} of the time",
                detail=(
                    f"Tool '{tool_name}' returned empty/null in {empty}/{total} calls "
                    f"({rate:.0%}). The agent continued processing after each empty "
                    f"result, burning ~{wasted_tokens:,} tokens for nothing."
                ),
                affected_sessions=affected_sessions,
                estimated_waste_usd_per_day=waste_usd,
                fix=(
                    f"Add a guard after calling '{tool_name}': if result is empty, "
                    f"stop the chain immediately and return a structured error. "
                    f"Don't pass empty context back to the LLM — it will hallucinate."
                ),
            ))

        return findings
