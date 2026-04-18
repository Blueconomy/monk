"""
Detector 7 — Error Cascade

Requires span-level data (OTEL format).

Fires when a tool or LLM span has status=error and the parent span
(or sibling spans after the error) continues making LLM calls — burning
tokens on a context that will never produce a useful result.

Two patterns detected:
  A) Error → continued LLM calls: a tool fails but the agent keeps thinking
     and calling the LLM rather than surfacing the error or using a fallback.
  B) Error propagation: an error in a child span causes the parent to fail,
     which causes the grandparent to re-try from scratch (wasteful retry).

Why it matters: in production agents, error handling is the #1 source of
unexpected cost spikes. A single unhandled tool error can triple token usage
because the agent hallucinates in a loop trying to recover.
"""
from __future__ import annotations

from collections import defaultdict
from monk.parsers.otel import Span
from monk.pricing import cost_for_call
from .base import BaseDetector, Finding

# How many LLM calls after an error = definitely not handled
LLM_CALLS_AFTER_ERROR_THRESHOLD = 1


class ErrorCascadeDetector(BaseDetector):
    name = "error_cascade"
    requires_spans = True

    def run_spans(self, roots: list[Span]) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_continued_after_error(roots))
        findings.extend(self._check_propagation_chains(roots))
        return findings

    def run(self, calls) -> list[Finding]:  # type: ignore[override]
        return []

    # ── Pattern A: LLM calls continue after error ─────────────────────────────

    def _check_continued_after_error(self, roots: list[Span]) -> list[Finding]:
        """
        Within each agent run (trace), find error spans and count LLM calls
        that occur after them in the same parent context.
        """
        findings: list[Finding] = []

        def analyse_children(parent: Span) -> None:
            children = parent.children
            for i, span in enumerate(children):
                if span.status != "error":
                    # Recurse into non-error spans
                    analyse_children(span)
                    continue

                # Found an error span — look at subsequent siblings
                subsequent = children[i + 1:]
                llm_after = [s for s in subsequent if s.kind == "llm"]
                if len(llm_after) <= LLM_CALLS_AFTER_ERROR_THRESHOLD:
                    analyse_children(span)
                    continue

                wasted_tokens = sum(s.input_tokens + s.output_tokens for s in llm_after)
                worst_model = max(llm_after, key=lambda s: s.input_tokens).model or "unknown"
                waste_usd = cost_for_call(worst_model, wasted_tokens, 0)

                error_name = span.tool_name or span.name
                findings.append(Finding(
                    detector=self.name,
                    severity="high",
                    title=f"Error cascade: '{error_name}' failed → {len(llm_after)} LLM call(s) wasted",
                    detail=(
                        f"Span '{error_name}' errored (msg: {span.error_message or 'no message'}) "
                        f"but the agent made {len(llm_after)} more LLM call(s) afterward in the same "
                        f"context, burning ~{wasted_tokens:,} tokens on a poisoned context. "
                        f"The LLM cannot fix a tool error by thinking harder about it."
                    ),
                    affected_sessions=[span.session_id],
                    estimated_waste_usd_per_day=waste_usd,
                    fix=(
                        f"Add an error guard after calling '{error_name}': check the result status "
                        f"before passing it to the LLM. On error, either raise immediately, use a "
                        f"fallback tool, or return a structured error message to the user. "
                        f"Never let an error silently propagate into context."
                    ),
                ))
                # Still recurse into the error span's own children
                analyse_children(span)

        for root in roots:
            analyse_children(root)

        return findings

    # ── Pattern B: error propagation chains ──────────────────────────────────

    def _check_propagation_chains(self, roots: list[Span]) -> list[Finding]:
        """
        Find chains where error propagates up: child error → parent errors →
        grandparent retries from scratch.
        """
        findings: list[Finding] = []

        # Collect all error spans and check if their parent is also an error
        all_spans = _collect_all_spans(roots)
        by_id: dict[str, Span] = {s.span_id: s for s in all_spans}

        # Group by trace_id to find retry patterns
        by_trace: dict[str, list[Span]] = defaultdict(list)
        for s in all_spans:
            by_trace[s.trace_id].append(s)

        for trace_id, spans in by_trace.items():
            error_spans = [s for s in spans if s.status == "error"]
            if not error_spans:
                continue

            # Check for propagation: error span has error parent
            chained = [
                s for s in error_spans
                if s.parent_span_id and by_id.get(s.parent_span_id, Span(
                    "", "", None, "", "unknown", 0, 0, "unset", "", {}
                )).status == "error"
            ]
            if len(chained) >= 1:
                affected_names = list({s.tool_name or s.name for s in chained[:5]})
                findings.append(Finding(
                    detector=self.name,
                    severity="medium",
                    title=f"Error propagation chain: {len(chained) + 1} errors cascading in trace {trace_id[:8]}",
                    detail=(
                        f"{len(chained)} spans errored in a chain (child → parent → grandparent) "
                        f"within trace '{trace_id[:12]}'. Affected spans: {', '.join(affected_names)}. "
                        f"This pattern means a single root failure is causing the whole agent run to fail "
                        f"unrecoverably instead of being contained."
                    ),
                    affected_sessions=[trace_id],
                    estimated_waste_usd_per_day=0.0,
                    fix=(
                        "Wrap each agent step in try/except and return a Result type (ok/error) rather than "
                        "raising exceptions. Use a supervisor pattern: if a sub-agent fails, the orchestrator "
                        "should reassign or skip — not fail entirely."
                    ),
                ))

        return findings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_all_spans(roots: list[Span]) -> list[Span]:
    result: list[Span] = []
    def walk(s: Span) -> None:
        result.append(s)
        for child in s.children:
            walk(child)
    for root in roots:
        walk(root)
    return result
