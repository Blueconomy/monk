"""
Detector 12 — Span-Output Consistency (Hallucination Proxy)

Covers: Language-only/Hallucination (11 in TRAIL), Tool Output Misinterpretation (3)
Also catches: the 5 "Context Handling Failures" where final_answer called but result not shown.

No LLM needed. Free alternative to LLM-as-judge. Strategy:

  A) Claim-without-evidence: model output contains verifiable claim patterns
     ("According to X", "I found that", "The results show", "I verified") but
     no corresponding tool span exists in the trace. Strong hallucination signal.

  B) Tool-result mismatch: model output references a specific value/fact that
     does NOT appear in any tool result span in the trace.
     (e.g. model says "The answer is 42" but tool returned "No results found")

  C) Final answer gap: `final_answer` tool was called but the call_index is
     the last span — model never shows the result. The tool result exists in
     the span but the model's output after it is absent/empty.

  D) Empty-result-accepted: a tool returned an empty or error result but the
     model's next output claims the tool succeeded and references data from it.

These are all rule-based checks on span data + output text. No LLM cost.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from monk.parsers.otel import Span
from .base import BaseDetector, Finding

# Patterns in model output that imply a preceding tool call verified something
CLAIM_PATTERNS = [
    re.compile(r"according to\s+(?:the\s+)?(?:search|results?|data|tool|api)", re.I),
    re.compile(r"I (?:found|verified|confirmed|checked|retrieved|obtained)", re.I),
    re.compile(r"the (?:results?|data|output|response) (?:show|indicate|reveal|confirm)", re.I),
    re.compile(r"based on (?:the\s+)?(?:search|results?|retrieved|fetched)", re.I),
    re.compile(r"(?:search|lookup|query) (?:shows?|reveals?|returns?|confirms?)", re.I),
]

# Patterns that indicate a model is claiming to have used a tool
TOOL_CLAIM_PATTERNS = [
    re.compile(r"I (?:called|used|ran|executed|invoked)\s+(?:the\s+)?(\w+)\s+tool", re.I),
    re.compile(r"using\s+(?:the\s+)?(\w+)\s+tool", re.I),
    re.compile(r"(?:the\s+)?(\w+)\s+tool\s+(?:returned|gave|showed|found)", re.I),
]

# Words that indicate empty/failed tool result
EMPTY_RESULT_SIGNALS = {
    "no results", "not found", "none found", "0 results", "empty", "null",
    "no data", "failed", "error", "exception", "no matches", "nothing found",
}


class SpanConsistencyDetector(BaseDetector):
    name = "span_consistency"
    requires_spans = True

    def run_spans(self, roots: list[Span]) -> list[Finding]:
        findings: list[Finding] = []

        sessions: dict[str, list[Span]] = defaultdict(list)
        for root in roots:
            self._collect_all(root, sessions)

        for session_id, spans in sessions.items():
            findings.extend(self._check_session(session_id, spans))

        return findings

    def run(self, calls) -> list[Finding]:  # type: ignore[override]
        return []

    def _collect_all(self, span: Span, out: dict) -> None:
        out[span.session_id].append(span)
        for child in span.children:
            self._collect_all(child, out)

    def _check_session(self, session_id: str, spans: list[Span]) -> list[Finding]:
        findings: list[Finding] = []

        llm_spans = sorted([s for s in spans if s.kind == "llm"], key=lambda s: s.start_time_ns)
        tool_spans = sorted([s for s in spans if s.kind == "tool"], key=lambda s: s.start_time_ns)

        # Build set of tool names actually called
        tool_names_called: set[str] = {
            (s.tool_name or s.name).lower() for s in tool_spans
        }

        # Collect all tool results
        all_tool_results = " ".join(
            s.tool_result.lower() for s in tool_spans if s.tool_result
        )

        # ── Check A: claim-without-evidence ──────────────────────────
        for llm_span in llm_spans:
            out_text = self._get_output_text(llm_span)
            if not out_text or len(out_text) < 50:
                continue

            has_claim = any(p.search(out_text) for p in CLAIM_PATTERNS)
            if not has_claim:
                continue

            # Was this LLM span preceded by any tool span?
            prior_tool_spans = [s for s in tool_spans if s.start_time_ns < llm_span.start_time_ns]
            if prior_tool_spans:
                continue  # claim is justified — tools were called before this output

            # LLM makes a factual claim but has NO prior tool calls
            # Check if it's the first LLM span (planning) — that's ok
            if llm_span == llm_spans[0]:
                continue

            findings.append(Finding(
                detector=self.name,
                severity="high",
                title=f"Unverified claim: model asserts facts with no prior tool call",
                detail=(
                    f"Session '{session_id[:12]}': an LLM output span makes verified-sounding "
                    f"claims (e.g. 'I found...', 'According to results...') but no tool call "
                    f"preceded this output. The model is likely hallucinating evidence. "
                    f"Sample output: '{out_text[:120]}...'"
                ),
                affected_sessions=[session_id],
                estimated_waste_usd_per_day=0.0,
                fix=(
                    "Require all factual claims to be grounded in a tool result. Add a "
                    "post-generation check: if the output contains claim phrases ('According to', "
                    "'I found', etc.), verify that a tool call with a non-empty result exists "
                    "in the same turn context. Flag and retry if not."
                ),
            ))
            break  # one finding per session for claim-without-evidence

        # ── Check B: tool-claimed-but-not-called ────────────────────
        for llm_span in llm_spans:
            out_text = self._get_output_text(llm_span)
            if not out_text:
                continue

            for pattern in TOOL_CLAIM_PATTERNS:
                for match in pattern.finditer(out_text):
                    claimed_tool = match.group(1).lower() if match.lastindex else ""
                    if not claimed_tool or len(claimed_tool) < 3:
                        continue
                    # Check if any tool span name contains this word
                    tool_was_called = any(claimed_tool in t for t in tool_names_called)
                    if not tool_was_called and claimed_tool not in ("the", "a", "an", "my"):
                        findings.append(Finding(
                            detector=self.name,
                            severity="high",
                            title=f"Hallucinated tool call: model claims it used '{claimed_tool}' — no span found",
                            detail=(
                                f"Session '{session_id[:12]}': model output says "
                                f"'{match.group(0)[:80]}' but no tool span with name "
                                f"containing '{claimed_tool}' exists in the trace. "
                                f"The model fabricated a tool interaction that never happened. "
                                f"Actual tools called: {list(tool_names_called)[:5]}"
                            ),
                            affected_sessions=[session_id],
                            estimated_waste_usd_per_day=0.0,
                            fix=(
                                "Add a grounding check: extract tool names from model output "
                                "and verify each has a corresponding span. If the model claims "
                                "it called a tool that doesn't appear in the trace, invalidate "
                                "the response and force a retry with explicit tool use."
                            ),
                        ))
                        break  # one finding per span

        # ── Check C: final_answer called but result not shown ────────
        final_answer_spans = [
            s for s in tool_spans
            if "final_answer" in (s.tool_name or s.name).lower()
        ]
        if final_answer_spans:
            last_final = max(final_answer_spans, key=lambda s: s.start_time_ns)
            # Any LLM output AFTER the final_answer call?
            subsequent_llm = [
                s for s in llm_spans if s.start_time_ns > last_final.start_time_ns
            ]
            if not subsequent_llm:
                # final_answer was the last thing — no output shown afterward
                result = last_final.tool_result or last_final.tool_args or ""
                findings.append(Finding(
                    detector=self.name,
                    severity="medium",
                    title="Final answer tool called but result never surfaced to output",
                    detail=(
                        f"Session '{session_id[:12]}': the 'final_answer' tool was called "
                        f"(args: '{result[:80]}') but there is no LLM output span after it. "
                        f"The agent completed the tool call but did not produce a visible "
                        f"response for the user. This is a context handling failure — the "
                        f"tool result is in the trace but was never shown."
                    ),
                    affected_sessions=[session_id],
                    estimated_waste_usd_per_day=0.0,
                    fix=(
                        "Ensure the orchestration layer always reads and displays the result "
                        "of the final_answer tool call. After calling final_answer, the agent "
                        "must produce at least one more LLM output turn that formats and "
                        "presents the result to the user."
                    ),
                ))

        # ── Check D: empty result accepted as valid ──────────────────
        for i, tool_span in enumerate(tool_spans):
            result = (tool_span.tool_result or "").lower()
            if not result:
                continue

            is_empty = any(sig in result for sig in EMPTY_RESULT_SIGNALS)
            if not is_empty:
                continue

            # Find the next LLM span after this tool
            next_llm = next(
                (s for s in llm_spans if s.start_time_ns > tool_span.start_time_ns),
                None
            )
            if not next_llm:
                continue

            next_out = self._get_output_text(next_llm).lower()
            # If next output makes a positive claim despite empty result
            if any(p.search(next_out) for p in CLAIM_PATTERNS):
                tool_name = tool_span.tool_name or tool_span.name
                findings.append(Finding(
                    detector=self.name,
                    severity="high",
                    title=f"Accepted empty result: '{tool_name}' returned nothing, model claims success",
                    detail=(
                        f"Session '{session_id[:12]}': tool '{tool_name}' returned an empty "
                        f"or error result ('{result[:60]}') but the very next LLM output "
                        f"makes a positive factual claim. The model misinterpreted or ignored "
                        f"the empty result and fabricated a successful outcome."
                    ),
                    affected_sessions=[session_id],
                    estimated_waste_usd_per_day=0.0,
                    fix=(
                        f"Add an empty-result guard for '{tool_name}': if the tool returns "
                        f"empty or error, the agent must explicitly handle it — retry with "
                        f"different parameters, use a fallback tool, or escalate to the user. "
                        f"Never pass an empty result silently into the LLM context."
                    ),
                ))
            break  # one finding per session for this check

        return findings

    def _get_output_text(self, span: Span) -> str:
        attrs = span.attributes
        out = (
            attrs.get("llm.output_messages.0.message.content") or
            attrs.get("output.value") or ""
        )
        out_str = str(out)
        if out_str.startswith("{"):
            try:
                parsed = json.loads(out_str)
                out_str = parsed.get("content", out_str)
            except (json.JSONDecodeError, AttributeError):
                pass
        return out_str
