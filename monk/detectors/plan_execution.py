"""
Detector 11 — Plan vs Execution Gap

Covers: Goal Deviation (12 in TRAIL), Task Orchestration (8 in TRAIL)

No LLM needed. Strategy:
  1. Find the agent's plan in LLM output spans — look for numbered step lists,
     Thought:/Code: patterns, or explicit "Step N:" labels.
  2. Extract the tools/actions the plan mentions.
  3. Compare against the tool spans that were actually executed.
  4. Flag if the agent planned to use a tool but never called it, OR
     if the agent abandoned its plan partway through.

Also detects: plan written but zero tool calls made (agent went straight to
answer without executing any steps — common in hallucination cases).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from monk.parsers.otel import Span
from .base import BaseDetector, Finding

# Regex to extract planned steps from LLM output
STEP_PATTERNS = [
    re.compile(r"^\s*(\d+)\.\s+(.+)", re.M),          # 1. Do X
    re.compile(r"Step\s+(\d+)[:\-]\s*(.+)", re.I),     # Step 1: Do X
    re.compile(r"^[-*]\s+(.+)", re.M),                  # - Do X (bullet)
]

# Tool/action keywords that appear in plans and map to span types
PLAN_TOOL_HINTS = {
    "search": ["search", "web_search", "SearchInformation", "search_agent"],
    "code": ["python_interpreter", "execute_code", "code", "run"],
    "file": ["read_file", "inspect_file", "file_reader"],
    "visit": ["visit_page", "web_browser", "navigate"],
    "download": ["download", "fetch"],
    "final_answer": ["final_answer", "finish", "answer"],
}

# Minimum steps in a plan to bother checking
MIN_PLAN_STEPS = 3


class PlanExecutionDetector(BaseDetector):
    name = "plan_execution"
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

        # Sort by start time to find the planning phase (first LLM output)
        llm_spans = sorted(
            [s for s in spans if s.kind == "llm"],
            key=lambda s: s.start_time_ns
        )
        tool_spans = [s for s in spans if s.kind == "tool"]

        if not llm_spans:
            return []

        # ── Extract plan from first LLM output ───────────────────────
        first_output = self._get_output_text(llm_spans[0])
        if not first_output or len(first_output) < 100:
            return []

        plan_steps = self._extract_plan_steps(first_output)
        if len(plan_steps) < MIN_PLAN_STEPS:
            return []

        # ── Check 1: planned tools were actually called ───────────────
        tool_names_executed = {
            (s.tool_name or s.name).lower()
            for s in tool_spans
        }

        planned_but_missing: list[str] = []
        for hint_key, hint_variants in PLAN_TOOL_HINTS.items():
            # Does plan mention this capability?
            plan_mentions = any(
                hint_key in step.lower() or
                any(v.lower() in step.lower() for v in hint_variants)
                for step in plan_steps
            )
            if not plan_mentions:
                continue
            # Was it actually executed?
            executed = any(
                any(v.lower() in tool_name for v in hint_variants)
                for tool_name in tool_names_executed
            )
            if not executed and hint_key != "final_answer":
                planned_but_missing.append(hint_key)

        if planned_but_missing:
            findings.append(Finding(
                detector=self.name,
                severity="high",
                title=(
                    f"Plan abandoned: {len(plan_steps)} steps planned, "
                    f"{len(planned_but_missing)} capability(ies) never executed"
                ),
                detail=(
                    f"Session '{session_id[:12]}' produced a {len(plan_steps)}-step plan "
                    f"but never executed: {', '.join(planned_but_missing)}. "
                    f"The agent deviated from its own plan — either hallucinating the result "
                    f"or giving up mid-execution. "
                    f"Planned steps were: {'; '.join(plan_steps[:3])}{'...' if len(plan_steps)>3 else ''}"
                ),
                affected_sessions=[session_id],
                estimated_waste_usd_per_day=0.0,
                fix=(
                    "Add an execution verifier: after each planned step, check that a "
                    "corresponding tool call appears in the trace. If a step is skipped, "
                    "force a retry or raise an alert. Never let the agent jump to a final "
                    "answer without having executed all planned verification steps."
                ),
            ))

        # ── Check 2: plan exists but zero tool calls made ─────────────
        if len(plan_steps) >= MIN_PLAN_STEPS and len(tool_spans) == 0:
            findings.append(Finding(
                detector=self.name,
                severity="high",
                title=f"Plan written but no tools executed ({len(plan_steps)} steps planned, 0 calls made)",
                detail=(
                    f"Session '{session_id[:12]}' produced a {len(plan_steps)}-step plan "
                    f"but made zero tool calls. This is a strong hallucination signal: "
                    f"the agent fabricated an answer without executing any verification steps. "
                    f"First planned step: '{plan_steps[0][:80]}'"
                ),
                affected_sessions=[session_id],
                estimated_waste_usd_per_day=0.0,
                fix=(
                    "Force tool use: set tool_choice='required' or add a system-prompt rule "
                    "'You must call at least one tool before providing a final answer.' "
                    "Validate that the agent executed at least the minimum required verification "
                    "steps before accepting its output."
                ),
            ))

        # ── Check 3: plan truncated mid-execution ────────────────────
        # Agent had N steps planned, executed tools < N/2 times
        if len(plan_steps) >= MIN_PLAN_STEPS and 0 < len(tool_spans) < len(plan_steps) // 2:
            findings.append(Finding(
                detector=self.name,
                severity="medium",
                title=(
                    f"Incomplete execution: {len(plan_steps)} steps planned, "
                    f"only {len(tool_spans)} tool call(s) made"
                ),
                detail=(
                    f"Session '{session_id[:12]}' planned {len(plan_steps)} steps "
                    f"but only made {len(tool_spans)} tool call(s) — less than half the plan. "
                    f"The agent likely short-circuited to an answer without completing "
                    f"planned verification steps."
                ),
                affected_sessions=[session_id],
                estimated_waste_usd_per_day=0.0,
                fix=(
                    "Add a completion check before accepting a final answer: count executed "
                    "steps and compare to planned steps. Require the agent to either complete "
                    "all steps or explicitly mark steps as skipped with a reason."
                ),
            ))

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

    def _extract_plan_steps(self, text: str) -> list[str]:
        """Extract numbered/bulleted plan steps from LLM output."""
        steps: list[str] = []

        # Look for numbered list
        numbered = re.findall(r"^\s*\d+\.\s*(.{10,120})", text, re.M)
        if len(numbered) >= MIN_PLAN_STEPS:
            return [s.strip() for s in numbered]

        # Look for Step N: pattern
        step_n = re.findall(r"Step\s*\d+[:\-]\s*(.{10,120})", text, re.I)
        if len(step_n) >= MIN_PLAN_STEPS:
            return [s.strip() for s in step_n]

        # Look for "Facts to look up" or similar sections with bullets
        bullets = re.findall(r"^[-•*]\s*(.{10,120})", text, re.M)
        if len(bullets) >= MIN_PLAN_STEPS:
            return [s.strip() for s in bullets]

        return steps
