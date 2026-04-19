"""
Detector 10 — Output Format Compliance

Covers: Formatting Errors (~60 in TRAIL), Instruction Non-compliance (~34 in TRAIL)

No LLM needed. Strategy:
  1. Read the system prompt from LLM input message spans.
  2. Extract explicit format rules using regex patterns:
     - Required tags: write the '<end_plan>' tag, end with </answer>, etc.
     - Required sections: must include "Thought:", "Code:", "Observation:" cycle
     - Required suffixes/prefixes, structured outputs
  3. Read the model's output from LLM output message spans.
  4. Check each extracted rule against the output.

Works on OTEL span data where llm.input_messages.* and llm.output_messages.* attributes
contain the full prompt and response text.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from monk.parsers.otel import Span
from .base import BaseDetector, Finding

# Patterns in system prompts that indicate required output elements
RULE_EXTRACTORS = [
    # Required XML/markdown tags
    (re.compile(r"write\s+the\s+['\"`]?(<[\w/_]+>)['\"`]?\s+tag", re.I),
     lambda m: (f"Required output tag: {m.group(1)}", m.group(1), "tag")),
    (re.compile(r"end\s+with\s+['\"`]?(<[\w/_]+>)['\"`]", re.I),
     lambda m: (f"Required closing tag: {m.group(1)}", m.group(1), "tag")),
    (re.compile(r"must\s+include\s+['\"`]?(<[\w/_]+>)['\"`]", re.I),
     lambda m: (f"Required tag: {m.group(1)}", m.group(1), "tag")),
    # Required thought/code/observation cycle
    (re.compile(r"cycle\s+of\s+'?Thought:'.*?'?Code:'.*?'?Observation:'", re.I | re.S),
     lambda m: ("Required Thought:/Code:/Observation: cycle", "Thought:", "keyword")),
    # Required final answer format
    (re.compile(r"use\s+the\s+['\"`]?(\w+)['\"`]?\s+tool\s+to\s+provide\s+(?:the\s+)?final", re.I),
     lambda m: (f"Required final answer via tool: {m.group(1)}", m.group(1), "tool")),
    # Must end with specific pattern
    (re.compile(r"stop\s+there[\.,]?\s*$", re.I | re.M),
     lambda m: None),  # Structural, harder to verify
]

# Known format keywords to check in output
FORMAT_KEYWORDS = {
    "<end_plan>":   "Required <end_plan> tag at end of planning phase",
    "<answer>":     "Required <answer> tag wrapping final answer",
    "</answer>":    "Required </answer> closing tag",
    "Thought:":     "Required Thought: prefix before reasoning steps",
    "Code:":        "Required Code: prefix before code blocks",
    "Observation:": "Required Observation: prefix after tool results",
    "Final Answer:": "Required 'Final Answer:' prefix",
}


class OutputFormatDetector(BaseDetector):
    name = "output_format"
    requires_spans = True

    def run_spans(self, roots: list[Span]) -> list[Finding]:
        findings: list[Finding] = []

        # Group all spans by session (trace)
        sessions: dict[str, list[Span]] = defaultdict(list)
        for root in roots:
            self._collect_all(root, sessions)

        for session_id, spans in sessions.items():
            findings.extend(self._check_session(session_id, spans))

        return findings

    def run(self, calls) -> list[Finding]:  # type: ignore[override]
        return []

    def _collect_all(self, span: Span, sessions: dict) -> None:
        sessions[span.session_id].append(span)
        for child in span.children:
            self._collect_all(child, sessions)

    def _check_session(self, session_id: str, spans: list[Span]) -> list[Finding]:
        findings: list[Finding] = []

        # Extract system prompt and output from LLM spans
        system_prompt = ""
        outputs: list[str] = []

        for span in spans:
            attrs = span.attributes
            # Collect system prompt from first input message
            sys_content = str(attrs.get("llm.input_messages.0.message.content", ""))
            if not system_prompt and len(sys_content) > 100:
                system_prompt = sys_content

            # Collect LLM outputs
            out = (
                attrs.get("llm.output_messages.0.message.content") or
                attrs.get("output.value") or ""
            )
            out_str = str(out)
            if isinstance(out, str) and out.startswith("{"):
                try:
                    parsed = json.loads(out)
                    out_str = parsed.get("content", out_str)
                except (json.JSONDecodeError, AttributeError):
                    pass
            if out_str and len(out_str) > 20:
                outputs.append(out_str)

        if not system_prompt or not outputs:
            return []

        full_output = "\n".join(outputs)

        # ── Extract required format rules from system prompt ──────────
        required = self._extract_requirements(system_prompt)
        if not required:
            return []

        # ── Check each requirement against actual output ──────────────
        violations: list[tuple[str, str]] = []
        for req_text, pattern, kind in required:
            if kind == "tag":
                if pattern not in full_output:
                    violations.append((req_text, pattern))
            elif kind == "keyword":
                if pattern not in full_output:
                    violations.append((req_text, pattern))
            elif kind == "tool":
                # Presence in output OR in tool span names
                tool_names = {s.tool_name for s in spans if s.tool_name}
                if pattern not in full_output and pattern not in tool_names:
                    violations.append((req_text, pattern))

        if violations:
            findings.append(Finding(
                detector=self.name,
                severity="medium",
                title=f"Output format violation: {len(violations)} required element(s) missing",
                detail=(
                    f"Session '{session_id[:12]}' violated {len(violations)} explicit format "
                    f"requirement(s) from the system prompt.\n"
                    + "\n".join(f"  • {r} (pattern: '{p}')" for r, p in violations[:4])
                    + (f"\n  • ...and {len(violations)-4} more" if len(violations) > 4 else "")
                ),
                affected_sessions=[session_id],
                estimated_waste_usd_per_day=0.0,
                fix=(
                    "Ensure the agent template enforces required output structure. "
                    "Add a post-generation validator that checks for required tags before "
                    "returning the response. If using a planning agent, verify the planning "
                    "loop template always appends required structural tags."
                ),
            ))

        return findings

    def _extract_requirements(self, system_prompt: str) -> list[tuple[str, str, str]]:
        """Extract (description, pattern, kind) tuples from system prompt text."""
        requirements: list[tuple[str, str, str]] = []

        # Check for known format keywords required in output
        for keyword, description in FORMAT_KEYWORDS.items():
            if keyword.lower() in system_prompt.lower():
                requirements.append((description, keyword, "keyword"))

        # Apply regex extractors
        for pattern, extractor in RULE_EXTRACTORS:
            for m in pattern.finditer(system_prompt):
                result = extractor(m)
                if result:
                    requirements.append(result)

        # De-duplicate
        seen: set[str] = set()
        deduped = []
        for item in requirements:
            key = item[1]
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        return deduped
