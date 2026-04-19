"""
Detector — Text I/O Length Analysis

Tracks input/output text lengths and token counts across a session.
Flags:
  - input_bloat: input text growing unboundedly turn-over-turn (not just tokens)
  - low_compression: output is very short relative to massive input (agent not doing work)
  - truncated_output: output suspiciously short (possible truncation / refusal)
"""
from __future__ import annotations
from collections import defaultdict
from monk.parsers.auto import TraceCall
from .base import BaseDetector, Finding


class TextIODetector(BaseDetector):
    name = "text_io"
    requires_spans = False

    # Thresholds
    LOW_COMPRESSION_RATIO = 0.02   # output < 2% of input tokens = agent not producing
    TRUNCATION_OUTPUT_TOKENS = 10  # output < 10 tokens = likely truncated/refused
    MIN_INPUT_TOKENS = 500         # only flag if input is substantial
    GROWTH_FACTOR = 3.0            # flag if input grows 3x within a session

    def run(self, calls: list[TraceCall]) -> list[Finding]:
        findings: list[Finding] = []

        # Group by session
        sessions: dict[str, list[TraceCall]] = defaultdict(list)
        for call in calls:
            sessions[call.session_id].append(call)

        for session_id, session_calls in sessions.items():
            session_calls = sorted(session_calls, key=lambda c: c.call_index)
            findings.extend(self._check_session(session_id, session_calls))

        return findings

    def _check_session(self, session_id: str, calls: list[TraceCall]) -> list[Finding]:
        findings = []

        input_lengths = [c.input_tokens for c in calls if c.input_tokens > 0]
        output_lengths = [c.output_tokens for c in calls if c.output_tokens > 0]

        if not input_lengths:
            return []

        # Check 1: Low compression — massive input, tiny output
        low_comp = [
            c for c in calls
            if c.input_tokens >= self.MIN_INPUT_TOKENS
            and c.output_tokens > 0
            and (c.output_tokens / c.input_tokens) < self.LOW_COMPRESSION_RATIO
        ]
        if low_comp:
            worst = min(low_comp, key=lambda c: c.output_tokens / max(c.input_tokens, 1))
            ratio = worst.output_tokens / max(worst.input_tokens, 1)
            findings.append(Finding(
                detector=self.name,
                severity="medium",
                title=f"Low output compression: {len(low_comp)} call(s) with output < 2% of input",
                detail=(
                    f"Session '{session_id[:12]}': {len(low_comp)} LLM call(s) received large "
                    f"inputs but produced very little output. Worst: {worst.input_tokens} input "
                    f"tokens → {worst.output_tokens} output tokens "
                    f"({ratio:.1%} compression ratio). This suggests the model may be "
                    f"overwhelmed by context, producing refusals, or the task is ill-formed."
                ),
                affected_sessions=[session_id],
                estimated_waste_usd_per_day=0.0,
                fix=(
                    "Review what's in the input context for these calls. Large inputs with tiny "
                    "outputs often indicate: (1) tool output flooding the context, (2) the model "
                    "hitting a reasoning dead-end, or (3) the task being too ambiguous. "
                    "Truncate tool outputs and validate that the task description is concrete."
                ),
            ))

        # Check 2: Truncated output (suspiciously short)
        truncated = [
            c for c in calls
            if c.input_tokens >= self.MIN_INPUT_TOKENS
            and 0 < c.output_tokens <= self.TRUNCATION_OUTPUT_TOKENS
        ]
        if truncated:
            findings.append(Finding(
                detector=self.name,
                severity="high",
                title=f"Possible output truncation: {len(truncated)} call(s) with \u226410 output tokens",
                detail=(
                    f"Session '{session_id[:12]}': {len(truncated)} call(s) produced \u226410 output "
                    f"tokens despite large inputs. This is a strong signal of truncation (max_tokens "
                    f"set too low), model refusal, or a malformed prompt that prevents generation."
                ),
                affected_sessions=[session_id],
                estimated_waste_usd_per_day=0.0,
                fix=(
                    "Check max_tokens parameter — it may be set too low. Also check for prompt "
                    "injection or content filter triggers that cause early stop. These calls "
                    "paid full input token cost and produced nothing useful."
                ),
            ))

        # Check 3: Input growing unboundedly
        if len(input_lengths) >= 3:
            first_third_avg = sum(input_lengths[:len(input_lengths)//3]) / max(len(input_lengths)//3, 1)
            last_third_avg = sum(input_lengths[-(len(input_lengths)//3):]) / max(len(input_lengths)//3, 1)
            if first_third_avg > 0 and last_third_avg / first_third_avg >= self.GROWTH_FACTOR:
                growth = last_third_avg / first_third_avg
                findings.append(Finding(
                    detector=self.name,
                    severity="medium",
                    title=f"Input text growing {growth:.1f}\u00d7 within session",
                    detail=(
                        f"Session '{session_id[:12]}': average input tokens grew from "
                        f"{first_third_avg:.0f} (first third) to {last_third_avg:.0f} (last third) "
                        f"— a {growth:.1f}\u00d7 increase. Unbounded input growth means later calls "
                        f"cost proportionally more while the model attends less to earlier context."
                    ),
                    affected_sessions=[session_id],
                    estimated_waste_usd_per_day=0.0,
                    fix=(
                        "Implement a rolling context window: keep the last 5 turns + a "
                        "running summary of earlier turns. Don't append raw tool outputs — "
                        "summarise them before adding to context."
                    ),
                ))

        return findings
