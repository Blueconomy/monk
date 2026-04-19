"""
Detector 4 — Context Bloat

Two sub-checks:
  A) System prompt consumes an outsized share of every token budget
  B) Conversation history grows unbounded within a session — each turn
     costs significantly more than the last with no summarisation
"""
from __future__ import annotations
from collections import defaultdict
from monk.parsers.auto import TraceCall
from monk.pricing import cost_for_call
from .base import BaseDetector, Finding

SYSTEM_RATIO_THRESHOLD = 0.55   # system tokens > 55% of input → bloated
GROWTH_RATIO_THRESHOLD = 2.5    # last call costs 2.5x the first → unbounded growth
MIN_CALLS_FOR_GROWTH = 4        # need enough calls to detect growth trend
LARGE_TOOL_RESULT_TOKENS = 2000 # tool result >2000 tokens injected verbatim is a smell
TOOL_RESULT_FRACTION = 0.40     # tool results > 40% of input = context dominated by raw outputs


class ContextBloatDetector(BaseDetector):
    name = "context_bloat"

    def run(self, calls: list[TraceCall]) -> list[Finding]:
        findings: list[Finding] = []

        # ── Check A: system prompt ratio ──────────────────────────────
        bloated = [
            c for c in calls
            if c.input_tokens > 0 and
               c.system_prompt_tokens / c.input_tokens > SYSTEM_RATIO_THRESHOLD
        ]
        if len(bloated) >= 3:
            avg_ratio = sum(
                c.system_prompt_tokens / c.input_tokens for c in bloated
            ) / len(bloated)
            wasted_tokens = sum(
                int(c.system_prompt_tokens - c.input_tokens * 0.3)
                for c in bloated
            )
            waste_usd = sum(
                cost_for_call(c.model, max(0, c.system_prompt_tokens - int(c.input_tokens * 0.3)), 0)
                for c in bloated
            )
            findings.append(Finding(
                detector=self.name,
                severity="medium",
                title=f"System prompt is {avg_ratio:.0%} of input tokens on average",
                detail=(
                    f"{len(bloated)} calls had system prompts consuming >{SYSTEM_RATIO_THRESHOLD:.0%} "
                    f"of the full token budget. You're paying to re-send the same instructions "
                    f"on every single call."
                ),
                affected_sessions=list({c.session_id for c in bloated}),
                estimated_waste_usd_per_day=waste_usd,
                fix=(
                    "Trim your system prompt: remove redundant instructions, move static "
                    "knowledge to a RAG lookup, or use prompt caching (Anthropic/OpenAI both "
                    "support it — cuts re-send cost by ~90%)."
                ),
            ))

        # ── Check B: unbounded history growth ────────────────────────
        sessions: dict[str, list[TraceCall]] = defaultdict(list)
        for c in calls:
            sessions[c.session_id].append(c)

        growing_sessions = []
        total_excess_usd = 0.0

        for sid, scalls in sessions.items():
            if len(scalls) < MIN_CALLS_FOR_GROWTH:
                continue
            first_tok = scalls[0].input_tokens or 1
            last_tok = scalls[-1].input_tokens
            if last_tok / first_tok >= GROWTH_RATIO_THRESHOLD:
                growing_sessions.append(sid)
                # Excess = tokens above what a summarised context would cost
                ideal = first_tok * 1.5
                excess = sum(max(0, c.input_tokens - ideal) for c in scalls[2:])
                total_excess_usd += cost_for_call(scalls[-1].model, int(excess), 0)

        if growing_sessions:
            findings.append(Finding(
                detector=self.name,
                severity="medium",
                title=f"Unbounded history growth in {len(growing_sessions)} session(s)",
                detail=(
                    f"In {len(growing_sessions)} session(s), input token count grew "
                    f"{GROWTH_RATIO_THRESHOLD:.1f}x+ from first to last call. "
                    f"You're re-sending the full conversation history every turn."
                ),
                affected_sessions=growing_sessions,
                estimated_waste_usd_per_day=total_excess_usd,
                fix=(
                    "Summarise conversation history every N turns rather than appending forever. "
                    "Or use a sliding window: keep the last K exchanges + a rolling summary. "
                    "LangChain's ConversationSummaryMemory and LlamaIndex's ChatMemoryBuffer "
                    "both do this out of the box."
                ),
            ))

        # ── Check C: large tool results injected verbatim ─────────────
        # Detects the "Context Handling Failure" pattern from TRAIL annotations:
        # agent dumps a full web page / file into context instead of extracting facts.
        bloated_tool_calls = []
        for c in calls:
            for tc in c.tool_calls:
                result_tokens = len(tc.result) // 4  # approx token estimate
                input_tok = c.input_tokens or 1
                if (result_tokens >= LARGE_TOOL_RESULT_TOKENS and
                        result_tokens / input_tok >= TOOL_RESULT_FRACTION):
                    bloated_tool_calls.append((c, tc, result_tokens))

        if bloated_tool_calls:
            sessions_affected = list({c.session_id for c, _, _ in bloated_tool_calls})
            worst_c, worst_tc, worst_tok = max(bloated_tool_calls, key=lambda x: x[2])
            waste_usd = sum(
                cost_for_call(c.model, result_tokens - LARGE_TOOL_RESULT_TOKENS // 2, 0)
                for c, _, result_tokens in bloated_tool_calls
            )
            findings.append(Finding(
                detector=self.name,
                severity="high",
                title=(
                    f"Verbatim tool output flooding context: "
                    f"'{worst_tc.name}' injected ~{worst_tok:,} tokens"
                ),
                detail=(
                    f"{len(bloated_tool_calls)} tool call(s) returned results large enough "
                    f"to dominate the LLM context (>{LARGE_TOOL_RESULT_TOKENS:,} tokens, "
                    f">{TOOL_RESULT_FRACTION:.0%} of input budget). "
                    f"Worst: '{worst_tc.name}' with ~{worst_tok:,} tokens. "
                    f"Injecting raw tool outputs verbatim wastes tokens and can cause the "
                    f"model to lose track of the actual task (context handling failure)."
                ),
                affected_sessions=sessions_affected,
                estimated_waste_usd_per_day=waste_usd,
                fix=(
                    f"Extract only the relevant facts from '{worst_tc.name}' output before "
                    f"injecting into context. Use a retrieval step (e.g. BM25 or embedding search) "
                    f"to pull the 3-5 most relevant chunks. For web pages, pass title + summary "
                    f"+ target paragraph rather than the full HTML content."
                ),
            ))

        return findings
