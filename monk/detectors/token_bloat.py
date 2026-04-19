"""
Detector 9 — Token Bloat (per-session context growth)

Distinct from context_bloat (which checks system prompt ratio and growth ratio),
this detector fires on span-level data when:

  A) A single LLM span has abnormally high token consumption vs the session median
     — suggests unbounded history or accidentally huge payloads being sent.

  B) Across a session, the token cost per call grows monotonically — no
     summarisation or window management, just infinite append.

  C) Total session cost exceeds a threshold that suggests runaway consumption
     (e.g. a 20-call session that has consumed $0.50+).

Why separate from context_bloat?
  context_bloat checks TraceCall-level data (ratio and growth across sessions).
  token_bloat uses span timing + absolute values to catch cases where one single
  massive call or a visually monotonic growth curve reveals the problem more
  precisely than a ratio check.
"""
from __future__ import annotations

from collections import defaultdict
from monk.parsers.otel import Span
from monk.pricing import cost_for_call
from .base import BaseDetector, Finding

SPIKE_MULTIPLIER = 4.0        # single call > 4x session median → spike
MIN_CALLS_FOR_TREND = 4       # need at least this many LLM calls to assess trend
MONOTONIC_GROWTH_PCTS = 0.75  # 75%+ of calls must be increasing to flag
SESSION_COST_THRESHOLD = 0.50 # $0.50+/session — only flag notably expensive sessions


class TokenBloatDetector(BaseDetector):
    name = "token_bloat"
    requires_spans = True

    def run_spans(self, roots: list[Span]) -> list[Finding]:
        findings: list[Finding] = []

        # Group LLM spans by session (trace_id)
        sessions: dict[str, list[Span]] = defaultdict(list)
        for root in roots:
            self._collect_llm_spans(root, sessions)

        for session_id, llm_spans in sessions.items():
            if not llm_spans:
                continue

            llm_spans.sort(key=lambda s: s.start_time_ns)
            token_counts = [s.input_tokens + s.output_tokens for s in llm_spans]
            total_tokens = sum(token_counts)
            costs = [cost_for_call(s.model or 'gpt-4o', s.input_tokens, s.output_tokens)
                     for s in llm_spans]
            total_cost = sum(costs)

            # ── A) Single-call spike ──────────────────────────────────
            if len(token_counts) >= 2:
                median_tok = sorted(token_counts)[len(token_counts) // 2]
                for i, (span, tok) in enumerate(zip(llm_spans, token_counts)):
                    if median_tok > 0 and tok / median_tok >= SPIKE_MULTIPLIER:
                        waste_usd = cost_for_call(span.model or 'gpt-4o', int(tok - median_tok), 0)
                        findings.append(Finding(
                            detector=self.name,
                            severity="high",
                            title=f"Token spike: call #{i+1} uses {tok:,} tokens ({tok/median_tok:.1f}x median)",
                            detail=(
                                f"LLM call #{i+1} in session '{session_id[:12]}' consumed {tok:,} tokens — "
                                f"{tok/median_tok:.1f}x the session median of {median_tok:,}. "
                                f"This typically indicates a full conversation history being passed "
                                f"where only a summary or subset was needed, or a very large tool "
                                f"output being injected verbatim into context."
                            ),
                            affected_sessions=[session_id],
                            estimated_waste_usd_per_day=waste_usd,
                            fix=(
                                "Truncate or summarise tool outputs before injecting them into context. "
                                "For large tool results (e.g. web pages, file contents), pass only the "
                                "relevant excerpt. Consider using a retrieval step to extract the 3-5 "
                                "most relevant paragraphs rather than the full document."
                            ),
                        ))
                        break  # one finding per session for A

            # ── B) Monotonic growth trend ─────────────────────────────
            if len(token_counts) >= MIN_CALLS_FOR_TREND:
                increases = sum(1 for a, b in zip(token_counts, token_counts[1:]) if b > a)
                growth_pct = increases / (len(token_counts) - 1)
                if growth_pct >= MONOTONIC_GROWTH_PCTS:
                    first, last = token_counts[0], token_counts[-1]
                    growth_factor = last / max(first, 1)
                    # Excess = what we'd save with a sliding-window of size first*1.5
                    ideal_window = first * 1.5
                    excess_tokens = sum(max(0, t - ideal_window) for t in token_counts[2:])
                    waste_usd = cost_for_call(
                        llm_spans[-1].model or 'gpt-4o', int(excess_tokens), 0
                    )
                    if excess_tokens > 500:  # only flag if material
                        findings.append(Finding(
                            detector=self.name,
                            severity="medium",
                            title=(
                                f"Monotonic token growth: {first:,}→{last:,} tokens "
                                f"({growth_factor:.1f}x) across {len(token_counts)} calls"
                            ),
                            detail=(
                                f"Session '{session_id[:12]}' shows {growth_pct:.0%} of LLM calls "
                                f"increasing in token count — a signature of unbounded context append. "
                                f"The first call cost {first:,} tokens; the last cost {last:,}. "
                                f"Without summarisation, costs compound every turn."
                            ),
                            affected_sessions=[session_id],
                            estimated_waste_usd_per_day=waste_usd,
                            fix=(
                                "Implement a sliding context window or periodic summarisation. "
                                "Keep the last K turns + a summary of everything before. "
                                "LangChain ConversationSummaryBufferMemory and LlamaIndex "
                                "ChatMemoryBuffer both implement this pattern. "
                                "Anthropic prompt caching can cut re-send cost by ~90%."
                            ),
                        ))

            # ── C) High-cost session ──────────────────────────────────
            if total_cost >= SESSION_COST_THRESHOLD and len(llm_spans) >= 3:
                findings.append(Finding(
                    detector=self.name,
                    severity="medium",
                    title=(
                        f"Expensive session: {total_tokens:,} tokens "
                        f"(~${total_cost:.3f}) in {len(llm_spans)} calls"
                    ),
                    detail=(
                        f"Session '{session_id[:12]}' consumed {total_tokens:,} total tokens "
                        f"across {len(llm_spans)} LLM calls, costing ~${total_cost:.3f}. "
                        f"At current volume, this single session type would cost "
                        f"~${total_cost*100:.2f}/day if repeated 100x."
                    ),
                    affected_sessions=[session_id],
                    estimated_waste_usd_per_day=total_cost,
                    fix=(
                        "Profile which calls are most expensive and whether they can be batched, "
                        "cached, or replaced with a smaller model. Use model_overkill detector "
                        "results alongside this finding."
                    ),
                ))

        return findings

    def run(self, calls) -> list[Finding]:  # type: ignore[override]
        return []

    def _collect_llm_spans(self, span: Span, sessions: dict) -> None:
        if span.kind == "llm" and (span.input_tokens > 0 or span.output_tokens > 0):
            sessions[span.session_id].append(span)
        for child in span.children:
            self._collect_llm_spans(child, sessions)
