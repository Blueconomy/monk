"""
Detector 6 — Latency Spike

Requires span-level timing data (OTEL format).

Fires when:
  A) A tool call takes >3x the median duration for that tool — outlier latency
     that indicates flaky external APIs, missing timeouts, or no retry budget.
  B) An LLM call takes >15 seconds — suggests the model is generating an
     unexpectedly long response or the provider is under load with no timeout guard.

Why it matters: slow tools stall the whole agent turn. One 30s tool call in a
voice AI pipeline means the user waits 30s for a response.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from monk.parsers.otel import Span
from .base import BaseDetector, Finding

# Thresholds
TOOL_SPIKE_MULTIPLIER = 3.0     # flag if duration > 3x median for that tool
LLM_SLOW_THRESHOLD_MS = 15_000  # 15 seconds
MIN_TOOL_SAMPLES = 3            # need at least this many calls to compute median


class LatencySpikeDetector(BaseDetector):
    name = "latency_spike"
    requires_spans = True        # marks this as a span-level detector

    def run_spans(self, roots: list[Span]) -> list[Finding]:
        all_spans = _collect_all_spans(roots)

        findings: list[Finding] = []
        findings.extend(self._check_tool_spikes(all_spans))
        findings.extend(self._check_slow_llm(all_spans))
        return findings

    # Keep base compatibility — returns empty if called with TraceCall list
    def run(self, calls) -> list[Finding]:  # type: ignore[override]
        return []

    # ── Tool spike detection ──────────────────────────────────────────────────

    def _check_tool_spikes(self, spans: list[Span]) -> list[Finding]:
        # tool_name → list of (duration_ms, session_id)
        tool_durations: dict[str, list[tuple[float, str]]] = defaultdict(list)

        for s in spans:
            if s.kind == "tool" and s.duration_ms > 0:
                name = s.tool_name or s.name
                tool_durations[name].append((s.duration_ms, s.session_id))

        findings: list[Finding] = []
        for tool_name, samples in tool_durations.items():
            if len(samples) < MIN_TOOL_SAMPLES:
                continue

            durations = [d for d, _ in samples]
            median = statistics.median(durations)
            if median == 0:
                continue

            spikes = [(d, sid) for d, sid in samples if d > median * TOOL_SPIKE_MULTIPLIER]
            if not spikes:
                continue

            worst_ms, worst_sid = max(spikes, key=lambda x: x[0])
            affected = list({sid for _, sid in spikes})
            spike_rate = len(spikes) / len(samples)

            findings.append(Finding(
                detector=self.name,
                severity="high" if spike_rate > 0.2 or worst_ms > 10_000 else "medium",
                title=f"Latency spike: '{tool_name}' hits {worst_ms/1000:.1f}s (median: {median/1000:.1f}s)",
                detail=(
                    f"'{tool_name}' spiked to {worst_ms:.0f}ms against a median of {median:.0f}ms "
                    f"({TOOL_SPIKE_MULTIPLIER:.0f}x threshold). This happened in {len(spikes)}/{len(samples)} "
                    f"calls ({spike_rate:.0%}). Slow tool calls block the entire agent turn."
                ),
                affected_sessions=affected,
                estimated_waste_usd_per_day=0.0,  # latency waste is time, not tokens
                fix=(
                    f"Add a timeout to '{tool_name}' (e.g. `asyncio.wait_for(..., timeout=5.0)`). "
                    f"If it's an external API, add a circuit breaker and fail fast with a cached "
                    f"or default result. For voice AI, anything >2s breaks conversational flow."
                ),
            ))

        return findings

    # ── Slow LLM detection ────────────────────────────────────────────────────

    def _check_slow_llm(self, spans: list[Span]) -> list[Finding]:
        slow = [
            s for s in spans
            if s.kind == "llm" and s.duration_ms > LLM_SLOW_THRESHOLD_MS
        ]
        if not slow:
            return []

        affected = list({s.session_id for s in slow})
        worst = max(slow, key=lambda s: s.duration_ms)
        avg_slow_ms = sum(s.duration_ms for s in slow) / len(slow)

        return [Finding(
            detector=self.name,
            severity="medium",
            title=f"{len(slow)} LLM call(s) exceeded {LLM_SLOW_THRESHOLD_MS/1000:.0f}s response time",
            detail=(
                f"Slowest: {worst.duration_ms/1000:.1f}s on model '{worst.model or 'unknown'}' "
                f"(session: {worst.session_id}). Average of slow calls: {avg_slow_ms/1000:.1f}s. "
                f"This is usually caused by unexpectedly long generated outputs or no request timeout."
            ),
            affected_sessions=affected,
            estimated_waste_usd_per_day=0.0,
            fix=(
                "Set a `max_tokens` cap on your LLM calls to prevent runaway generation. "
                "Add a request timeout at the HTTP client level (e.g. `httpx.Timeout(30.0)`). "
                "For latency-sensitive pipelines, stream responses instead of waiting for completion."
            ),
        )]


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
