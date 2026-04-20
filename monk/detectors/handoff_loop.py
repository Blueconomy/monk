"""
Detector 15 — Handoff Loop

Detects multi-agent handoff cycling: agents repeatedly transferring control
back and forth (A→B→A→B) without making progress toward task completion.

Applies to any trace where tool calls contain transfer_to_* or
transfer_back_to_* patterns — the convention used by LangGraph Supervisor,
LangGraph Swarm, and OpenAI Swarm.

Heuristic: within a session, if the same pair of agent targets appears 3+
times in the handoff sequence, the agents are bouncing without resolving.
"""
from __future__ import annotations

from collections import defaultdict

from monk.parsers.auto import TraceCall
from monk.pricing import cost_for_call
from .base import BaseDetector, Finding

LOOP_THRESHOLD = 3   # same handoff pair seen this many times → flag
_TRANSFER_PREFIXES = ("transfer_to_", "transfer_back_to_", "handoff_to_", "route_to_")


def _is_transfer(name: str) -> bool:
    return any(name.lower().startswith(p) for p in _TRANSFER_PREFIXES)


def _target(name: str) -> str:
    """Normalise transfer_to_X / transfer_back_to_X → X."""
    n = name.lower()
    for p in _TRANSFER_PREFIXES:
        if n.startswith(p):
            return name[len(p):]
    return name


class HandoffLoopDetector(BaseDetector):
    name = "handoff_loop"

    def run(self, calls: list[TraceCall]) -> list[Finding]:
        sessions: dict[str, list[TraceCall]] = defaultdict(list)
        for c in calls:
            sessions[c.session_id].append(c)

        findings: list[Finding] = []

        for sid, scalls in sessions.items():
            result = _detect_handoff_loop(scalls)
            if result:
                pair, count, wasted_tokens, model = result
                waste_usd = cost_for_call(model, wasted_tokens, 0)
                findings.append(Finding(
                    detector=self.name,
                    severity="high",
                    title=f"Handoff loop: agents '{pair[0]}' ↔ '{pair[1]}' cycled {count}x",
                    detail=(
                        f"Session '{sid}': agents '{pair[0]}' and '{pair[1]}' "
                        f"transferred control to each other {count} times without "
                        f"resolving the task. This is a swarm/supervisor routing "
                        f"failure — agents are bouncing rather than progressing."
                    ),
                    affected_sessions=[sid],
                    estimated_waste_usd_per_day=waste_usd,
                    fix=(
                        f"Add a transfer guard: track which agents have been active "
                        f"this session and block re-transfer to an agent that already "
                        f"declined or returned. Consider a max_handoffs limit "
                        f"(recommended: ≤3) at the orchestration level. "
                        f"In LangGraph Swarm, use InMemorySaver checkpointing to "
                        f"persist intermediate results so agents don't restart from scratch."
                    ),
                ))

        return findings


def _detect_handoff_loop(
    calls: list[TraceCall],
) -> tuple[tuple[str, str], int, int, str] | None:
    """
    Extract the handoff sequence and look for repeated A↔B pairs.
    Returns (pair, count, wasted_input_tokens, model) or None.
    """
    handoff_seq: list[tuple[str, int, str]] = []  # (target_agent, input_tokens, model)

    for call in calls:
        for tc in call.tool_calls:
            if _is_transfer(tc.name):
                handoff_seq.append((_target(tc.name), call.input_tokens, call.model))

    if len(handoff_seq) < LOOP_THRESHOLD * 2:
        return None

    targets = [h[0] for h in handoff_seq]

    # Count consecutive pairs (A, B) in the sequence
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for i in range(len(targets) - 1):
        a, b = targets[i], targets[i + 1]
        # Normalise pair order so A↔B and B↔A are the same pair
        key: tuple[str, str] = tuple(sorted([a, b]))  # type: ignore[assignment]
        pair_counts[key] += 1

    if not pair_counts:
        return None

    worst_pair, worst_count = max(pair_counts.items(), key=lambda x: x[1])

    if worst_count < LOOP_THRESHOLD:
        return None

    # Estimate wasted tokens: all handoff calls after the first round-trip
    wasted = sum(
        handoff_seq[i][1]
        for i in range(2, len(handoff_seq))  # first 2 transfers are "expected"
        if targets[i] in worst_pair
    )
    model = handoff_seq[0][2] if handoff_seq else "unknown"

    return worst_pair, worst_count, wasted, model
