"""
Detector 3 — Model Overkill

Fires when an expensive model (GPT-4o, Claude Sonnet/Opus) is used for
calls where output is short and simple — a strong signal that a cheaper
model would perform identically at a fraction of the cost.
"""
from __future__ import annotations
from collections import defaultdict
from monk.parsers.auto import TraceCall
from monk.pricing import EXPENSIVE_MODELS, CHEAPER_ALTERNATIVE, cost_for_call, cost_difference
from .base import BaseDetector, Finding

# If output < this many tokens and input < this many tokens → likely simple task
SIMPLE_OUTPUT_THRESHOLD = 150
SIMPLE_INPUT_THRESHOLD = 800
MIN_CALLS_TO_FLAG = 5         # only flag if pattern is consistent


class ModelOverkillDetector(BaseDetector):
    name = "model_overkill"

    def run(self, calls: list[TraceCall]) -> list[Finding]:
        # model -> (total_calls, overkill_calls, total_savings_usd)
        stats: dict[str, list] = defaultdict(lambda: [0, 0, 0.0])

        _expensive_lower = {m.lower() for m in EXPENSIVE_MODELS}
        for call in calls:
            model_lower = call.model.lower()
            # Exact match only — prevents "gpt-4o" substring-matching "gpt-4o-mini"
            if model_lower not in _expensive_lower:
                continue
            stats[call.model][0] += 1

            is_simple = (
                call.output_tokens < SIMPLE_OUTPUT_THRESHOLD and
                call.input_tokens < SIMPLE_INPUT_THRESHOLD
            )
            if is_simple:
                alt = _get_alt(call.model)
                savings = cost_difference(call.model, alt, call.input_tokens, call.output_tokens)
                stats[call.model][1] += 1
                stats[call.model][2] += savings

        findings: list[Finding] = []
        affected = list({c.session_id for c in calls})

        for model, (total, overkill, total_savings) in stats.items():
            if overkill < MIN_CALLS_TO_FLAG:
                continue
            rate = overkill / total if total > 0 else 0
            if rate < 0.25:  # less than 25% overkill → skip
                continue
            alt = _get_alt(model)
            findings.append(Finding(
                detector=self.name,
                severity="high" if rate > 0.5 else "medium",
                title=f"Model overkill: {model} used for simple tasks ({rate:.0%} of calls)",
                detail=(
                    f"{overkill}/{total} calls to '{model}' produced short outputs "
                    f"(avg <{SIMPLE_OUTPUT_THRESHOLD} tokens) — likely simple lookups, "
                    f"routing decisions, or classification steps that don't need a powerful model."
                ),
                affected_sessions=affected,
                estimated_waste_usd_per_day=total_savings,
                fix=(
                    f"Route simple/short tasks to '{alt}' instead of '{model}'. "
                    f"A/B test quality on your eval set first — for most routing, "
                    f"classification, and formatting calls the cheaper model is identical."
                ),
            ))

        return findings


def _get_alt(model: str) -> str:
    for key, alt in CHEAPER_ALTERNATIVE.items():
        if key.lower() in model.lower() or model.lower() in key.lower():
            return alt
    return "a cheaper model"
