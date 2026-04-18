from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monk.parsers.auto import TraceCall
    from monk.parsers.otel import Span


@dataclass
class Finding:
    detector: str
    severity: str          # "high" | "medium" | "low"
    title: str
    detail: str
    affected_sessions: list[str]
    estimated_waste_usd_per_day: float = 0.0
    fix: str = ""


class BaseDetector:
    name: str = "base"
    requires_spans: bool = False   # True = needs Span data; False = works on TraceCall

    def run(self, calls: list["TraceCall"]) -> list[Finding]:
        """Run detector on normalised TraceCall list. Override in TraceCall-based detectors."""
        return []

    def run_spans(self, roots: list["Span"]) -> list[Finding]:
        """Run detector on OTEL span trees. Override in span-aware detectors."""
        return []
