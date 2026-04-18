from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monk.parsers.auto import TraceCall


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

    def run(self, calls: list["TraceCall"]) -> list[Finding]:
        raise NotImplementedError
