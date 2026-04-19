"""
In-process span buffer for real-time monk instrumentation.
Accumulates TraceCall records and flushes them to detectors on demand.
"""
from __future__ import annotations
import threading
import time
from collections import defaultdict
from monk.parsers.auto import TraceCall, ToolCall


class SpanBuffer:
    """Thread-safe buffer that accumulates TraceCalls and triggers analysis."""

    def __init__(self, flush_every: int = 5, on_finding=None):
        """
        flush_every: run detectors every N new calls
        on_finding: callback(finding) called when a finding is detected
        """
        self._calls: list[TraceCall] = []
        self._lock = threading.Lock()
        self._flush_every = flush_every
        self._on_finding = on_finding or self._default_handler
        self._call_count = 0

    def add(self, call: TraceCall) -> None:
        with self._lock:
            self._calls.append(call)
            self._call_count += 1
            if self._call_count % self._flush_every == 0:
                self._analyze()

    def flush(self) -> None:
        with self._lock:
            self._analyze()

    def _analyze(self) -> None:
        from monk.detectors import TRACE_DETECTORS
        calls = list(self._calls)
        for detector in TRACE_DETECTORS:
            try:
                findings = detector.run(calls)
                for f in findings:
                    self._on_finding(f)
            except Exception:
                pass

    def get_calls(self) -> list[TraceCall]:
        with self._lock:
            return list(self._calls)

    def _default_handler(self, finding) -> None:
        sev = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(finding.severity, "⚪")
        print(f"[monk] {sev} {finding.title}")
        if finding.estimated_waste_usd_per_day > 0:
            print(f"       Est. waste: ${finding.estimated_waste_usd_per_day:.2f}/day")
        print(f"       Fix: {finding.fix}")
