"""
monk serve — watches trace files and exposes Prometheus metrics + JSON endpoint.

Exposes:
  GET /metrics   — Prometheus text format
  GET /findings  — JSON summary of all findings
  GET /health    — liveness check

Usage:
  monk serve ./traces/ --port 9090 --interval 30
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from monk.detectors import ALL_DETECTORS
from monk.parsers.auto import parse_traces


class MetricsStore:
    """Thread-safe Prometheus metrics store."""

    def __init__(self):
        self._lock = threading.Lock()
        self._findings_count: dict[tuple, int] = {}
        self._waste: dict[str, float] = {}
        self._sessions_total = 0
        self._calls_total = 0
        self._last_scan_ts = 0.0
        self._high_total = 0
        self._recent: list[dict] = []

    def update(self, findings, calls, sessions: int) -> None:
        with self._lock:
            self._sessions_total += sessions
            self._calls_total += len(calls)
            self._last_scan_ts = time.time()

            for f in findings:
                key = (f.detector, f.severity)
                self._findings_count[key] = self._findings_count.get(key, 0) + 1
                self._waste[f.detector] = (
                    self._waste.get(f.detector, 0.0) + f.estimated_waste_usd_per_day
                )
                if f.severity == "high":
                    self._high_total += 1

                self._recent.append({
                    "detector": f.detector,
                    "severity": f.severity,
                    "title": f.title,
                    "detail": f.detail,
                    "fix": f.fix,
                    "sessions": f.affected_sessions[:3],
                    "waste_per_day": round(f.estimated_waste_usd_per_day, 4),
                    "ts": self._last_scan_ts,
                })

            if len(self._recent) > 500:
                self._recent = self._recent[-500:]

    def to_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP monk_findings_total Total findings by detector and severity",
                "# TYPE monk_findings_total counter",
            ]
            for (det, sev), count in sorted(self._findings_count.items()):
                lines.append(f'monk_findings_total{{detector="{det}",severity="{sev}"}} {count}')

            lines += [
                "# HELP monk_waste_usd_per_day Estimated daily waste USD by detector",
                "# TYPE monk_waste_usd_per_day gauge",
            ]
            for det, waste in sorted(self._waste.items()):
                lines.append(f'monk_waste_usd_per_day{{detector="{det}"}} {waste:.4f}')

            lines += [
                "# HELP monk_sessions_analyzed_total Total agent sessions analyzed",
                "# TYPE monk_sessions_analyzed_total counter",
                f"monk_sessions_analyzed_total {self._sessions_total}",
                "# HELP monk_calls_analyzed_total Total LLM calls analyzed",
                "# TYPE monk_calls_analyzed_total counter",
                f"monk_calls_analyzed_total {self._calls_total}",
                "# HELP monk_last_scan_timestamp Unix timestamp of last scan",
                "# TYPE monk_last_scan_timestamp gauge",
                f"monk_last_scan_timestamp {self._last_scan_ts:.0f}",
                "# HELP monk_high_severity_findings_total High-severity findings total",
                "# TYPE monk_high_severity_findings_total counter",
                f"monk_high_severity_findings_total {self._high_total}",
            ]
            return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        with self._lock:
            total_waste = sum(self._waste.values())
            total = sum(self._findings_count.values())
            return json.dumps({
                "summary": {
                    "total_findings": total,
                    "high": sum(v for (_, s), v in self._findings_count.items() if s == "high"),
                    "medium": sum(v for (_, s), v in self._findings_count.items() if s == "medium"),
                    "low": sum(v for (_, s), v in self._findings_count.items() if s == "low"),
                    "sessions_analyzed": self._sessions_total,
                    "calls_analyzed": self._calls_total,
                    "waste_usd_per_day": round(total_waste, 2),
                    "waste_usd_per_month": round(total_waste * 30, 2),
                    "last_scan": self._last_scan_ts,
                },
                "by_detector": {
                    det: {"waste_per_day": round(w, 4)}
                    for det, w in sorted(self._waste.items())
                },
                "recent_findings": self._recent[-20:],
            }, indent=2)


class _Handler(BaseHTTPRequestHandler):
    store: MetricsStore

    def do_GET(self):
        if self.path == "/metrics":
            body = self.store.to_prometheus().encode()
            self._respond(200, "text/plain; version=0.0.4", body)
        elif self.path == "/findings":
            body = self.store.to_json().encode()
            self._respond(200, "application/json", body)
        elif self.path in ("/health", "/"):
            self._respond(200, "text/plain", b"ok")
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _scan(path: Path, store: MetricsStore) -> None:
    from monk.parsers.otel import is_otel_format, parse_spans, spans_to_trace_calls

    files = list(path.rglob("*.jsonl")) if path.is_dir() else [path]

    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            if is_otel_format(text):
                roots = parse_spans(text)
                calls = spans_to_trace_calls(roots)
                findings = []
                for det in ALL_DETECTORS:
                    findings.extend(
                        det.run_spans(roots) if det.requires_spans else det.run(calls)
                    )
            else:
                calls = parse_traces(str(f))
                findings = []
                for det in ALL_DETECTORS:
                    if not det.requires_spans:
                        findings.extend(det.run(calls))

            sessions = len({c.session_id for c in calls})
            store.update(findings, calls, sessions)
            print(f"[monk] {f.name}: {len(calls)} calls, {len(findings)} findings")
        except Exception as e:
            print(f"[monk] error scanning {f.name}: {e}")


def serve(path: str, port: int = 9090, interval: int = 30) -> None:
    store = MetricsStore()
    target = Path(path)

    if target.exists():
        _scan(target, store)

    def _watch():
        while True:
            time.sleep(interval)
            if target.exists():
                _scan(target, store)

    threading.Thread(target=_watch, daemon=True).start()

    class Handler(_Handler):
        pass
    Handler.store = store

    httpd = HTTPServer(("0.0.0.0", port), Handler)
    print(f"[monk] metrics  -> http://0.0.0.0:{port}/metrics")
    print(f"[monk] findings -> http://0.0.0.0:{port}/findings")
    print(f"[monk] watching -> {target}  (every {interval}s)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[monk] stopped.")
