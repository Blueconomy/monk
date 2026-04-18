"""
monk CLI — entry point.

Usage:
    monk run ./traces/
    monk run session.jsonl
    monk run otel_trace.jsonl --format otel
    monk run --help
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from monk import __version__
from monk.parsers import parse_traces, parse_spans, spans_to_trace_calls, is_otel_format
from monk.detectors import TRACE_DETECTORS, SPAN_DETECTORS, ALL_DETECTORS
from monk.report import render_report

console = Console()

SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".log", ".txt"}

DETECTOR_NAMES = {
    # TraceCall detectors
    "retry_loop", "empty_return", "model_overkill", "context_bloat", "agent_loop",
    # Span detectors
    "latency_spike", "error_cascade", "tool_dependency", "cross_turn_memory",
}


@click.group()
@click.version_option(__version__, prog_name="monk")
def main():
    """
    🕵️  monk — Find hidden cost leaks and blind spots in your agentic AI workflows.

    Drop monk on any trace file or folder and get a plain-English report
    of what's wasting tokens — and exactly how to fix it.

    Supports: OpenAI, Anthropic, LangSmith, OpenTelemetry (OTEL/OTLP), generic JSONL.

    Examples:

    \b
        monk run ./traces/
        monk run agent_session.jsonl
        monk run otel_spans.jsonl --format otel
        monk run --json report.json traces/
        monk run --min-severity high traces/
    """
    pass


@main.command()
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--json", "output_json", default=None,
    help="Also write findings to a JSON file.",
    metavar="FILE",
)
@click.option(
    "--min-severity", default="low",
    type=click.Choice(["low", "medium", "high"]),
    help="Only show findings at or above this severity.",
)
@click.option(
    "--detectors", default=None,
    help=(
        "Comma-separated detectors to run (default: all). "
        "TraceCall: retry_loop, empty_return, model_overkill, context_bloat, agent_loop. "
        "Span: latency_spike, error_cascade, tool_dependency, cross_turn_memory."
    ),
    metavar="LIST",
)
@click.option(
    "--format", "trace_format", default="auto",
    type=click.Choice(["auto", "otel", "openai", "anthropic", "langsmith", "jsonl"]),
    help=(
        "Force a specific trace format instead of auto-detecting. "
        "'otel' enables span-level analysis (latency, error cascade, dependency graph)."
    ),
)
def run(
    source: str,
    output_json: str | None,
    min_severity: str,
    detectors: str | None,
    trace_format: str,
):
    """
    Analyse trace files for blind spots and cost leaks.

    SOURCE can be a single .jsonl/.json file or a directory of trace files.
    Use --format otel to enable deep span-level analysis on OpenTelemetry traces.
    """
    source_path = Path(source)

    # ── Collect files ────────────────────────────────────────────────
    if source_path.is_dir():
        files = sorted([
            f for f in source_path.rglob("*")
            if f.suffix in SUPPORTED_EXTENSIONS
        ])
        if not files:
            console.print(
                f"[yellow]No trace files found in '{source}'.[/yellow]\n"
                f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
            sys.exit(1)
    else:
        files = [source_path]

    # ── Determine if OTEL path is active ────────────────────────────
    use_otel = trace_format == "otel"

    # Auto-detect OTEL on first file if format=auto
    if trace_format == "auto" and files:
        try:
            sample = files[0].read_text(encoding="utf-8")
            use_otel = is_otel_format(sample)
            if use_otel:
                console.print(
                    "[dim]Auto-detected OpenTelemetry format — enabling span-level analysis.[/dim]"
                )
        except Exception:
            pass

    # ── Select detectors ──────────────────────────────────────────────
    if detectors:
        requested = {d.strip() for d in detectors.split(",")}
        unknown = requested - DETECTOR_NAMES
        if unknown:
            console.print(f"[red]Unknown detector(s): {', '.join(sorted(unknown))}[/red]")
            console.print(f"Available: {', '.join(sorted(DETECTOR_NAMES))}")
            sys.exit(1)
        active_trace_detectors = [d for d in TRACE_DETECTORS if d.name in requested]
        active_span_detectors = [d for d in SPAN_DETECTORS if d.name in requested]
    else:
        active_trace_detectors = TRACE_DETECTORS
        active_span_detectors = SPAN_DETECTORS if use_otel else []

    severity_order = {"low": 0, "medium": 1, "high": 2}
    min_sev = severity_order[min_severity]

    findings = []
    total_calls = 0

    # ── OTEL path: parse spans + run both detector sets ──────────────
    if use_otel:
        all_roots = []
        for f in files:
            try:
                roots = parse_spans(f)
                all_roots.extend(roots)
            except Exception as e:
                console.print(f"[yellow]Warning: could not parse '{f.name}': {e}[/yellow]")

        if not all_roots:
            console.print(
                "[red]No valid OTEL spans found.[/red]\n"
                "monk expects OTLP proto-JSON or JSONL with traceId/spanId fields."
            )
            sys.exit(1)

        # Extract TraceCall objects for trace-level detectors
        all_calls = spans_to_trace_calls(all_roots)
        total_calls = len(all_calls) + sum(
            len(r.all_descendants()) + 1 for r in all_roots
        )

        # Run TraceCall detectors on extracted calls
        for detector in active_trace_detectors:
            try:
                results = detector.run(all_calls)
                findings.extend([
                    f for f in results
                    if severity_order.get(f.severity, 0) >= min_sev
                ])
            except Exception as e:
                console.print(f"[yellow]Warning: detector '{detector.name}' failed: {e}[/yellow]")

        # Run span detectors on span trees
        for detector in active_span_detectors:
            try:
                results = detector.run_spans(all_roots)
                findings.extend([
                    f for f in results
                    if severity_order.get(f.severity, 0) >= min_sev
                ])
            except Exception as e:
                console.print(f"[yellow]Warning: detector '{detector.name}' failed: {e}[/yellow]")

    # ── Standard path: parse TraceCall objects ───────────────────────
    else:
        all_calls = []
        for f in files:
            try:
                calls = parse_traces(f)
                all_calls.extend(calls)
            except Exception as e:
                console.print(f"[yellow]Warning: could not parse '{f.name}': {e}[/yellow]")

        if not all_calls:
            console.print(
                "[red]No valid trace records found.[/red]\n"
                "monk expects JSONL files with fields: model, input_tokens, output_tokens, session_id.\n"
                "For OpenTelemetry traces, use: monk run <file> --format otel\n"
                "See: github.com/Blueconomy/monk#trace-format"
            )
            sys.exit(1)

        total_calls = len(all_calls)

        for detector in active_trace_detectors:
            try:
                results = detector.run(all_calls)
                findings.extend([
                    f for f in results
                    if severity_order.get(f.severity, 0) >= min_sev
                ])
            except Exception as e:
                console.print(f"[yellow]Warning: detector '{detector.name}' failed: {e}[/yellow]")

        if active_span_detectors and not use_otel:
            console.print(
                "[dim]Note: span detectors (latency_spike, error_cascade, tool_dependency, "
                "cross_turn_memory) require OTEL format. Use --format otel.[/dim]"
            )

    # ── Sort findings ─────────────────────────────────────────────────
    findings.sort(key=lambda f: (
        -severity_order.get(f.severity, 0),
        -f.estimated_waste_usd_per_day,
    ))

    # ── Render report ─────────────────────────────────────────────────
    render_report(findings, total_calls, str(source))

    # ── Optional JSON output ──────────────────────────────────────────
    if output_json:
        import json as json_lib
        data = [
            {
                "detector": f.detector,
                "severity": f.severity,
                "title": f.title,
                "detail": f.detail,
                "affected_sessions": f.affected_sessions,
                "estimated_waste_usd_per_day": round(f.estimated_waste_usd_per_day, 4),
                "fix": f.fix,
            }
            for f in findings
        ]
        Path(output_json).write_text(json_lib.dumps(data, indent=2))
        console.print(f"[dim]JSON report written to: {output_json}[/dim]")

    # Exit code: 1 if high-severity findings exist (useful in CI)
    if any(f.severity == "high" for f in findings):
        sys.exit(1)
