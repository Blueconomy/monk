"""
monk CLI — entry point.

Usage:
    monk run ./traces/
    monk run session.jsonl
    monk run --help
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from monk import __version__
from monk.parsers import parse_traces
from monk.detectors import ALL_DETECTORS
from monk.report import render_report

console = Console()

SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".log", ".txt"}


@click.group()
@click.version_option(__version__, prog_name="monk")
def main():
    """
    🕵️  monk — Find hidden cost leaks and blind spots in your agentic AI workflows.

    Drop monk on any trace file or folder and get a plain-English report
    of what's wasting tokens — and exactly how to fix it.

    Examples:

    \b
        monk run ./traces/
        monk run agent_session.jsonl
        monk run --json report.json traces/
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
    help="Comma-separated list of detectors to run (default: all). "
         "Choices: retry_loop, empty_return, model_overkill, context_bloat, agent_loop",
    metavar="LIST",
)
def run(source: str, output_json: str | None, min_severity: str, detectors: str | None):
    """
    Analyse trace files for blind spots and cost leaks.

    SOURCE can be a single .jsonl file or a directory of trace files.
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

    # ── Parse ────────────────────────────────────────────────────────
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
            "monk expects JSONL files with fields like: model, input_tokens, output_tokens, session_id.\n"
            "See: github.com/blueconomy-ai/monk#trace-format"
        )
        sys.exit(1)

    # ── Run detectors ────────────────────────────────────────────────
    active_detectors = ALL_DETECTORS
    if detectors:
        names = {d.strip() for d in detectors.split(",")}
        active_detectors = [d for d in ALL_DETECTORS if d.name in names]
        if not active_detectors:
            console.print(f"[red]No matching detectors for: {detectors}[/red]")
            sys.exit(1)

    severity_order = {"low": 0, "medium": 1, "high": 2}
    min_sev = severity_order[min_severity]

    findings = []
    for detector in active_detectors:
        try:
            results = detector.run(all_calls)
            findings.extend([
                f for f in results
                if severity_order.get(f.severity, 0) >= min_sev
            ])
        except Exception as e:
            console.print(f"[yellow]Warning: detector '{detector.name}' failed: {e}[/yellow]")

    # Sort: high first, then by estimated waste desc
    findings.sort(key=lambda f: (
        -severity_order.get(f.severity, 0),
        -f.estimated_waste_usd_per_day
    ))

    # ── Render ───────────────────────────────────────────────────────
    render_report(findings, len(all_calls), str(source))

    # ── Optional JSON output ─────────────────────────────────────────
    if output_json:
        import json
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
        Path(output_json).write_text(json.dumps(data, indent=2))
        console.print(f"[dim]JSON report written to: {output_json}[/dim]")

    # Exit code: 1 if high-severity findings exist (useful in CI)
    if any(f.severity == "high" for f in findings):
        sys.exit(1)
