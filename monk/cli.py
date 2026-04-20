"""
monk CLI — entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.text import Text

from monk import __version__
from monk.parsers import parse_traces, parse_spans, spans_to_trace_calls, is_otel_format
from monk.detectors import TRACE_DETECTORS, SPAN_DETECTORS, ALL_DETECTORS
from monk.report import render_report

console = Console()

SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".log", ".txt"}

DETECTOR_NAMES = {
    # TraceCall detectors
    "retry_loop", "empty_return", "model_overkill", "context_bloat", "agent_loop", "text_io",
    # Span detectors
    "latency_spike", "error_cascade", "tool_dependency", "cross_turn_memory",
}

# ── Public HuggingFace dataset files ──────────────────────────────────────────
HF_BASE = "https://huggingface.co/datasets/Blueconomy/monk-benchmarks/resolve/main"
DEMO_DATASETS = [
    {
        "name":    "taubench_traces.jsonl",
        "label":   "tau-bench",
        "desc":    "17,932 calls · banking + e-commerce agents",
        "size_mb": 9.9,
    },
    {
        "name":    "finance_traces.jsonl",
        "label":   "finance (10-K ReAct)",
        "desc":    "4,610 calls · LangGraph financial analysis",
        "size_mb": 3.9,
    },
    {
        "name":    "trail_otel.jsonl",
        "label":   "TRAIL (PatronusAI)",
        "desc":    "879 spans · ground-truth benchmark (OTEL)",
        "size_mb": 25.8,
    },
]

# Synthetic sample — used as fallback when HF download fails
SYNTHETIC_SAMPLE = [
    {"session_id":"sess_a","model":"gpt-4o","input_tokens":1800,"output_tokens":120,"tool_name":"calculator","tool_result":"42"},
    {"session_id":"sess_a","model":"gpt-4o","input_tokens":1820,"output_tokens":110,"tool_name":"calculator","tool_result":"42"},
    {"session_id":"sess_a","model":"gpt-4o","input_tokens":1840,"output_tokens":115,"tool_name":"calculator","tool_result":"42"},
    {"session_id":"sess_a","model":"gpt-4o","input_tokens":1860,"output_tokens":118,"tool_name":"calculator","tool_result":"42"},
    {"session_id":"sess_b","model":"gpt-4o","input_tokens":2100,"output_tokens":130,"tool_name":"web_search","tool_result":""},
    {"session_id":"sess_b","model":"gpt-4o","input_tokens":2120,"output_tokens":140,"tool_name":"web_search","tool_result":None},
    {"session_id":"sess_b","model":"gpt-4o","input_tokens":2140,"output_tokens":125,"tool_name":"web_search","tool_result":""},
    {"session_id":"sess_c","model":"gpt-4o","input_tokens":18000,"output_tokens":200,"system_prompt_tokens":13000,"tool_name":"lookup","tool_result":"result"},
    {"session_id":"sess_c","model":"gpt-4o","input_tokens":19200,"output_tokens":180,"system_prompt_tokens":13000,"tool_name":"lookup","tool_result":"result2"},
    {"session_id":"sess_d","model":"gpt-4o","input_tokens":1600,"output_tokens":100,"tool_name":"search","tool_result":"results"},
    {"session_id":"sess_d","model":"gpt-4o","input_tokens":1700,"output_tokens":110,"tool_name":"summarize","tool_result":"summary"},
    {"session_id":"sess_d","model":"gpt-4o","input_tokens":1650,"output_tokens":105,"tool_name":"search","tool_result":"results"},
    {"session_id":"sess_d","model":"gpt-4o","input_tokens":1720,"output_tokens":112,"tool_name":"summarize","tool_result":"summary"},
    {"session_id":"sess_d","model":"gpt-4o","input_tokens":1660,"output_tokens":108,"tool_name":"search","tool_result":"results"},
    {"session_id":"sess_d","model":"gpt-4o","input_tokens":1730,"output_tokens":115,"tool_name":"summarize","tool_result":"summary"},
    {"session_id":"sess_e","model":"gpt-4o","input_tokens":300,"output_tokens":40,"tool_name":None,"tool_result":None},
    {"session_id":"sess_e","model":"gpt-4o","input_tokens":280,"output_tokens":35,"tool_name":None,"tool_result":None},
    {"session_id":"sess_f","model":"gpt-4o-mini","input_tokens":900,"output_tokens":200,"tool_name":"db_query","tool_result":"[{id:1}]"},
]


def _print_home():
    """Print the monk home screen — shown when monk is run with no arguments."""
    console.print()
    console.print("  🕵️  [bold white]monk[/bold white] [dim]v{}[/dim]".format(__version__))
    console.print()
    console.print("  [dim]Find hidden cost leaks and blind spots in your AI agents.[/dim]")
    console.print()
    console.print("  [dim]─────────────────────────────────────────────────────────[/dim]")
    console.print()

    console.print("  [bold white]Getting started[/bold white]")
    console.print()
    console.print("    [bold orange1]monk quickstart[/bold orange1]                  demo data + live dashboard in one command")
    console.print("    [bold orange1]monk demo[/bold orange1]                        download real agent data and run analysis")
    console.print("    [bold orange1]monk init[/bold orange1]                        scaffold a traces/ folder + config")
    console.print()
    console.print("  [bold white]Core commands[/bold white]")
    console.print()
    console.print("    [orange1]monk run[/orange1] [cyan]./traces/[/cyan]               analyse a folder of trace files")
    console.print("    [orange1]monk run[/orange1] [cyan]agent.jsonl[/cyan]             analyse a single file")
    console.print("    [orange1]monk run[/orange1] [cyan]traces/[/cyan] [dim]--min-severity high[/dim]  surface only critical findings")
    console.print("    [orange1]monk run[/orange1] [cyan]traces/[/cyan] [dim]--json report.json[/dim]   export findings for CI")
    console.print()
    console.print("  [bold white]Live dashboard[/bold white]")
    console.print()
    console.print("    [orange1]monk serve[/orange1] [cyan]./traces/[/cyan] [dim]--port 9090[/dim]     start dashboard at localhost:9090")
    console.print("    [dim]Auto-refreshes every 15s · Prometheus /metrics · dataset downloader[/dim]")
    console.print()
    console.print("  [bold white]Real-time instrumentation[/bold white] [dim](zero config)[/dim]")
    console.print()
    console.print("    [dim]import[/dim] [orange1]monk[/orange1]")
    console.print("    [orange1]monk[/orange1][dim].instrument()    # patches openai + anthropic at import time[/dim]")
    console.print()
    console.print("  [dim]─────────────────────────────────────────────────────────[/dim]")
    console.print()
    console.print("  [dim]Docs · github.com/Blueconomy/monk[/dim]")
    console.print("  [dim]PyPI · pip install monk-ai[/dim]")
    console.print("  [dim]MIT · Blueconomy AI · Techstars '25[/dim]")
    console.print()


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="monk")
@click.pass_context
def main(ctx):
    """🕵️  monk — Find hidden cost leaks in your AI agents."""
    if ctx.invoked_subcommand is None:
        _print_home()


@main.command()
def init():
    """Scaffold a traces/ folder and show next steps."""
    import os

    console.print()
    console.print("  🕵️  [bold white]monk init[/bold white]")
    console.print()

    # Create traces/ folder
    traces_dir = Path("traces")
    created = not traces_dir.exists()
    traces_dir.mkdir(exist_ok=True)

    if created:
        console.print("  [green]✓[/green]  Created [cyan]./traces/[/cyan]")
    else:
        console.print("  [dim]·[/dim]  [cyan]./traces/[/cyan] already exists")

    # Create .gitignore entry
    gitignore = Path(".gitignore")
    gitignore_entry = "traces/*.jsonl\n"
    if gitignore.exists():
        if gitignore_entry.strip() not in gitignore.read_text():
            gitignore.write_text(gitignore.read_text() + gitignore_entry)
            console.print("  [green]✓[/green]  Added traces/*.jsonl to [cyan].gitignore[/cyan]")
    else:
        gitignore.write_text(gitignore_entry)
        console.print("  [green]✓[/green]  Created [cyan].gitignore[/cyan]")

    console.print()
    console.print("  [dim]─────────────────────────────────────────[/dim]")
    console.print()
    console.print("  [bold white]Next steps[/bold white]")
    console.print()
    console.print("  [dim]1.[/dim]  Download sample data to try monk immediately:")
    console.print()
    console.print("       [bold orange1]monk demo[/bold orange1]")
    console.print()
    console.print("  [dim]2.[/dim]  Or drop your own trace files into [cyan]./traces/[/cyan]")
    console.print("       monk accepts: OpenAI, Anthropic, LangSmith, OTEL, generic JSONL")
    console.print()
    console.print("  [dim]3.[/dim]  Run analysis:")
    console.print()
    console.print("       [orange1]monk run[/orange1] [cyan]./traces/[/cyan]")
    console.print()
    console.print("  [dim]4.[/dim]  Start the live metrics server:")
    console.print()
    console.print("       [orange1]monk serve[/orange1] [cyan]./traces/[/cyan] [dim]--port 9090[/dim]")
    console.print("       [dim]Then open monk_dashboard.html in your browser.[/dim]")
    console.print()
    console.print("  [dim]5.[/dim]  Instrument your agent (zero config):")
    console.print()
    console.print("       [dim]import[/dim] [orange1]monk[/orange1]")
    console.print("       [orange1]monk[/orange1][dim].instrument()  # patches openai + anthropic[/dim]")
    console.print()


@main.command()
@click.option("--dataset", default="all",
              type=click.Choice(["all", "taubench", "finance", "trail"]),
              help="Which sample dataset to download (default: all).")
@click.option("--dir", "dest_dir", default="./traces",
              help="Destination folder (default: ./traces).")
@click.option("--no-run", is_flag=True, default=False,
              help="Download only, skip analysis.")
def demo(dataset, dest_dir, no_run):
    """Download real agent trace data and run analysis.

    \b
    Pulls from: huggingface.co/datasets/Blueconomy/monk-benchmarks

    Datasets available:
      taubench  — 17,932 calls, banking + e-commerce agents
      finance   — 4,610 calls, LangGraph financial analysis (10-K ReAct)
      trail     — 879 spans, PatronusAI TRAIL benchmark (OpenTelemetry)
    """
    import urllib.request
    import urllib.error

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Pick which datasets to download
    if dataset == "all":
        targets = DEMO_DATASETS
    else:
        key_map = {"taubench": 0, "finance": 1, "trail": 2}
        targets = [DEMO_DATASETS[key_map[dataset]]]

    console.print()
    console.print("  🕵️  [bold white]monk demo[/bold white]")
    console.print()
    console.print(
        "  Downloading real agent traces from "
        "[cyan]huggingface.co/datasets/Blueconomy/monk-benchmarks[/cyan]"
    )
    console.print()

    downloaded = []
    for ds in targets:
        url  = f"{HF_BASE}/{ds['name']}"
        dest_file = dest / ds["name"]

        if dest_file.exists():
            console.print(
                f"  [dim]·[/dim]  [cyan]{ds['name']}[/cyan]  "
                f"[dim]already downloaded — skipping[/dim]"
            )
            downloaded.append(dest_file)
            continue

        console.print(
            f"  [dim]↓[/dim]  [cyan]{ds['name']}[/cyan]  "
            f"[dim]{ds['desc']} · ~{ds['size_mb']}MB[/dim]",
            end="",
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"monk/{__version__}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            dest_file.write_bytes(data)
            kb = len(data) // 1024
            console.print(f"  [green]✓[/green]  ({kb:,} KB)")
            downloaded.append(dest_file)
        except urllib.error.HTTPError as e:
            console.print(f"  [red]✗[/red]  HTTP {e.code}")
        except Exception as e:
            console.print(f"  [red]✗[/red]  {e}")

    if not downloaded:
        console.print()
        console.print("  [yellow]Could not reach HuggingFace — generating a synthetic sample instead.[/yellow]")
        import json as _json
        fallback = dest / "sample_agent.jsonl"
        fallback.write_text("\n".join(_json.dumps(r) for r in SYNTHETIC_SAMPLE))
        console.print(f"  [green]✓[/green]  Created [cyan]{fallback}[/cyan]  (18 records, 6 sessions)")
        downloaded.append(fallback)
        console.print()

    console.print()

    if no_run:
        console.print(f"  Files saved to [cyan]{dest}[/cyan]")
        console.print()
        console.print(f"  Run:  [orange1]monk run[/orange1] [cyan]{dest}[/cyan]")
        console.print()
        return

    # Run analysis on each file individually so we see per-file results
    console.print("  [dim]─────────────────────────────────────────[/dim]")
    console.print()

    from monk.parsers.auto import parse_traces
    from monk.parsers.otel import is_otel_format, parse_spans, spans_to_trace_calls
    from monk.report import render_report

    severity_order = {"low": 0, "medium": 1, "high": 2}

    for f in downloaded:
        text = f.read_text(encoding="utf-8", errors="replace")

        if is_otel_format(text):
            roots = parse_spans(f)
            calls = spans_to_trace_calls(roots)
            findings = []
            for det in ALL_DETECTORS:
                if det.requires_spans:
                    findings.extend(det.run_spans(roots))
                else:
                    findings.extend(det.run(calls))
        else:
            calls = parse_traces(str(f))
            findings = []
            for det in ALL_DETECTORS:
                if not det.requires_spans:
                    findings.extend(det.run(calls))

        findings.sort(key=lambda x: (
            -severity_order.get(x.severity, 0),
            -x.estimated_waste_usd_per_day,
        ))
        render_report(findings, len(calls), str(f))

    console.print()
    console.print(
        "  [dim]Tip:[/dim] start the live dashboard with "
        "[orange1]monk serve[/orange1] [cyan]./traces/[/cyan] [dim]--port 9090[/dim]"
    )
    console.print()


@main.command()
@click.option("--port", default=9090, show_default=True, help="Dashboard port.")
@click.option("--dir", "dest_dir", default="./traces", show_default=True,
              help="Directory to write demo traces and watch.")
def quickstart(port: int, dest_dir: str) -> None:
    """Write built-in demo data, analyze it, and open the live dashboard.

    \b
    This is the fastest way to see monk in action:

      monk quickstart          # uses ./traces on port 9090
      monk quickstart --port 8080
    """
    import json as _json
    from monk.serve import DEMO_TRACES_JSONL, serve as _serve

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    sample_file = dest / "demo_traces.jsonl"

    console.print()
    console.print("  🕵️  [bold white]monk quickstart[/bold white]")
    console.print()

    if not sample_file.exists():
        sample_file.write_text(DEMO_TRACES_JSONL)
        console.print(f"  [green]✓[/green]  Generated demo traces → [cyan]{sample_file}[/cyan]")
    else:
        console.print(f"  [dim]·[/dim]  Using existing [cyan]{sample_file}[/cyan]")

    console.print()

    # Quick analysis pass
    from monk.parsers.auto import parse_traces as _pt
    from monk.report import render_report as _rr

    calls = _pt(str(sample_file))
    findings = []
    for det in ALL_DETECTORS:
        if not det.requires_spans:
            findings.extend(det.run(calls))

    findings.sort(key=lambda x: (
        -{"low": 0, "medium": 1, "high": 2}.get(x.severity, 0),
        -x.estimated_waste_usd_per_day,
    ))
    _rr(findings, len(calls), str(sample_file))

    console.print()
    console.print(f"  Starting live dashboard on [orange1]http://localhost:{port}/[/orange1]")
    console.print(f"  Watching [cyan]{dest}[/cyan] for new trace files  [dim](Ctrl+C to stop)[/dim]")
    console.print()

    _serve(dest_dir, port=port)


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
@click.option(
    "--samples", "samples_path", default=None,
    help="Save labeled text I/O samples (for LLM judge training) to this JSONL file.",
    metavar="FILE",
)
def run(
    source: str,
    output_json: str | None,
    min_severity: str,
    detectors: str | None,
    trace_format: str,
    samples_path: str | None,
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

    # ── Optional samples output ───────────────────────────────────────
    if samples_path:
        from monk.samples import collect_samples_from_run
        source_name = Path(source).name
        collector = collect_samples_from_run(
            calls=all_calls,
            findings=findings,
            output_path=samples_path,
            source=source_name,
        )
        stats = collector.stats()
        n = collector.flush()
        console.print(
            f"[dim]Saved {n} samples to {samples_path} "
            f"({stats['bad']} bad, {stats['good']} good)[/dim]"
        )

    # Exit code: 1 if high-severity findings exist (useful in CI)
    if any(f.severity == "high" for f in findings):
        sys.exit(1)


@main.command()
@click.argument("path", default="./traces")
@click.option("--port", default=9090, help="Metrics server port")
@click.option("--interval", default=30, help="Rescan interval in seconds")
def serve(path, port, interval):
    """Start a Prometheus metrics server that watches PATH for trace files."""
    from monk.serve import serve as _serve
    _serve(path, port=port, interval=interval)
