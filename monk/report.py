"""
Rich terminal report renderer for monk findings.
"""
from __future__ import annotations

from collections import defaultdict
from monk.detectors.base import Finding
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.rule import Rule

console = Console()

SEVERITY_COLOURS = {
    "high":   "bold red",
    "medium": "bold yellow",
    "low":    "bold green",
}

SEVERITY_ICONS = {
    "high":   "🔴",
    "medium": "🟡",
    "low":    "🟢",
}

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def render_report(findings: list[Finding], total_calls: int, source: str) -> None:
    console.print()
    console.print(Panel.fit(
        "[bold white]🕵️  monk — Agentic Workflow Blind Spot Detector[/bold white]\n"
        f"[dim]Source: {source}   |   Calls analysed: {total_calls:,}[/dim]",
        border_style="bright_blue",
    ))
    console.print()

    if not findings:
        console.print(
            Panel(
                "[bold green]✅  No blind spots detected.[/bold green]\n"
                "[dim]monk analysed your traces and found nothing suspicious. "
                "Consider running on more sessions for higher confidence.[/dim]",
                border_style="green",
            )
        )
        return

    # Sort: high first, then by estimated waste descending
    findings = sorted(
        findings,
        key=lambda f: (SEVERITY_ORDER.get(f.severity, 3), -f.estimated_waste_usd_per_day),
    )

    total_waste = sum(f.estimated_waste_usd_per_day for f in findings)
    monthly_waste = total_waste * 30
    high_count = sum(1 for f in findings if f.severity == "high")
    med_count = sum(1 for f in findings if f.severity == "medium")
    low_count = sum(1 for f in findings if f.severity == "low")

    # ── Summary banner ──────────────────────────────────────────────
    summary = Table.grid(padding=(0, 2))
    severity_str = (
        f"[bold red]{high_count} high[/bold red]  "
        f"[bold yellow]{med_count} medium[/bold yellow]  "
        f"[bold green]{low_count} low[/bold green]"
    )
    summary.add_row(
        f"[bold white]{len(findings)} blind spot(s)[/bold white]  {severity_str}",
        f"[bold yellow]~${total_waste:.2f}/day waste[/bold yellow]",
        f"[bold magenta]~${monthly_waste:.0f}/month[/bold magenta]",
    )
    console.print(Panel(summary, border_style="yellow", title="[bold]Summary[/bold]"))
    console.print()

    # ── Individual findings ──────────────────────────────────────────
    for i, f in enumerate(findings, 1):
        colour = SEVERITY_COLOURS.get(f.severity, "white")
        icon = SEVERITY_ICONS.get(f.severity, "⚪")

        header = Text()
        header.append(f"  {icon} [{i}] ", style=colour)
        header.append(f.title, style="bold white")
        if f.estimated_waste_usd_per_day > 0:
            header.append(
                f"  ·  ~${f.estimated_waste_usd_per_day:.2f}/day",
                style="bold yellow"
            )

        body = (
            f"[dim]{f.detail}[/dim]\n\n"
            f"[bold cyan]Fix:[/bold cyan] {f.fix}"
        )
        if f.affected_sessions:
            sessions_str = ", ".join(f.affected_sessions[:3])
            if len(f.affected_sessions) > 3:
                sessions_str += f" (+{len(f.affected_sessions) - 3} more)"
            body += f"\n[dim]Sessions: {sessions_str}[/dim]"

        console.print(Panel(
            body,
            title=header,
            title_align="left",
            border_style=colour.replace("bold ", ""),
            padding=(1, 2),
        ))
        console.print()

    # ── Detector breakdown table ─────────────────────────────────────
    from collections import Counter
    detector_waste: dict[str, float] = defaultdict(float)
    detector_count: dict[str, int] = defaultdict(int)
    detector_sev: dict[str, str] = {}
    for f in findings:
        detector_waste[f.detector] += f.estimated_waste_usd_per_day
        detector_count[f.detector] += 1
        # Track worst severity per detector
        existing = detector_sev.get(f.detector, "low")
        if SEVERITY_ORDER.get(f.severity, 3) < SEVERITY_ORDER.get(existing, 3):
            detector_sev[f.detector] = f.severity

    table = Table(
        title="Blind Spot Breakdown by Detector",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Detector", style="white")
    table.add_column("Count", justify="center")
    table.add_column("Worst Severity", justify="center")
    table.add_column("Waste / day", justify="right", style="yellow")
    table.add_column("Waste / month", justify="right", style="magenta")

    for det in sorted(detector_waste, key=lambda d: -detector_waste[d]):
        sev = detector_sev.get(det, "low")
        icon = SEVERITY_ICONS.get(sev, "⚪")
        table.add_row(
            det,
            str(detector_count[det]),
            f"{icon} {sev}",
            f"${detector_waste[det]:.2f}",
            f"${detector_waste[det] * 30:.0f}",
        )

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{len(findings)}[/bold]",
        "",
        f"[bold yellow]${total_waste:.2f}[/bold yellow]",
        f"[bold magenta]${monthly_waste:.0f}[/bold magenta]",
    )
    console.print(table)
    console.print()

    # ── Top wasteful sessions ────────────────────────────────────────
    session_waste: dict[str, float] = defaultdict(float)
    session_findings: dict[str, int] = defaultdict(int)
    for f in findings:
        for sid in f.affected_sessions:
            session_waste[sid] += f.estimated_waste_usd_per_day
            session_findings[sid] += 1

    if session_waste and max(session_waste.values()) > 0:
        top_sessions = sorted(session_waste.items(), key=lambda x: -x[1])[:5]
        stab = Table(
            title="Top Wasteful Sessions",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
        )
        stab.add_column("Session ID", style="dim white")
        stab.add_column("Findings", justify="center")
        stab.add_column("Waste / day", justify="right", style="yellow")
        stab.add_column("Waste / month", justify="right", style="magenta")

        for sid, waste in top_sessions:
            stab.add_row(
                sid[:24] + ("…" if len(sid) > 24 else ""),
                str(session_findings[sid]),
                f"${waste:.3f}",
                f"${waste * 30:.2f}",
            )
        console.print(stab)
        console.print()

    console.print(Rule(style="dim"))
    console.print(
        "[dim]monk is open source. Found something useful? "
        "⭐ github.com/Blueconomy/monk[/dim]"
    )
    console.print()
