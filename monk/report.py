"""
Rich terminal report renderer for monk findings.
"""
from __future__ import annotations

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

    total_waste = sum(f.estimated_waste_usd_per_day for f in findings)
    monthly_waste = total_waste * 30

    # ── Summary banner ──────────────────────────────────────────────
    summary = Table.grid(padding=(0, 2))
    summary.add_row(
        f"[bold red]{len(findings)} blind spot(s) found[/bold red]",
        f"[bold yellow]~${total_waste:.2f}/day estimated waste[/bold yellow]",
        f"[bold magenta]~${monthly_waste:.0f}/month at current volume[/bold magenta]",
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

    # ── Savings table ────────────────────────────────────────────────
    table = Table(
        title="Potential Savings Breakdown",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Detector", style="white")
    table.add_column("Severity", justify="center")
    table.add_column("Waste / day", justify="right", style="yellow")
    table.add_column("Waste / month", justify="right", style="magenta")

    for f in sorted(findings, key=lambda x: -x.estimated_waste_usd_per_day):
        icon = SEVERITY_ICONS.get(f.severity, "⚪")
        table.add_row(
            f.title[:55] + ("…" if len(f.title) > 55 else ""),
            f"{icon} {f.severity}",
            f"${f.estimated_waste_usd_per_day:.2f}",
            f"${f.estimated_waste_usd_per_day * 30:.0f}",
        )

    table.add_row(
        "[bold]TOTAL[/bold]", "",
        f"[bold yellow]${total_waste:.2f}[/bold yellow]",
        f"[bold magenta]${monthly_waste:.0f}[/bold magenta]",
    )
    console.print(table)
    console.print()

    console.print(Rule(style="dim"))
    console.print(
        "[dim]monk is open source. Found something useful? "
        "⭐ github.com/Blueconomy/monk[/dim]"
    )
    console.print()
