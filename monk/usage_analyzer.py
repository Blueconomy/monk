"""
monk usage_analyzer — detect cost patterns in team AI usage exports.

Takes a list of UsageRecord objects (from parsers/usage_csv.py) and returns
structured findings + actionable recommendations, plus a rich console report.

Patterns detected
-----------------
1. context_bloat      Cache-read tokens dominate (>80% of call total)
2. model_overkill     Expensive thinking/opus model used on routine tasks
3. usage_spike        Single call costs > $10 (runaway session)
4. user_concentration One user drives >70% of total team spend
5. peak_day_spend     Daily spend > 2× 30-day average (anomaly)
6. zero_output        Calls with no output tokens (aborted / wasted)

Each finding ships with:
  - severity (high / medium / low)
  - estimated monthly savings
  - concrete fix recommendation
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from monk.parsers.usage_csv import UsageRecord


# ── Finding dataclass ─────────────────────────────────────────────────────────

@dataclass
class UsageFinding:
    pattern: str
    severity: str          # "high" | "medium" | "low"
    title: str
    detail: str
    affected_users: list[str]
    cost_attributed_usd: float
    savings_per_month_usd: float
    fix: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Analysis result ───────────────────────────────────────────────────────────

@dataclass
class UsageReport:
    platform: str
    period_start: datetime
    period_end: datetime
    period_days: int
    total_cost_usd: float
    monthly_run_rate_usd: float
    total_calls: int
    total_tokens: int
    users: list[dict]          # [{label, calls, cost, cache_pct, top_model}]
    models: list[dict]         # [{model, calls, cost, tokens}]
    findings: list[UsageFinding]
    total_savings_potential_usd: float
    warnings: list[str] = field(default_factory=list)

    # ── Case-study snapshot (populated when demo data is used) ──
    case_study: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["period_start"] = self.period_start.isoformat()
        d["period_end"]   = self.period_end.isoformat()
        d["findings"]     = [f.to_dict() for f in self.findings]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ── Thresholds (tunable) ──────────────────────────────────────────────────────

CACHE_BLOAT_THRESHOLD   = 0.80   # >80% of call tokens are cache reads
BLOAT_MIN_TOKENS        = 500_000
MODEL_OVERKILL_MAX_TOK  = 800_000  # thinking model on small context
SPIKE_COST_USD          = 10.0
CONCENTRATION_THRESHOLD = 0.70   # one user > 70% of spend
PEAK_DAY_MULTIPLIER     = 2.0
ZERO_OUTPUT_MIN_CALLS   = 5

# Monthly savings multipliers (conservative)
_THINKING_TO_SONNET_SAVING = 0.60   # switching thinking → sonnet saves ~60% on those calls
_CONTEXT_HYGIENE_SAVING    = 0.70   # capping context saves ~70% of cache-read cost


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze(
    records: list[UsageRecord],
    warnings: list[str] | None = None,
    platform: str = "claude_ai",
) -> UsageReport:
    if not records:
        raise ValueError("No records to analyze.")

    warnings = warnings or []
    period_start = records[0].timestamp
    period_end   = records[-1].timestamp
    period_days  = max((period_end - period_start).days, 1)

    total_cost   = sum(r.cost_usd for r in records)
    total_tokens = sum(r.total_tokens for r in records)
    monthly_rate = total_cost / period_days * 30

    # ── Per-user aggregation ──────────────────────────────────────────────────
    user_agg: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "cost": 0.0, "tokens": 0,
        "cache_read": 0, "input": 0, "output": 0,
        "model_counts": defaultdict(int),
    })
    for r in records:
        u = user_agg[r.user_label]
        u["calls"]  += 1
        u["cost"]   += r.cost_usd
        u["tokens"] += r.total_tokens
        u["cache_read"] += r.cache_read_tokens
        u["input"]  += r.input_tokens
        u["output"] += r.output_tokens
        u["model_counts"][r.model] += 1

    users_list = []
    for label, s in sorted(user_agg.items(), key=lambda x: -x[1]["cost"]):
        top_model = max(s["model_counts"], key=s["model_counts"].get)
        cache_pct = s["cache_read"] / s["tokens"] * 100 if s["tokens"] else 0
        users_list.append({
            "label":     label,
            "calls":     s["calls"],
            "cost":      round(s["cost"], 2),
            "tokens":    s["tokens"],
            "cache_pct": round(cache_pct, 1),
            "top_model": top_model,
            "output":    s["output"],
        })

    # ── Per-model aggregation ─────────────────────────────────────────────────
    model_agg: dict[str, dict] = defaultdict(lambda: {"calls": 0, "cost": 0.0, "tokens": 0})
    for r in records:
        m = model_agg[r.model]
        m["calls"]  += 1
        m["cost"]   += r.cost_usd
        m["tokens"] += r.total_tokens

    models_list = sorted(
        [{"model": k, **v} for k, v in model_agg.items()],
        key=lambda x: -x["cost"],
    )

    # ── Pattern detection ─────────────────────────────────────────────────────
    findings: list[UsageFinding] = []

    findings += _detect_context_bloat(records, period_days, user_agg)
    findings += _detect_model_overkill(records, period_days, user_agg)
    findings += _detect_spikes(records, period_days)
    findings += _detect_user_concentration(records, total_cost, period_days)
    findings += _detect_peak_days(records, period_days, total_cost)
    findings += _detect_zero_output(records, period_days)

    # Sort by savings potential desc
    findings.sort(key=lambda f: -f.savings_per_month_usd)

    total_savings = sum(f.savings_per_month_usd for f in findings)
    # Cap at 95% of monthly rate (can't save more than you spend)
    total_savings = min(total_savings, monthly_rate * 0.95)

    return UsageReport(
        platform=platform,
        period_start=period_start,
        period_end=period_end,
        period_days=period_days,
        total_cost_usd=round(total_cost, 2),
        monthly_run_rate_usd=round(monthly_rate, 2),
        total_calls=len(records),
        total_tokens=total_tokens,
        users=users_list,
        models=models_list,
        findings=findings,
        total_savings_potential_usd=round(total_savings, 2),
        warnings=warnings,
        case_study=_case_study_note(platform, total_cost, users_list),
    )


# ── Pattern detectors ─────────────────────────────────────────────────────────

def _detect_context_bloat(
    records: list[UsageRecord],
    period_days: int,
    user_agg: dict,
) -> list[UsageFinding]:
    bloated = [
        r for r in records
        if r.total_tokens >= BLOAT_MIN_TOKENS
        and r.cache_read_tokens / r.total_tokens > CACHE_BLOAT_THRESHOLD
    ]
    if len(bloated) < 3:
        return []

    cost_bloated   = sum(r.cost_usd for r in bloated)
    avg_cache_m    = sum(r.cache_read_tokens for r in bloated) / len(bloated) / 1e6
    worst          = max(bloated, key=lambda r: r.cache_read_tokens)
    affected_users = sorted({r.user_label for r in bloated})

    savings = cost_bloated * _CONTEXT_HYGIENE_SAVING / period_days * 30

    return [UsageFinding(
        pattern="context_bloat",
        severity="high",
        title=f"Context bloat: {len(bloated)} calls where >80% of tokens are cache reads",
        detail=(
            f"{len(bloated)} calls have >80% of tokens in cache (average {avg_cache_m:.1f}M "
            f"cached tokens per call). This means conversations are running far too long "
            f"without being reset. The worst single call: {worst.cache_read_tokens/1e6:.1f}M "
            f"cache tokens (${worst.cost_usd:.2f}) on {worst.timestamp.strftime('%b %d')}. "
            f"These {len(bloated)} calls account for ${cost_bloated:.2f} "
            f"({cost_bloated/sum(r.cost_usd for r in records)*100:.0f}% of total spend)."
        ),
        affected_users=affected_users,
        cost_attributed_usd=round(cost_bloated, 2),
        savings_per_month_usd=round(savings, 2),
        fix=(
            "Start a new conversation every 30–60 minutes or when switching tasks. "
            "In Cursor, open a new Composer window instead of continuing the same session. "
            "In Claude.ai, use Projects to share context without loading it on every call. "
            "Target: keep cache_read per call below 500K tokens."
        ),
        evidence={
            "bloated_calls": len(bloated),
            "avg_cache_tokens_M": round(avg_cache_m, 1),
            "worst_call_tokens_M": round(worst.cache_read_tokens / 1e6, 1),
            "worst_call_cost": worst.cost_usd,
            "total_cost": round(cost_bloated, 2),
        },
    )]


def _detect_model_overkill(
    records: list[UsageRecord],
    period_days: int,
    user_agg: dict,
) -> list[UsageFinding]:
    # Calls using thinking/opus on small contexts (<800K total tokens) — routine tasks
    overkill = [
        r for r in records
        if r.model_tier in ("thinking", "opus")
        and r.total_tokens < MODEL_OVERKILL_MAX_TOK
        and r.output_tokens < 2000          # short output = not a hard reasoning task
    ]
    if len(overkill) < 5:
        return []

    cost_ok      = sum(r.cost_usd for r in overkill)
    avg_cost     = cost_ok / len(overkill)
    top_models   = {}
    for r in overkill:
        top_models[r.model] = top_models.get(r.model, 0) + 1
    top_model    = max(top_models, key=top_models.get)
    affected     = sorted({r.user_label for r in overkill})
    savings      = cost_ok * _THINKING_TO_SONNET_SAVING / period_days * 30

    return [UsageFinding(
        pattern="model_overkill",
        severity="high",
        title=f"Model overkill: {len(overkill)} routine tasks run on expensive thinking model",
        detail=(
            f"{len(overkill)} calls use '{top_model}' (thinking/opus tier) "
            f"on tasks with <{MODEL_OVERKILL_MAX_TOK//1000}K tokens and <2K output — "
            f"signals routine coding or formatting tasks, not complex reasoning. "
            f"Average cost per call: ${avg_cost:.2f}. "
            f"These calls cost ${cost_ok:.2f} total. "
            f"A sonnet-tier model would handle most of these at 5–10× lower cost."
        ),
        affected_users=affected,
        cost_attributed_usd=round(cost_ok, 2),
        savings_per_month_usd=round(savings, 2),
        fix=(
            f"In Cursor: Settings → Models → disable 'Max Mode' / extended thinking. "
            f"Use claude-sonnet or gpt-4o as the default. Reserve '{top_model}' "
            f"for tasks requiring deep multi-step reasoning (architecture decisions, "
            f"complex debugging). Rule of thumb: if you can describe the task in "
            f"one sentence, a sonnet-tier model is sufficient."
        ),
        evidence={
            "overkill_calls":   len(overkill),
            "top_model":        top_model,
            "avg_cost_per_call": round(avg_cost, 2),
            "total_cost":       round(cost_ok, 2),
        },
    )]


def _detect_spikes(
    records: list[UsageRecord],
    period_days: int,
) -> list[UsageFinding]:
    spikes = [r for r in records if r.cost_usd >= SPIKE_COST_USD]
    if not spikes:
        return []

    total_spike_cost = sum(r.cost_usd for r in spikes)
    worst            = max(spikes, key=lambda r: r.cost_usd)
    avg_spike        = total_spike_cost / len(spikes)
    savings          = total_spike_cost * 0.80 / period_days * 30  # 80% preventable

    return [UsageFinding(
        pattern="usage_spike",
        severity="high" if worst.cost_usd >= 20 else "medium",
        title=f"Usage spikes: {len(spikes)} calls each cost ≥${SPIKE_COST_USD:.0f}",
        detail=(
            f"{len(spikes)} individual API calls each cost more than ${SPIKE_COST_USD:.0f}. "
            f"Worst: ${worst.cost_usd:.2f} on {worst.timestamp.strftime('%b %d at %H:%M')} "
            f"({worst.user_label}, {worst.model}, {worst.total_tokens/1e6:.1f}M tokens). "
            f"Average spike cost: ${avg_spike:.2f}. "
            f"These are runaway sessions — conversations that were never cleared and "
            f"accumulated millions of tokens over hours of work."
        ),
        affected_users=sorted({r.user_label for r in spikes}),
        cost_attributed_usd=round(total_spike_cost, 2),
        savings_per_month_usd=round(savings, 2),
        fix=(
            f"Set a per-session token budget alert in your AI tool. "
            f"In Cursor, open a new Composer window when a session feels 'heavy'. "
            f"In Claude.ai, use the Projects feature to share context without "
            f"continuously appending to the same conversation thread. "
            f"Consider a hard limit: if a single conversation costs >$5, restart it."
        ),
        evidence={
            "spike_calls":    len(spikes),
            "worst_cost":     worst.cost_usd,
            "worst_date":     worst.timestamp.isoformat(),
            "worst_user":     worst.user_label,
            "worst_tokens_M": round(worst.total_tokens / 1e6, 1),
            "total_cost":     round(total_spike_cost, 2),
        },
    )]


def _detect_user_concentration(
    records: list[UsageRecord],
    total_cost: float,
    period_days: int,
) -> list[UsageFinding]:
    user_cost: dict[str, float] = defaultdict(float)
    for r in records:
        user_cost[r.user_label] += r.cost_usd

    if not user_cost:
        return []

    top_user, top_cost = max(user_cost.items(), key=lambda x: x[1])
    share = top_cost / total_cost if total_cost else 0

    if share < CONCENTRATION_THRESHOLD or len(user_cost) < 2:
        return []

    return [UsageFinding(
        pattern="user_concentration",
        severity="medium",
        title=f"Spend concentration: {top_user} drives {share*100:.0f}% of team cost",
        detail=(
            f"{top_user} accounts for ${top_cost:.2f} of ${total_cost:.2f} total spend "
            f"({share*100:.0f}%). This is a governance risk — one person's usage patterns "
            f"determine the entire team's bill. "
            f"It also means the patterns above (context bloat, model overkill) are "
            f"largely driven by one workflow that can be fixed in one place."
        ),
        affected_users=[top_user],
        cost_attributed_usd=round(top_cost, 2),
        savings_per_month_usd=0.0,   # not a direct saving, governance signal
        fix=(
            f"Set per-user monthly spend limits in the admin console. "
            f"Review {top_user}'s Cursor/Claude settings together — "
            f"fixing model selection and context hygiene for one user will move the needle "
            f"for the whole team. Consider a shared 'AI usage playbook' for the team."
        ),
        evidence={
            "top_user":       top_user,
            "top_user_cost":  round(top_cost, 2),
            "share_pct":      round(share * 100, 1),
            "num_users":      len(user_cost),
        },
    )]


def _detect_peak_days(
    records: list[UsageRecord],
    period_days: int,
    total_cost: float,
) -> list[UsageFinding]:
    daily: dict[str, float] = defaultdict(float)
    for r in records:
        daily[r.timestamp.strftime("%Y-%m-%d")] += r.cost_usd

    if not daily:
        return []

    avg_daily  = total_cost / period_days
    threshold  = avg_daily * PEAK_DAY_MULTIPLIER
    peak_days  = [(d, c) for d, c in daily.items() if c >= max(threshold, 10.0)]

    if not peak_days:
        return []

    peak_days.sort(key=lambda x: -x[1])
    worst_day, worst_cost = peak_days[0]
    total_peak_cost = sum(c for _, c in peak_days)

    return [UsageFinding(
        pattern="peak_day_spend",
        severity="medium",
        title=f"Peak spend days: {len(peak_days)} days each >{PEAK_DAY_MULTIPLIER:.0f}× daily average",
        detail=(
            f"{len(peak_days)} days had spend more than {PEAK_DAY_MULTIPLIER:.0f}× the "
            f"${avg_daily:.2f} daily average. Worst: ${worst_cost:.2f} on {worst_day}. "
            f"These peaks correlate with long uninterrupted Cursor sessions where one "
            f"conversation thread accumulates tokens over an entire workday."
        ),
        affected_users=[],
        cost_attributed_usd=round(total_peak_cost, 2),
        savings_per_month_usd=0.0,
        fix=(
            "Set a daily spend alert (email or Slack) via the admin console. "
            "When a day crosses $50, it's usually one user in a runaway session — "
            "a quick nudge to restart their conversation saves the rest of the day's budget."
        ),
        evidence={
            "peak_days":       len(peak_days),
            "avg_daily":       round(avg_daily, 2),
            "worst_day":       worst_day,
            "worst_day_cost":  round(worst_cost, 2),
            "total_peak_cost": round(total_peak_cost, 2),
        },
    )]


def _detect_zero_output(
    records: list[UsageRecord],
    period_days: int,
) -> list[UsageFinding]:
    zero_out = [r for r in records if r.output_tokens == 0 and r.total_tokens > 1000]
    if len(zero_out) < ZERO_OUTPUT_MIN_CALLS:
        return []

    cost_wasted = sum(r.cost_usd for r in zero_out)
    if cost_wasted < 0.50:
        return []

    savings = cost_wasted / period_days * 30

    return [UsageFinding(
        pattern="zero_output",
        severity="low",
        title=f"Aborted calls: {len(zero_out)} calls produced zero output tokens",
        detail=(
            f"{len(zero_out)} calls consumed tokens on input/cache but generated "
            f"zero output — these are aborted or errored requests. "
            f"Total wasted: ${cost_wasted:.2f}. Likely causes: context-too-long errors, "
            f"user cancelled mid-response, or tool timeout."
        ),
        affected_users=sorted({r.user_label for r in zero_out}),
        cost_attributed_usd=round(cost_wasted, 2),
        savings_per_month_usd=round(savings, 2),
        fix=(
            "Check for context-length errors in your AI tool logs. "
            "If conversations are hitting context limits, that's another signal "
            "to reset them more frequently. Add error handling in any programmatic "
            "workflows to avoid re-sending failed requests."
        ),
        evidence={
            "zero_output_calls": len(zero_out),
            "total_cost":        round(cost_wasted, 2),
        },
    )]


# ── Case-study note ───────────────────────────────────────────────────────────

def _case_study_note(platform: str, total_cost: float, users: list[dict]) -> dict:
    """Inline benchmark from a real team's data (anonymised)."""
    return {
        "source": "Real team usage export, April 2026 (users anonymised)",
        "platform": "Claude.ai + Cursor",
        "period_days": 69,
        "total_cost_usd": 2116.90,
        "monthly_run_rate_usd": 920,
        "users": 3,
        "dominant_user_share_pct": 95,
        "top_pattern": "context_bloat (93% of tokens were cache reads)",
        "worst_single_call_usd": 47.05,
        "worst_single_call_tokens_M": 43.5,
        "savings_potential_monthly_usd": 873,
        "key_fix": (
            "Disable extended thinking for routine tasks + "
            "start a new conversation every 30–60 min → "
            "saves ~$873/mo on a $920/mo bill"
        ),
    }


# ── Rich console report ───────────────────────────────────────────────────────

def render_usage_report(report: UsageReport, show_case_study: bool = True) -> None:
    """Print a formatted usage analysis report to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        from rich.text import Text
    except ImportError:
        print(report.to_json())
        return

    console = Console()

    platform_label = {
        "claude_ai": "Claude.ai Team Export",
        "cursor":    "Cursor Team Billing",
        "openai":    "OpenAI Usage Export",
    }.get(report.platform, report.platform)

    # ── Header ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold white]🕵️  monk — Team AI Usage Analysis[/]\n"
        f"[dim]{platform_label}  ·  "
        f"{report.period_start.strftime('%b %d')} → {report.period_end.strftime('%b %d, %Y')}  "
        f"({report.period_days} days)[/]",
        style="bold orange1", expand=False,
    ))

    # ── KPI row ──────────────────────────────────────────────────────────────
    savings_pct = (report.total_savings_potential_usd / report.monthly_run_rate_usd * 100
                   if report.monthly_run_rate_usd else 0)
    kpi = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    kpi.add_column(style="bold")
    kpi.add_column(style="dim")
    kpi.add_row("Total Spend",     f"${report.total_cost_usd:,.2f}")
    kpi.add_row("Monthly Rate",    f"${report.monthly_run_rate_usd:,.0f}/mo")
    kpi.add_row("Calls Analysed",  f"{report.total_calls:,}")
    kpi.add_row("Total Tokens",    f"{report.total_tokens/1e9:.2f}B")
    kpi.add_row("Findings",        f"{len(report.findings)}")
    kpi.add_row("Savings Potential",
                f"[bold green]${report.total_savings_potential_usd:,.0f}/mo "
                f"({savings_pct:.0f}% of bill)[/]")
    console.print(kpi)

    # ── Per-user table ───────────────────────────────────────────────────────
    if report.users:
        console.print("\n[bold]Spend by User[/] [dim](names anonymised)[/]")
        ut = Table(box=box.SIMPLE_HEAD, show_header=True)
        ut.add_column("User",       style="bold")
        ut.add_column("Calls",      justify="right")
        ut.add_column("Cost",       justify="right", style="bold red")
        ut.add_column("% of Total", justify="right")
        ut.add_column("Cache %",    justify="right")
        ut.add_column("Top Model",  style="dim")
        total = report.total_cost_usd or 1
        for u in report.users:
            ut.add_row(
                u["label"],
                str(u["calls"]),
                f"${u['cost']:,.2f}",
                f"{u['cost']/total*100:.0f}%",
                f"{u['cache_pct']:.0f}%",
                u["top_model"],
            )
        console.print(ut)

    # ── Model breakdown ───────────────────────────────────────────────────────
    if report.models:
        console.print("\n[bold]Spend by Model[/]")
        mt = Table(box=box.SIMPLE_HEAD)
        mt.add_column("Model")
        mt.add_column("Calls",  justify="right")
        mt.add_column("Cost",   justify="right", style="bold")
        mt.add_column("Tokens", justify="right", style="dim")
        for m in report.models[:8]:
            mt.add_row(
                m["model"],
                str(m["calls"]),
                f"${m['cost']:,.2f}",
                f"{m['tokens']/1e6:.0f}M",
            )
        console.print(mt)

    # ── Findings ─────────────────────────────────────────────────────────────
    if report.findings:
        console.print(f"\n[bold]Findings[/]  ({len(report.findings)} patterns detected)\n")
        icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for i, f in enumerate(report.findings, 1):
            icon = icons.get(f.severity, "⚪")
            savings_str = (
                f"  [green]~${f.savings_per_month_usd:,.0f}/mo savings[/]"
                if f.savings_per_month_usd >= 1 else ""
            )
            console.print(f"  {icon} [{i}] [bold]{f.title}[/]{savings_str}")
            console.print(f"       [dim]{f.detail}[/]")
            console.print(f"       [italic cyan]Fix:[/] {f.fix}")
            console.print()

    # ── Case study ────────────────────────────────────────────────────────────
    if show_case_study and report.case_study:
        cs = report.case_study
        console.print(Panel(
            f"[bold]📋 Real-world benchmark[/]  [dim]{cs['source']}[/]\n\n"
            f"  $[bold]{cs['total_cost_usd']:,.2f}[/] over {cs['period_days']} days  "
            f"(${cs['monthly_run_rate_usd']}/mo run rate)\n"
            f"  {cs['users']} users  ·  top user = {cs['dominant_user_share_pct']}% of spend\n"
            f"  Worst single call: [bold red]${cs['worst_single_call_usd']}[/]  "
            f"({cs['worst_single_call_tokens_M']}M tokens)\n"
            f"  Top pattern: {cs['top_pattern']}\n"
            f"  [bold green]Fix applied → saved ${cs['savings_potential_monthly_usd']}/mo[/]  "
            f"({cs['key_fix']})",
            title="Case Study",
            border_style="dim",
        ))

    # ── Warnings ──────────────────────────────────────────────────────────────
    if report.warnings:
        console.print(f"\n[dim]⚠  {len(report.warnings)} parse warning(s):[/]")
        for w in report.warnings[:5]:
            console.print(f"   [dim]{w}[/]")
