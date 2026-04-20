"""
monk usage_csv — parser for AI platform team usage CSV exports.

Supported formats (auto-detected):
  - Claude.ai / Anthropic Console team usage export
  - Cursor team billing export  (future: same column family)
  - OpenAI usage export         (future)

Output: list of UsageRecord dataclasses, users anonymised by default.

Error handling:
  - Missing columns       → FormatError with actionable message
  - Unparseable values    → row skipped, warning collected
  - All-zero rows         → silently dropped
  - Unknown model names   → kept as-is, pricing estimated from context
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    """One normalised API call from any supported billing export."""
    timestamp: datetime
    user_label: str          # anonymised: "User A", "User B", …
    user_raw: str            # original email / id (empty when anonymised)
    platform: str            # "claude_ai" | "cursor" | "openai"
    kind: str                # "included" | "on_demand" | "unknown"
    model: str               # normalised model string
    model_tier: str          # "thinking" | "opus" | "sonnet" | "haiku" | "other"
    cache_write_tokens: int
    input_tokens: int        # fresh (non-cached) input
    cache_read_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    raw: dict = field(default_factory=dict, repr=False)


class FormatError(ValueError):
    """Raised when the CSV cannot be recognised or is missing required columns."""


# ── Model tier classification ────────────────────────────────────────────────

_THINKING_PATTERNS   = re.compile(r"thinking|extended", re.I)
_OPUS_PATTERNS       = re.compile(r"opus", re.I)
_SONNET_PATTERNS     = re.compile(r"sonnet", re.I)
_HAIKU_PATTERNS      = re.compile(r"haiku|flash|mini|fast|composer", re.I)


def _model_tier(model: str) -> str:
    if _THINKING_PATTERNS.search(model):
        return "thinking"
    if _OPUS_PATTERNS.search(model):
        return "opus"
    if _SONNET_PATTERNS.search(model):
        return "sonnet"
    if _HAIKU_PATTERNS.search(model):
        return "haiku"
    return "other"


# ── Format signatures ────────────────────────────────────────────────────────

_CLAUDE_AI_REQUIRED = {
    "Date", "User", "Model",
    "Input (w/o Cache Write)", "Cache Read", "Output Tokens",
    "Total Tokens", "Cost",
}

_CURSOR_REQUIRED = {
    "Date", "User", "Model",
    "Input Tokens", "Output Tokens", "Total Tokens", "Cost",
}

_OPENAI_REQUIRED = {
    "Date", "Organization Name", "Project Name",
    "Model", "Input Tokens", "Output Tokens", "Total Tokens", "Cost",
}


def _detect_platform(headers: set[str]) -> str:
    if _CLAUDE_AI_REQUIRED.issubset(headers):
        return "claude_ai"
    if "Cache Read" not in headers and _CURSOR_REQUIRED.issubset(headers):
        return "cursor"
    if _OPENAI_REQUIRED.issubset(headers):
        return "openai"
    raise FormatError(
        "Unrecognised CSV format. Supported: Claude.ai team usage export, "
        "Cursor billing export, OpenAI usage export.\n"
        f"  Found columns: {sorted(headers)}\n"
        f"  Expected (Claude.ai): {sorted(_CLAUDE_AI_REQUIRED)}"
    )


# ── Safe value parsers ────────────────────────────────────────────────────────

def _int(v: str) -> int:
    try:
        return int(str(v).replace(",", "").strip() or 0)
    except (ValueError, TypeError):
        return 0


def _float(v: str) -> float:
    try:
        s = str(v).replace(",", "").strip()
        return float(s) if s not in ("", "-", "N/A") else 0.0
    except (ValueError, TypeError):
        return 0.0


def _ts(v: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(v.strip(), fmt)
        except ValueError:
            continue
    return None


# ── Platform-specific row parsers ─────────────────────────────────────────────

def _parse_claude_ai_row(row: dict) -> UsageRecord | None:
    ts = _ts(row.get("Date", ""))
    if ts is None:
        return None
    total = _int(row.get("Total Tokens", "0"))
    if total == 0:
        return None

    kind_raw = row.get("Kind", "").strip().lower()
    kind = "on_demand" if "demand" in kind_raw else "included"
    model = row.get("Model", "auto").strip()

    return UsageRecord(
        timestamp=ts,
        user_label="",        # filled by anonymiser
        user_raw=row.get("User", "").strip(),
        platform="claude_ai",
        kind=kind,
        model=model,
        model_tier=_model_tier(model),
        cache_write_tokens=_int(row.get("Input (w/ Cache Write)", "0")),
        input_tokens=_int(row.get("Input (w/o Cache Write)", "0")),
        cache_read_tokens=_int(row.get("Cache Read", "0")),
        output_tokens=_int(row.get("Output Tokens", "0")),
        total_tokens=total,
        cost_usd=_float(row.get("Cost", "0")),
        raw=dict(row),
    )


def _parse_cursor_row(row: dict) -> UsageRecord | None:
    ts = _ts(row.get("Date", ""))
    if ts is None:
        return None
    total = _int(row.get("Total Tokens", "0"))
    if total == 0:
        return None

    model = row.get("Model", "unknown").strip()
    return UsageRecord(
        timestamp=ts,
        user_label="",
        user_raw=row.get("User", row.get("Email", "")).strip(),
        platform="cursor",
        kind="on_demand",
        model=model,
        model_tier=_model_tier(model),
        cache_write_tokens=0,
        input_tokens=_int(row.get("Input Tokens", "0")),
        cache_read_tokens=_int(row.get("Cache Read Tokens", "0")),
        output_tokens=_int(row.get("Output Tokens", "0")),
        total_tokens=total,
        cost_usd=_float(row.get("Cost", "0")),
        raw=dict(row),
    )


def _parse_openai_row(row: dict) -> UsageRecord | None:
    ts = _ts(row.get("Date", ""))
    if ts is None:
        return None
    total = _int(row.get("Total Tokens", "0"))
    if total == 0:
        return None

    model = row.get("Model", "unknown").strip()
    return UsageRecord(
        timestamp=ts,
        user_label="",
        user_raw=row.get("Organization Name", row.get("Project Name", "")).strip(),
        platform="openai",
        kind="on_demand",
        model=model,
        model_tier=_model_tier(model),
        cache_write_tokens=0,
        input_tokens=_int(row.get("Input Tokens", "0")),
        cache_read_tokens=_int(row.get("Cached Input Tokens", "0")),
        output_tokens=_int(row.get("Output Tokens", "0")),
        total_tokens=total,
        cost_usd=_float(row.get("Cost", "0")),
        raw=dict(row),
    )


_PARSERS = {
    "claude_ai": _parse_claude_ai_row,
    "cursor":    _parse_cursor_row,
    "openai":    _parse_openai_row,
}


# ── Anonymiser ────────────────────────────────────────────────────────────────

def _anonymise(records: list[UsageRecord]) -> list[UsageRecord]:
    """Replace real user identifiers with 'User A', 'User B', … ordered by spend."""
    spend: dict[str, float] = {}
    for r in records:
        spend[r.user_raw] = spend.get(r.user_raw, 0.0) + r.cost_usd

    labels = {
        user: f"User {chr(65 + i)}"
        for i, (user, _) in enumerate(sorted(spend.items(), key=lambda x: -x[1]))
    }
    for r in records:
        r.user_label = labels.get(r.user_raw, "User ?")
    return records


# ── Public API ────────────────────────────────────────────────────────────────

def parse_usage_csv(
    source: str | Path,
    anonymise: bool = True,
) -> tuple[list[UsageRecord], list[str], str]:
    """
    Parse a team usage CSV export.

    Parameters
    ----------
    source      : file path or raw CSV string
    anonymise   : replace real user names with 'User A/B/C' (default True)

    Returns
    -------
    (records, warnings, platform)
      records   : list of UsageRecord
      warnings  : list of human-readable parse warnings
      platform  : detected platform string
    """
    source = str(source)
    # Distinguish a file path from raw CSV text.
    # Path(source).exists() throws [Errno 63] on long strings (i.e. raw CSV text).
    _is_file = False
    try:
        _is_file = len(source) < 4096 and Path(source).exists()
    except OSError:
        pass

    if _is_file:
        text = Path(source).read_text(encoding="utf-8-sig")  # handles BOM
    else:
        text = source
        if text.startswith("\ufeff"):   # strip UTF-8 BOM if present in raw text
            text = text[1:]

    if not text.strip():
        raise FormatError("File is empty.")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise FormatError("CSV has no header row.")

    headers = {h.strip() for h in reader.fieldnames}
    platform = _detect_platform(headers)
    parse_row = _PARSERS[platform]

    records: list[UsageRecord] = []
    warnings: list[str] = []

    for i, row in enumerate(reader, start=2):  # row 1 = header
        try:
            rec = parse_row({k.strip(): v for k, v in row.items()})
            if rec is not None:
                records.append(rec)
        except Exception as e:
            warnings.append(f"Row {i}: skipped — {e}")

    if not records:
        raise FormatError(
            f"No valid records found in {platform} CSV. "
            "All rows may be zero-token or malformed."
        )

    records.sort(key=lambda r: r.timestamp)

    if anonymise:
        _anonymise(records)

    return records, warnings, platform
