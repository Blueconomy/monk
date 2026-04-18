"""
OpenTelemetry span parser for monk.

Handles two common OTEL export formats:
  1. OTLP proto-JSON  — { resourceSpans: [...] } (Jaeger/Tempo/OTEL collector exports)
  2. Simple span dicts — flat list of span objects (LangChain, TRAIL dataset, custom)

Produces:
  - list[Span]       — the full span tree, for span-aware detectors
  - list[TraceCall]  — LLM spans normalised to TraceCall, for existing detectors

Span kind classification:
  "llm"     — has gen_ai.* or llm.* attributes, or name matches model patterns
  "tool"    — has tool.name attribute or name contains "tool"
  "agent"   — top-level orchestration span
  "chain"   — intermediate orchestration
  "unknown" — everything else
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monk.parsers.auto import TraceCall, ToolCall


# ── Span dataclass ────────────────────────────────────────────────────────────

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str                        # "llm" | "tool" | "agent" | "chain" | "unknown"
    start_time_ns: int               # Unix nanoseconds
    end_time_ns: int
    status: str                      # "ok" | "error" | "unset"
    error_message: str
    attributes: dict[str, Any]
    children: list["Span"] = field(default_factory=list, repr=False)

    # Cached parsed fields (populated by _enrich)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    tool_name: str = ""
    tool_args: str = ""
    tool_result: str = ""

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.end_time_ns - self.start_time_ns) / 1_000_000)

    @property
    def session_id(self) -> str:
        """trace_id groups all spans in one agent run."""
        return self.trace_id

    def to_trace_call(self, call_index: int = 0) -> TraceCall | None:
        """Convert an LLM span to TraceCall for use by existing detectors."""
        if self.kind != "llm" or not self.model:
            return None
        tool_calls = []
        for child in self.children:
            if child.kind == "tool" and child.tool_name:
                tool_calls.append(ToolCall(
                    name=child.tool_name,
                    arguments=child.tool_args,
                    result=child.tool_result,
                ))
        return TraceCall(
            session_id=self.session_id,
            call_index=call_index,
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            system_prompt_tokens=int(self.attributes.get("llm.system_prompt_tokens", 0)),
            tool_calls=tool_calls,
            raw=self.attributes,
        )

    def all_descendants(self) -> list["Span"]:
        """Return all descendant spans in DFS order."""
        result: list[Span] = []
        for child in self.children:
            result.append(child)
            result.extend(child.all_descendants())
        return result


# ── Public entry point ────────────────────────────────────────────────────────

def parse_spans(source: str | Path) -> list[Span]:
    """
    Parse an OTEL trace file and return a list of root Span objects (span trees).
    Each root represents one top-level agent execution.
    Source can be a file path or raw JSON/JSONL string.
    """
    source = str(source)
    text = source
    # Only attempt filesystem access for short strings that look like paths
    if len(source) < 4096 and not source.strip().startswith(("{", "[")):
        try:
            p = Path(source)
            if p.exists():
                text = p.read_text(encoding="utf-8")
        except (OSError, ValueError):
            pass

    raw_spans = _load_raw_spans(text)
    if not raw_spans:
        return []

    spans = [_parse_span(r) for r in raw_spans if r]
    spans = [s for s in spans if s is not None]
    return _build_tree(spans)


def spans_to_trace_calls(roots: list[Span]) -> list[TraceCall]:
    """
    Extract TraceCall objects from a span tree for use by existing detectors.
    Walks all LLM spans, assigns call_index per session.
    """
    session_counters: dict[str, int] = {}
    calls: list[TraceCall] = []

    def walk(span: Span) -> None:
        if span.kind == "llm":
            sid = span.session_id
            idx = session_counters.get(sid, 0)
            session_counters[sid] = idx + 1
            tc = span.to_trace_call(idx)
            if tc:
                calls.append(tc)
        for child in span.children:
            walk(child)

    for root in roots:
        walk(root)
    return calls


def is_otel_format(text: str) -> bool:
    """Quick check — is this text an OTEL trace file?"""
    stripped = text.strip()
    # OTLP proto-JSON envelope
    if '"resourceSpans"' in stripped or '"resource_spans"' in stripped:
        return True
    # Flat JSONL — check first few records for OTEL fingerprints
    for line in stripped.splitlines()[:5]:
        try:
            obj = json.loads(line)
            if _has_otel_fields(obj):
                return True
        except json.JSONDecodeError:
            continue
    # JSON array
    if stripped.startswith("["):
        try:
            arr = json.loads(stripped)
            if arr and isinstance(arr, list) and _has_otel_fields(arr[0]):
                return True
        except json.JSONDecodeError:
            pass
    return False


# ── Raw record loading ────────────────────────────────────────────────────────

def _load_raw_spans(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []

    # OTLP proto-JSON envelope
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return _extract_otlp_spans(obj)
        if isinstance(obj, list):
            # Could be a JSON array of spans
            if obj and _has_otel_fields(obj[0]):
                return obj
    except json.JSONDecodeError:
        pass

    # JSONL
    spans = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                spans.append(rec)
        except json.JSONDecodeError:
            continue
    return spans


def _extract_otlp_spans(obj: dict) -> list[dict]:
    """Unwrap OTLP proto-JSON envelope: resourceSpans → scopeSpans → spans."""
    spans = []
    resource_spans = obj.get("resourceSpans") or obj.get("resource_spans", [])
    for rs in resource_spans:
        scope_spans = rs.get("scopeSpans") or rs.get("scope_spans", [])
        for ss in scope_spans:
            for span in ss.get("spans", []):
                spans.append(span)
    return spans


# ── Single span parsing ───────────────────────────────────────────────────────

def _parse_span(rec: dict) -> Span | None:
    if not isinstance(rec, dict):
        return None

    # IDs — handle both camelCase and snake_case
    trace_id = str(rec.get("traceId") or rec.get("trace_id") or "")
    span_id = str(rec.get("spanId") or rec.get("span_id") or rec.get("id") or "")
    parent_id = rec.get("parentSpanId") or rec.get("parent_span_id") or rec.get("parent_id")
    parent_id = str(parent_id) if parent_id else None

    if not span_id:
        return None

    name = str(rec.get("name") or rec.get("operation_name") or "unknown")

    # Timing — handle nanoseconds (OTLP), microseconds, milliseconds, ISO strings
    start_ns = _parse_time_ns(rec.get("startTimeUnixNano") or rec.get("start_time") or 0)
    end_ns = _parse_time_ns(rec.get("endTimeUnixNano") or rec.get("end_time") or 0)

    # Attributes — normalise OTLP key-value array to flat dict
    raw_attrs = rec.get("attributes") or {}
    attrs = _normalise_attributes(raw_attrs)

    # Status
    status_obj = rec.get("status") or {}
    if isinstance(status_obj, dict):
        code = str(status_obj.get("code") or status_obj.get("status_code") or "").lower()
        error_msg = str(status_obj.get("message") or "")
    elif isinstance(status_obj, str):
        code = status_obj.lower()
        error_msg = ""
    else:
        code = ""
        error_msg = ""

    if "error" in code or "2" == code:
        status = "error"
    elif "ok" in code or "1" == code:
        status = "ok"
    else:
        # Check events for exceptions
        events = rec.get("events", [])
        has_exception = any(
            "exception" in str(e.get("name", "")).lower()
            for e in events if isinstance(e, dict)
        )
        status = "error" if has_exception else "unset"
        if has_exception and not error_msg:
            for e in events:
                if isinstance(e, dict) and "exception" in str(e.get("name", "")).lower():
                    e_attrs = _normalise_attributes(e.get("attributes", {}))
                    error_msg = str(e_attrs.get("exception.message") or e_attrs.get("exception.type") or "")
                    break

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_id,
        name=name,
        kind="unknown",       # classified below
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        status=status,
        error_message=error_msg,
        attributes=attrs,
    )
    _enrich_span(span)
    return span


def _enrich_span(span: Span) -> None:
    """Classify span kind and extract semantic fields from attributes."""
    attrs = span.attributes
    name_lower = span.name.lower()

    # Model / LLM detection
    model = (
        attrs.get("gen_ai.request.model") or
        attrs.get("llm.model_name") or
        attrs.get("model") or
        attrs.get("gen_ai.model") or
        ""
    )
    input_tok = int(
        attrs.get("gen_ai.usage.prompt_tokens") or
        attrs.get("llm.token_count.prompt") or
        attrs.get("input_tokens") or
        attrs.get("prompt_tokens") or 0
    )
    output_tok = int(
        attrs.get("gen_ai.usage.completion_tokens") or
        attrs.get("llm.token_count.completion") or
        attrs.get("output_tokens") or
        attrs.get("completion_tokens") or 0
    )

    # Tool detection
    tool_name = (
        attrs.get("tool.name") or
        attrs.get("tool_name") or
        attrs.get("function.name") or
        ""
    )
    tool_args = str(attrs.get("tool.arguments") or attrs.get("tool_arguments") or "")
    tool_result = str(attrs.get("tool.result") or attrs.get("tool_output") or attrs.get("tool_result") or "")

    # Classify kind
    if model or input_tok or "llm" in name_lower or "chat" in name_lower or "completion" in name_lower:
        span.kind = "llm"
        span.model = str(model)
        span.input_tokens = input_tok
        span.output_tokens = output_tok
    elif tool_name or "tool" in name_lower or "function_call" in name_lower:
        span.kind = "tool"
        span.tool_name = str(tool_name) or span.name
        span.tool_args = tool_args
        span.tool_result = tool_result
    elif any(k in name_lower for k in ("agent", "run", "executor", "planner")):
        span.kind = "agent"
    elif any(k in name_lower for k in ("chain", "pipeline", "workflow", "step")):
        span.kind = "chain"
    else:
        span.kind = "unknown"


# ── Tree building ─────────────────────────────────────────────────────────────

def _build_tree(spans: list[Span]) -> list[Span]:
    """
    Assemble flat span list into a forest of trees.
    Returns root spans (those with no parent or whose parent isn't in the set).
    """
    by_id: dict[str, Span] = {s.span_id: s for s in spans}
    roots: list[Span] = []

    for span in spans:
        if span.parent_span_id and span.parent_span_id in by_id:
            by_id[span.parent_span_id].children.append(span)
        else:
            roots.append(span)

    # Sort children by start time within each parent
    def sort_children(s: Span) -> None:
        s.children.sort(key=lambda c: c.start_time_ns)
        for child in s.children:
            sort_children(child)

    for root in roots:
        sort_children(root)

    return roots


# ── Attribute normalisation ───────────────────────────────────────────────────

def _normalise_attributes(attrs: Any) -> dict[str, Any]:
    """
    OTLP proto-JSON attributes are a list of {key, value: {stringValue/intValue/...}}.
    Simple formats use plain dicts. Normalise both to flat key→value dict.
    """
    if isinstance(attrs, dict):
        return attrs

    if isinstance(attrs, list):
        result: dict[str, Any] = {}
        for item in attrs:
            if not isinstance(item, dict):
                continue
            key = item.get("key", "")
            val_obj = item.get("value", {})
            if isinstance(val_obj, dict):
                # OTLP typed value
                for vtype in ("stringValue", "intValue", "doubleValue", "boolValue",
                              "string_value", "int_value", "double_value", "bool_value"):
                    if vtype in val_obj:
                        result[key] = val_obj[vtype]
                        break
            else:
                result[key] = val_obj
        return result

    return {}


# ── Time parsing ──────────────────────────────────────────────────────────────

_ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})?")


def _parse_time_ns(val: Any) -> int:
    """Parse various time formats to nanoseconds since epoch."""
    if val is None or val == 0:
        return 0
    if isinstance(val, (int, float)):
        v = int(val)
        # Heuristic based on realistic timestamp ranges:
        # ns:  >= 1e18  (year 2001+)
        # µs:  >= 1e15  (year 2001+)
        # ms:  >= 1e12  (year 2001+)
        # s:   >= 1e9   (year 2001+)
        if v >= 1_000_000_000_000_000_000:
            return v                        # already ns
        if v >= 1_000_000_000_000_000:
            return v * 1_000               # µs → ns
        if v >= 1_000_000_000_000:
            return v * 1_000_000           # ms → ns
        if v >= 1_000_000_000:
            return v * 1_000_000_000       # s → ns
        return v
    if isinstance(val, str):
        # ISO 8601
        if _ISO_RE.match(val.strip()):
            from datetime import datetime, timezone
            try:
                # Normalise to UTC float seconds
                ts = val.strip().rstrip("Z")
                dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1_000_000_000)
            except ValueError:
                pass
        # Plain int string
        try:
            return _parse_time_ns(int(val))
        except ValueError:
            pass
    return 0


# ── Format detection helper ───────────────────────────────────────────────────

def _has_otel_fields(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    otel_keys = {"traceId", "spanId", "parentSpanId", "trace_id", "span_id",
                 "startTimeUnixNano", "endTimeUnixNano", "start_time", "end_time",
                 "resourceSpans", "resource_spans"}
    return bool(otel_keys & obj.keys())
