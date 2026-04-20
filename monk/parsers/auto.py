"""
Auto-detecting trace parser.

Normalises OpenAI, Anthropic, LangSmith, OpenTelemetry, and generic JSONL logs
into a single internal TraceCall schema so detectors stay framework-agnostic.

Format detection order:
  1. OpenTelemetry (OTLP proto-JSON or OTEL-field JSONL) → delegates to otel.py
  2. OpenAI Chat Completions API response
  3. Anthropic Messages API response
  4. LangSmith run export
  5. Generic JSONL (fallback)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: str = ""
    result: str = ""


@dataclass
class TraceCall:
    """Normalised representation of one LLM call inside an agentic workflow."""
    session_id: str
    call_index: int            # 0-based position within the session
    model: str
    input_tokens: int
    output_tokens: int
    system_prompt_tokens: int  # tokens attributable to system message
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def parse_traces(source: str | Path) -> list[TraceCall]:
    """
    Parse a trace file (JSONL or JSON array) and return normalised TraceCall list.
    Source can be a file path or a raw JSON string.

    Automatically delegates to the OTEL parser when OpenTelemetry format is detected.
    """
    source = str(source)
    if Path(source).exists():
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = source  # treat as raw string

    # Detect OTEL format and delegate — import here to avoid circular imports
    from monk.parsers.otel import is_otel_format, parse_spans, spans_to_trace_calls
    if is_otel_format(text):
        roots = parse_spans(text)  # always pass text, parse_spans handles file paths internally
        return spans_to_trace_calls(roots)

    records = _load_records(text)
    calls: list[TraceCall] = []
    session_counters: dict[str, int] = {}

    for rec in records:
        call = _parse_record(rec, session_counters)
        if call:
            calls.append(call)
            # LangGraph responses embed multiple AI turns — drain them
            for extra in rec.pop("_lg_extra", []):
                calls.append(extra)

    return calls


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _load_records(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    # Try JSON array
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Try JSONL
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _parse_record(rec: dict, session_counters: dict[str, int]) -> TraceCall | None:
    """Dispatch to the right parser based on record shape."""
    if _is_langgraph_format(rec):
        return _parse_langgraph_record(rec, session_counters)
    if _is_openai_format(rec):
        return _parse_openai(rec, session_counters)
    if _is_anthropic_format(rec):
        return _parse_anthropic(rec, session_counters)
    if _is_langsmith_format(rec):
        return _parse_langsmith(rec, session_counters)
    return _parse_generic(rec, session_counters)


def _is_langgraph_format(rec: dict) -> bool:
    """LangGraph invoke response: {"messages": [...]} where AIMessages have usage_metadata."""
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    return any(
        isinstance(m, dict) and "usage_metadata" in m
        for m in msgs
    )


def _is_openai_format(rec: dict) -> bool:
    return "choices" in rec or ("model" in rec and "usage" in rec and "messages" in rec)


def _is_anthropic_format(rec: dict) -> bool:
    return rec.get("type") in ("message_start", "message") or \
           ("usage" in rec and "input_tokens" in rec.get("usage", {}))


def _is_langsmith_format(rec: dict) -> bool:
    return "run_id" in rec or ("inputs" in rec and "outputs" in rec and "run_type" in rec)


# ──────────────────────────────────────────────
# Format-specific parsers
# ──────────────────────────────────────────────

def _parse_langgraph_record(rec: dict, counters: dict) -> TraceCall | None:
    """
    Parse a LangGraph invoke() response into the *first* TraceCall we find.
    Because one LangGraph response contains multiple AI turns, we inject all of
    them into the counter and return only the first so the caller loop stays
    one-record → one-call.  Extra calls are stashed in rec['_lg_extra'] for
    the outer loop to consume if present (handled in parse_traces below).
    """
    messages = rec.get("messages", [])
    session = str(
        rec.get("session_id") or rec.get("thread_id") or rec.get("configurable", {}).get("thread_id") or "langgraph-default"
    )

    # Build tool-result lookup: tool_call_id → content
    tool_results: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, dict):
            continue
        mtype = (m.get("type") or m.get("__class__", "")).lower()
        if mtype in ("tool", "toolmessage"):
            tid = m.get("tool_call_id", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(b.get("text", b)) for b in content if isinstance(b, dict))
            tool_results[tid] = str(content)

    calls: list[TraceCall] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        mtype = (m.get("type") or m.get("__class__", "")).lower()
        if mtype not in ("ai", "aimessage"):
            continue
        meta = m.get("usage_metadata")
        if not meta:
            continue

        input_tok = int(meta.get("input_tokens", 0))
        output_tok = int(meta.get("output_tokens", 0))

        # System prompt tokens live in response_metadata if available
        resp_meta = m.get("response_metadata", {}) or {}
        system_tok = int(resp_meta.get("system_prompt_tokens", 0))

        # Extract tool calls + resolve results
        raw_tcs = m.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for tc in raw_tcs:
            name = tc.get("name", "unknown")
            args = tc.get("args", tc.get("arguments", {}))
            tc_id = tc.get("id", "")
            result = tool_results.get(tc_id, "")
            tool_calls.append(ToolCall(
                name=name,
                arguments=json.dumps(args) if isinstance(args, dict) else str(args),
                result=result,
            ))

        # Derive model — LangGraph stores it in response_metadata
        model = (
            resp_meta.get("model_name")
            or resp_meta.get("model_id")
            or m.get("name", "langgraph-agent")  # agent name as fallback
            or "unknown"
        )

        idx = _next_idx(counters, session)
        calls.append(TraceCall(
            session_id=session,
            call_index=idx,
            model=str(model),
            input_tokens=input_tok,
            output_tokens=output_tok,
            system_prompt_tokens=system_tok,
            tool_calls=tool_calls,
            raw=m,
        ))

    if not calls:
        return None
    # Stash extras so parse_traces can surface them
    if len(calls) > 1:
        rec["_lg_extra"] = calls[1:]
    return calls[0]


def _parse_openai(rec: dict, counters: dict) -> TraceCall:
    session = str(rec.get("session_id") or rec.get("thread_id") or "default")
    idx = _next_idx(counters, session)

    usage = rec.get("usage", {})
    input_tok = int(usage.get("prompt_tokens", 0))
    output_tok = int(usage.get("completion_tokens", 0))

    # Estimate system prompt tokens from messages
    messages = rec.get("messages", [])
    system_tok = _estimate_system_tokens(messages)

    tool_calls: list[ToolCall] = []
    for choice in rec.get("choices", []):
        msg = choice.get("message", {})
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            tool_calls.append(ToolCall(
                name=fn.get("name", "unknown"),
                arguments=fn.get("arguments", ""),
            ))
    # Attach tool results from subsequent messages if available
    _attach_results_from_messages(tool_calls, messages)

    return TraceCall(
        session_id=session,
        call_index=idx,
        model=rec.get("model", "unknown"),
        input_tokens=input_tok,
        output_tokens=output_tok,
        system_prompt_tokens=system_tok,
        tool_calls=tool_calls,
        raw=rec,
    )


def _parse_anthropic(rec: dict, counters: dict) -> TraceCall:
    session = str(rec.get("session_id") or "default")
    idx = _next_idx(counters, session)

    usage = rec.get("usage", {})
    input_tok = int(usage.get("input_tokens", 0))
    output_tok = int(usage.get("output_tokens", 0))

    messages = rec.get("messages", [])
    system = rec.get("system", "")
    system_tok = _count_tokens_approx(system) if system else _estimate_system_tokens(messages)

    tool_calls: list[ToolCall] = []
    for block in rec.get("content", []):
        if block.get("type") == "tool_use":
            tool_calls.append(ToolCall(
                name=block.get("name", "unknown"),
                arguments=json.dumps(block.get("input", {})),
            ))

    return TraceCall(
        session_id=session,
        call_index=idx,
        model=rec.get("model", "unknown"),
        input_tokens=input_tok,
        output_tokens=output_tok,
        system_prompt_tokens=system_tok,
        tool_calls=tool_calls,
        raw=rec,
    )


def _parse_langsmith(rec: dict, counters: dict) -> TraceCall:
    session = str(rec.get("session_id") or rec.get("session_name") or "default")
    idx = _next_idx(counters, session)

    extra = rec.get("extra", {})
    usage = extra.get("usage", {}) if extra else {}
    input_tok = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    output_tok = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))

    inputs = rec.get("inputs", {})
    messages = inputs.get("messages", []) if isinstance(inputs, dict) else []
    system_tok = _estimate_system_tokens(messages)

    return TraceCall(
        session_id=session,
        call_index=idx,
        model=extra.get("invocation_params", {}).get("model_name", "unknown") if extra else "unknown",
        input_tokens=input_tok,
        output_tokens=output_tok,
        system_prompt_tokens=system_tok,
        tool_calls=[],
        raw=rec,
    )


def _parse_generic(rec: dict, counters: dict) -> TraceCall | None:
    """Fallback: try common field names used in custom logging."""
    model = (rec.get("model") or rec.get("llm_model") or rec.get("engine") or "unknown")
    session = str(rec.get("session_id") or rec.get("trace_id") or
                  rec.get("conversation_id") or "default")
    idx = _next_idx(counters, session)

    input_tok = int(
        rec.get("input_tokens") or rec.get("prompt_tokens") or
        rec.get("tokens_in") or rec.get("usage", {}).get("input_tokens", 0)
    )
    output_tok = int(
        rec.get("output_tokens") or rec.get("completion_tokens") or
        rec.get("tokens_out") or rec.get("usage", {}).get("output_tokens", 0)
    )

    system_tok = int(rec.get("system_prompt_tokens", 0))

    # Tool call info
    tool_calls: list[ToolCall] = []
    tool_name = rec.get("tool_name") or rec.get("tool") or rec.get("function_name")
    if tool_name:
        result = str(rec.get("tool_result") or rec.get("tool_output") or
                     rec.get("result") or "")
        tool_calls.append(ToolCall(name=str(tool_name), result=result))

    return TraceCall(
        session_id=session,
        call_index=idx,
        model=str(model),
        input_tokens=input_tok,
        output_tokens=output_tok,
        system_prompt_tokens=system_tok,
        tool_calls=tool_calls,
        raw=rec,
    )


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def _next_idx(counters: dict, session: str) -> int:
    counters[session] = counters.get(session, -1) + 1
    return counters[session]


def _count_tokens_approx(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(0, len(text) // 4)


def _estimate_system_tokens(messages: list[Any]) -> int:
    total = 0
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str):
                total += _count_tokens_approx(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += _count_tokens_approx(str(block.get("text", "")))
    return total


def _attach_results_from_messages(tool_calls: list[ToolCall], messages: list[Any]) -> None:
    """Try to populate tool_call.result from tool messages in the conversation."""
    tool_results: dict[str, str] = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            name = str(m.get("name", ""))
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            tool_results[name] = str(content)

    for tc in tool_calls:
        if tc.name in tool_results:
            tc.result = tool_results[tc.name]
