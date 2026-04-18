"""
Tests for the OTEL span parser and span tree building.
"""
import json
import pytest
from monk.parsers.otel import (
    parse_spans, spans_to_trace_calls, is_otel_format,
    Span, _parse_span, _build_tree, _normalise_attributes, _parse_time_ns,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_span_dict(
    span_id="s1", trace_id="t1", parent_id=None, name="ChatCompletion",
    start_ns=1_000_000_000_000_000_000, end_ns=1_002_000_000_000_000_000,
    status_code="STATUS_CODE_OK", attrs=None,
):
    d = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": end_ns,
        "status": {"code": status_code},
        "attributes": attrs or [],
    }
    if parent_id:
        d["parentSpanId"] = parent_id
    return d


def make_attr(key, value):
    if isinstance(value, str):
        return {"key": key, "value": {"stringValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


# ── is_otel_format ────────────────────────────────────────────────────────────

class TestIsOtelFormat:
    def test_detects_otlp_envelope(self):
        text = json.dumps({"resourceSpans": []})
        assert is_otel_format(text) is True

    def test_detects_jsonl_with_span_fields(self):
        line = json.dumps({"traceId": "abc", "spanId": "def", "name": "test"})
        assert is_otel_format(line) is True

    def test_rejects_plain_jsonl(self):
        line = json.dumps({"model": "gpt-4o", "input_tokens": 100, "output_tokens": 50})
        assert is_otel_format(line) is False

    def test_rejects_empty(self):
        assert is_otel_format("") is False


# ── Attribute normalisation ───────────────────────────────────────────────────

class TestNormaliseAttributes:
    def test_otlp_array_to_dict(self):
        attrs = [
            make_attr("gen_ai.request.model", "gpt-4o"),
            make_attr("gen_ai.usage.prompt_tokens", 500),
        ]
        result = _normalise_attributes(attrs)
        assert result["gen_ai.request.model"] == "gpt-4o"
        assert result["gen_ai.usage.prompt_tokens"] == 500

    def test_plain_dict_passthrough(self):
        attrs = {"model": "claude-sonnet-4-6", "input_tokens": 300}
        result = _normalise_attributes(attrs)
        assert result == attrs

    def test_empty_list(self):
        assert _normalise_attributes([]) == {}


# ── Time parsing ──────────────────────────────────────────────────────────────

class TestParseTimeNs:
    def test_nanoseconds_passthrough(self):
        ns = 1_700_000_000_000_000_000
        assert _parse_time_ns(ns) == ns

    def test_milliseconds_upscaled(self):
        ms = 1_700_000_000_000
        assert _parse_time_ns(ms) == ms * 1_000_000

    def test_seconds_upscaled(self):
        s = 1_700_000_000
        assert _parse_time_ns(s) == s * 1_000_000_000

    def test_zero_returns_zero(self):
        assert _parse_time_ns(0) == 0

    def test_none_returns_zero(self):
        assert _parse_time_ns(None) == 0


# ── Single span parsing ───────────────────────────────────────────────────────

class TestParseSpan:
    def test_llm_span_classified(self):
        d = make_span_dict(
            name="ChatCompletion",
            attrs=[
                make_attr("gen_ai.request.model", "gpt-4o"),
                make_attr("gen_ai.usage.prompt_tokens", 1000),
                make_attr("gen_ai.usage.completion_tokens", 200),
            ]
        )
        span = _parse_span(d)
        assert span is not None
        assert span.kind == "llm"
        assert span.model == "gpt-4o"
        assert span.input_tokens == 1000
        assert span.output_tokens == 200

    def test_tool_span_classified(self):
        d = make_span_dict(
            name="tool_call",
            attrs=[make_attr("tool.name", "web_search")]
        )
        span = _parse_span(d)
        assert span is not None
        assert span.kind == "tool"
        assert span.tool_name == "web_search"

    def test_error_span_status(self):
        d = make_span_dict(status_code="STATUS_CODE_ERROR")
        span = _parse_span(d)
        assert span is not None
        assert span.status == "error"

    def test_ok_span_status(self):
        d = make_span_dict(status_code="STATUS_CODE_OK")
        span = _parse_span(d)
        assert span is not None
        assert span.status == "ok"

    def test_duration_computed(self):
        start = 1_000_000_000_000_000_000          # valid ns timestamp
        end   = start + 500_000_000                # 500ms = 500_000_000 ns
        d = make_span_dict(start_ns=start, end_ns=end)
        span = _parse_span(d)
        assert span is not None
        assert abs(span.duration_ms - 500.0) < 1.0

    def test_missing_span_id_returns_none(self):
        d = {"traceId": "t1", "name": "test"}  # no spanId
        assert _parse_span(d) is None


# ── Tree building ─────────────────────────────────────────────────────────────

class TestBuildTree:
    def test_parent_child_relationship(self):
        parent_d = make_span_dict(span_id="p1", name="agent")
        child_d  = make_span_dict(span_id="c1", parent_id="p1", name="tool_call",
                                   attrs=[make_attr("tool.name", "search")])
        spans = [_parse_span(parent_d), _parse_span(child_d)]
        roots = _build_tree([s for s in spans if s])
        assert len(roots) == 1
        assert roots[0].span_id == "p1"
        assert len(roots[0].children) == 1
        assert roots[0].children[0].span_id == "c1"

    def test_orphan_becomes_root(self):
        child_d = make_span_dict(span_id="c1", parent_id="missing_parent")
        spans = [_parse_span(child_d)]
        roots = _build_tree([s for s in spans if s])
        assert len(roots) == 1

    def test_multiple_roots(self):
        a = make_span_dict(span_id="a1", trace_id="t1")
        b = make_span_dict(span_id="b1", trace_id="t2")
        spans = [_parse_span(a), _parse_span(b)]
        roots = _build_tree([s for s in spans if s])
        assert len(roots) == 2


# ── spans_to_trace_calls ──────────────────────────────────────────────────────

class TestSpansToTraceCalls:
    def test_extracts_llm_calls(self):
        d = make_span_dict(
            name="ChatCompletion",
            attrs=[
                make_attr("gen_ai.request.model", "gpt-4o"),
                make_attr("gen_ai.usage.prompt_tokens", 800),
                make_attr("gen_ai.usage.completion_tokens", 150),
            ]
        )
        roots = parse_spans(json.dumps([d]))
        calls = spans_to_trace_calls(roots)
        assert len(calls) == 1
        assert calls[0].model == "gpt-4o"
        assert calls[0].input_tokens == 800

    def test_tool_calls_attached_to_llm(self):
        parent_d = make_span_dict(
            span_id="llm1", name="ChatCompletion",
            attrs=[make_attr("gen_ai.request.model", "gpt-4o")]
        )
        tool_d = make_span_dict(
            span_id="tool1", parent_id="llm1", name="tool_call",
            attrs=[make_attr("tool.name", "web_search")]
        )
        roots = parse_spans(json.dumps([parent_d, tool_d]))
        calls = spans_to_trace_calls(roots)
        assert len(calls) == 1
        assert len(calls[0].tool_calls) == 1
        assert calls[0].tool_calls[0].name == "web_search"

    def test_non_llm_spans_excluded(self):
        d = make_span_dict(name="tool_call", attrs=[make_attr("tool.name", "search")])
        roots = parse_spans(json.dumps([d]))
        calls = spans_to_trace_calls(roots)
        assert len(calls) == 0


# ── OTLP envelope ─────────────────────────────────────────────────────────────

class TestOTLPEnvelope:
    def test_extracts_from_resource_spans(self):
        otlp = {
            "resourceSpans": [{
                "scopeSpans": [{
                    "spans": [
                        make_span_dict(
                            attrs=[
                                make_attr("gen_ai.request.model", "claude-sonnet-4-6"),
                                make_attr("gen_ai.usage.prompt_tokens", 500),
                            ]
                        )
                    ]
                }]
            }]
        }
        roots = parse_spans(json.dumps(otlp))
        assert len(roots) == 1
        assert roots[0].model == "claude-sonnet-4-6"
