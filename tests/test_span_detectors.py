"""
Tests for span-level detectors:
  - LatencySpikeDetector
  - ErrorCascadeDetector
  - ToolDependencyDetector
  - CrossTurnMemoryDetector
"""
import pytest
from monk.parsers.otel import Span
from monk.detectors.latency_spike import LatencySpikeDetector
from monk.detectors.error_cascade import ErrorCascadeDetector
from monk.detectors.tool_dependency import ToolDependencyDetector
from monk.detectors.cross_turn_memory import CrossTurnMemoryDetector


# ── Span builders ─────────────────────────────────────────────────────────────

_id_counter = 0

def _uid() -> str:
    global _id_counter
    _id_counter += 1
    return f"span_{_id_counter:04d}"


def make_span(
    name="test", kind="unknown", duration_ms=100,
    status="ok", error_msg="",
    trace_id="trace1", span_id=None, parent_id=None,
    model="", input_tokens=0, output_tokens=0,
    tool_name="", tool_args="", tool_result="",
    children=None,
    start_ns=None,
):
    sid = span_id or _uid()
    start = start_ns if start_ns is not None else 1_000_000_000_000_000_000
    end = start + int(duration_ms * 1_000_000)
    s = Span(
        trace_id=trace_id,
        span_id=sid,
        parent_span_id=parent_id,
        name=name,
        kind=kind,
        start_time_ns=start,
        end_time_ns=end,
        status=status,
        error_message=error_msg,
        attributes={},
        children=children or [],
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
    )
    return s


# ── LatencySpikeDetector ──────────────────────────────────────────────────────

class TestLatencySpike:
    def test_detects_tool_spike(self):
        # 5 calls at 100ms, 1 spike at 500ms (5x median)
        normal = [make_span(kind="tool", tool_name="web_search", duration_ms=100)
                  for _ in range(5)]
        spike = make_span(kind="tool", tool_name="web_search", duration_ms=500)
        root = make_span(kind="agent", children=normal + [spike])
        findings = LatencySpikeDetector().run_spans([root])
        assert any("web_search" in f.title for f in findings)

    def test_no_spike_for_consistent_durations(self):
        tools = [make_span(kind="tool", tool_name="search", duration_ms=100 + i * 5)
                 for i in range(6)]
        root = make_span(kind="agent", children=tools)
        findings = LatencySpikeDetector().run_spans([root])
        assert len(findings) == 0

    def test_detects_slow_llm(self):
        slow_llm = make_span(kind="llm", model="gpt-4o", duration_ms=20_000)
        root = make_span(kind="agent", children=[slow_llm])
        findings = LatencySpikeDetector().run_spans([root])
        assert any("LLM" in f.title for f in findings)

    def test_fast_llm_not_flagged(self):
        fast_llm = make_span(kind="llm", model="gpt-4o", duration_ms=2_000)
        root = make_span(kind="agent", children=[fast_llm])
        findings = LatencySpikeDetector().run_spans([root])
        llm_findings = [f for f in findings if "LLM" in f.title]
        assert len(llm_findings) == 0

    def test_run_tracecall_returns_empty(self):
        assert LatencySpikeDetector().run([]) == []


# ── ErrorCascadeDetector ──────────────────────────────────────────────────────

class TestErrorCascade:
    def test_detects_llm_calls_after_error(self):
        error_tool = make_span(kind="tool", tool_name="get_order",
                               status="error", error_msg="timeout")
        llm1 = make_span(kind="llm", model="gpt-4o", input_tokens=1000)
        llm2 = make_span(kind="llm", model="gpt-4o", input_tokens=1000)
        root = make_span(kind="agent", children=[error_tool, llm1, llm2])
        findings = ErrorCascadeDetector().run_spans([root])
        assert len(findings) >= 1
        assert "get_order" in findings[0].title

    def test_no_finding_when_error_is_last(self):
        llm1 = make_span(kind="llm", model="gpt-4o")
        error_tool = make_span(kind="tool", tool_name="search", status="error")
        root = make_span(kind="agent", children=[llm1, error_tool])
        findings = ErrorCascadeDetector().run_spans([root])
        # No LLM calls after error
        cascade = [f for f in findings if "cascade" in f.detector]
        assert len(cascade) == 0

    def test_no_finding_when_no_errors(self):
        children = [
            make_span(kind="tool", tool_name="search", status="ok"),
            make_span(kind="llm", model="gpt-4o", status="ok"),
        ]
        root = make_span(kind="agent", children=children)
        findings = ErrorCascadeDetector().run_spans([root])
        assert len(findings) == 0

    def test_detects_propagation_chain(self):
        # 3 chained errors in same trace
        # grandchild errors → child errors → root errors
        gc = make_span(span_id="gc", kind="tool", status="error", trace_id="trace_chain")
        child = make_span(span_id="ch", kind="chain", status="error",
                          trace_id="trace_chain", parent_id="gc", children=[gc])
        root = make_span(span_id="rt", kind="agent", status="error",
                         trace_id="trace_chain", children=[child])
        # Need at least 2 chained errors (child has error parent)
        findings = ErrorCascadeDetector().run_spans([root])
        prop = [f for f in findings if "propagation" in f.title.lower()]
        assert len(prop) >= 1

    def test_run_tracecall_returns_empty(self):
        assert ErrorCascadeDetector().run([]) == []


# ── ToolDependencyDetector ────────────────────────────────────────────────────

class TestToolDependency:
    def test_detects_cycle(self):
        # Create a span tree where tool A → B → A would be a cycle in the DAG
        # We simulate this via the adjacency builder: A is parent of B, B is parent of A-child
        a1 = make_span(span_id="a1", kind="tool", tool_name="tool_a")
        b1 = make_span(span_id="b1", kind="tool", tool_name="tool_b", parent_id="a1")
        a2 = make_span(span_id="a2", kind="tool", tool_name="tool_a", parent_id="b1")
        # Wire children manually
        b1.children = [a2]
        a1.children = [b1]
        root = make_span(kind="agent", children=[a1])
        findings = ToolDependencyDetector().run_spans([root])
        cycle_findings = [f for f in findings if "cycle" in f.title.lower()]
        assert len(cycle_findings) >= 1

    def test_detects_fanout(self):
        # Same tool called 4 times with same args as siblings
        siblings = [
            make_span(kind="tool", tool_name="fetch_user", tool_args='{"id": 42}')
            for _ in range(4)
        ]
        root = make_span(kind="agent", children=siblings)
        findings = ToolDependencyDetector().run_spans([root])
        fanout = [f for f in findings if "fan-out" in f.title.lower()]
        assert len(fanout) >= 1

    def test_no_fanout_for_different_args(self):
        siblings = [
            make_span(kind="tool", tool_name="fetch_user", tool_args=f'{{"id": {i}}}')
            for i in range(4)
        ]
        root = make_span(kind="agent", children=siblings)
        findings = ToolDependencyDetector().run_spans([root])
        fanout = [f for f in findings if "fan-out" in f.title.lower()]
        assert len(fanout) == 0

    def test_detects_deep_chain(self):
        # Build a 7-level deep chain
        current = make_span(kind="tool", tool_name="leaf")
        for i in range(6):
            current = make_span(kind="chain", name=f"step_{i}", children=[current])
        root = make_span(kind="agent", children=[current])
        findings = ToolDependencyDetector().run_spans([root])
        deep = [f for f in findings if "deep" in f.title.lower()]
        assert len(deep) >= 1

    def test_run_tracecall_returns_empty(self):
        assert ToolDependencyDetector().run([]) == []


# ── CrossTurnMemoryDetector ───────────────────────────────────────────────────

class TestCrossTurnMemory:
    def test_detects_repeated_tool_same_args(self):
        args = '{"user_id": "123", "query": "order status"}'
        t = 1_000_000_000_000_000_000
        step = 1_000_000_000_000_000  # 1ms in ns
        tools = [
            make_span(kind="tool", tool_name="get_order", tool_args=args,
                      start_ns=t + i * step)
            for i in range(3)
        ]
        root = make_span(kind="agent", children=tools, trace_id="trace_mem")
        findings = CrossTurnMemoryDetector().run_spans([root])
        assert len(findings) >= 1
        assert "get_order" in findings[0].title

    def test_no_finding_for_different_args(self):
        t = 1_000_000_000_000_000_000
        step = 1_000_000_000_000_000
        tools = [
            make_span(kind="tool", tool_name="search",
                      tool_args=f'{{"query": "topic_{i}"}}',
                      start_ns=t + i * step)
            for i in range(3)
        ]
        root = make_span(kind="agent", children=tools, trace_id="trace_diff")
        findings = CrossTurnMemoryDetector().run_spans([root])
        assert len(findings) == 0

    def test_no_finding_for_single_call(self):
        tool = make_span(kind="tool", tool_name="fetch", tool_args='{"id": 1}')
        root = make_span(kind="agent", children=[tool])
        findings = CrossTurnMemoryDetector().run_spans([root])
        assert len(findings) == 0

    def test_severity_high_for_many_repeats(self):
        args = '{"key": "value"}'
        t = 1_000_000_000_000_000_000
        step = 1_000_000_000_000_000
        tools = [
            make_span(kind="tool", tool_name="expensive_api", tool_args=args,
                      start_ns=t + i * step)
            for i in range(5)
        ]
        root = make_span(kind="agent", children=tools, trace_id="trace_hi")
        findings = CrossTurnMemoryDetector().run_spans([root])
        high = [f for f in findings if f.severity == "high"]
        assert len(high) >= 1

    def test_run_tracecall_returns_empty(self):
        assert CrossTurnMemoryDetector().run([]) == []
