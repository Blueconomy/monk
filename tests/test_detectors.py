"""
Tests for all 5 monk detectors.
Run with: pytest tests/ -v
"""
import pytest
from monk.parsers.auto import TraceCall, ToolCall
from monk.detectors.retry_loop import RetryLoopDetector
from monk.detectors.empty_return import EmptyReturnDetector
from monk.detectors.model_overkill import ModelOverkillDetector
from monk.detectors.context_bloat import ContextBloatDetector
from monk.detectors.agent_loop import AgentLoopDetector


# ── Helpers ──────────────────────────────────────────────────────────

def make_call(
    session="s1", idx=0, model="gpt-4o",
    input_tokens=1000, output_tokens=100,
    system_tokens=0, tools=None
):
    return TraceCall(
        session_id=session,
        call_index=idx,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        system_prompt_tokens=system_tokens,
        tool_calls=tools or [],
    )


def tc(name, result="ok"):
    return ToolCall(name=name, result=result)


# ── RetryLoopDetector ────────────────────────────────────────────────

class TestRetryLoop:
    def test_detects_three_consecutive_same_tool(self):
        calls = [
            make_call(idx=i, tools=[tc("web_search")]) for i in range(4)
        ]
        findings = RetryLoopDetector().run(calls)
        assert len(findings) == 1
        assert "web_search" in findings[0].title
        assert findings[0].severity == "high"

    def test_no_finding_for_two_consecutive(self):
        calls = [
            make_call(idx=0, tools=[tc("search")]),
            make_call(idx=1, tools=[tc("search")]),
            make_call(idx=2, tools=[tc("other")]),
        ]
        findings = RetryLoopDetector().run(calls)
        assert len(findings) == 0

    def test_no_finding_for_alternating_tools(self):
        calls = [
            make_call(idx=i, tools=[tc("a") if i % 2 == 0 else tc("b")])
            for i in range(6)
        ]
        findings = RetryLoopDetector().run(calls)
        assert len(findings) == 0

    def test_separate_sessions_not_merged(self):
        calls = [
            make_call(session="s1", idx=0, tools=[tc("search")]),
            make_call(session="s1", idx=1, tools=[tc("search")]),
            make_call(session="s2", idx=0, tools=[tc("search")]),
            make_call(session="s2", idx=1, tools=[tc("search")]),
        ]
        findings = RetryLoopDetector().run(calls)
        assert len(findings) == 0  # only 2 per session, below threshold


# ── EmptyReturnDetector ──────────────────────────────────────────────

class TestEmptyReturn:
    def test_detects_high_empty_rate(self):
        calls = [
            make_call(idx=i, tools=[tc("get_user", result="" if i < 4 else "data")])
            for i in range(5)
        ]
        findings = EmptyReturnDetector().run(calls)
        assert len(findings) == 1
        assert "get_user" in findings[0].title

    def test_no_finding_for_low_empty_rate(self):
        calls = [
            make_call(idx=i, tools=[tc("tool", result="" if i == 0 else "data")])
            for i in range(10)
        ]
        findings = EmptyReturnDetector().run(calls)
        assert len(findings) == 0

    def test_null_string_treated_as_empty(self):
        calls = [
            make_call(idx=i, tools=[tc("api_call", result="null")])
            for i in range(5)
        ]
        findings = EmptyReturnDetector().run(calls)
        assert len(findings) == 1


# ── ModelOverkillDetector ────────────────────────────────────────────

class TestModelOverkill:
    def test_detects_expensive_model_for_simple_tasks(self):
        calls = [
            make_call(idx=i, model="gpt-4o", input_tokens=300, output_tokens=50)
            for i in range(10)
        ]
        findings = ModelOverkillDetector().run(calls)
        assert len(findings) == 1
        assert "gpt-4o" in findings[0].title

    def test_no_finding_for_cheap_model(self):
        calls = [
            make_call(idx=i, model="gpt-4o-mini", input_tokens=300, output_tokens=50)
            for i in range(10)
        ]
        findings = ModelOverkillDetector().run(calls)
        assert len(findings) == 0

    def test_no_finding_for_complex_task(self):
        calls = [
            make_call(idx=i, model="gpt-4o", input_tokens=2000, output_tokens=800)
            for i in range(10)
        ]
        findings = ModelOverkillDetector().run(calls)
        assert len(findings) == 0


# ── ContextBloatDetector ─────────────────────────────────────────────

class TestContextBloat:
    def test_detects_high_system_prompt_ratio(self):
        calls = [
            make_call(idx=i, input_tokens=1000, system_tokens=700)
            for i in range(5)
        ]
        findings = ContextBloatDetector().run(calls)
        system_findings = [f for f in findings if "System prompt" in f.title]
        assert len(system_findings) >= 1

    def test_detects_unbounded_history_growth(self):
        calls = [
            make_call(idx=i, session="s1", input_tokens=500 * (i + 1))
            for i in range(5)
        ]
        findings = ContextBloatDetector().run(calls)
        growth_findings = [f for f in findings if "growth" in f.title.lower()]
        assert len(growth_findings) >= 1

    def test_no_finding_for_stable_context(self):
        calls = [
            make_call(idx=i, session="s1", input_tokens=1000, system_tokens=200)
            for i in range(5)
        ]
        findings = ContextBloatDetector().run(calls)
        assert all("growth" not in f.title.lower() for f in findings)


# ── AgentLoopDetector ────────────────────────────────────────────────

class TestAgentLoop:
    def test_detects_repeated_tool_pair(self):
        pattern = [tc("search"), tc("fetch")]
        calls = []
        for i in range(6):
            calls.append(make_call(idx=i, tools=[pattern[i % 2]]))
        findings = AgentLoopDetector().run(calls)
        assert len(findings) >= 1

    def test_detects_single_tool_loop(self):
        calls = [
            make_call(idx=i, tools=[tc("think")])
            for i in range(4)
        ]
        findings = AgentLoopDetector().run(calls)
        assert len(findings) >= 1

    def test_no_finding_for_varied_sequence(self):
        tools_seq = ["search", "fetch", "parse", "summarise", "respond"]
        calls = [
            make_call(idx=i, tools=[tc(t)]) for i, t in enumerate(tools_seq)
        ]
        findings = AgentLoopDetector().run(calls)
        assert len(findings) == 0
