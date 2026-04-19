"""
Tests for all monk detectors.
Run with: pytest tests/ -v
"""
import pytest
from monk.parsers.auto import TraceCall, ToolCall
from monk.parsers.otel import Span
from monk.detectors.retry_loop import RetryLoopDetector
from monk.detectors.empty_return import EmptyReturnDetector
from monk.detectors.model_overkill import ModelOverkillDetector
from monk.detectors.context_bloat import ContextBloatDetector
from monk.detectors.agent_loop import AgentLoopDetector
from monk.detectors.output_format import OutputFormatDetector
from monk.detectors.plan_execution import PlanExecutionDetector
from monk.detectors.span_consistency import SpanConsistencyDetector


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


# ── Span helpers ─────────────────────────────────────────────────────

_T = 1_000_000_000  # 1 second in nanoseconds

def make_span(
    trace_id="trace1",
    span_id="s1",
    name="llm_call",
    kind="llm",
    t_start=0,
    t_end=500_000_000,
    attributes=None,
    tool_name="",
    tool_result="",
    tool_args="",
    model="gpt-4o",
    status="ok",
) -> Span:
    s = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        name=name,
        kind=kind,
        start_time_ns=t_start,
        end_time_ns=t_end,
        status=status,
        error_message="",
        attributes=attributes or {},
    )
    s.tool_name = tool_name
    s.tool_result = tool_result
    s.tool_args = tool_args
    s.model = model
    return s


# ── OutputFormatDetector ─────────────────────────────────────────────

class TestOutputFormat:
    def _session_spans(self, sys_prompt: str, output: str) -> list[Span]:
        """One LLM span with system prompt input and model output."""
        span = make_span(
            kind="llm",
            attributes={
                "llm.input_messages.0.message.content": sys_prompt,
                "llm.output_messages.0.message.content": output,
            },
        )
        return [span]

    def test_detects_missing_thought_tag(self):
        sys = (
            "You are an AI agent. You must use a strict cycle of 'Thought:' then 'Code:' "
            "then 'Observation:' in every response. Do not skip any of these prefixes. "
            "Failure to follow this format will cause downstream parsing errors."
        )
        out = "I will now look up the data and provide an answer to the question asked."
        spans = self._session_spans(sys, out)
        findings = OutputFormatDetector().run_spans(spans)
        assert len(findings) >= 1
        assert any("Thought:" in f.detail for f in findings)

    def test_no_finding_when_thought_present(self):
        sys = (
            "You are an AI agent. You must use a strict cycle of 'Thought:' then 'Code:' "
            "then 'Observation:' in every response. Do not skip any of these prefixes. "
            "Failure to follow this format will cause downstream parsing errors."
        )
        out = "Thought: I need to search for X.\nCode: search('X')\nObservation: Found X."
        spans = self._session_spans(sys, out)
        findings = OutputFormatDetector().run_spans(spans)
        assert len(findings) == 0

    def test_detects_missing_end_plan_tag(self):
        sys = (
            "You are a planning agent. After you have written your plan, you must write the "
            "'<end_plan>' tag to signal the end of the planning phase. Then proceed with "
            "tool calls to execute your plan step by step. Never skip the <end_plan> marker."
        )
        out = "I plan to search for the answer and then verify it using code."
        spans = self._session_spans(sys, out)
        findings = OutputFormatDetector().run_spans(spans)
        assert any("<end_plan>" in f.detail for f in findings)

    def test_no_finding_when_tag_present(self):
        sys = (
            "You are a planning agent. After you have written your plan, you must write the "
            "'<end_plan>' tag to signal the end of the planning phase. Then proceed with "
            "tool calls to execute your plan step by step. Never skip the <end_plan> marker."
        )
        out = "My plan: search the web.\n<end_plan>\nExecuting plan now."
        spans = self._session_spans(sys, out)
        findings = OutputFormatDetector().run_spans(spans)
        assert len(findings) == 0

    def test_no_finding_when_no_system_prompt(self):
        span = make_span(
            kind="llm",
            attributes={"llm.output_messages.0.message.content": "Some output text here."},
        )
        findings = OutputFormatDetector().run_spans([span])
        assert len(findings) == 0


# ── PlanExecutionDetector ────────────────────────────────────────────

class TestPlanExecution:
    def _make_session(
        self,
        plan_output: str,
        tool_names: list[str],
        trace_id: str = "trace1",
    ) -> list[Span]:
        """LLM span with plan + tool spans for execution."""
        llm = make_span(
            trace_id=trace_id,
            span_id="llm0",
            kind="llm",
            t_start=0,
            t_end=_T,
            attributes={"llm.output_messages.0.message.content": plan_output},
        )
        spans = [llm]
        for i, tname in enumerate(tool_names):
            spans.append(make_span(
                trace_id=trace_id,
                span_id=f"tool{i}",
                kind="tool",
                t_start=_T * (i + 1),
                t_end=_T * (i + 1) + 500_000_000,
                tool_name=tname,
            ))
        return spans

    def test_detects_plan_with_zero_tool_calls(self):
        plan = (
            "1. Search for information about the topic\n"
            "2. Visit the relevant web pages to gather details\n"
            "3. Download the relevant file for analysis\n"
            "4. Write code to process the data\n"
        )
        findings = PlanExecutionDetector().run_spans(self._make_session(plan, []))
        assert any("no tools" in f.title.lower() or "0 calls" in f.title for f in findings)
        assert all(f.severity == "high" for f in findings)

    def test_detects_plan_abandoned_midway(self):
        plan = (
            "1. Search for recent news articles\n"
            "2. Visit the top 3 pages for details\n"
            "3. Download any relevant data files\n"
            "4. Execute code to analyze the data\n"
            "5. Verify results with another search\n"
            "6. Compile final answer\n"
        )
        # Only one tool call made — plan had 6 steps, < 6//2 = 3
        findings = PlanExecutionDetector().run_spans(
            self._make_session(plan, ["web_search"])
        )
        assert len(findings) >= 1

    def test_no_finding_when_all_steps_executed(self):
        plan = (
            "1. Search for information\n"
            "2. Visit the page\n"
            "3. Write code to verify\n"
            "4. Provide final answer\n"
        )
        findings = PlanExecutionDetector().run_spans(
            self._make_session(plan, ["web_search", "visit_page", "python_interpreter"])
        )
        # No "plan abandoned" findings (capability coverage met)
        abandoned = [f for f in findings if "abandoned" in f.title.lower()]
        assert len(abandoned) == 0

    def test_no_finding_for_short_plans(self):
        plan = "1. Search\n2. Answer\n"  # only 2 steps, below MIN_PLAN_STEPS=3
        findings = PlanExecutionDetector().run_spans(self._make_session(plan, []))
        assert len(findings) == 0


# ── SpanConsistencyDetector ──────────────────────────────────────────

class TestSpanConsistency:
    def test_detects_claim_without_tool_call(self):
        """LLM makes 'I found...' claim but no tool was called before it."""
        llm0 = make_span(span_id="llm0", kind="llm", t_start=0, t_end=_T,
            attributes={"llm.output_messages.0.message.content":
                "I will now look up information and plan my approach."})
        llm1 = make_span(span_id="llm1", kind="llm", t_start=2*_T, t_end=3*_T,
            attributes={"llm.output_messages.0.message.content":
                "I found that the answer is 42. According to the results from the search, "
                "the stock price has increased significantly over the past year."})
        # No tool span between llm0 and llm1
        findings = SpanConsistencyDetector().run_spans([llm0, llm1])
        assert any("unverified" in f.title.lower() or "claim" in f.title.lower()
                   for f in findings)

    def test_no_claim_finding_when_tool_precedes_llm(self):
        """LLM makes a claim but a tool was called first — this is fine."""
        llm0 = make_span(span_id="llm0", kind="llm", t_start=0, t_end=_T,
            attributes={"llm.output_messages.0.message.content": "I will search."})
        tool = make_span(span_id="t0", kind="tool", t_start=_T, t_end=2*_T,
            tool_name="web_search", tool_result="Paris is the capital of France.")
        llm1 = make_span(span_id="llm1", kind="llm", t_start=2*_T, t_end=3*_T,
            attributes={"llm.output_messages.0.message.content":
                "I found that Paris is the capital of France. "
                "According to the results from the search, this is confirmed."})
        findings = SpanConsistencyDetector().run_spans([llm0, tool, llm1])
        claim_findings = [f for f in findings if "unverified" in f.title.lower()]
        assert len(claim_findings) == 0

    def test_detects_hallucinated_tool_call(self):
        """Model claims it called 'database' tool but no such span exists."""
        llm = make_span(span_id="llm0", kind="llm", t_start=0, t_end=_T,
            attributes={"llm.output_messages.0.message.content":
                "I called the database tool to look up the user record and found the data."})
        findings = SpanConsistencyDetector().run_spans([llm])
        hallucinated = [f for f in findings if "hallucinated" in f.title.lower()]
        assert len(hallucinated) >= 1

    def test_detects_empty_result_accepted(self):
        """Tool returns 'no results found', but model next claims success."""
        llm0 = make_span(span_id="llm0", kind="llm", t_start=0, t_end=_T,
            attributes={"llm.output_messages.0.message.content": "I will search now."})
        tool = make_span(span_id="t0", kind="tool", t_start=_T, t_end=2*_T,
            tool_name="web_search", tool_result="no results found for this query")
        llm1 = make_span(span_id="llm1", kind="llm", t_start=2*_T, t_end=3*_T,
            attributes={"llm.output_messages.0.message.content":
                "I found extensive data on the topic. According to the results, "
                "the information confirms our hypothesis is correct."})
        findings = SpanConsistencyDetector().run_spans([llm0, tool, llm1])
        empty_findings = [f for f in findings if "empty" in f.title.lower()]
        assert len(empty_findings) >= 1

    def test_detects_final_answer_gap(self):
        """final_answer tool called but no LLM output follows it."""
        llm0 = make_span(span_id="llm0", kind="llm", t_start=0, t_end=_T,
            attributes={"llm.output_messages.0.message.content": "I will answer now."})
        fa = make_span(span_id="fa0", kind="tool", t_start=_T, t_end=2*_T,
            name="final_answer", tool_name="final_answer",
            tool_result="The answer is 42.")
        # No LLM span after final_answer
        findings = SpanConsistencyDetector().run_spans([llm0, fa])
        gap_findings = [f for f in findings if "final answer" in f.title.lower()]
        assert len(gap_findings) >= 1

    def test_no_final_answer_gap_when_llm_follows(self):
        """final_answer tool called and LLM output follows — no gap."""
        llm0 = make_span(span_id="llm0", kind="llm", t_start=0, t_end=_T,
            attributes={"llm.output_messages.0.message.content": "Planning."})
        fa = make_span(span_id="fa0", kind="tool", t_start=_T, t_end=2*_T,
            name="final_answer", tool_name="final_answer",
            tool_result="The answer is 42.")
        llm1 = make_span(span_id="llm1", kind="llm", t_start=2*_T, t_end=3*_T,
            attributes={"llm.output_messages.0.message.content": "The answer is 42."})
        findings = SpanConsistencyDetector().run_spans([llm0, fa, llm1])
        gap_findings = [f for f in findings if "final answer" in f.title.lower()]
        assert len(gap_findings) == 0
