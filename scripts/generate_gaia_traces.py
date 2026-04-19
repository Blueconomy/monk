"""
Generate GAIA-style smolagents OTEL traces for monk evaluation.

Uses MATH-500 (public) as question source — same multi-step reasoning
structure as GAIA Level 1-3. Injects realistic failure patterns that monk
is designed to detect.

Output: tests/fixtures/gaia_smolagents_traces.jsonl  (OTEL JSONL, one span per line)

Usage:
    python scripts/generate_gaia_traces.py
    python scripts/generate_gaia_traces.py --tasks 200 --seed 99
"""
from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path

# ── Question bank ─────────────────────────────────────────────────────────────

def load_questions(n: int, seed: int) -> list[tuple[int, str, int]]:
    """
    Load questions from MATH-500 (public HF dataset) and map their difficulty
    levels (1-5) to GAIA levels (1-3).  Falls back to a curated synthetic
    question bank if the dataset is unavailable.

    Returns: list of (task_id, question_text, level)  where level ∈ {1, 2, 3}
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rng = random.Random(seed)
        rows = list(ds)
        rng.shuffle(rows)
        rows = rows[:n]
        questions = []
        for i, row in enumerate(rows):
            # MATH levels 1-2 → GAIA level 1, 3 → GAIA level 2, 4-5 → GAIA level 3
            math_level = int(row.get("level", 2))
            gaia_level = 1 if math_level <= 2 else (2 if math_level == 3 else 3)
            q = row["problem"]
            questions.append((i, q, gaia_level))
        print(f"Loaded {len(questions)} tasks from MATH-500 (HuggingFaceH4/MATH-500)")
        return questions
    except Exception as exc:
        print(f"MATH-500 unavailable ({exc}), using synthetic GAIA-style questions")
        return _synthetic_questions(n, seed)


def _synthetic_questions(n: int, seed: int) -> list[tuple[int, str, int]]:
    """
    Curated GAIA-style questions drawn from published GAIA paper examples
    and the publicly-visible leaderboard questions.
    """
    bank = [
        # Level 1 — single-step lookup
        ("How many studio albums did the Beatles release between 1963 and 1966?", 1),
        ("What is the capital of the country that hosted the 2022 FIFA World Cup?", 1),
        ("Which element has atomic number 79?", 1),
        ("In what year was the Eiffel Tower completed?", 1),
        ("What programming language was Python named after?", 1),
        ("How many legs does a spider have?", 1),
        ("What is the chemical formula for water?", 1),
        ("Who wrote 'Pride and Prejudice'?", 1),
        ("What is the largest planet in the solar system?", 1),
        ("In which country was Marie Curie born?", 1),
        ("What is the speed of light in a vacuum in m/s?", 1),
        ("How many bones are in the adult human body?", 1),
        ("What year did World War II end?", 1),
        ("What is the currency of Japan?", 1),
        ("Who painted the Mona Lisa?", 1),
        # Level 2 — multi-step reasoning
        ("A train travels 120 miles in 2 hours. If it then travels 180 miles in 3 hours, what is its average speed over the whole journey?", 2),
        ("What is the total number of prime numbers less than 50?", 2),
        ("If a square has a diagonal of 10 cm, what is its area?", 2),
        ("How many US states share a border with Canada?", 2),
        ("What is the sum of interior angles of a regular hexagon?", 2),
        ("How many days are there in a leap year?", 2),
        ("What is the cube root of 512?", 2),
        ("If you invest $1000 at 5% annual compound interest, how much do you have after 3 years?", 2),
        ("What is the product of the first 5 prime numbers?", 2),
        ("A rectangle has perimeter 36cm and width 8cm. What is its area?", 2),
        ("How many edges does a regular octahedron have?", 2),
        ("What is the GCD of 48 and 36?", 2),
        ("If a clock shows 3:45, what is the angle between the hour and minute hands?", 2),
        ("How many ways can you arrange 5 books on a shelf?", 2),
        ("What is the median of the first 10 odd numbers?", 2),
        # Level 3 — research + multi-step + verification
        ("What is the Yola word for 'gimlie' and what is its Latin root? What Spanish word shares that spelling? What is the Google translation of the 1994 Collins example sentence source title?", 3),
        ("What is the atomic weight of the element whose symbol is the abbreviation for the Latin word for 'tin'?", 3),
        ("Find the smallest prime greater than 1000 and compute the sum of its digits.", 3),
        ("What is the GDP of the country with the highest Human Development Index that is not in Europe?", 3),
        ("How many Academy Awards has the director of the highest-grossing film of 2019 won in total?", 3),
        ("What is the name of the theorem that states every even integer greater than 2 can be expressed as the sum of two primes?", 3),
        ("What is the capital of the province in Canada that has the largest area?", 3),
        ("In Python, what is the output of: sorted(set([3,1,4,1,5,9,2,6,5,3,5]), reverse=True)[:3]?", 3),
        ("Which country has the most UNESCO World Heritage Sites and how many does it have?", 3),
        ("What is the derivative of x^3 * sin(x) evaluated at x=0?", 3),
        ("How many languages are spoken in Papua New Guinea according to Ethnologue?", 3),
        ("What year did the first iPhone ship and who was the CEO of Apple at the time?", 3),
        ("What is the chemical name of the compound with molecular formula C6H12O6 and what biological process produces it?", 3),
        ("What is the name of the largest moon of Neptune and who discovered it?", 3),
        ("How many moves is the longest possible game of chess theoretically?", 3),
    ]
    rng = random.Random(seed)
    rng.shuffle(bank)
    result = []
    for i, (q, lvl) in enumerate(bank * ((n // len(bank)) + 1)):
        if i >= n:
            break
        result.append((i, q, lvl))
    return result


# ── Span builders ─────────────────────────────────────────────────────────────

def _sid() -> str:
    return uuid.uuid4().hex[:16]


def _make_agent_root(trace_id: str, span_id: str, question: str,
                     base_t: int, duration_ns: int, model: str,
                     total_prompt: int, total_completion: int) -> dict:
    # NOTE: deliberately omit gen_ai.usage.* and gen_ai.request.model from the
    # root agent span so that the OTEL parser classifies it as kind="agent"
    # rather than kind="llm".  This ensures plan_execution and other span-aware
    # detectors see the child LLM spans (not the root) as the planning step.
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": None,
        "name": "CodeAgent.run",
        "startTimeUnixNano": str(base_t),
        "endTimeUnixNano": str(base_t + duration_ns),
        "status": {"code": "STATUS_CODE_OK"},
        "attributes": {
            "openinference.span.kind": "AGENT",
            "input.value": question,
            "smolagents.max_steps": "12",
            "smolagents.tools_names": '["web_search","visit_webpage","python_interpreter","final_answer"]',
            # Store aggregates in smolagents-style keys (not gen_ai.*) so the
            # OTEL _enrich_span classifier does not promote this to kind="llm"
            "smolagents.llm_calls.prompt_tokens_total": total_prompt,
            "smolagents.llm_calls.completion_tokens_total": total_completion,
            "smolagents.model": model,
        },
        "events": [],
    }


def _make_llm_span(trace_id: str, parent_id: str, model: str,
                   start_t: int, duration_ns: int,
                   prompt_tokens: int, completion_tokens: int,
                   output_content: str, input_content: str) -> dict:
    return {
        "traceId": trace_id,
        "spanId": _sid(),
        "parentSpanId": parent_id,
        "name": "LLMCall",
        "startTimeUnixNano": str(start_t),
        "endTimeUnixNano": str(start_t + duration_ns),
        "status": {"code": "STATUS_CODE_OK"},
        "attributes": {
            "openinference.span.kind": "LLM",
            "gen_ai.request.model": model,
            "gen_ai.usage.prompt_tokens": prompt_tokens,
            "gen_ai.usage.completion_tokens": completion_tokens,
            "llm.output_messages.0.message.content": output_content,
            "llm.input_messages.0.message.content": input_content,
        },
        "events": [],
    }


def _make_tool_span(trace_id: str, parent_id: str, tool_name: str,
                    start_t: int, duration_ns: int,
                    tool_args: str, tool_result: str,
                    status_code: str = "STATUS_CODE_OK") -> dict:
    return {
        "traceId": trace_id,
        "spanId": _sid(),
        "parentSpanId": parent_id,
        "name": tool_name,
        "startTimeUnixNano": str(start_t),
        "endTimeUnixNano": str(start_t + duration_ns),
        "status": {"code": status_code},
        "attributes": {
            "openinference.span.kind": "TOOL",
            "tool.name": tool_name,
            "tool.arguments": tool_args,
            "tool.result": tool_result,
        },
        "events": [],
    }


# ── Trace pattern generators ──────────────────────────────────────────────────

def gen_normal_trace(task_id: int, question: str, level: int, model: str) -> list[dict]:
    """Clean, successful agent run — no defects."""
    trace_id = uuid.uuid4().hex
    base_t = int(time.time() * 1e9) - random.randint(0, 86400) * int(1e9)
    root_id = _sid()
    spans = []

    # 1. Root agent span (placeholder, added last with correct totals)
    n_tools = level + 1  # level 1 → 2 tools, level 3 → 4 tools
    tool_names = ["web_search", "visit_webpage", "python_interpreter", "final_answer"][:n_tools]

    prompt_tokens = [350 + level * 150 + i * 80 for i in range(n_tools + 1)]
    completion_tokens = [180, *[60 for _ in range(n_tools - 1)], 120]

    # 2. Planning LLM call
    plan_text = (
        f"Thought: I need to find information to answer this question step by step.\n"
        f"1. Search for relevant information about: {question[:60]}\n"
        f"2. Visit the most relevant results to extract details\n"
        f"3. Process and verify the information found\n"
        f"4. Formulate the final answer"
    )
    t = base_t + int(0.1e9)
    planning_span = _make_llm_span(
        trace_id, root_id, model, t, int(1.5e9),
        prompt_tokens[0], completion_tokens[0],
        plan_text,
        f"System: You are a helpful agent. Answer accurately.\n\nTask: {question}",
    )
    spans.append(planning_span)

    # 3. Tool calls
    t += int(1.8e9)
    for i, tool_name in enumerate(tool_names[:-1]):
        args = json.dumps({"query": question[:60]} if "search" in tool_name
                          else {"url": f"https://example.com/result{i}"})
        result = f"Result {i+1}: Found relevant information about the topic. Detail: {'X' * 50}"
        tool_span = _make_tool_span(trace_id, root_id, tool_name, t, int(1.2e9), args, result)
        spans.append(tool_span)
        t += int(1.5e9)

        # Follow-up LLM call after each tool (except last)
        if i < len(tool_names) - 2:
            followup = _make_llm_span(
                trace_id, root_id, model, t, int(1.0e9),
                prompt_tokens[i + 1], completion_tokens[i + 1],
                f"Thought: I have some information. Let me continue researching.\nCode: {tool_names[i+1]}(query='{question[:40]}')",
                f"System: Continue.\n\nPrevious result: {result[:100]}",
            )
            spans.append(followup)
            t += int(1.2e9)

    # 4. Final answer LLM call
    final_llm = _make_llm_span(
        trace_id, root_id, model, t, int(2.0e9),
        prompt_tokens[-1], completion_tokens[-1],
        "Based on my research, I can now provide the final answer.",
        f"System: Provide final answer.\n\nQuestion: {question}\n\nResearch complete.",
    )
    spans.append(final_llm)
    t += int(2.1e9)

    # 5. final_answer tool
    final_tool = _make_tool_span(
        trace_id, root_id, "final_answer", t, int(0.1e9),
        json.dumps({"answer": "The computed answer based on research."}),
        "Answer submitted successfully.",
    )
    spans.append(final_tool)

    total_duration = t + int(0.2e9) - base_t
    total_prompt = sum(prompt_tokens)
    total_completion = sum(completion_tokens)

    root_span = _make_agent_root(
        trace_id, root_id, question, base_t, total_duration,
        model, total_prompt, total_completion,
    )
    return [root_span] + spans


def gen_retry_loop_trace(task_id: int, question: str, level: int, model: str) -> list[dict]:
    """Same tool called 4+ consecutive times — triggers retry_loop detector."""
    trace_id = uuid.uuid4().hex
    base_t = int(time.time() * 1e9) - random.randint(0, 43200) * int(1e9)
    root_id = _sid()
    spans = []

    n_retries = random.randint(4, 6)
    tool_name = "web_search"
    prompt_tokens = 500 + level * 200

    # Planning LLM call
    plan = (
        "Thought: I need to search for information.\n"
        "1. Search for the answer online\n"
        "2. Visit relevant pages\n"
        "3. Verify and return the answer"
    )
    t = base_t + int(0.2e9)
    spans.append(_make_llm_span(
        trace_id, root_id, model, t, int(1.5e9),
        prompt_tokens, 150,
        plan,
        f"System: Answer accurately.\n\nTask: {question}",
    ))
    t += int(1.8e9)

    # Retry loop: each LLM call spawns the same tool as a child span.
    # This ensures to_trace_call() finds tool_calls nested under each LLM span,
    # so the trace-level retry_loop and agent_loop detectors fire correctly.
    for i in range(n_retries):
        args = json.dumps({"query": f"{question[:40]} attempt {i+1}"})
        # Result gets emptier as retries progress — realistic failure pattern
        result = (f"Search attempt {i+1}: No relevant results found." if i > 1
                  else f"Search result {i+1}: Partial information found about the topic.")

        # LLM span that decides to call the tool
        llm_span_id = _sid()
        llm_span = _make_llm_span(
            trace_id, root_id, model, t, int(0.8e9),
            prompt_tokens + i * 100, 80,
            f"Thought: The search didn't return good results, let me try again.\nCode: {tool_name}(query='{question[:35]} more specific')",
            f"Previous result was insufficient. Attempt {i+1}. Task: {question}",
        )
        llm_span["spanId"] = llm_span_id
        spans.append(llm_span)
        t += int(0.9e9)

        # Tool span as child of this LLM span
        tool_span = _make_tool_span(trace_id, llm_span_id, tool_name, t, int(1.0e9), args, result)
        spans.append(tool_span)
        t += int(1.2e9)

    # Give up with a weak final answer
    spans.append(_make_llm_span(
        trace_id, root_id, model, t, int(1.5e9),
        prompt_tokens + n_retries * 100, 100,
        "I was unable to find definitive information. Based on my knowledge, the answer is approximately correct.",
        f"Task: {question}\n\nAfter {n_retries} search attempts, providing best-effort answer.",
    ))
    t += int(1.7e9)

    root_span = _make_agent_root(
        trace_id, root_id, question, base_t, t - base_t,
        model, prompt_tokens * n_retries, 150 * n_retries,
    )
    return [root_span] + spans


def gen_empty_return_trace(task_id: int, question: str, level: int, model: str) -> list[dict]:
    """Tool returns empty/null 4+ out of 5 calls — triggers empty_return detector."""
    trace_id = uuid.uuid4().hex
    base_t = int(time.time() * 1e9) - random.randint(0, 43200) * int(1e9)
    root_id = _sid()
    spans = []
    prompt_tokens = 400 + level * 150
    t = base_t + int(0.2e9)

    # Planning
    spans.append(_make_llm_span(
        trace_id, root_id, model, t, int(1.5e9),
        prompt_tokens, 140,
        "Thought: Let me look up the information step by step.\n1. Use the calculator\n2. Search for context\n3. Return answer",
        f"Task: {question}",
    ))
    t += int(1.8e9)

    tool_name = "python_interpreter"
    n_calls = 5
    empty_results = ["", "null", "None", "[]", ""]  # 5 mostly empty results
    real_result = "42"

    for i in range(n_calls):
        result = empty_results[i] if i < len(empty_results) else real_result
        args = json.dumps({"code": f"# attempt {i+1}\nprint(solve('{question[:30]}'))"})

        # LLM span that invokes the tool — tool is a child so to_trace_call picks it up
        llm_span_id = _sid()
        llm_span = _make_llm_span(
            trace_id, root_id, model, t, int(1.0e9),
            prompt_tokens + i * 50, 90,
            f"Thought: Got result: '{result}'. Let me try a different approach.\nCode: {tool_name}(code='...')",
            f"Previous result: '{result}'. Task: {question}",
        )
        llm_span["spanId"] = llm_span_id
        spans.append(llm_span)
        t += int(1.1e9)

        # Tool as child of LLM span — the empty result is visible to trace-level detectors
        spans.append(_make_tool_span(trace_id, llm_span_id, tool_name, t, int(0.5e9), args, result))
        t += int(0.8e9)

    root_span = _make_agent_root(
        trace_id, root_id, question, base_t, t - base_t,
        model, prompt_tokens * n_calls, 90 * n_calls,
    )
    return [root_span] + spans


def gen_plan_abandonment_trace(task_id: int, question: str, level: int, model: str) -> list[dict]:
    """
    4-step plan written, only 1 tool call made (< half the plan steps).
    Triggers plan_execution detector (incomplete execution check).
    """
    trace_id = uuid.uuid4().hex
    base_t = int(time.time() * 1e9) - random.randint(0, 43200) * int(1e9)
    root_id = _sid()
    spans = []
    prompt_tokens = 600 + level * 200
    t = base_t + int(0.2e9)

    # Planning LLM — detailed 4-step plan
    plan_text = (
        f"Thought: I need to solve this step-by-step.\n"
        f"1. Search for background information on: {question[:50]}\n"
        f"2. Visit the Wikipedia page to get authoritative facts\n"
        f"3. Use python_interpreter to compute or verify the numerical result\n"
        f"4. Cross-check with an additional search to confirm accuracy\n"
        f"Let me start with step 1."
    )
    spans.append(_make_llm_span(
        trace_id, root_id, model, t, int(2.0e9),
        prompt_tokens, 200,
        plan_text,
        f"System: You are a careful, methodical agent. Answer: {question}",
    ))
    t += int(2.2e9)

    # Only 1 tool call (step 1 of 4)
    spans.append(_make_tool_span(
        trace_id, root_id, "web_search", t, int(1.5e9),
        json.dumps({"query": question[:50]}),
        "Search result: Found some general information about this topic.",
    ))
    t += int(1.7e9)

    # Agent immediately jumps to answer without steps 2-4
    spans.append(_make_llm_span(
        trace_id, root_id, model, t, int(2.0e9),
        prompt_tokens + 300, 150,
        "Based on the search result, I can provide the answer without further verification.",
        f"Search result obtained. Task: {question}",
    ))
    t += int(2.2e9)

    spans.append(_make_tool_span(
        trace_id, root_id, "final_answer", t, int(0.1e9),
        json.dumps({"answer": "Premature answer without full plan execution."}),
        "Submitted.",
    ))
    t += int(0.2e9)

    root_span = _make_agent_root(
        trace_id, root_id, question, base_t, t - base_t,
        model, prompt_tokens * 2, 350,
    )
    return [root_span] + spans


def gen_plan_abandoned_zero_tools(task_id: int, question: str, level: int, model: str) -> list[dict]:
    """
    Plan written but zero tool calls — triggers plan_execution 'no tools' check.
    Classic hallucination: agent invents the answer without any grounding.
    """
    trace_id = uuid.uuid4().hex
    base_t = int(time.time() * 1e9) - random.randint(0, 43200) * int(1e9)
    root_id = _sid()
    spans = []
    prompt_tokens = 700 + level * 250
    t = base_t + int(0.3e9)

    # Detailed plan — but then immediately answers from memory
    plan_text = (
        f"Thought: To answer this, I should:\n"
        f"1. Search for current data on: {question[:50]}\n"
        f"2. Visit authoritative sources for verification\n"
        f"3. Cross-reference multiple sources\n"
        f"4. Synthesize and provide the final verified answer\n\n"
        f"Actually, I know the answer from my training data. "
        f"The answer is: [fabricated answer without any tool use]."
    )
    spans.append(_make_llm_span(
        trace_id, root_id, model, t, int(2.5e9),
        prompt_tokens, 300,
        plan_text,
        f"System: You must verify all factual claims with tool calls.\n\nTask: {question}",
    ))
    t += int(2.7e9)

    # No tool calls at all — straight to final answer
    spans.append(_make_llm_span(
        trace_id, root_id, model, t, int(1.0e9),
        prompt_tokens + 400, 100,
        "Final answer: The answer is [hallucinated value].",
        "Providing answer based on training knowledge.",
    ))
    t += int(1.2e9)

    root_span = _make_agent_root(
        trace_id, root_id, question, base_t, t - base_t,
        model, prompt_tokens * 2, 400,
    )
    return [root_span] + spans


def gen_error_cascade_trace(task_id: int, question: str, level: int, model: str) -> list[dict]:
    """
    Tool errors are ignored and agent keeps burning tokens — triggers error_cascade.
    """
    trace_id = uuid.uuid4().hex
    base_t = int(time.time() * 1e9) - random.randint(0, 43200) * int(1e9)
    root_id = _sid()
    spans = []
    prompt_tokens = 450 + level * 180
    t = base_t + int(0.2e9)

    # Planning
    spans.append(_make_llm_span(
        trace_id, root_id, model, t, int(1.5e9),
        prompt_tokens, 150,
        "Thought: I will search for and process the information.\n1. Search\n2. Process\n3. Answer",
        f"Task: {question}",
    ))
    t += int(1.7e9)

    # First tool: ERROR
    spans.append(_make_tool_span(
        trace_id, root_id, "web_search", t, int(0.8e9),
        json.dumps({"query": question[:50]}),
        "Error: Connection timeout after 30s",
        status_code="STATUS_CODE_ERROR",
    ))
    t += int(1.0e9)

    # Agent ignores error and calls LLM anyway (×3 times — the cascade)
    for i in range(3):
        spans.append(_make_llm_span(
            trace_id, root_id, model, t, int(1.2e9),
            prompt_tokens + i * 200, 120,
            f"Thought: I got an error but let me try to proceed with what I know. Attempt {i+2}.",
            f"Error from tool: Connection timeout. Task: {question}",
        ))
        t += int(1.4e9)

        if i < 2:
            # More failed tool attempts
            spans.append(_make_tool_span(
                trace_id, root_id, "visit_webpage", t, int(0.5e9),
                json.dumps({"url": f"https://example.com/fallback{i}"}),
                "Error: 404 Not Found",
                status_code="STATUS_CODE_ERROR",
            ))
            t += int(0.7e9)

    root_span = _make_agent_root(
        trace_id, root_id, question, base_t, t - base_t,
        model, prompt_tokens * 4, 120 * 3,
    )
    return [root_span] + spans


def gen_token_bloat_trace(task_id: int, question: str, level: int, model: str) -> list[dict]:
    """
    Token counts grow monotonically across calls — triggers token_bloat detector.
    Simulates unbounded context window (no summarisation).
    """
    trace_id = uuid.uuid4().hex
    base_t = int(time.time() * 1e9) - random.randint(0, 43200) * int(1e9)
    root_id = _sid()
    spans = []
    t = base_t + int(0.2e9)

    # 6 LLM calls with strictly growing token counts
    base_tokens = 500
    growth_per_call = 600  # keeps growing — no summarisation
    n_calls = 6

    for i in range(n_calls):
        input_tok = base_tokens + i * growth_per_call
        output_tok = 150 + i * 20

        content = (
            f"Thought: Continuing analysis (step {i+1}). "
            f"Context so far: {'[accumulated history] ' * (i + 1)}"
        )
        spans.append(_make_llm_span(
            trace_id, root_id, model, t, int(2.0e9),
            input_tok, output_tok,
            content,
            f"Task: {question}\n\n" + ("Previous results... " * (i * 5)),
        ))
        t += int(2.3e9)

        if i < n_calls - 1:
            tool_name = ["web_search", "visit_webpage", "python_interpreter"][i % 3]
            spans.append(_make_tool_span(
                trace_id, root_id, tool_name, t, int(1.0e9),
                json.dumps({"query": question[:40]}),
                f"Result {i+1}: " + ("data " * 50),  # big results added to context verbatim
            ))
            t += int(1.2e9)

    total_prompt = sum(base_tokens + i * growth_per_call for i in range(n_calls))
    total_completion = sum(150 + i * 20 for i in range(n_calls))

    root_span = _make_agent_root(
        trace_id, root_id, question, base_t, t - base_t,
        model, total_prompt, total_completion,
    )
    return [root_span] + spans


def gen_cross_turn_memory_trace(task_id: int, question: str, level: int, model: str) -> list[dict]:
    """
    Same tool+args re-fetched across multiple turns — triggers cross_turn_memory.
    Agent has no memory of previous results and re-fetches identical data.
    """
    trace_id = uuid.uuid4().hex
    base_t = int(time.time() * 1e9) - random.randint(0, 43200) * int(1e9)
    root_id = _sid()
    spans = []
    prompt_tokens = 400 + level * 150
    t = base_t + int(0.2e9)
    repeated_args = json.dumps({"query": question[:50]})

    for i in range(4):
        # LLM call that "forgets" previous searches
        spans.append(_make_llm_span(
            trace_id, root_id, model, t, int(1.5e9),
            prompt_tokens, 130,
            f"Thought: I need to search for information about this question (turn {i+1}).\nCode: web_search(query='{question[:40]}')",
            f"System: Answer accurately.\n\nTask: {question}",
        ))
        t += int(1.7e9)

        # Identical tool call — same args every turn
        spans.append(_make_tool_span(
            trace_id, root_id, "web_search", t, int(1.2e9),
            repeated_args,  # exact same args each time
            f"Search result: Same information as before — found {i+1} relevant articles about the topic.",
        ))
        t += int(1.4e9)

    # Final answer
    spans.append(_make_llm_span(
        trace_id, root_id, model, t, int(1.5e9),
        prompt_tokens + 200, 100,
        "Based on the research, the answer is [value].",
        f"Task: {question}. Research complete.",
    ))
    t += int(1.7e9)

    root_span = _make_agent_root(
        trace_id, root_id, question, base_t, t - base_t,
        model, prompt_tokens * 5, 130 * 5,
    )
    return [root_span] + spans


def gen_model_overkill_trace(task_id: int, question: str, level: int, model: str) -> list[dict]:
    """
    GPT-4o used for trivial formatting tasks — triggers model_overkill.
    Heavy model doing cheap work (simple JSON formatting or string operations).
    """
    trace_id = uuid.uuid4().hex
    base_t = int(time.time() * 1e9) - random.randint(0, 43200) * int(1e9)
    root_id = _sid()
    spans = []
    # Force expensive model regardless of input level
    expensive_model = "gpt-4o"
    t = base_t + int(0.2e9)

    trivial_tasks = [
        ("Format this JSON output", '{"result": "42"}', 20),
        ("Convert this to uppercase", "hello world", 15),
        ("Add a period to this sentence", "The answer is 42", 18),
        ("Extract the number from this string", "The count is 7 items", 12),
    ]

    for task_desc, inp, out_tok in trivial_tasks:
        spans.append(_make_llm_span(
            trace_id, root_id, expensive_model, t, int(0.5e9),
            45, out_tok,  # very low token counts — trivial task
            inp.upper() if "upper" in task_desc.lower() else inp + ".",
            f"System: {task_desc}.\n\nInput: {inp}",
        ))
        t += int(0.7e9)

    root_span = _make_agent_root(
        trace_id, root_id, question, base_t, t - base_t,
        expensive_model, 45 * len(trivial_tasks), 20 * len(trivial_tasks),
    )
    return [root_span] + spans


# ── Pattern catalogue ─────────────────────────────────────────────────────────

PATTERNS = {
    "normal":             (gen_normal_trace,              "Clean run — no defects"),
    "retry_loop":         (gen_retry_loop_trace,          "Same tool 4-6x consecutive"),
    "empty_return":       (gen_empty_return_trace,        "Tool returns empty 4/5 calls, agent continues"),
    "plan_abandonment":   (gen_plan_abandonment_trace,    "4-step plan, only 1 tool call made"),
    "plan_no_tools":      (gen_plan_abandoned_zero_tools, "Plan written, zero tool calls (hallucination)"),
    "error_cascade":      (gen_error_cascade_trace,       "Tool errors ignored, 3 wasted LLM calls"),
    "token_bloat":        (gen_token_bloat_trace,         "Monotonic token growth, no summarisation"),
    "cross_turn_memory":  (gen_cross_turn_memory_trace,   "Same tool+args re-fetched across 4 turns"),
    "model_overkill":     (gen_model_overkill_trace,      "GPT-4o for trivial formatting tasks"),
}

# Distribution: 25% normal, rest split across defect types
PATTERN_WEIGHTS = {
    "normal":            0.25,
    "retry_loop":        0.10,
    "empty_return":      0.10,
    "plan_abandonment":  0.12,
    "plan_no_tools":     0.08,
    "error_cascade":     0.10,
    "token_bloat":       0.10,
    "cross_turn_memory": 0.08,
    "model_overkill":    0.07,
}

# Models used by GAIA leaderboard submissions (from results_public)
GAIA_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-6",
    "claude-3-5-haiku-20241022",
    "meta-llama/Llama-3.3-70B-Instruct",
]


# ── Main generation ───────────────────────────────────────────────────────────

def generate_traces(n_tasks: int, seed: int, output_path: Path) -> dict:
    """Generate n_tasks traces and write to output_path. Returns summary stats."""
    rng = random.Random(seed)

    questions = load_questions(n_tasks, seed)
    pattern_names = list(PATTERN_WEIGHTS.keys())
    pattern_weights = [PATTERN_WEIGHTS[p] for p in pattern_names]

    stats: dict[str, int] = {p: 0 for p in pattern_names}
    total_spans = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for task_id, question, level in questions:
            pattern_name = rng.choices(pattern_names, weights=pattern_weights, k=1)[0]
            gen_fn, _ = PATTERNS[pattern_name]
            model = rng.choice(GAIA_MODELS)

            try:
                spans = gen_fn(task_id, question, level, model)
            except Exception as exc:
                print(f"  Warning: task {task_id} ({pattern_name}) failed: {exc}")
                spans = gen_normal_trace(task_id, question, level, model)
                pattern_name = "normal"

            for span in spans:
                f.write(json.dumps(span) + "\n")

            stats[pattern_name] += 1
            total_spans += len(spans)

    return {"stats": stats, "total_spans": total_spans, "total_traces": n_tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GAIA-style smolagents OTEL traces")
    parser.add_argument("--tasks", type=int, default=150, help="Number of traces to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str,
                        default="/sessions/practical-funny-euler/mnt/monk/tests/fixtures/gaia_smolagents_traces.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()

    output_path = Path(args.output)
    print(f"\nGenerating {args.tasks} GAIA-style traces (seed={args.seed})...")
    print(f"Output: {output_path}\n")

    result = generate_traces(args.tasks, args.seed, output_path)

    print(f"\nGeneration complete:")
    print(f"  Total traces:  {result['total_traces']}")
    print(f"  Total spans:   {result['total_spans']}")
    print(f"\nPattern distribution:")
    for pattern, count in sorted(result["stats"].items(), key=lambda x: -x[1]):
        desc = PATTERNS[pattern][1]
        print(f"  {pattern:<25} {count:>4} traces  — {desc}")
    print(f"\nFile written: {output_path}")


if __name__ == "__main__":
    main()
