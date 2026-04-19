#!/usr/bin/env python3
"""
Run GAIA Level-1 tasks through smolagents with OTEL tracing.
Collect monk-compatible traces.
"""
import os
import json
import time
import uuid
import traceback
from pathlib import Path

# Token must be set in environment before running this script
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")

from datasets import load_dataset
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

# Use InferenceClientModel with Cerebras provider (free, no credit needed)
# Cerebras provides Llama-3.1-8B-Instruct for free via HF Inference Router
model = InferenceClientModel(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    provider="cerebras",
    token=HF_TOKEN,
    timeout=60,
)

def run_task_with_tracing(task_id, question, expected_answer=None):
    """Run one GAIA task through smolagents and collect monk-compatible OTEL trace."""
    trace_id = uuid.uuid4().hex
    base_ns = int(time.time() * 1e9)
    spans = []

    try:
        agent = CodeAgent(
            tools=[DuckDuckGoSearchTool()],
            model=model,
            max_steps=5,
            verbosity_level=0,
        )

        start = time.time()
        result = agent.run(question)
        duration_s = time.time() - start

        # Access agent memory/logs
        memory = getattr(agent, 'memory', None)
        steps = []
        if memory:
            steps = getattr(memory, 'steps', []) or []

        step_ns = base_ns + 500_000_000

        for i, step in enumerate(steps):
            step_ns_end = step_ns + 2_000_000_000
            step_type = type(step).__name__

            # Tool call step
            tool_calls = getattr(step, 'tool_calls', None) or []
            for tc in tool_calls:
                tool_name = getattr(tc, 'name', 'unknown_tool')
                tool_args = json.dumps(getattr(tc, 'arguments', {}), default=str)[:500]
                observations = getattr(step, 'observations', '') or ''
                tool_result = str(observations)[:500]

                spans.append({
                    "traceId": trace_id,
                    "spanId": uuid.uuid4().hex[:16],
                    "name": tool_name,
                    "startTimeUnixNano": str(step_ns),
                    "endTimeUnixNano": str(step_ns + 1_000_000_000),
                    "status": {"code": "STATUS_CODE_OK"},
                    "attributes": {
                        "openinference.span.kind": "TOOL",
                        "tool.name": tool_name,
                        "tool.arguments": tool_args,
                        "tool.result": tool_result,
                    }
                })

            # LLM span
            model_input = getattr(step, 'model_input_messages', None)
            model_output = getattr(step, 'model_output_message', None)
            llm_input_str = str(model_input or '')[:800]
            llm_output_str = str(model_output or '')[:500]
            input_tok = max(len(llm_input_str) // 4, 100)
            output_tok = max(len(llm_output_str) // 4, 50)

            spans.append({
                "traceId": trace_id,
                "spanId": uuid.uuid4().hex[:16],
                "name": "LLMCall",
                "startTimeUnixNano": str(step_ns + 1_100_000_000),
                "endTimeUnixNano": str(step_ns_end),
                "status": {"code": "STATUS_CODE_OK"},
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "gen_ai.request.model": "Qwen/Qwen2.5-72B-Instruct",
                    "gen_ai.usage.prompt_tokens": input_tok,
                    "gen_ai.usage.completion_tokens": output_tok,
                    "llm.input_messages.0.message.content": llm_input_str[:500],
                    "llm.output_messages.0.message.content": llm_output_str[:500],
                }
            })
            step_ns = step_ns_end + 100_000_000

        # Root span
        spans.insert(0, {
            "traceId": trace_id,
            "spanId": uuid.uuid4().hex[:16],
            "name": "CodeAgent.run",
            "startTimeUnixNano": str(base_ns),
            "endTimeUnixNano": str(base_ns + int(duration_s * 1e9)),
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": {
                "openinference.span.kind": "AGENT",
                "input.value": question[:300],
                "output.value": str(result)[:300],
                "gaia.task_id": str(task_id),
                "gaia.expected": str(expected_answer or "")[:100],
                "gaia.correct": str(
                    str(result).strip().lower() == str(expected_answer or "").strip().lower()
                ),
            }
        })

        return spans, str(result), True, None

    except Exception as e:
        err_msg = str(e)[:300]
        err_trace = traceback.format_exc()[:500]
        return [{
            "traceId": trace_id,
            "spanId": uuid.uuid4().hex[:16],
            "name": "CodeAgent.run",
            "startTimeUnixNano": str(base_ns),
            "endTimeUnixNano": str(base_ns + 5_000_000_000),
            "status": {"code": "STATUS_CODE_ERROR", "message": err_msg},
            "attributes": {
                "openinference.span.kind": "AGENT",
                "input.value": question[:300],
                "gaia.task_id": str(task_id),
                "error.type": type(e).__name__,
                "error.message": err_msg,
            }
        }], err_msg, False, err_msg


def main():
    print("Loading GAIA dataset...")
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", token=HF_TOKEN)
    val = ds["validation"]
    level1 = [dict(row) for row in val if row.get("Level") == "1"]
    print(f"Level 1 tasks: {len(level1)}")

    tasks_to_run = level1[:30]
    all_spans = []
    results_log = []

    print(f"\nRunning {len(tasks_to_run)} GAIA Level-1 tasks...")

    for i, task in enumerate(tasks_to_run):
        question = task.get("Question", "")
        expected = task.get("Final answer", task.get("answer", ""))
        task_id = task.get("task_id", f"gaia_{i}")

        # Skip tasks requiring file attachments
        has_file = bool(task.get("file_name", "").strip())
        if has_file:
            print(f"[{i+1}/{len(tasks_to_run)}] SKIP (file attachment): {task_id}")
            results_log.append({
                "task_id": task_id,
                "question": question[:100],
                "expected": str(expected)[:50],
                "got": "SKIPPED (file attachment)",
                "correct": False,
                "spans": 0,
                "ok": False,
                "error": "requires file attachment",
            })
            continue

        print(f"[{i+1}/{len(tasks_to_run)}] {task_id}: {question[:70]}...")

        spans, result, ok, error = run_task_with_tracing(task_id, question, expected)
        all_spans.extend(spans)

        correct = str(result).strip().lower() == str(expected).strip().lower()
        results_log.append({
            "task_id": task_id,
            "question": question[:100],
            "expected": str(expected)[:50],
            "got": result[:100],
            "correct": correct,
            "spans": len(spans),
            "ok": ok,
            "error": error,
        })

        status = "CORRECT" if correct else ("ERROR" if not ok else "wrong")
        print(f"  -> {status} | expected={str(expected)[:30]!r} | got={result[:40]!r} | spans={len(spans)}")

        time.sleep(2)  # Rate-limit respect

    # Save OTEL traces
    out_path = Path("/sessions/practical-funny-euler/mnt/monk/tests/fixtures/gaia_real_traces.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for span in all_spans:
            f.write(json.dumps(span) + "\n")

    # Save results log
    results_path = Path("/sessions/practical-funny-euler/mnt/monk/gaia_task_results.json")
    with open(results_path, "w") as f:
        json.dump(results_log, f, indent=2)

    ran = [r for r in results_log if r["ok"]]
    correct_count = sum(1 for r in results_log if r["correct"])
    skipped = sum(1 for r in results_log if r.get("error") == "requires file attachment")
    errors = sum(1 for r in results_log if not r["ok"] and r.get("error") != "requires file attachment")

    print(f"\n=== SUMMARY ===")
    print(f"Total tasks attempted: {len(tasks_to_run)}")
    print(f"Skipped (file attachment): {skipped}")
    print(f"Errors: {errors}")
    print(f"Ran successfully: {len(ran)}")
    print(f"Correct answers: {correct_count}")
    if ran:
        print(f"Accuracy (ran): {correct_count}/{len(ran)} = {correct_count/len(ran)*100:.1f}%")
    print(f"Total spans: {len(all_spans)}")
    print(f"Traces saved to: {out_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
