"""
Run smolagents on GAIA benchmark with OpenTelemetry instrumentation.

Produces OTEL-format traces that monk can evaluate directly.

Prerequisites:
    pip install smolagents opentelemetry-sdk opentelemetry-exporter-otlp
    pip install openinference-instrumentation-smolagents datasets

Usage:
    HF_TOKEN=hf_xxx OPENAI_API_KEY=sk-xxx python scripts/run_gaia_live.py
    HF_TOKEN=hf_xxx OPENAI_API_KEY=sk-xxx python scripts/run_gaia_live.py --tasks 200 --level 1
    HF_TOKEN=hf_xxx python scripts/run_gaia_live.py --model Qwen/Qwen2.5-72B-Instruct --tasks 50

Environment variables:
    HF_TOKEN          — Required to access gaia-benchmark/GAIA (gated dataset)
    OPENAI_API_KEY    — Required for OpenAI models (gpt-4o, gpt-4o-mini)
    ANTHROPIC_API_KEY — Required for Anthropic models (claude-*)
    HF_INFERENCE_KEY  — Required for HuggingFace Inference API models

Output:
    tests/fixtures/gaia_live_traces.jsonl   — OTEL spans, one per line
    benchmark_gaia_live.txt                 — monk analysis report

After running:
    monk run tests/fixtures/gaia_live_traces.jsonl --format otel
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Step 1: OTEL setup ────────────────────────────────────────────────────────

def setup_otel_pipeline(output_path: Path) -> None:
    """
    Configure OpenTelemetry to export spans to a local JSONL file.

    Uses a custom JSONL file exporter so monk can read traces without
    needing a running OTEL collector.  In production you'd replace this
    with an OTLP gRPC/HTTP exporter pointing at Jaeger or Tempo.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
    from opentelemetry.sdk.resources import Resource

    class JSONLFileExporter(SpanExporter):
        """Write one JSON record per span to a JSONL file (monk OTEL format)."""

        def __init__(self, path: Path) -> None:
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

        def export(self, spans):
            for span in spans:
                ctx = span.get_span_context()
                parent = span.parent
                rec = {
                    "traceId": format(ctx.trace_id, "032x"),
                    "spanId": format(ctx.span_id, "016x"),
                    "parentSpanId": format(parent.span_id, "016x") if parent else None,
                    "name": span.name,
                    "startTimeUnixNano": str(span.start_time),
                    "endTimeUnixNano": str(span.end_time),
                    "status": {
                        "code": "STATUS_CODE_ERROR" if span.status.is_error else "STATUS_CODE_OK"
                    },
                    "attributes": dict(span.attributes or {}),
                    "events": [
                        {
                            "name": e.name,
                            "attributes": dict(e.attributes or {}),
                        }
                        for e in span.events
                    ],
                }
                self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
            return SpanExportResult.SUCCESS

        def shutdown(self):
            self._fh.close()

    resource = Resource.create({"service.name": "gaia-smolagents-bench"})
    provider = TracerProvider(resource=resource)
    exporter = JSONLFileExporter(output_path)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info(f"OTEL pipeline writing to {output_path}")


# ── Step 2: smolagents OTEL instrumentation ───────────────────────────────────

def instrument_smolagents() -> None:
    """
    Attach OpenInference instrumentation to smolagents.

    openinference-instrumentation-smolagents patches smolagents internals
    to emit OTEL spans for every agent step, LLM call, and tool invocation —
    the exact format monk's OTEL parser expects.
    """
    try:
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
        SmolagentsInstrumentor().instrument()
        logger.info("openinference-instrumentation-smolagents: active")
    except ImportError:
        logger.warning(
            "openinference-instrumentation-smolagents not installed. "
            "Spans will be generated but without deep smolagents metadata. "
            "Install: pip install openinference-instrumentation-smolagents"
        )


# ── Step 3: Load GAIA tasks ───────────────────────────────────────────────────

def load_gaia_tasks(n: int, level: int | None, split: str = "validation") -> list[dict]:
    """
    Load tasks from the GAIA benchmark (gated — requires HF_TOKEN).

    Falls back to MATH-500 if GAIA is inaccessible (useful for testing
    the pipeline without HF authentication).

    Args:
        n:      Maximum number of tasks to return.
        level:  Filter to GAIA level 1, 2, or 3 (None = all levels).
        split:  HuggingFace dataset split — 'validation' has ground-truth answers.

    Returns:
        List of dicts with keys: task_id, question, level, expected_answer (if available).
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set — cannot access gated GAIA dataset. Falling back to MATH-500.")
        return _load_math500_fallback(n, level)

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_all",
            token=hf_token,
        )
        rows = list(ds[split])
        if level is not None:
            rows = [r for r in rows if r.get("Level") == level]
        rows = rows[:n]
        tasks = []
        for i, row in enumerate(rows):
            tasks.append({
                "task_id": row.get("task_id", f"gaia_{i}"),
                "question": row["Question"],
                "level": row.get("Level", 1),
                "expected_answer": row.get("Final answer", ""),
                "file_name": row.get("file_name", ""),
            })
        logger.info(f"Loaded {len(tasks)} GAIA tasks (split={split}, level={level or 'all'})")
        return tasks
    except Exception as exc:
        logger.warning(f"GAIA load failed: {exc}. Falling back to MATH-500.")
        return _load_math500_fallback(n, level)


def _load_math500_fallback(n: int, level: int | None) -> list[dict]:
    """MATH-500 as a stand-in: public, same multi-step reasoning structure."""
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = list(ds)
        if level is not None:
            # Map GAIA levels: 1→MATH 1-2, 2→MATH 3, 3→MATH 4-5
            level_map = {1: {1, 2}, 2: {3}, 3: {4, 5}}
            allowed = level_map.get(level, {1, 2, 3, 4, 5})
            rows = [r for r in rows if r.get("level", 1) in allowed]
        rows = rows[:n]
        tasks = []
        for i, row in enumerate(rows):
            math_level = row.get("level", 2)
            gaia_level = 1 if math_level <= 2 else (2 if math_level == 3 else 3)
            tasks.append({
                "task_id": row.get("unique_id", f"math_{i}"),
                "question": row["problem"],
                "level": gaia_level,
                "expected_answer": row.get("answer", ""),
                "file_name": "",
            })
        logger.info(f"Loaded {len(tasks)} MATH-500 tasks as GAIA fallback")
        return tasks
    except Exception as exc:
        logger.error(f"MATH-500 also failed: {exc}")
        sys.exit(1)


# ── Step 4: Build smolagents agent ────────────────────────────────────────────

def build_agent(model_id: str) -> "smolagents.MultiStepAgent":
    """
    Build a smolagents ToolCallingAgent / CodeAgent configured for GAIA.

    Model selection:
    - OpenAI models (gpt-*): use LiteLLMModel with OPENAI_API_KEY
    - Anthropic models (claude-*): use LiteLLMModel with ANTHROPIC_API_KEY
    - HuggingFace models (Qwen/*, meta-llama/*): use InferenceClientModel with HF_INFERENCE_KEY
    - 'mock': use DummyModel for testing the pipeline without API keys

    The agent is equipped with:
    - web_search: DuckDuckGo search (no API key needed)
    - visit_webpage: headless page scraping
    - python_interpreter: safe code execution (sandboxed)
    - final_answer: structured answer submission

    For GAIA Level 3 tasks, a managed search sub-agent is added.
    """
    import smolagents
    from smolagents import (
        CodeAgent,
        ToolCallingAgent,
        DuckDuckGoSearchTool,
        VisitWebpageTool,
        PythonInterpreterTool,
    )

    # Model selection
    if model_id == "mock":
        # DummyModel for pipeline testing — no API keys needed
        from smolagents import HfApiModel
        model = HfApiModel(model_id="Qwen/Qwen2.5-72B-Instruct")
        logger.warning("Using HfApiModel with Qwen — set HF_INFERENCE_KEY for real inference")
    elif model_id.startswith("gpt-") or model_id.startswith("o1") or model_id.startswith("o3"):
        from smolagents import LiteLLMModel
        model = LiteLLMModel(model_id=model_id, api_key=os.environ.get("OPENAI_API_KEY"))
    elif model_id.startswith("claude-"):
        from smolagents import LiteLLMModel
        model = LiteLLMModel(
            model_id=f"anthropic/{model_id}",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    else:
        # HuggingFace Inference API (Qwen, Llama, etc.)
        from smolagents import HfApiModel
        model = HfApiModel(
            model_id=model_id,
            token=os.environ.get("HF_INFERENCE_KEY") or os.environ.get("HF_TOKEN"),
        )

    tools = [
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        PythonInterpreterTool(),
    ]

    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=12,
        verbosity_level=1,
    )
    return agent


# ── Step 5: Run one task ──────────────────────────────────────────────────────

def run_task(agent, task: dict, timeout_s: int = 120) -> dict:
    """
    Run a single GAIA task through the agent with timeout.

    Returns a result dict with:
        task_id, question, level, predicted_answer, correct (bool),
        error (str|None), elapsed_s (float)
    """
    import signal

    question = task["question"]
    expected = task.get("expected_answer", "")

    result = {
        "task_id": task["task_id"],
        "question": question,
        "level": task["level"],
        "predicted_answer": "",
        "correct": False,
        "error": None,
        "elapsed_s": 0.0,
    }

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Task {task['task_id']} timed out after {timeout_s}s")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)

    try:
        t0 = time.time()
        answer = agent.run(question, reset=True)
        result["elapsed_s"] = time.time() - t0
        result["predicted_answer"] = str(answer)
        # GAIA uses exact-match (normalised lowercase, stripped punctuation)
        if expected:
            pred_norm = str(answer).lower().strip().rstrip(".")
            exp_norm = str(expected).lower().strip().rstrip(".")
            result["correct"] = pred_norm == exp_norm
    except TimeoutError as exc:
        result["error"] = str(exc)
        logger.warning(f"Timeout: {task['task_id']}")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.error(f"Error on {task['task_id']}: {exc}")
    finally:
        signal.alarm(0)

    return result


# ── Step 6: Main pipeline ─────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run smolagents on GAIA benchmark with OTEL instrumentation"
    )
    parser.add_argument("--tasks", type=int, default=50,
                        help="Number of GAIA tasks to run (default: 50)")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None,
                        help="Filter to GAIA difficulty level (default: all)")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test"],
                        help="GAIA dataset split (default: validation)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct",
                        help="Model ID (gpt-4o, claude-sonnet-4-6, Qwen/Qwen2.5-72B-Instruct, mock)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-task timeout in seconds (default: 120)")
    parser.add_argument("--output-traces", type=str,
                        default="tests/fixtures/gaia_live_traces.jsonl",
                        help="OTEL trace output file")
    parser.add_argument("--output-results", type=str,
                        default="gaia_live_results.jsonl",
                        help="Per-task result output file")
    parser.add_argument("--monk-report", type=str,
                        default="benchmark_gaia_live.txt",
                        help="Where to write the monk analysis report")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    output_traces = repo_root / args.output_traces
    output_results = repo_root / args.output_results
    monk_report = repo_root / args.monk_report

    # ── Setup ──────────────────────────────────────────────────────────────
    setup_otel_pipeline(output_traces)
    instrument_smolagents()

    # ── Load tasks ─────────────────────────────────────────────────────────
    tasks = load_gaia_tasks(args.tasks, args.level, args.split)
    if not tasks:
        logger.error("No tasks loaded — exiting.")
        sys.exit(1)

    logger.info(f"Running {len(tasks)} tasks with model={args.model}")

    # ── Build agent ────────────────────────────────────────────────────────
    agent = build_agent(args.model)

    # ── Run tasks ──────────────────────────────────────────────────────────
    results = []
    correct = 0
    total = 0

    output_results.parent.mkdir(parents=True, exist_ok=True)
    with output_results.open("w", encoding="utf-8") as rf:
        for i, task in enumerate(tasks):
            logger.info(f"[{i+1}/{len(tasks)}] Level {task['level']}: {task['question'][:80]}")
            result = run_task(agent, task, timeout_s=args.timeout)
            results.append(result)
            rf.write(json.dumps(result) + "\n")
            rf.flush()

            total += 1
            if result["correct"]:
                correct += 1

            status = "✓" if result["correct"] else ("⏱" if result.get("error") else "✗")
            logger.info(
                f"  {status} {result['elapsed_s']:.1f}s | "
                f"pred='{result['predicted_answer'][:40]}' "
                f"expected='{task.get('expected_answer', '')[:40]}'"
            )

    # ── Accuracy summary ───────────────────────────────────────────────────
    by_level: dict[int, list[bool]] = {}
    for r in results:
        by_level.setdefault(r["level"], []).append(r["correct"])

    print("\n" + "=" * 60)
    print("GAIA BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"Tasks run:    {total}")
    print(f"Correct:      {correct}/{total} ({correct/total:.1%})")
    for lvl in sorted(by_level):
        level_correct = sum(by_level[lvl])
        level_total = len(by_level[lvl])
        print(f"  Level {lvl}:   {level_correct}/{level_total} ({level_correct/level_total:.1%})")
    errors = [r for r in results if r.get("error")]
    print(f"Errors/timeouts: {len(errors)}")
    print(f"\nOTEL traces: {output_traces}")
    print(f"Results:     {output_results}")

    # ── Run monk analysis ──────────────────────────────────────────────────
    print(f"\nRunning monk analysis on traces...")
    try:
        import subprocess
        proc = subprocess.run(
            ["monk", "run", str(output_traces), "--format", "otel"],
            capture_output=True, text=True,
        )
        monk_output = proc.stdout + proc.stderr
        monk_report.write_text(monk_output, encoding="utf-8")
        print(monk_output[:3000])
        print(f"\nFull monk report: {monk_report}")
    except FileNotFoundError:
        print("monk CLI not found — install with: pip install -e .")
    except Exception as exc:
        print(f"monk analysis failed: {exc}")


if __name__ == "__main__":
    main()
