"""
monk simulate — generate synthetic agent traces with configurable failure patterns.

Useful for:
  - Testing your detectors before you have real traces
  - Reproducing specific failure modes in isolation
  - Demoing monk to stakeholders with realistic data
  - Benchmarking detector sensitivity

Usage:
  monk simulate --pattern all -o traces/sim.jsonl
  monk simulate --pattern retry_loop,agent_loop --sessions 20
  monk simulate --healthy                         # clean traces, no findings expected
"""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Literal

# ── Models available for simulation ───────────────────────────────────────────
CHEAP_MODELS  = ["gpt-4o-mini", "claude-haiku-4-5-20251001"]
MID_MODELS    = ["gpt-4o", "claude-sonnet-4-6"]
EXPENSIVE     = ["o3", "claude-opus-4-6"]

ALL_PATTERNS = [
    "retry_loop",
    "empty_return",
    "agent_loop",
    "context_bloat",
    "model_overkill",
    "healthy",
]


def _rnd_session() -> str:
    return "sim-" + uuid.uuid4().hex[:8]


def _call(session_id: str, model: str, input_tok: int, output_tok: int,
          tool_name: str | None = None, tool_result=None,
          system_prompt_tokens: int | None = None) -> dict:
    rec: dict = {
        "session_id": session_id,
        "model": model,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
    }
    if tool_name:
        rec["tool_name"] = tool_name
        rec["tool_result"] = tool_result
    if system_prompt_tokens is not None:
        rec["system_prompt_tokens"] = system_prompt_tokens
    return rec


# ── Pattern generators ────────────────────────────────────────────────────────

def _gen_retry_loop(n_sessions: int = 3) -> list[dict]:
    """Same tool called 4–6× consecutively (connection error / 503 pattern)."""
    records = []
    tools = ["database_query", "call_payment_api", "fetch_user_profile", "search_index"]
    errors = ["Connection timeout", "503 Service Unavailable", "Rate limit exceeded"]
    for _ in range(n_sessions):
        sid  = _rnd_session()
        tool = random.choice(tools)
        err  = random.choice(errors)
        reps = random.randint(4, 6)
        base = random.randint(1800, 3200)
        model = random.choice(MID_MODELS)
        for i in range(reps):
            records.append(_call(sid, model, base + i * 50, random.randint(60, 130),
                                 tool, err))
    return records


def _gen_empty_return(n_sessions: int = 3) -> list[dict]:
    """Tool returns empty/null, agent keeps retrying."""
    records = []
    for _ in range(n_sessions):
        sid  = _rnd_session()
        tool = random.choice(["web_search", "document_lookup", "get_context"])
        base = random.randint(1400, 2400)
        model = random.choice(MID_MODELS)
        # 3–4 empty results, then one real one
        for i in range(random.randint(3, 4)):
            records.append(_call(sid, model, base + i * 60, 110,
                                 tool, None if i % 2 == 0 else ""))
        records.append(_call(sid, model, base + 300, 180,
                             tool, "Here are the results: " + "x" * 200))
    return records


def _gen_agent_loop(n_sessions: int = 3) -> list[dict]:
    """A→B→A→B cycle repeated 3+ times — no progress."""
    records = []
    tool_pairs = [
        ("search_documents", "rank_results"),
        ("retrieve_context", "summarize_chunk"),
        ("query_db", "validate_row"),
    ]
    for _ in range(n_sessions):
        sid = _rnd_session()
        a, b = random.choice(tool_pairs)
        model = random.choice(MID_MODELS)
        base = random.randint(1800, 2800)
        reps = random.randint(3, 5)
        for i in range(reps):
            records.append(_call(sid, model, base + i * 80, 140,
                                 a, f"result_{i * 2}"))
            records.append(_call(sid, model, base + i * 80 + 40, 140,
                                 b, f"result_{i * 2 + 1}"))
    return records


def _gen_context_bloat(n_sessions: int = 3) -> list[dict]:
    """System prompt consuming 60–75% of token budget."""
    records = []
    for _ in range(n_sessions):
        sid   = _rnd_session()
        model = random.choice(MID_MODELS)
        for j in range(random.randint(3, 6)):
            total  = random.randint(12000, 20000)
            syspct = random.uniform(0.60, 0.75)
            sys_tok = int(total * syspct)
            records.append(_call(sid, model, total, random.randint(150, 250),
                                 "lookup", f"data_{j}",
                                 system_prompt_tokens=sys_tok))
    return records


def _gen_model_overkill(n_sessions: int = 3) -> list[dict]:
    """Flagship model doing trivial formatting / classification tasks."""
    records = []
    trivial_tools = ["format_json", "classify_intent", "extract_date", "format_currency"]
    for _ in range(n_sessions):
        sid   = _rnd_session()
        model = random.choice(EXPENSIVE)   # expensive model for cheap work
        for _ in range(random.randint(4, 8)):
            records.append(_call(sid, model,
                                 random.randint(600, 1000),
                                 random.randint(20, 45),
                                 random.choice(trivial_tools),
                                 '{"ok":true}'))
    return records


def _gen_healthy(n_sessions: int = 5) -> list[dict]:
    """Clean sessions with no failure patterns — should produce zero findings."""
    records = []
    tools = ["tool_a", "tool_b", "tool_c", "tool_d"]
    for _ in range(n_sessions):
        sid   = _rnd_session()
        model = random.choice(CHEAP_MODELS)
        n     = random.randint(3, 8)
        base  = random.randint(800, 1600)
        for i in range(n):
            records.append(_call(sid, model,
                                 base + i * random.randint(50, 150),
                                 random.randint(150, 350),
                                 random.choice(tools),
                                 "success"))
    return records


# ── Public API ────────────────────────────────────────────────────────────────

GENERATORS = {
    "retry_loop":    _gen_retry_loop,
    "empty_return":  _gen_empty_return,
    "agent_loop":    _gen_agent_loop,
    "context_bloat": _gen_context_bloat,
    "model_overkill": _gen_model_overkill,
    "healthy":       _gen_healthy,
}


def generate(
    patterns: list[str] | Literal["all"] = "all",
    sessions_per_pattern: int = 3,
    seed: int | None = 42,
) -> list[dict]:
    """
    Generate synthetic trace records for the specified patterns.

    Args:
        patterns:  list of pattern names, or "all"
        sessions_per_pattern:  number of sessions per pattern type
        seed:  random seed for reproducibility (None = random)

    Returns:
        list of trace record dicts (each is one LLM call)
    """
    if seed is not None:
        random.seed(seed)

    selected = list(GENERATORS.keys()) if patterns == "all" else patterns
    unknown  = [p for p in selected if p not in GENERATORS]
    if unknown:
        raise ValueError(f"Unknown patterns: {unknown}. Available: {list(GENERATORS)}")

    records: list[dict] = []
    for pattern in selected:
        records.extend(GENERATORS[pattern](sessions_per_pattern))

    # Shuffle so patterns aren't grouped by type (more realistic)
    random.shuffle(records)
    return records


def write_jsonl(records: list[dict], path: str | Path) -> Path:
    """Write trace records to a JSONL file. Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out
