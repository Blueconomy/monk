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


# ── OTEL workflow simulation ───────────────────────────────────────────────────
import time as _time_module
from collections import defaultdict as _defaultdict


def _hex(n: int) -> str:
    return "".join(random.choices("0123456789abcdef", k=n))


# Preset workflow graph definitions ─────────────────────────────────────────
WORKFLOW_PRESETS: dict[str, dict] = {
    "retry_loop": {
        "name": "Retry Loop",
        "desc": "Same tool called 4× in a row — 503/timeout pattern",
        "nodes": [
            {"id": "n1", "type": "llm", "label": "LLM Call", "x": 80, "y": 130,
             "config": {"model": "gpt-4o", "tokens_in": 2100, "tokens_out": 95, "latency_ms": 800}},
            {"id": "n2", "type": "tool", "label": "database_query", "x": 310, "y": 130,
             "config": {"tool_name": "database_query", "failure": "timeout",
                        "failure_rate": 0.9, "failure_result": "Connection timeout",
                        "success_result": "[{id:1,data:'ok'}]", "latency_ms": 1200, "max_retries": 4}},
        ],
        "edges": [{"src": "n1", "dst": "n2"}, {"src": "n2", "dst": "n2"}],
    },
    "agent_loop": {
        "name": "Agent Loop",
        "desc": "search→rank→search→rank cycling with no progress",
        "nodes": [
            {"id": "n1", "type": "llm", "label": "LLM Plan", "x": 60, "y": 150,
             "config": {"model": "gpt-4o", "tokens_in": 1900, "tokens_out": 140, "latency_ms": 700}},
            {"id": "n2", "type": "tool", "label": "search_docs", "x": 270, "y": 60,
             "config": {"tool_name": "search_documents", "failure": "none", "failure_rate": 0,
                        "success_result": "result_set", "latency_ms": 400}},
            {"id": "n3", "type": "llm", "label": "LLM Rank", "x": 480, "y": 60,
             "config": {"model": "gpt-4o", "tokens_in": 2100, "tokens_out": 120, "latency_ms": 700}},
            {"id": "n4", "type": "tool", "label": "rank_results", "x": 480, "y": 230,
             "config": {"tool_name": "rank_results", "failure": "none", "failure_rate": 0,
                        "success_result": "ranked_items", "latency_ms": 300}},
        ],
        "edges": [
            {"src": "n1", "dst": "n2"}, {"src": "n2", "dst": "n3"},
            {"src": "n3", "dst": "n4"}, {"src": "n4", "dst": "n1"},
        ],
    },
    "empty_return": {
        "name": "Empty Return",
        "desc": "Tool returns empty 80% of the time — agent retries wastefully",
        "nodes": [
            {"id": "n1", "type": "llm", "label": "LLM Call", "x": 80, "y": 130,
             "config": {"model": "gpt-4o", "tokens_in": 1800, "tokens_out": 110, "latency_ms": 600}},
            {"id": "n2", "type": "tool", "label": "web_search", "x": 310, "y": 130,
             "config": {"tool_name": "web_search", "failure": "empty", "failure_rate": 0.8,
                        "failure_result": "", "success_result": "Search results found...",
                        "latency_ms": 600, "max_retries": 5}},
        ],
        "edges": [{"src": "n1", "dst": "n2"}, {"src": "n2", "dst": "n2"}],
    },
    "context_bloat": {
        "name": "Context Bloat",
        "desc": "System prompt eating 65%+ of token budget every call",
        "nodes": [
            {"id": "n1", "type": "llm", "label": "LLM (bloated)", "x": 80, "y": 130,
             "config": {"model": "gpt-4o", "tokens_in": 14000, "tokens_out": 200,
                        "system_tokens": 9100, "latency_ms": 1800}},
            {"id": "n2", "type": "tool", "label": "lookup", "x": 310, "y": 130,
             "config": {"tool_name": "lookup", "failure": "none", "failure_rate": 0,
                        "success_result": "data", "latency_ms": 300}},
        ],
        "edges": [{"src": "n1", "dst": "n2"}, {"src": "n2", "dst": "n1"}],
    },
    "healthy": {
        "name": "Healthy Agent",
        "desc": "Clean, well-behaved agent — expect zero findings",
        "nodes": [
            {"id": "n1", "type": "llm", "label": "LLM Call", "x": 80, "y": 130,
             "config": {"model": "gpt-4o-mini", "tokens_in": 900, "tokens_out": 250, "latency_ms": 400}},
            {"id": "n2", "type": "tool", "label": "fetch_data", "x": 310, "y": 70,
             "config": {"tool_name": "fetch_data", "failure": "none", "failure_rate": 0,
                        "success_result": "data payload", "latency_ms": 200}},
            {"id": "n3", "type": "tool", "label": "write_output", "x": 310, "y": 200,
             "config": {"tool_name": "write_output", "failure": "none", "failure_rate": 0,
                        "success_result": "written", "latency_ms": 150}},
        ],
        "edges": [{"src": "n1", "dst": "n2"}, {"src": "n1", "dst": "n3"}],
    },
    "supervisor": {
        "name": "Supervisor (Model Overkill)",
        "desc": "gpt-4o supervisor routes to gpt-4o-mini specialist — expensive routing",
        "nodes": [
            {"id": "n1", "type": "llm", "label": "Supervisor LLM", "x": 60, "y": 140,
             "config": {"model": "gpt-4o", "tokens_in": 1200, "tokens_out": 60,
                        "latency_ms": 700}},
            {"id": "n2", "type": "tool", "label": "transfer_to_specialist", "x": 260, "y": 140,
             "config": {"tool_name": "transfer_to_specialist", "failure": "none",
                        "failure_rate": 0, "success_result": "Transferred to specialist",
                        "latency_ms": 50}},
            {"id": "n3", "type": "llm", "label": "Specialist LLM", "x": 440, "y": 140,
             "config": {"model": "gpt-4o-mini", "tokens_in": 800, "tokens_out": 200,
                        "latency_ms": 400}},
            {"id": "n4", "type": "tool", "label": "compute", "x": 600, "y": 70,
             "config": {"tool_name": "compute", "failure": "none", "failure_rate": 0,
                        "success_result": "result", "latency_ms": 180}},
            {"id": "n5", "type": "tool", "label": "transfer_back_to_supervisor", "x": 600, "y": 210,
             "config": {"tool_name": "transfer_back_to_supervisor", "failure": "none",
                        "failure_rate": 0, "success_result": "Transferred back",
                        "latency_ms": 50}},
        ],
        "edges": [
            {"src": "n1", "dst": "n2"}, {"src": "n2", "dst": "n3"},
            {"src": "n3", "dst": "n4"}, {"src": "n3", "dst": "n5"},
        ],
    },
    "swarm": {
        "name": "Swarm (Handoff Loop)",
        "desc": "Peer agents bouncing back and forth without resolving",
        "nodes": [
            {"id": "n1", "type": "llm", "label": "agent_A", "x": 60, "y": 140,
             "config": {"model": "gpt-4o", "tokens_in": 1400, "tokens_out": 80,
                        "latency_ms": 600}},
            {"id": "n2", "type": "tool", "label": "transfer_to_agent_B", "x": 250, "y": 140,
             "config": {"tool_name": "transfer_to_agent_B", "failure": "none",
                        "failure_rate": 0, "success_result": "Transferred to agent_B",
                        "latency_ms": 40}},
            {"id": "n3", "type": "llm", "label": "agent_B", "x": 430, "y": 140,
             "config": {"model": "gpt-4o", "tokens_in": 1600, "tokens_out": 90,
                        "latency_ms": 650}},
            {"id": "n4", "type": "tool", "label": "partial_work", "x": 580, "y": 60,
             "config": {"tool_name": "partial_work", "failure": "empty",
                        "failure_rate": 0.7, "failure_result": "",
                        "success_result": "partial result", "latency_ms": 300}},
            {"id": "n5", "type": "tool", "label": "transfer_back_to_agent_A", "x": 430, "y": 240,
             "config": {"tool_name": "transfer_back_to_agent_A", "failure": "none",
                        "failure_rate": 0, "success_result": "Transferred back to agent_A",
                        "latency_ms": 40}},
        ],
        "edges": [
            {"src": "n1", "dst": "n2"}, {"src": "n2", "dst": "n3"},
            {"src": "n3", "dst": "n4"}, {"src": "n3", "dst": "n5"},
            {"src": "n5", "dst": "n1"},   # back-edge: creates the handoff cycle
        ],
    },
}


def simulate_workflow_otel(
    nodes: list[dict],
    edges: list[dict],
    sessions: int = 5,
    model: str = "gpt-4o",
    seed: int | None = 42,
) -> list[dict]:
    """
    Execute a workflow graph definition and generate OTEL-format spans.

    nodes: list of {id, type ('llm'|'tool'), label, config}
    edges: list of {src, dst}  — self-loop edges mean "retry"
    Returns flat list of OTEL span dicts (JSONL-compatible).
    """
    if seed is not None:
        random.seed(seed)
    if not nodes:
        return []

    node_map = {n["id"]: n for n in nodes}

    # Build adjacency
    adj: dict[str, list[str]] = _defaultdict(list)
    in_deg: dict[str, int] = _defaultdict(int)
    for e in edges:
        adj[e["src"]].append(e["dst"])
        if e["src"] != e["dst"]:
            in_deg[e["dst"]] += 1

    # Entry: first node with no incoming (non-self) edges
    entry = next((n["id"] for n in nodes if in_deg.get(n["id"], 0) == 0), nodes[0]["id"])

    # Detect back-edges: self-loops + edges to earlier nodes
    order = {n["id"]: i for i, n in enumerate(nodes)}
    back_edges: set[tuple[str, str]] = set()
    for e in edges:
        s, d = e["src"], e["dst"]
        if s == d or order.get(d, 999) <= order.get(s, 0):
            back_edges.add((s, d))

    all_spans: list[dict] = []
    base_ns = int(_time_module.time() * 1_000_000_000) - sessions * 60_000_000_000

    for idx in range(sessions):
        trace_id = _hex(32)
        session_id = f"sim-{trace_id[:8]}"
        t0 = base_ns + idx * 60_000_000_000
        agent_sid = _hex(16)
        sess_spans: list[dict] = []
        vc: dict[str, int] = _defaultdict(int)

        final_t, _ = _exec(
            entry, node_map, adj, back_edges, vc,
            trace_id, agent_sid, session_id, model, agent_sid, t0, sess_spans,
        )

        sess_spans.insert(0, {
            "traceId": trace_id, "spanId": agent_sid, "parentSpanId": None,
            "name": "agent.run",
            "startTimeUnixNano": t0, "endTimeUnixNano": final_t + 50_000_000,
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": {"session_id": session_id, "gen_ai.agent.name": "sim-agent"},
        })
        all_spans.extend(sess_spans)

    return all_spans


def _exec(nid, node_map, adj, back_edges, vc, trace_id, agent_sid, session_id,
          default_model, llm_ctx, t, spans):
    """Recursively execute a workflow node, return (final_t, llm_ctx)."""
    node = node_map.get(nid)
    if not node:
        return t, llm_ctx

    cfg = node.get("config", {})
    ntype = node.get("type", "tool")
    max_v = 5 if ntype == "tool" else 4
    vc[nid] += 1
    if vc[nid] > max_v:
        return t, llm_ctx

    lat = int((cfg.get("latency_ms", 500) + random.randint(-50, 250)) * 1_000_000)
    sid = _hex(16)

    if ntype == "llm":
        m = cfg.get("model", default_model)
        it = max(100, cfg.get("tokens_in", 1200) + random.randint(-200, 300))
        ot = max(10, cfg.get("tokens_out", 100) + random.randint(-20, 50))
        spt = cfg.get("system_tokens", 0)
        spans.append({
            "traceId": trace_id, "spanId": sid, "parentSpanId": agent_sid,
            "name": "LLMCall",
            "startTimeUnixNano": t, "endTimeUnixNano": t + lat,
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": {
                "gen_ai.request.model": m,
                "gen_ai.usage.prompt_tokens": it,
                "gen_ai.usage.completion_tokens": ot,
                "llm.system_prompt_tokens": spt,
                "session_id": session_id,
            },
        })
        t += lat
        cur_ctx = sid  # tools after this LLM become its children
        for nbr in adj.get(nid, []):
            if (nid, nbr) not in back_edges:
                t, cur_ctx = _exec(nbr, node_map, adj, back_edges, vc,
                                   trace_id, agent_sid, session_id, default_model, cur_ctx, t, spans)
        for nbr in adj.get(nid, []):
            if (nid, nbr) in back_edges and vc.get(nbr, 0) < 3:
                t, cur_ctx = _exec(nbr, node_map, adj, back_edges, vc,
                                   trace_id, agent_sid, session_id, default_model, cur_ctx, t, spans)
        return t, cur_ctx

    # tool node
    tool_name = cfg.get("tool_name", node.get("label", "tool"))
    fail_mode = cfg.get("failure", "none")
    fail_rate = float(cfg.get("failure_rate", 0))
    has_retry = (nid, nid) in back_edges
    iters = int(cfg.get("max_retries", 4)) if has_retry else 1

    for i in range(iters):
        is_last = i == iters - 1
        is_fail = not is_last and fail_mode != "none" and random.random() < fail_rate
        iter_end = t + lat + random.randint(0, 80_000_000)

        if fail_mode == "empty":
            result = "" if is_fail else cfg.get("success_result", "data")
            code = "STATUS_CODE_OK"
        else:
            result = cfg.get("failure_result", "Error") if is_fail else cfg.get("success_result", "data")
            code = "STATUS_CODE_ERROR" if is_fail else "STATUS_CODE_OK"

        spans.append({
            "traceId": trace_id, "spanId": _hex(16), "parentSpanId": llm_ctx,
            "name": tool_name,
            "startTimeUnixNano": t, "endTimeUnixNano": iter_end,
            "status": {"code": code},
            "attributes": {
                "tool.name": tool_name, "tool.result": result,
                "tool_result": result, "session_id": session_id,
            },
        })
        t = iter_end + 5_000_000

    for nbr in adj.get(nid, []):
        if nbr != nid and (nid, nbr) not in back_edges:
            t, llm_ctx = _exec(nbr, node_map, adj, back_edges, vc,
                               trace_id, agent_sid, session_id, default_model, llm_ctx, t, spans)
    for nbr in adj.get(nid, []):
        if nbr != nid and (nid, nbr) in back_edges and vc.get(nbr, 0) < 4:
            t, llm_ctx = _exec(nbr, node_map, adj, back_edges, vc,
                               trace_id, agent_sid, session_id, default_model, llm_ctx, t, spans)
    return t, llm_ctx
