"""
Convert HuggingFace agent trace datasets to monk JSONL format.

Dataset 1: sammshen/wildclaw-opus-traces
  - HTTP proxy log format: paired request/response records
  - Each response is one LLM call with OpenAI-compatible body
  - 686 records total (344 requests, 342 responses), 290 matched (200 OK) pairs
  - Task name used as session_id grouping

Dataset 2: snorkelai/agent-finance-reasoning
  - LangGraph/ReAct traces with messages array (human/ai/tool)
  - 357 traces, each with 10-40 turns
  - No per-call token counts — estimated from system prompt + turn length heuristic
  - Model: claude-opus-4-20250514
"""

import json
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────
# Dataset 1: wildclaw-opus-traces
# ─────────────────────────────────────────────────────────────

def convert_wildclaw(output_path: str) -> int:
    """Convert wildclaw traces to monk JSONL. Returns record count."""
    ds = load_dataset("sammshen/wildclaw-opus-traces", trust_remote_code=True)
    train = ds["train"]

    # Collect requests and responses indexed by request_id
    requests = {}
    responses = {}
    for item in train:
        rid = item["request_id"]
        if item["type"] == "request":
            requests[rid] = item
        elif item["type"] == "response" and item["status_code"] == 200:
            responses[rid] = item

    records = []
    for rid in responses:
        if rid not in requests:
            continue

        req = requests[rid]
        resp = responses[rid]

        req_body = req["body"]
        resp_body = resp["body"]

        if not isinstance(req_body, dict) or not isinstance(resp_body, dict):
            continue

        usage = resp_body.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Model normalisation: strip router prefix
        model_raw = resp_body.get("model", req_body.get("model", "claude-opus-4-6"))
        model = model_raw.split("/")[-1]  # e.g. anthropic/claude-opus-4-6 -> claude-opus-4-6

        # session_id = task_name (groups all HTTP calls in one task together)
        session_id = req.get("task_name", rid)

        # Extract tool calls from response choices
        choices = resp_body.get("choices", [])
        tool_calls = []
        system_prompt_tokens = 0

        if choices:
            msg = choices[0].get("message", {})
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                tool_calls.append({"name": fn.get("name", ""), "result": None})

        # Extract tool results from request messages (role=tool)
        req_msgs = req_body.get("messages", [])

        # Count system prompt tokens (first system message)
        system_content = ""
        for m in req_msgs:
            if m.get("role") == "system":
                system_content = m.get("content", "")
                break
        if system_content:
            # Rough estimate: 1 token ≈ 4 chars
            system_prompt_tokens = len(system_content) // 4

        # Pair tool results with calls by position
        tool_result_msgs = [m for m in req_msgs if m.get("role") == "tool"]
        for i, tr in enumerate(tool_result_msgs):
            result_content = tr.get("content", "")
            if i < len(tool_calls):
                tool_calls[i]["result"] = result_content
            else:
                # Extra tool result not paired with a call in this turn — skip
                pass

        record = {
            "session_id": session_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tool_calls": tool_calls if tool_calls else [],
            "system_prompt_tokens": system_prompt_tokens,
            "cost": usage.get("cost"),
            "task_name": req.get("task_name"),
            "category": req.get("category"),
        }
        records.append(record)

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[wildclaw] Written {len(records)} records to {output_path}")
    return len(records)


# ─────────────────────────────────────────────────────────────
# Dataset 2: agent-finance-reasoning
# ─────────────────────────────────────────────────────────────

def convert_finance(output_path: str) -> int:
    """Convert finance reasoning traces to monk JSONL. Returns record count."""
    ds2 = load_dataset("snorkelai/agent-finance-reasoning", trust_remote_code=True)
    train2 = ds2["train"]

    records = []
    for row in train2:
        session_id = f"finance_{row['id']}"
        model = row["model"]
        trace = json.loads(row["trace"])

        # Gather system prompt token estimate from row-level system_prompt field
        sys_prompt = row.get("system_prompt", "")
        system_prompt_tokens = len(sys_prompt) // 4 if sys_prompt else 0

        # Walk through trace messages; emit one record per AI turn
        # Pair each AI turn with subsequent tool responses
        i = 0
        turn_index = 0
        while i < len(trace):
            msg = trace[i]
            if msg["type"] != "ai":
                i += 1
                continue

            content = msg.get("content", [])

            # Extract tool calls from this AI turn
            tool_calls = []
            text_len = 0
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "tool_use":
                            tool_calls.append({
                                "name": c.get("name", ""),
                                "input": c.get("input", {}),
                                "result": None,
                            })
                        elif c.get("type") == "text":
                            text_len += len(c.get("text", ""))
            elif isinstance(content, str):
                text_len = len(content)

            # Collect tool responses that follow this AI turn
            j = i + 1
            tool_results = []
            while j < len(trace) and trace[j]["type"] == "tool":
                tool_results.append(trace[j])
                j += 1

            # Pair results with calls
            for k, tr in enumerate(tool_results):
                result_content = tr.get("content", "")
                if k < len(tool_calls):
                    tool_calls[k]["result"] = result_content

            # Estimate tokens: no usage metadata available in this dataset
            # Use character-count heuristics (rough but consistent)
            # For input: accumulate all preceding messages' content length
            preceding_chars = sum(
                len(str(trace[x].get("content", "")))
                for x in range(i)
            )
            estimated_input = max(100, preceding_chars // 4)
            estimated_output = max(10, (text_len + sum(
                len(str(tc.get("input", ""))) for tc in tool_calls
            )) // 4)

            # tool_name / tool_result for single-tool records (monk convenience)
            single_tool_name = tool_calls[0]["name"] if len(tool_calls) == 1 else None
            single_tool_result = tool_calls[0]["result"] if len(tool_calls) == 1 else None

            record = {
                "session_id": session_id,
                "model": model,
                "input_tokens": estimated_input,
                "output_tokens": estimated_output,
                "tool_calls": tool_calls,
                "system_prompt_tokens": system_prompt_tokens if turn_index == 0 else 0,
                "company": row.get("company"),
                "correctness": row.get("correctness"),
                "turn_index": turn_index,
            }

            if single_tool_name:
                record["tool_name"] = single_tool_name
            if single_tool_result is not None:
                record["tool_result"] = single_tool_result

            records.append(record)
            turn_index += 1
            i = j  # skip past tool responses to next AI turn

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[finance] Written {len(records)} records to {output_path}")
    return len(records)


if __name__ == "__main__":
    wildclaw_out = "/sessions/practical-funny-euler/mnt/monk/tests/fixtures/wildclaw_traces.jsonl"
    finance_out = "/sessions/practical-funny-euler/mnt/monk/tests/fixtures/finance_traces.jsonl"

    n1 = convert_wildclaw(wildclaw_out)
    n2 = convert_finance(finance_out)

    print(f"\nSummary: wildclaw={n1} records, finance={n2} records")
