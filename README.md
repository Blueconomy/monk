# 🕵️ monk

**Find hidden cost leaks and blind spots in your agentic AI workflows.**

Drop `monk` on any trace file. Get a plain-English report of what's wasting tokens — and exactly how to fix it.

```
$ monk run ./traces/

  🕵️  monk — Agentic Workflow Blind Spot Detector
  Source: ./traces/   |   Calls analysed: 2,847

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  3 blind spots found  ·  ~$62.40/day estimated waste  ·  ~$1,872/month  │
  └─────────────────────────────────────────────────────────────────────────┘

  🔴 [1] Retry loop: 'web_search' called 4x in a row  ·  ~$38.20/day
  Fix: Add a max-retries guard before calling 'web_search'.

  🔴 [2] 'get_user_profile' returns empty 80% of the time  ·  ~$19.10/day
  Fix: Guard against empty returns — don't pass null context back to the LLM.

  🟡 [3] Model overkill: gpt-4o used for simple tasks (62% of calls)  ·  ~$5.10/day
  Fix: Route classify/format calls to gpt-4o-mini — identical quality, 16x cheaper.
```

---

## What monk detects

| Detector | What it finds |
|---|---|
| **retry_loop** | Same tool called 3+ times in a row — agent stuck, burning tokens |
| **empty_return** | Tool returns null/empty and agent retries anyway |
| **model_overkill** | Expensive model used for short, simple tasks |
| **context_bloat** | System prompt consuming >55% of token budget, or unbounded history growth |
| **agent_loop** | Agent cycling through the same step sequence without progress |

---

## Install

```bash
pip install monk-ai
```

Or run from source:

```bash
git clone https://github.com/Blueconomy/monk
cd monk
pip install -e .
```

---

## Usage

**Analyse a single file:**
```bash
monk run agent_traces.jsonl
```

**Analyse a folder of traces:**
```bash
monk run ./traces/
```

**Run specific detectors only:**
```bash
monk run traces/ --detectors retry_loop,model_overkill
```

**Export findings as JSON (great for CI):**
```bash
monk run traces/ --json findings.json
```

**Only show high-severity findings:**
```bash
monk run traces/ --min-severity high
```

**Use in CI — monk exits with code 1 if high-severity findings exist:**
```yaml
- name: Run monk
  run: monk run ./traces/ --min-severity high
```

---

## Trace format

monk auto-detects OpenAI, Anthropic, and LangSmith trace formats.

For custom logging, any JSONL file with these fields works:

```jsonl
{"session_id": "abc123", "model": "gpt-4o", "input_tokens": 1200, "output_tokens": 80, "tool_name": "web_search", "tool_result": "some result"}
```

**Supported fields:**

| Field | Required | Notes |
|---|---|---|
| `model` | ✅ | e.g. `gpt-4o`, `claude-sonnet-4-6` |
| `input_tokens` | ✅ | Prompt token count |
| `output_tokens` | ✅ | Completion token count |
| `session_id` | Recommended | Groups calls into sessions |
| `tool_name` | Optional | Name of tool called |
| `tool_result` | Optional | Result returned by tool |
| `system_prompt_tokens` | Optional | Enables context_bloat detection |

monk also natively parses:
- OpenAI Chat Completions API response format
- Anthropic Messages API response format
- LangSmith run export format

---

## Why monk?

Most observability tools show you dashboards and traces. monk finds the things you don't know to look for.

The patterns monk detects are the ones we found repeatedly when auditing real agentic workflows — retry loops that nobody noticed, tools returning empty results silently, expensive models doing simple jobs. These don't show up as errors. They just quietly burn money.

---

## Roadmap

- [ ] Live mode: instrument running agents via OpenTelemetry
- [ ] Prompt compression suggestions
- [ ] Cross-workflow benchmarking
- [ ] Slack / PagerDuty alerts
- [ ] Web dashboard

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

To add a new detector:
1. Create `monk/detectors/your_detector.py` extending `BaseDetector`
2. Add it to `monk/detectors/__init__.py`
3. Add tests in `tests/test_detectors.py`

---

## License

MIT — free to use, modify, and distribute.

---

*Built by [Blueconomy AI](https://theblueconomy.ai) — Techstars '25.*  
*If monk finds something useful, give us a ⭐ — it helps more people find it.*
