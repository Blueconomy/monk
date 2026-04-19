# 🕵️ monk

**Find the money your AI agents are silently burning.**

monk analyzes trace logs from any AI agent — LangGraph, smolagents, MemGPT, custom — and surfaces the cost leaks and behavioral failures that dashboards miss.

```
$ monk run ./traces/

  🕵️  monk — Agentic Workflow Blind Spot Detector
  Source: ./traces/   |   Calls analysed: 4,610

  ┌────────────────────────────────────────────────────────────────────────────┐
  │  12 blind spots found  ·  ~$118.61/day estimated waste  ·  ~$3,558/month  │
  └────────────────────────────────────────────────────────────────────────────┘

  🔴 [1] Retry loop: 'calculator_tool' called 5x in a row across 38 sessions
  Fix: Add a result-cache keyed on (tool, args). Eliminate re-computation.

  🔴 [2] Error cascade: tool failure ignored → 8 downstream LLM calls wasted
  Fix: Guard every tool call — if status=error, short-circuit before next LLM call.

  🔴 [3] Token spike: single web_search injected 583K tokens (26x session median)
  Fix: Truncate tool outputs to 1,000 tokens before injecting into context.
```

---

## Benchmark results

We evaluated monk on **7 real-world agentic trace datasets** — including PatronusAI's TRAIL benchmark, which provides human-labeled error annotations across 20 error categories.

| Dataset | Traces | Findings | Est. waste/day |
|---|---|---|---|
| TRAIL (GAIA + SWE-bench agents) | 879 spans / 33 traces | 137 | $13.48 |
| Finance / 10-K ReAct (LangGraph) | 4,610 calls | 558 | **$118.61** |
| GAIA smolagents | 1,253 spans | 296 | $0.74 |
| MemGPT (multi-turn conversations) | 500 calls | 22 | $0.41 |
| Nvidia Nemotron (customer service) | 413 calls | 14 | $0.00 |
| WildClaw (Claude Opus 4.6) | 288 calls | 1 | $0.00 |
| **Total** | **7,972** | **1,041** | **~$133/day** |

**~$3,990/month in avoidable agent costs identified across 7 datasets.**

> WildClaw produced 1 finding — a well-tuned production agent. monk correctly fires rarely on clean traces. That's the signal working as intended.

### TRAIL precision / recall

TRAIL is the most rigorous public benchmark for agentic error detection, with human-labeled ground truth across 20 error categories.

| Version | Precision | Recall | F1 | Detectors |
|---|---|---|---|---|
| v0.1 | 84.85% | 84.85% | 84.85% | 5 |
| **v0.2 (current)** | **100%** | **100%** | **100%** | **13** |

**v0.2 catches all 33 error-containing TRAIL traces with zero false positives.**  
Full methodology and per-detector breakdown: [`BENCHMARK.md`](BENCHMARK.md)

---

## What monk detects

13 detectors across two levels — trace-level (any format) and span-level (OpenTelemetry):

**Trace detectors** — work on OpenAI, Anthropic, LangSmith, or raw JSONL:

| Detector | What it finds |
|---|---|
| `retry_loop` | Same tool called 3+ consecutive times — agent stuck |
| `empty_return` | Tool returns null, agent retries anyway |
| `model_overkill` | gpt-4o / claude-opus doing formatting, summarising, or classification |
| `context_bloat` | System prompt >55% of token budget, or unbounded history growth |
| `agent_loop` | Agent cycling A→B→A→B without making progress |

**Span detectors** — require OpenTelemetry traces:

| Detector | What it finds |
|---|---|
| `error_cascade` | Tool fails silently → agent continues making 6–8 more LLM calls on a poisoned context |
| `token_bloat` | Single-call token spikes (worst seen: 583K, 26× session median) |
| `latency_spike` | Outlier call latency vs. session median |
| `cross_turn_memory` | Same tool + args re-fetched across turns (pure cache waste) |
| `tool_dependency` | Cycles and deep chains in the tool call graph |
| `output_format` | Model violates explicit format rules in its own system prompt |
| `plan_execution` | Model writes a plan, then executes none of it |
| `span_consistency` | Model asserts facts with no supporting tool call (hallucinated evidence) |

All detectors are deterministic — no LLM-as-judge, no API calls, no surprises.

---

## Install

```bash
pip install monk-ai
```

---

## Usage

```bash
# Analyse a trace file or folder
monk run agent_traces.jsonl
monk run ./traces/

# Run specific detectors
monk run traces/ --detectors retry_loop,error_cascade,token_bloat

# Export findings as JSON for CI
monk run traces/ --json findings.json

# Only surface high-severity findings
monk run traces/ --min-severity high
```

**CI integration** — monk exits `1` if high-severity findings exist:

```yaml
- name: monk trace audit
  run: monk run ./traces/ --min-severity high
```

---

## Trace format

monk auto-detects OpenAI, Anthropic, LangSmith, and OpenTelemetry formats.

For custom logging, any JSONL with these fields works:

```jsonl
{"session_id": "abc123", "model": "gpt-4o", "input_tokens": 1200, "output_tokens": 80, "tool_name": "web_search", "tool_result": "..."}
```

For full span-level analysis (recommended), export OpenTelemetry traces — monk parses both OTLP proto-JSON and flat JSONL span formats.

---

## Why we built this

Most observability tools show you what happened. monk finds what's *costing* you.

The patterns here — retry loops, silent tool failures, token spikes, agents re-fetching the same data turn after turn — came from auditing real production agentic workflows. They don't show up as errors in your logs. They don't trigger alerts. They just quietly multiply your inference bill.

87% of the GAIA and SWE-bench agent runs we analyzed had at least one unhandled tool error that caused downstream LLM calls to be wasted. The worst token spike we saw was 583,787 tokens — 26× the session median — from a single unfiltered web page injected into context.

These are solvable problems. monk finds them.

---

## Benchmark datasets

All evaluation datasets are publicly available:

- **PatronusAI/TRAIL** — [github.com/patronus-ai/trail-benchmark](https://github.com/patronus-ai/trail-benchmark)
- **monk benchmark fixtures** (TRAIL, MemGPT, Nemotron, Finance, WildClaw, GAIA) — [huggingface.co/datasets/BenTu22/monk-benchmarks](https://huggingface.co/datasets/BenTu22/monk-benchmarks)

---

## Roadmap

- [ ] Real-time mode via OpenTelemetry SDK (auto-instrument your agent in 2 lines)
- [ ] Prompt compression suggestions
- [ ] Slack / PagerDuty alerts on finding threshold breaches
- [ ] Web dashboard

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

To add a detector:
1. Create `monk/detectors/your_detector.py` extending `BaseDetector`
2. Register in `monk/detectors/__init__.py`
3. Add tests in `tests/test_detectors.py`

Detectors must be deterministic — same traces → same findings.

---

## License

MIT

---

*Built by [Blueconomy AI](https://theblueconomy.ai) — Techstars '25*  
*If monk saves you money, a ⭐ helps others find it.*
