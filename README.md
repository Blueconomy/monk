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

  🔴 [3] Token spike: single web_search injected 583K tokens (26× session median)
  Fix: Truncate tool outputs to 1,000 tokens before injecting into context.
```

---

## Install

```bash
pip install monk-ai
```

---

## Quickstart

```bash
monk quickstart
```

Writes 33 built-in demo traces, runs analysis, and opens the live dashboard at `http://localhost:9090` — all in one command.

---

## Benchmark results

Evaluated on **8 real-world agentic trace datasets** — including PatronusAI's TRAIL benchmark with human-labeled ground truth across 20 error categories.

| Dataset | Records | Findings | Est. waste/day |
|---|---|---|---|
| taubench (banking / e-commerce agents) | 17,932 calls | 7,864 | $68.49 |
| Finance / 10-K ReAct (LangGraph) | 4,610 calls | 558 | **$118.61** |
| GAIA smolagents | 1,253 spans | 296 | $0.74 |
| TRAIL — GAIA + SWE-bench (ground truth) | 879 spans | 137 | $13.48 |
| MemGPT (multi-turn) | 500 calls | 22 | $0.41 |
| Nvidia Nemotron (customer service) | 413 calls | 14 | — |
| WildClaw (Claude Opus 4.6) | 288 calls | 1 | — |
| **Total** | **25,875** | **8,892** | **~$201/day** |

**~$6,000/month in avoidable agent costs identified across 8 datasets.**

> WildClaw — a well-tuned production Claude agent — produced exactly 1 finding. monk correctly fires rarely on clean traces.

### TRAIL precision / recall (ground truth benchmark)

| Version | Precision | Recall | F1 | Detectors |
|---|---|---|---|---|
| v0.1 | 84.85% | 84.85% | 84.85% | 5 |
| **v0.4.6 (current)** | **100%** | **100%** | **100%** | **14** |

Zero false positives. All 33 error-containing TRAIL traces caught.  
Full methodology: [BENCHMARK.md](https://github.com/Blueconomy/monk/blob/main/BENCHMARK.md)

---

## What monk detects

14 detectors. All deterministic — no LLM-as-judge, no external API calls.

**Trace detectors** — work on OpenAI, Anthropic, LangSmith, or raw JSONL:

| Detector | What it finds |
|---|---|
| `retry_loop` | Same tool called 3+ consecutive times |
| `empty_return` | Tool returns null/empty, agent retries anyway |
| `model_overkill` | Expensive model doing formatting or classification |
| `context_bloat` | System prompt >55% of budget, or unbounded history growth |
| `agent_loop` | Agent cycling A→B→A→B without progress |
| `text_io` | Low output compression, unbounded input growth |

**Span detectors** — require OpenTelemetry traces:

| Detector | What it finds |
|---|---|
| `error_cascade` | Tool fails silently → downstream LLM calls wasted on poisoned context |
| `token_bloat` | Token spikes (worst seen: 583K — 26× the session median) |
| `latency_spike` | Single-call outlier latency vs. session median |
| `cross_turn_memory` | Same tool + args re-fetched across turns |
| `tool_dependency` | Cycles and deep chains in the tool call graph |
| `output_format` | Model violates its own system prompt's format rules |
| `plan_execution` | Model writes a plan, then never executes it |
| `span_consistency` | Model asserts facts with no supporting tool call |

---

## Usage

```bash
# Fastest path — built-in demo, analysis, live dashboard
monk quickstart

# Analyse a trace file or folder
monk run agent_traces.jsonl
monk run ./traces/

# Run specific detectors
monk run traces/ --detectors retry_loop,error_cascade,token_bloat

# Export findings as JSON for CI
monk run traces/ --json findings.json

# Only surface high-severity findings
monk run traces/ --min-severity high

# Download real benchmark datasets and analyze them
monk demo

# Generate synthetic traces with configurable failure patterns
monk simulate                                  # all patterns
monk simulate --pattern retry_loop,agent_loop  # specific patterns
monk simulate --sessions 10 --run              # generate + analyze immediately

# Start the live dashboard
monk serve ./traces/ --port 9090
```

**CI integration** — monk exits `1` if high-severity findings exist:

```yaml
- name: monk trace audit
  run: monk run ./traces/ --min-severity high
```

**Real-time instrumentation** — catch issues as they happen, not after:

```python
import monk
monk.instrument()  # patches openai + anthropic automatically

# monk prints findings live as your agent runs
```

---

## Live dashboard

```bash
monk serve ./traces/ --port 9090
```

Opens a web dashboard at `http://localhost:9090` with:

- KPI cards: waste/day, projected/month, total findings, calls analyzed
- Severity breakdown with color-coded cards (high / medium / low)
- Waste ranked by detector with gradient bars
- Recent findings feed with fix suggestions
- Dataset downloader (tau-bench, Finance, TRAIL, GAIA, MemGPT, Nemotron)
- Prometheus metrics at `/metrics` for Grafana integration
- Auto-refreshes every 15 seconds
- "⚡ Load sample" button — populates demo data in one click

---

## Simulate workflows

```bash
monk simulate --pattern retry_loop,empty_return --sessions 5 --run
```

Generate synthetic trace data with specific failure patterns. Useful for:

- Testing your detectors before you have real production traces
- Reproducing a specific failure mode in isolation to verify a fix
- Demoing cost leaks to stakeholders with realistic numbers
- Validating that a code change actually eliminated a pattern

Available patterns: `retry_loop`, `empty_return`, `agent_loop`, `context_bloat`, `model_overkill`, `healthy`

The `healthy` pattern generates clean sessions with no failures — verifying monk produces zero findings on well-behaved agents.

---

## Trace format

monk auto-detects OpenAI, Anthropic, LangSmith, and OpenTelemetry formats. For custom logs, any JSONL with these fields works:

```jsonl
{"session_id": "abc123", "model": "gpt-4o", "input_tokens": 1200, "output_tokens": 80, "tool_name": "web_search", "tool_result": "..."}
```

For full span-level analysis, export OpenTelemetry traces — monk parses both OTLP proto-JSON and flat JSONL span formats.

---

## Why we built this

Most observability tools show you what happened. monk finds what's *costing* you.

The patterns here — retry loops, silent tool failures, token spikes, agents re-fetching the same data — don't show up as errors. They don't trigger alerts. They just quietly multiply your inference bill.

87% of the GAIA and SWE-bench agent runs we analyzed had at least one unhandled tool error that caused downstream LLM calls to be wasted. The worst token spike: 583,787 tokens from a single unfiltered web page, 26× the session median. These are solvable problems. monk finds them.

---

## Datasets

All benchmark fixtures are public:

- **PatronusAI/TRAIL** — [github.com/patronus-ai/trail-benchmark](https://github.com/patronus-ai/trail-benchmark)
- **monk benchmark fixtures** (TRAIL, MemGPT, Nemotron, Finance, WildClaw, GAIA, taubench) — [huggingface.co/datasets/Blueconomy/monk-benchmarks](https://huggingface.co/datasets/Blueconomy/monk-benchmarks)

---

## Roadmap

- [x] 14 deterministic detectors (trace + span level)
- [x] Live dashboard with dataset downloader
- [x] Real-time instrumentation (`monk.instrument()`)
- [x] `monk simulate` — synthetic workflow sandbox
- [x] Prometheus metrics + Grafana-ready `/metrics`
- [ ] Prompt compression suggestions
- [ ] Slack / PagerDuty alerts
- [ ] Confidence scores per finding

---

## Contributing

To add a detector: create `monk/detectors/your_detector.py` extending `BaseDetector`, register it in `monk/detectors/__init__.py`, add tests. Detectors must be deterministic — same traces → same findings.

See the full guide: [CONTRIBUTING.md](https://github.com/Blueconomy/monk/blob/main/CONTRIBUTING.md)

---

## License

MIT — [github.com/Blueconomy/monk](https://github.com/Blueconomy/monk)

---

*Built by [Blueconomy AI](https://theblueconomy.ai) — Techstars '25*  
*If monk saves you money, a ⭐ helps others find it.*
