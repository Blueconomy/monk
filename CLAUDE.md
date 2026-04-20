# 🕵️ monk — Project Intelligence

**Last updated:** 2026-04-21

---

## Project Summary

**monk** is an agentic AI workflow blind spot detector — a Python CLI tool (pip: `monk-ai`) that analyzes trace logs from AI agent executions to find hidden cost leaks and inefficiencies.

- **Status:** v0.4.8 (Alpha, Techstars '25)
- **Owner/Org:** Blueconomy AI
- **License:** MIT
- **Repository:** https://github.com/Blueconomy/monk
- **Primary Language:** Python 3.9+

---

## The Problem monk Solves

Most observability tools show dashboards and traces. monk finds the patterns you don't know to look for:
- Retry loops (same tool called 3+ times in a row)
- Tools silently returning empty results (agent retries anyway, wastes tokens)
- Expensive models doing cheap tasks (gpt-4o for formatting, not analysis)
- System prompts consuming >55% of token budget
- Agent cycling without progress

These don't show as errors — they silently burn money. monk detects them automatically.

---

## Architecture & Core Modules

```
monk/
├── cli.py                    # Entry point — command interface
├── pricing.py                # Cost calculations per model
├── report.py                 # Report formatting & display (rich)
├── parsers/
│   ├── __init__.py
│   ├── auto.py               # Auto-detect format (OpenAI, Anthropic, LangGraph, LangSmith, JSONL)
│   └── otel.py               # OTEL span parser — Span dataclass + tree builder
├── detectors/
│   ├── base.py               # BaseDetector abstract class + Finding dataclass
│   ├── __init__.py           # Exports TRACE_DETECTORS, SPAN_DETECTORS, ALL_DETECTORS
│   │
│   ├── ── TRACE DETECTORS (work on all formats) ──
│   ├── retry_loop.py         # Same tool called 3+ consecutive times
│   ├── empty_return.py       # Tool returns null/empty → agent retries
│   ├── model_overkill.py     # Expensive model doing trivial tasks
│   ├── context_bloat.py      # System prompt bloat or unbounded history growth
│   ├── agent_loop.py         # Multi-step cycle repeated 3x+ (A→B→A→B)
│   ├── handoff_loop.py       # Multi-agent transfer cycling (Supervisor/Swarm A↔B bounce)
│   │
│   └── ── SPAN DETECTORS (OTEL only — require span trees) ──
│       ├── latency_spike.py      # Single call outlier vs session median
│       ├── error_cascade.py      # Tool error ignored → LLM calls wasted
│       ├── tool_dependency.py    # Cycles/chains in tool call graph
│       ├── cross_turn_memory.py  # Same tool+args re-fetched across turns
│       ├── token_bloat.py        # Token spike or monotonic growth per session
│       ├── output_format.py      # Model violates format rules in system prompt
│       ├── plan_execution.py     # Model planned steps it never executed
│       └── span_consistency.py   # Model claims facts with no supporting tool call
└── __init__.py

tests/
├── test_detectors.py      # Unit tests for all 15 detectors + OTEL parser
├── fixtures/
│   ├── sample_traces.jsonl      # Synthetic baseline (29 records)
│   ├── trail_otel.jsonl         # PatronusAI/TRAIL — 879 spans, 38 real traces
│   ├── memgpt_traces.jsonl      # MemGPT — 500 real multi-turn conversations
│   └── nemotron_traces.jsonl    # Nvidia Nemotron — 413 customer-service traces
```

### Key Design Patterns
- **BaseDetector:** All detectors inherit, implement `detect(traces)` → `Findings` list
- **Auto-parser:** Detects format (OpenAI API response, Anthropic Messages, LangGraph invoke response, LangSmith export, or raw JSONL)
- **Rich formatting:** Pretty console output with severity badges (🔴 high, 🟡 medium, 🟢 low)
- **JSON export:** CI-friendly format with `--json findings.json` flag

---

## CLI Interface

```bash
monk run <path>                          # Analyze file or directory
monk run traces/ --detectors retry_loop,model_overkill  # Specific detectors
monk run traces/ --json findings.json    # Export findings
monk run traces/ --min-severity high     # Filter by severity
```

Exit codes for CI:
- `0` — No high-severity findings
- `1` — High-severity findings found (exits on `--min-severity high`)

---

## Trace Format

monk accepts:
- **Native formats:** OpenAI Chat Completions API responses, Anthropic Messages API responses, LangSmith run exports
- **Custom JSONL:** One record per line, auto-parser extracts relevant fields

Required fields:
- `model` — e.g., `gpt-4o`, `claude-sonnet-4-6`
- `input_tokens` — Prompt token count
- `output_tokens` — Completion token count

Optional/recommended:
- `session_id` — Groups calls into sessions (enables agent_loop detection)
- `tool_name` — Name of tool called
- `tool_result` — Result returned (enables empty_return detection)
- `system_prompt_tokens` — For context_bloat detection

---

## Dependencies

- **click ≥8.1** — CLI framework
- **rich ≥13.0** — Console formatting
- **openai ≥1.0** — OpenAI SDK (trace parsing + pricing)
- **anthropic ≥0.25** — Anthropic SDK (trace parsing + pricing)

Dev:
- **pytest ≥7.0**
- **pytest-cov ≥4.0**

---

## Roadmap (Post v0.1.0)

- [ ] Live mode: instrument running agents via OpenTelemetry
- [ ] Prompt compression suggestions
- [ ] Cross-workflow benchmarking
- [ ] Slack / PagerDuty alerts
- [ ] Web dashboard

---

## Working Agreements & Patterns

### Code Organization
- New detectors go in `monk/detectors/` extending `BaseDetector`
- Register in `monk/detectors/__init__.py`
- Add tests in `tests/test_detectors.py`
- Detectors must be deterministic (same traces → same findings)

### Testing
- All detectors tested against `tests/fixtures/sample_traces.jsonl`
- New fixture files document specific patterns (e.g., `voice_ai_traces.jsonl` for voice workflow patterns)
- Tests use pytest; CI requires 100% pass

### Git Workflow
- Branch naming: `feature/detector-name` or `fix/issue-description`
- Commit messages: imperative mood, e.g., "Add retry_loop detector", "Fix empty_return timing bug"
- PR titles include emoji: ✨ feature, 🐛 fix, 📚 docs, ♻️ refactor
- All PRs require passing tests before merge

### Documentation
- README stays current with usage examples
- CONTRIBUTING.md updated for new workflows
- CLAUDE.md syncs context across Cowork ↔ Claude Code ↔ Claude Chat

---

## Benchmark Results (TRAIL, as of 2026-04-19)

- **v2 F1 = 100%** — 33/33 TRAIL error traces detected (TP=33, FP=0, FN=0)
- **v1 F1 = 84.85%** — 28/33 traces detected (5 false negatives now closed)
- The TRAIL fixture contains only error-containing traces (all 33 are ground-truth positives)
- v2 adds 3 output-level detectors (output_format, plan_execution, span_consistency) that cover previously-missed categories
- 137 total TRAIL findings (up from 86), 31/31 unit tests passing
- Full benchmark report: `BENCHMARK.md`

---

## Recent Changes (Cowork sessions 2026-04-18 → 2026-04-21)

**v0.4.8 — LangGraph support + handoff_loop detector (2026-04-21)**
- `parsers/auto.py`: LangGraph format parser — detects `messages[]` with `usage_metadata`, extracts one TraceCall per AIMessage, resolves tool results via `tool_call_id` matching
- `detectors/handoff_loop.py`: new TRACE_DETECTORS entry — catches `transfer_to_*` / `transfer_back_to_*` cycling between agents (A↔B bouncing 3+ times). Fires on LangGraph Supervisor and Swarm traces
- `simulate.py`: `supervisor` preset (gpt-4o routing to gpt-4o-mini specialist, linear — surfaces model_overkill) and `swarm` preset (peer agents cycling with back-edge — surfaces handoff_loop)
- `serve.py`: 🏢 Supervisor and 🐝 Swarm preset buttons added to Simulate tab

**v0.4.7 — simulate command + bug fixes (2026-04-20)**
- `monk simulate` CLI command with `--pattern`, `--sessions`, `-o`, `--seed`, `--run` flags
- OTEL workflow simulation engine in `simulate.py` — converts node+edge graph to valid OTEL spans
- Simulate tab in dashboard — visual graph editor, 5 presets, SVG bezier edges, `/simulate` endpoint
- `agent_loop.py` fix: skip single-tool n-grams (was double-firing with retry_loop)
- Scan debounce (8s) in `serve.py` to prevent log spam

**v0.4.6 — 9 OTEL span detectors + dashboard overhaul (2026-04-19)**
- `latency_spike`, `error_cascade`, `tool_dependency`, `cross_turn_memory`, `token_bloat`, `output_format`, `plan_execution`, `span_consistency` — all OTEL span detectors
- White+orange dashboard redesign, quickstart command, dataset downloader
- 100% F1 on TRAIL benchmark (33/33 error traces, 0 FP)

**Current branch:** `main` (pending commit — user needs to push)

---

## Next Priorities

- [ ] Commit v0.4.8 and push to GitHub:
  ```bash
  cd ~/Documents/ClaudeProject/monk
  rm -f .git/index.lock .git/HEAD.lock
  git add monk/__init__.py monk/parsers/auto.py monk/detectors/handoff_loop.py \
          monk/detectors/__init__.py monk/simulate.py monk/serve.py \
          pyproject.toml README.md CLAUDE.md
  git commit -m "✨ v0.4.8: LangGraph parser, handoff_loop detector, supervisor/swarm presets"
  git push https://baman95:GITHUB_TOKEN@github.com/Blueconomy/monk.git main
  ```
- [ ] Add unit tests for `handoff_loop` and LangGraph parser to `tests/test_detectors.py`
- [ ] Consider real-time mode (OTEL SDK integration) as next major feature
- [ ] Consider adding confidence scores to findings (high-confidence vs heuristic-only)
- [ ] Explore Slack / PagerDuty alerts for real-time monitoring

---

## How to Sync This File

This file is the shared brain. When decisions are made in Cowork:
1. Update this CLAUDE.md with the decision/insight
2. Commit & push: `git commit -m "docs: update CLAUDE.md with [decision]"`
3. Claude Code picks it up on next invocation
4. Claude Chat can reference it for drafts

All three instances stay in sync without re-explaining context.
