# 🕵️ monk — Project Intelligence

**Last updated:** 2026-04-19

---

## Project Summary

**monk** is an agentic AI workflow blind spot detector — a Python CLI tool (pip: `monk-ai`) that analyzes trace logs from AI agent executions to find hidden cost leaks and inefficiencies.

- **Status:** v0.2.0-dev (Alpha, Techstars '25)
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
│   ├── auto.py               # Auto-detect format (OpenAI, Anthropic, LangSmith, JSONL)
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
├── test_detectors.py      # Unit tests for all 13 detectors + OTEL parser
├── fixtures/
│   ├── sample_traces.jsonl      # Synthetic baseline (29 records)
│   ├── trail_otel.jsonl         # PatronusAI/TRAIL — 879 spans, 38 real traces
│   ├── memgpt_traces.jsonl      # MemGPT — 500 real multi-turn conversations
│   └── nemotron_traces.jsonl    # Nvidia Nemotron — 413 customer-service traces
```

### Key Design Patterns
- **BaseDetector:** All detectors inherit, implement `detect(traces)` → `Findings` list
- **Auto-parser:** Detects format (OpenAI API response, Anthropic Messages, LangSmith export, or raw JSONL)
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

## Recent Changes (Cowork sessions 2026-04-18 → 2026-04-19)

**New detectors built:**
- `latency_spike.py` — single-call outlier vs session median (OTEL)
- `error_cascade.py` — tool error ignored, downstream LLM calls wasted (OTEL)
- `tool_dependency.py` — cycles and deep chains in tool call graph (OTEL)
- `cross_turn_memory.py` — same tool+args re-fetched across turns (OTEL)
- `token_bloat.py` — per-session token spike or monotonic growth (OTEL)
- `output_format.py` — model violates format rules extracted from system prompt
- `plan_execution.py` — planned steps never executed, plan abandoned
- `span_consistency.py` — model claims verified facts with no preceding tool call

**Bug fixes:**
- `agent_loop.py`: was double-counting with `retry_loop` on single-tool repetitions. Fixed: agent_loop now only checks patterns of length ≥ 2.
- `tool_dependency.py`: false positives on OTEL orchestration wrappers ("Step 1", "ToolCallingAgent.run"). Fixed: `_is_orchestration()` filter added.
- `context_bloat.py`: added Check C — verbatim tool output flooding context (covers TRAIL Context Handling Failures).
- `parsers/otel.py`: spans with `tool_result` attribute now correctly classified as "tool" kind even if named "Step N". Fixes `cross_turn_memory` missing SWE-bench action spans.
- CI `smoke test`: bash `set -e` was killing exit-code capture. Fixed with `||` operator.

**Infrastructure:**
- `parsers/otel.py` — full OTEL span parser with Span dataclass and tree builder
- `report.py` — overhauled: severity summary, detector breakdown table, top wasteful sessions
- `BENCHMARK.md` — full analysis of 4 datasets (TRAIL, MemGPT, Nemotron, Sample)
- Benchmark fixtures: `trail_otel.jsonl` (879 spans), `memgpt_traces.jsonl` (500 convos), `nemotron_traces.jsonl` (413 convos)

**Current branch:** `main` (pending commit — user needs to push)

---

## Next Priorities

- [ ] Commit all pending changes (clear git index.lock first): `rm .git/index.lock && git add -A && git commit -m "✨ v0.2.0: output-level detectors, bug fixes, 100% TRAIL F1"`
- [ ] Consider real-time mode (OTEL SDK integration) as next major feature (v0.2 roadmap item)
- [ ] Consider adding confidence scores to findings (high-confidence vs heuristic-only)
- [ ] Explore web dashboard / Slack alerts for real-time monitoring

---

## How to Sync This File

This file is the shared brain. When decisions are made in Cowork:
1. Update this CLAUDE.md with the decision/insight
2. Commit & push: `git commit -m "docs: update CLAUDE.md with [decision]"`
3. Claude Code picks it up on next invocation
4. Claude Chat can reference it for drafts

All three instances stay in sync without re-explaining context.
