# 🕵️ monk — Project Intelligence

**Last updated:** 2026-04-18

---

## Project Summary

**monk** is an agentic AI workflow blind spot detector — a Python CLI tool (pip: `monk-ai`) that analyzes trace logs from AI agent executions to find hidden cost leaks and inefficiencies.

- **Status:** v0.1.0 (Alpha, Techstars '25)
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
├── cli.py                 # Entry point — command interface
├── pricing.py             # Cost calculations per model
├── report.py              # Report formatting & display (rich)
├── parsers/
│   ├── __init__.py
│   └── auto.py            # Auto-detect trace format (OpenAI, Anthropic, LangSmith, custom JSONL)
├── detectors/
│   ├── base.py            # BaseDetector abstract class
│   ├── retry_loop.py      # Same tool called 3+ times in a row
│   ├── empty_return.py    # Tool returns null/empty → agent retries
│   ├── model_overkill.py  # Expensive model on simple tasks
│   ├── context_bloat.py   # System prompt bloat or unbounded history
│   ├── agent_loop.py      # Agent cycling without progress
│   └── __init__.py
└── __init__.py

tests/
├── test_detectors.py      # Detector unit tests
├── fixtures/
│   ├── sample_traces.jsonl
│   └── voice_ai_traces.jsonl (to be added)
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

## Recent Changes

**Latest commits:**
- `ac22eb5` — fix: update repo URLs from blueconomy-ai/monk to Blueconomy/monk
- `74baaa1` — 🕵️ monk v0.1.0 — initial release

**Current branch:** `main` (stable)

---

## Next Priorities

*To be filled in from Cowork session decisions.*

---

## How to Sync This File

This file is the shared brain. When decisions are made in Cowork:
1. Update this CLAUDE.md with the decision/insight
2. Commit & push: `git commit -m "docs: update CLAUDE.md with [decision]"`
3. Claude Code picks it up on next invocation
4. Claude Chat can reference it for drafts

All three instances stay in sync without re-explaining context.
