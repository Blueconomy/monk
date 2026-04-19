"""
Text I/O sample collector for LLM judge training data.

When monk detects a finding, the associated LLM input/output text is captured
as a labeled example. Clean traces (no findings) produce negative examples.
Together these form a labeled dataset for fine-tuning or prompting an LLM judge.

Output format (JSONL):
  {
    "id": "uuid",
    "finding_type": "retry_loop" | "output_format" | ... | "clean",
    "severity": "high" | "medium" | "low" | "none",
    "label": "bad" | "good",
    "session_id": "...",
    "call_index": 0,
    "model": "gpt-4o",
    "input_text": "...",
    "output_text": "...",
    "input_length_chars": 1234,
    "output_length_chars": 456,
    "input_tokens": 310,
    "output_tokens": 82,
    "compression_ratio": 0.26,   # output_tokens / input_tokens
    "tool_calls": [...],
    "detector_evidence": "...",  # the finding detail — why monk flagged it
    "fix": "...",                # the recommended fix
    "source": "trail_otel",      # dataset name
    "timestamp": "2026-04-19T..."
  }
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monk.parsers.auto import TraceCall
from monk.detectors.base import Finding


class SampleCollector:
    """
    Collects labeled text I/O samples from monk runs for LLM judge training.

    Usage:
        collector = SampleCollector("samples.jsonl", source="my_dataset")
        collector.add_finding(call, finding)          # labeled bad
        collector.add_clean(call)                     # labeled good
        collector.flush()
    """

    def __init__(self, output_path: str | Path, source: str = "unknown", max_text_chars: int = 4000):
        self.output_path = Path(output_path)
        self.source = source
        self.max_text_chars = max_text_chars
        self._samples: list[dict] = []

    def add_finding(self, call: TraceCall, finding: Finding) -> None:
        """Add a labeled-bad sample: this call was associated with a finding."""
        sample = self._make_sample(call, label="bad", finding=finding)
        self._samples.append(sample)

    def add_clean(self, call: TraceCall) -> None:
        """Add a labeled-good sample: this call was in a clean trace."""
        sample = self._make_sample(call, label="good", finding=None)
        self._samples.append(sample)

    def add_from_span(self, span: Any, finding: Finding | None = None) -> None:
        """Add a sample from an OTEL Span object."""
        attrs = span.attributes
        input_text = str(attrs.get("llm.input_messages.0.message.content", ""))
        output_text = str(
            attrs.get("llm.output_messages.0.message.content") or
            attrs.get("output.value", "")
        )

        sample = {
            "id": str(uuid.uuid4()),
            "finding_type": finding.detector if finding else "clean",
            "severity": finding.severity if finding else "none",
            "label": "bad" if finding else "good",
            "session_id": span.session_id,
            "call_index": 0,
            "model": span.model or "",
            "input_text": input_text[:self.max_text_chars],
            "output_text": output_text[:self.max_text_chars],
            "input_length_chars": len(input_text),
            "output_length_chars": len(output_text),
            "input_tokens": span.input_tokens,
            "output_tokens": span.output_tokens,
            "compression_ratio": round(span.output_tokens / max(span.input_tokens, 1), 4),
            "tool_calls": [{"name": c.tool_name, "result_length": len(c.tool_result)}
                          for c in span.children if hasattr(c, 'tool_name') and c.tool_name],
            "detector_evidence": finding.detail if finding else "",
            "fix": finding.fix if finding else "",
            "source": self.source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._samples.append(sample)

    def flush(self) -> int:
        """Write all samples to disk. Returns count written."""
        if not self._samples:
            return 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            for s in self._samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        count = len(self._samples)
        self._samples.clear()
        return count

    def stats(self) -> dict:
        """Return stats about collected samples."""
        bad = sum(1 for s in self._samples if s["label"] == "bad")
        good = sum(1 for s in self._samples if s["label"] == "good")
        types = {}
        for s in self._samples:
            t = s["finding_type"]
            types[t] = types.get(t, 0) + 1
        return {"total": len(self._samples), "bad": bad, "good": good, "by_type": types}

    def _make_sample(self, call: TraceCall, label: str, finding: Finding | None) -> dict:
        raw = call.raw or {}
        input_text = str(raw.get("input_text", ""))
        output_text = str(raw.get("output_text", ""))

        # Fallback: reconstruct from raw record if texts aren't in raw
        if not input_text and call.raw:
            messages = call.raw.get("messages", [])
            for m in messages:
                if isinstance(m, dict):
                    input_text += f"[{m.get('role','')}]: {str(m.get('content',''))}\n"

        if not output_text and call.raw:
            choices = call.raw.get("choices", [])
            for ch in choices:
                if isinstance(ch, dict):
                    msg = ch.get("message", {})
                    output_text += str(msg.get("content", ""))

        input_text = input_text[:self.max_text_chars]
        output_text = output_text[:self.max_text_chars]

        return {
            "id": str(uuid.uuid4()),
            "finding_type": finding.detector if finding else "clean",
            "severity": finding.severity if finding else "none",
            "label": label,
            "session_id": call.session_id,
            "call_index": call.call_index,
            "model": call.model,
            "input_text": input_text,
            "output_text": output_text,
            "input_length_chars": len(input_text),
            "output_length_chars": len(output_text),
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "compression_ratio": round(call.output_tokens / max(call.input_tokens, 1), 4),
            "tool_calls": [{"name": tc.name, "args_length": len(tc.arguments), "result_length": len(tc.result)}
                          for tc in call.tool_calls],
            "detector_evidence": finding.detail if finding else "",
            "fix": finding.fix if finding else "",
            "source": self.source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def collect_samples_from_run(
    calls: list[TraceCall],
    findings: list[Finding],
    output_path: str | Path,
    source: str = "unknown",
    max_clean_samples: int = 50,
) -> SampleCollector:
    """
    Given the output of a monk run, collect text I/O samples.

    Calls associated with findings → labeled 'bad'.
    Other calls (up to max_clean_samples) → labeled 'good'.
    """
    collector = SampleCollector(output_path, source=source)

    # Build a set of session_ids that have findings
    flagged_sessions = {sid for f in findings for sid in f.affected_sessions}

    clean_count = 0
    for call in calls:
        # Find findings for this session
        session_findings = [f for f in findings if call.session_id in f.affected_sessions]
        if session_findings:
            for finding in session_findings:
                collector.add_finding(call, finding)
        elif clean_count < max_clean_samples:
            collector.add_clean(call)
            clean_count += 1

    return collector
