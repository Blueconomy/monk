"""
monk serve — live metrics server with embedded dashboard.

Endpoints:
  GET  /            — HTML control room dashboard
  GET  /findings    — JSON summary of all findings
  GET  /metrics     — Prometheus text format
  GET  /datasets    — available datasets + download status
  POST /download    — download a dataset to traces/ dir (body: {"name":"taubench"})
  GET  /health      — liveness check
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from monk import __version__
from monk.detectors import ALL_DETECTORS
from monk.parsers.auto import parse_traces


# ── Available datasets on HuggingFace ─────────────────────────────────────────
HF_BASE = "https://huggingface.co/datasets/Blueconomy/monk-benchmarks/resolve/main"
CATALOG = [
    {
        "id":      "taubench",
        "file":    "taubench_traces.jsonl",
        "label":   "tau-bench",
        "desc":    "Banking + e-commerce agents",
        "calls":   "17,932",
        "format":  "Trace",
        "size_mb": 9.9,
    },
    {
        "id":      "finance",
        "file":    "finance_traces.jsonl",
        "label":   "Finance (10-K ReAct)",
        "desc":    "LangGraph financial analysis",
        "calls":   "4,610",
        "format":  "Trace",
        "size_mb": 3.9,
    },
    {
        "id":      "trail",
        "file":    "trail_otel.jsonl",
        "label":   "TRAIL benchmark",
        "desc":    "PatronusAI ground-truth · 100% F1",
        "calls":   "879 spans",
        "format":  "OTEL",
        "size_mb": 25.8,
    },
    {
        "id":      "gaia",
        "file":    "gaia_smolagents_traces.jsonl",
        "label":   "GAIA smolagents",
        "desc":    "HuggingFace smolagents benchmark",
        "calls":   "621",
        "format":  "OTEL",
        "size_mb": 0.9,
    },
    {
        "id":      "memgpt",
        "file":    "memgpt_traces.jsonl",
        "label":   "MemGPT",
        "desc":    "Multi-turn memory conversations",
        "calls":   "500",
        "format":  "Trace",
        "size_mb": 5.6,
    },
    {
        "id":      "nemotron",
        "file":    "nemotron_traces.jsonl",
        "label":   "Nemotron",
        "desc":    "Nvidia customer-service agents",
        "calls":   "413",
        "format":  "Trace",
        "size_mb": 3.6,
    },
    {
        "id":      "sample",
        "file":    "sample_traces.jsonl",
        "label":   "Sample (synthetic)",
        "desc":    "Minimal example — good starting point",
        "calls":   "29",
        "format":  "Trace",
        "size_mb": 0.01,
    },
]


# ── Built-in demo sample (33 records, 7 sessions, triggers all 5 trace detectors) ──
DEMO_TRACES_JSONL = (
    '{"session_id": "prod-db-crash", "model": "gpt-4o", "input_tokens": 2100, "output_tokens": 95, "tool_name": "database_query", "tool_result": "Connection timeout"}\n'
    '{"session_id": "prod-db-crash", "model": "gpt-4o", "input_tokens": 2150, "output_tokens": 100, "tool_name": "database_query", "tool_result": "Connection timeout"}\n'
    '{"session_id": "prod-db-crash", "model": "gpt-4o", "input_tokens": 2200, "output_tokens": 98, "tool_name": "database_query", "tool_result": "Connection timeout"}\n'
    '{"session_id": "prod-db-crash", "model": "gpt-4o", "input_tokens": 2250, "output_tokens": 92, "tool_name": "database_query", "tool_result": "Connection timeout"}\n'
    '{"session_id": "prod-db-crash", "model": "gpt-4o", "input_tokens": 2300, "output_tokens": 90, "tool_name": "database_query", "tool_result": "Connection timeout"}\n'
    '{"session_id": "prod-search-fail", "model": "gpt-4o", "input_tokens": 1800, "output_tokens": 110, "tool_name": "web_search", "tool_result": ""}\n'
    '{"session_id": "prod-search-fail", "model": "gpt-4o", "input_tokens": 1860, "output_tokens": 110, "tool_name": "web_search", "tool_result": null}\n'
    '{"session_id": "prod-search-fail", "model": "gpt-4o", "input_tokens": 1920, "output_tokens": 110, "tool_name": "web_search", "tool_result": ""}\n'
    '{"session_id": "prod-search-fail", "model": "gpt-4o", "input_tokens": 1980, "output_tokens": 110, "tool_name": "web_search", "tool_result": null}\n'
    '{"session_id": "prod-search-fail", "model": "gpt-4o", "input_tokens": 2040, "output_tokens": 180, "tool_name": "web_search", "tool_result": "[1] OpenAI announces GPT-5..."}\n'
    '{"session_id": "prod-bloated-prompt", "model": "gpt-4o", "input_tokens": 14000, "output_tokens": 200, "system_prompt_tokens": 8680, "tool_name": "lookup", "tool_result": "data"}\n'
    '{"session_id": "prod-bloated-prompt", "model": "gpt-4o", "input_tokens": 14400, "output_tokens": 200, "system_prompt_tokens": 8928, "tool_name": "lookup", "tool_result": "data2"}\n'
    '{"session_id": "prod-bloated-prompt", "model": "gpt-4o", "input_tokens": 14800, "output_tokens": 200, "system_prompt_tokens": 9176, "tool_name": "lookup", "tool_result": "data3"}\n'
    '{"session_id": "prod-bloated-prompt", "model": "gpt-4o", "input_tokens": 15200, "output_tokens": 200, "system_prompt_tokens": 9424, "tool_name": "lookup", "tool_result": "data4"}\n'
    '{"session_id": "prod-loop-agent", "model": "gpt-4o", "input_tokens": 1900, "output_tokens": 140, "tool_name": "search_documents", "tool_result": "result_0"}\n'
    '{"session_id": "prod-loop-agent", "model": "gpt-4o", "input_tokens": 1980, "output_tokens": 140, "tool_name": "rank_results", "tool_result": "result_1"}\n'
    '{"session_id": "prod-loop-agent", "model": "gpt-4o", "input_tokens": 2060, "output_tokens": 140, "tool_name": "search_documents", "tool_result": "result_2"}\n'
    '{"session_id": "prod-loop-agent", "model": "gpt-4o", "input_tokens": 2140, "output_tokens": 140, "tool_name": "rank_results", "tool_result": "result_3"}\n'
    '{"session_id": "prod-loop-agent", "model": "gpt-4o", "input_tokens": 2220, "output_tokens": 140, "tool_name": "search_documents", "tool_result": "result_4"}\n'
    '{"session_id": "prod-loop-agent", "model": "gpt-4o", "input_tokens": 2300, "output_tokens": 140, "tool_name": "rank_results", "tool_result": "result_5"}\n'
    '{"session_id": "prod-overpriced-format", "model": "gpt-4o", "input_tokens": 800, "output_tokens": 30, "tool_name": "format_json", "tool_result": "{\\"ok\\":true}"}\n'
    '{"session_id": "prod-overpriced-format", "model": "gpt-4o", "input_tokens": 820, "output_tokens": 32, "tool_name": "format_json", "tool_result": "{\\"ok\\":true}"}\n'
    '{"session_id": "prod-overpriced-format", "model": "gpt-4o", "input_tokens": 840, "output_tokens": 28, "tool_name": "format_json", "tool_result": "{\\"ok\\":true}"}\n'
    '{"session_id": "prod-overpriced-format", "model": "gpt-4o", "input_tokens": 860, "output_tokens": 31, "tool_name": "format_json", "tool_result": "{\\"ok\\":true}"}\n'
    '{"session_id": "prod-overpriced-format", "model": "gpt-4o", "input_tokens": 880, "output_tokens": 29, "tool_name": "format_json", "tool_result": "{\\"ok\\":true}"}\n'
    '{"session_id": "prod-api-retry", "model": "claude-opus-4-6", "input_tokens": 2500, "output_tokens": 120, "tool_name": "call_payment_api", "tool_result": "503 Service Unavailable"}\n'
    '{"session_id": "prod-api-retry", "model": "claude-opus-4-6", "input_tokens": 2600, "output_tokens": 120, "tool_name": "call_payment_api", "tool_result": "503 Service Unavailable"}\n'
    '{"session_id": "prod-api-retry", "model": "claude-opus-4-6", "input_tokens": 2700, "output_tokens": 120, "tool_name": "call_payment_api", "tool_result": "503 Service Unavailable"}\n'
    '{"session_id": "prod-api-retry", "model": "claude-opus-4-6", "input_tokens": 2800, "output_tokens": 120, "tool_name": "call_payment_api", "tool_result": "503 Service Unavailable"}\n'
    '{"session_id": "prod-healthy", "model": "gpt-4o-mini", "input_tokens": 1200, "output_tokens": 250, "tool_name": "tool_a", "tool_result": "success"}\n'
    '{"session_id": "prod-healthy", "model": "gpt-4o-mini", "input_tokens": 1300, "output_tokens": 270, "tool_name": "tool_b", "tool_result": "success"}\n'
    '{"session_id": "prod-healthy", "model": "gpt-4o-mini", "input_tokens": 1400, "output_tokens": 290, "tool_name": "tool_c", "tool_result": "success"}\n'
    '{"session_id": "prod-healthy", "model": "gpt-4o-mini", "input_tokens": 1500, "output_tokens": 310, "tool_name": "tool_a", "tool_result": "success"}'
)


class DownloadManager:
    """Tracks background downloads of HuggingFace datasets."""

    def __init__(self, traces_dir: Path):
        self._dir   = traces_dir
        self._lock  = threading.Lock()
        self._state: dict[str, dict] = {}   # id -> {status, progress, error}

    def status(self) -> list[dict]:
        with self._lock:
            out = []
            for ds in CATALOG:
                fid   = ds["id"]
                fpath = self._dir / ds["file"]
                st    = self._state.get(fid, {})
                out.append({
                    **ds,
                    "loaded":    fpath.exists(),
                    "size_on_disk": round(fpath.stat().st_size / 1024 / 1024, 1) if fpath.exists() else 0,
                    "status":    st.get("status", "ready" if fpath.exists() else "available"),
                    "progress":  st.get("progress", 100 if fpath.exists() else 0),
                    "error":     st.get("error"),
                })
            return out

    def start_download(self, dataset_id: str) -> bool:
        ds = next((d for d in CATALOG if d["id"] == dataset_id), None)
        if not ds:
            return False
        with self._lock:
            if self._state.get(dataset_id, {}).get("status") == "downloading":
                return True  # already running
            self._state[dataset_id] = {"status": "downloading", "progress": 0}
        threading.Thread(target=self._download, args=(ds,), daemon=True).start()
        return True

    def _download(self, ds: dict):
        fid  = ds["id"]
        url  = f"{HF_BASE}/{ds['file']}"
        dest = self._dir / ds["file"]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "monk/0.4.5"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            dest.write_bytes(data)
            with self._lock:
                self._state[fid] = {"status": "ready", "progress": 100}
            print(f"[monk] downloaded {ds['file']}  ({len(data)//1024:,} KB)")
        except Exception as e:
            with self._lock:
                self._state[fid] = {"status": "error", "progress": 0, "error": str(e)}
            print(f"[monk] download failed {ds['file']}: {e}")


class MetricsStore:
    """Thread-safe store for Prometheus metrics and findings."""

    def __init__(self):
        self._lock = threading.Lock()
        self._findings_count: dict[tuple, int] = {}
        self._waste: dict[str, float] = {}
        self._sessions_total = 0
        self._calls_total = 0
        self._last_scan_ts = 0.0
        self._high_total = 0
        self._recent: list[dict] = []

    def update(self, findings, calls, sessions: int) -> None:
        with self._lock:
            self._sessions_total += sessions
            self._calls_total    += len(calls)
            self._last_scan_ts    = time.time()
            for f in findings:
                key = (f.detector, f.severity)
                self._findings_count[key] = self._findings_count.get(key, 0) + 1
                self._waste[f.detector]   = self._waste.get(f.detector, 0.0) + f.estimated_waste_usd_per_day
                if f.severity == "high":
                    self._high_total += 1
                self._recent.append({
                    "detector":    f.detector,
                    "severity":    f.severity,
                    "title":       f.title,
                    "detail":      f.detail,
                    "fix":         f.fix,
                    "sessions":    f.affected_sessions[:3],
                    "waste_per_day": round(f.estimated_waste_usd_per_day, 4),
                    "ts":          self._last_scan_ts,
                })
            if len(self._recent) > 500:
                self._recent = self._recent[-500:]

    def to_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP monk_findings_total Total findings by detector and severity",
                "# TYPE monk_findings_total counter",
            ]
            for (det, sev), count in sorted(self._findings_count.items()):
                lines.append(f'monk_findings_total{{detector="{det}",severity="{sev}"}} {count}')
            lines += [
                "# HELP monk_waste_usd_per_day Estimated daily waste USD by detector",
                "# TYPE monk_waste_usd_per_day gauge",
            ]
            for det, waste in sorted(self._waste.items()):
                lines.append(f'monk_waste_usd_per_day{{detector="{det}"}} {waste:.4f}')
            lines += [
                "# HELP monk_sessions_analyzed_total Sessions analyzed",
                "# TYPE monk_sessions_analyzed_total counter",
                f"monk_sessions_analyzed_total {self._sessions_total}",
                "# HELP monk_calls_analyzed_total LLM calls analyzed",
                "# TYPE monk_calls_analyzed_total counter",
                f"monk_calls_analyzed_total {self._calls_total}",
                "# HELP monk_high_severity_findings_total High-severity findings",
                "# TYPE monk_high_severity_findings_total counter",
                f"monk_high_severity_findings_total {self._high_total}",
                "# HELP monk_last_scan_timestamp Unix timestamp of last scan",
                "# TYPE monk_last_scan_timestamp gauge",
                f"monk_last_scan_timestamp {self._last_scan_ts:.0f}",
            ]
            return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        with self._lock:
            total_waste = sum(self._waste.values())
            total       = sum(self._findings_count.values())
            return json.dumps({
                "summary": {
                    "total_findings":    total,
                    "high":   sum(v for (_, s), v in self._findings_count.items() if s == "high"),
                    "medium": sum(v for (_, s), v in self._findings_count.items() if s == "medium"),
                    "low":    sum(v for (_, s), v in self._findings_count.items() if s == "low"),
                    "sessions_analyzed": self._sessions_total,
                    "calls_analyzed":    self._calls_total,
                    "waste_usd_per_day": round(total_waste, 2),
                    "waste_usd_per_month": round(total_waste * 30, 2),
                    "last_scan":         self._last_scan_ts,
                },
                "by_detector": {
                    det: {"waste_per_day": round(w, 4)}
                    for det, w in sorted(self._waste.items())
                },
                "recent_findings": self._recent[-50:],
            }, indent=2)


# ── Dashboard HTML ─────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>monk — Agent Cost Control Room</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0c0c0c;--s1:#111;--s2:#161616;--s3:#1c1c1c;
  --b1:#1f1f1f;--b2:#2a2a2a;--b3:#333;
  --t1:#fff;--t2:#a0a0a0;--t3:#555;--t4:#333;
  --or:#f97316;--or2:#f9731612;--or3:#f9731628;--or4:#f9731645;
  --red:#ff4444;--grn:#22c55e;--blue:#60a5fa;
  --mono:'JetBrains Mono',monospace;
}
html{font-size:14px}
body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--t1);-webkit-font-smoothing:antialiased}

/* NAV */
nav{position:sticky;top:0;z-index:100;background:rgba(12,12,12,.95);backdrop-filter:blur(24px);border-bottom:1px solid var(--b1);height:54px;display:flex;align-items:center;justify-content:space-between;padding:0 28px}
.nav-l{display:flex;align-items:center;gap:12px}
.logo{font-size:15px;font-weight:700;letter-spacing:-.2px;display:flex;align-items:center;gap:8px}
.logo-icon{font-size:18px}
.logo-name span{color:var(--or)}
.pill{font-size:10px;font-weight:600;padding:2px 7px;border-radius:99px;border:1px solid var(--b2);color:var(--t3);font-family:var(--mono);letter-spacing:.3px}
.nav-r{display:flex;align-items:center;gap:12px}
#live-badge{display:flex;align-items:center;gap:6px;font-size:11px;padding:4px 11px;border-radius:99px;border:1px solid var(--b2);color:var(--t3);font-family:var(--mono);transition:all .3s}
#live-badge.live{color:var(--grn);border-color:#22c55e28;background:#22c55e06}
#live-badge.error{color:var(--red);border-color:#ff444428}
.badge-dot{width:5px;height:5px;border-radius:50%;background:currentColor;flex-shrink:0}
#live-badge.live .badge-dot{box-shadow:0 0 5px var(--grn);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.ts{font-size:11px;color:var(--t4);font-family:var(--mono)}
.nav-btn{font-size:11px;padding:4px 11px;border-radius:6px;border:1px solid var(--b2);background:transparent;color:var(--t2);cursor:pointer;font-family:var(--mono);transition:all .15s;display:flex;align-items:center;gap:5px}
.nav-btn:hover{border-color:var(--or3);color:var(--or)}
.nav-divider{width:1px;height:20px;background:var(--b2)}

/* TABS */
.tabs{display:flex;gap:1px;background:var(--b1);border-bottom:1px solid var(--b1);padding:0 28px}
.tab{font-size:12px;font-weight:500;padding:10px 16px;cursor:pointer;color:var(--t3);border-bottom:2px solid transparent;transition:all .15s;letter-spacing:.2px}
.tab:hover{color:var(--t2)}
.tab.active{color:var(--or);border-bottom-color:var(--or)}

/* PAGE */
.page{max-width:1440px;margin:0 auto;padding:24px 28px 60px}
.tab-pane{display:none}.tab-pane.active{display:block}

/* SECTION */
.sec{margin-bottom:24px}
.sec-label{font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--t3);margin-bottom:14px;display:flex;align-items:center;gap:10px}
.sec-label::after{content:'';flex:1;height:1px;background:var(--b1)}

/* KPIs */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--b1);border:1px solid var(--b1);border-radius:14px;overflow:hidden;margin-bottom:20px}
.kpi{background:var(--s1);padding:22px 20px;position:relative;transition:background .18s;cursor:default}
.kpi:hover{background:var(--s2)}
.kpi-label{font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;color:var(--t3);margin-bottom:10px}
.kpi-val{font-size:34px;font-weight:800;letter-spacing:-1.5px;line-height:1;margin-bottom:5px;transition:all .4s}
.kpi-val.or{color:var(--or)}
.kpi-val.red{color:var(--red)}
.kpi-sub{font-size:11px;color:var(--t3);line-height:1.5}
.kpi::after{content:'';position:absolute;bottom:0;left:20px;right:20px;height:1px;background:var(--or);transform:scaleX(0);transform-origin:left;transition:transform .25s}
.kpi:hover::after{transform:scaleX(1)}

/* GRIDS */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
.g13{display:grid;grid-template-columns:1fr 3fr;gap:14px;margin-bottom:14px}
.g31{display:grid;grid-template-columns:3fr 1fr;gap:14px;margin-bottom:14px}
.g21{display:grid;grid-template-columns:2fr 1fr;gap:14px;margin-bottom:14px}

/* PANEL */
.panel{background:var(--s1);border:1px solid var(--b1);border-radius:12px;overflow:hidden}
.ph{padding:14px 18px 0;display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.ph-title{font-size:11px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--t3)}
.ph-right{display:flex;align-items:center;gap:8px}
.pb{padding:0 18px 18px}
.pb-full{padding:0}

/* DATASET CARDS */
.ds-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.ds-card{background:var(--s2);border:1px solid var(--b1);border-radius:10px;padding:14px 16px;transition:border-color .15s,background .15s}
.ds-card:hover{border-color:var(--b3);background:var(--s3)}
.ds-card.loaded{border-color:var(--or3)}
.ds-top{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:8px}
.ds-name{font-size:13px;font-weight:600;color:var(--t1)}
.ds-status{font-size:10px;font-weight:600;padding:2px 7px;border-radius:99px;font-family:var(--mono);letter-spacing:.3px}
.ds-status.loaded{background:#22c55e15;color:var(--grn);border:1px solid #22c55e28}
.ds-status.available{background:var(--b1);color:var(--t3);border:1px solid var(--b2)}
.ds-status.downloading{background:#f9731612;color:var(--or);border:1px solid var(--or3);animation:blink 1.2s infinite}
.ds-status.error{background:#ff444412;color:var(--red);border:1px solid #ff444428}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.5}}
.ds-desc{font-size:12px;color:var(--t3);margin-bottom:10px;line-height:1.5}
.ds-meta{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.ds-tag{font-size:10px;font-family:var(--mono);padding:2px 6px;border-radius:4px;color:var(--t3);background:var(--b1);border:1px solid var(--b2)}
.ds-tag.otel{color:var(--blue);background:#3b82f610;border-color:#3b82f628}
.ds-bar-wrap{margin-bottom:12px}
.ds-bar-label{display:flex;justify-content:space-between;font-size:10px;color:var(--t3);margin-bottom:4px;font-family:var(--mono)}
.ds-bar-track{height:2px;background:var(--b2);border-radius:1px;overflow:hidden}
.ds-bar-fill{height:100%;border-radius:1px;background:var(--or);transition:width .6s}
.ds-btn{width:100%;padding:7px;border-radius:7px;border:1px solid var(--b2);background:transparent;color:var(--t2);font-size:12px;font-family:var(--mono);cursor:pointer;transition:all .15s}
.ds-btn:hover:not(:disabled){border-color:var(--or3);color:var(--or);background:var(--or2)}
.ds-btn:disabled{opacity:.4;cursor:not-allowed}
.ds-btn.loaded{border-color:#22c55e28;color:var(--grn);background:#22c55e08}

/* FINDINGS STREAM */
.stream{max-height:360px;overflow-y:auto;display:flex;flex-direction:column;gap:7px}
.stream::-webkit-scrollbar{width:3px}
.stream::-webkit-scrollbar-thumb{background:var(--b2);border-radius:2px}
.stream-item{background:var(--s2);border:1px solid var(--b1);border-radius:8px;padding:10px 13px;border-left:2px solid var(--b2);cursor:default;transition:background .1s}
.stream-item:hover{background:var(--s3)}
.stream-item.high{border-left-color:var(--red)}
.stream-item.medium{border-left-color:var(--or)}
.stream-item.low{border-left-color:var(--grn)}
.si-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}
.si-det{font-family:var(--mono);font-size:11px;color:var(--or)}
.si-sev{font-size:10px;font-weight:600;color:var(--t3);letter-spacing:.5px;text-transform:uppercase}
.si-title{font-size:12px;color:var(--t2);line-height:1.45;margin-bottom:3px}
.si-fix{font-size:11px;color:var(--t3);line-height:1.4}
.si-fix::before{content:'→ ';color:var(--or)}

/* WASTE BARS */
.wrow{margin-bottom:11px}
.wrow:last-child{margin-bottom:0}
.wrow-top{display:flex;justify-content:space-between;margin-bottom:4px}
.wrow-name{font-family:var(--mono);font-size:11px;color:var(--t2)}
.wrow-val{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--or)}
.wrow-val.dim{color:var(--t3)}
.wtrack{height:2px;background:var(--b2);border-radius:1px;overflow:hidden}
.wfill{height:100%;border-radius:1px;background:var(--or);transition:width .6s ease}

/* TABLE */
.tbl{overflow-x:auto}
table{width:100%;border-collapse:collapse}
thead th{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--t3);padding:0 16px 10px;text-align:left;border-bottom:1px solid var(--b1)}
thead th:first-child{padding-left:18px}
thead th:last-child{padding-right:18px}
tbody tr{border-bottom:1px solid var(--b1);transition:background .1s}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--s2)}
tbody td{padding:11px 16px;font-size:12px;color:var(--t2);vertical-align:middle}
tbody td:first-child{padding-left:18px}
tbody td:last-child{padding-right:18px}
.chip{display:inline-block;font-family:var(--mono);font-size:10px;font-weight:600;padding:2px 7px;border-radius:4px}
.chip-or{background:var(--or2);color:var(--or);border:1px solid var(--or3)}
.chip-w{background:#fff06;color:#555;border:1px solid var(--b2)}
.sev-badge{font-size:11px;font-weight:700;display:flex;align-items:center;gap:5px}
.dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.sev-h{color:var(--red)}.sev-h .dot{background:var(--red);box-shadow:0 0 4px var(--red)}
.sev-m{color:var(--or)}.sev-m .dot{background:var(--or);box-shadow:0 0 4px var(--or)}
.mn{font-family:var(--mono);font-size:11px}

/* SEV TILES */
.sev-tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.sev-tile{background:var(--s2);border:1px solid var(--b1);border-radius:8px;padding:12px;text-align:center}
.st-val{font-size:22px;font-weight:800;letter-spacing:-.5px}
.st-lbl{font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--t3);margin-top:3px}

/* EMPTY STATE */
.empty{text-align:center;padding:40px 20px;color:var(--t3)}
.empty-icon{font-size:32px;margin-bottom:12px}
.empty-text{font-size:13px;line-height:1.6}

/* FOOTER */
footer{border-top:1px solid var(--b1);padding:18px 28px;display:flex;align-items:center;justify-content:space-between}
footer .left{font-size:11px;color:var(--t3)}
footer .right{display:flex;gap:16px}
footer a{font-size:11px;color:var(--t3);text-decoration:none;transition:color .15s}
footer a:hover{color:var(--or)}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="nav-l">
    <div class="logo">
      <span class="logo-icon">🕵️</span>
      <span class="logo-name">mon<span>k</span></span>
    </div>
    <div class="pill">v__VERSION__</div>
    <div class="pill" id="nav-files">— files</div>
  </div>
  <div class="nav-r">
    <span class="ts" id="ts"></span>
    <div class="nav-divider"></div>
    <button class="nav-btn" onclick="loadAll()">↻ Refresh</button>
    <div id="live-badge"><div class="badge-dot"></div><span id="badge-text">connecting</span></div>
  </div>
</nav>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('overview')">Overview</div>
  <div class="tab" onclick="switchTab('findings')">Findings</div>
  <div class="tab" onclick="switchTab('datasets')">Datasets</div>
</div>

<div class="page">

  <!-- ══ OVERVIEW ══════════════════════════════════════════════════════════ -->
  <div class="tab-pane active" id="tab-overview">

    <div class="sec">
      <div class="kpis">
        <div class="kpi">
          <div class="kpi-label">Total Findings</div>
          <div class="kpi-val" id="k-total">—</div>
          <div class="kpi-sub" id="k-total-sub">—</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Waste / Day</div>
          <div class="kpi-val or" id="k-day">—</div>
          <div class="kpi-sub" id="k-day-sub">estimated avoidable spend</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Projected / Month</div>
          <div class="kpi-val or" id="k-month">—</div>
          <div class="kpi-sub">all addressable with fixes</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">LLM Calls</div>
          <div class="kpi-val" id="k-calls">—</div>
          <div class="kpi-sub" id="k-calls-sub">across all trace files</div>
        </div>
      </div>
    </div>

    <div class="g2">
      <!-- Detector bar -->
      <div class="panel">
        <div class="ph"><div class="ph-title">Findings by Detector</div></div>
        <div class="pb"><canvas id="det-chart" height="270"></canvas></div>
      </div>
      <!-- Severity + donut -->
      <div class="panel">
        <div class="ph"><div class="ph-title">Severity</div></div>
        <div class="pb">
          <div class="sev-tiles">
            <div class="sev-tile"><div class="st-val" id="s-h" style="color:var(--red)">—</div><div class="st-lbl">High</div></div>
            <div class="sev-tile"><div class="st-val" id="s-m" style="color:var(--or)">—</div><div class="st-lbl">Medium</div></div>
            <div class="sev-tile"><div class="st-val" id="s-l" style="color:var(--grn)">—</div><div class="st-lbl">Low</div></div>
          </div>
          <canvas id="sev-chart" height="180"></canvas>
        </div>
      </div>
    </div>

    <div class="g21">
      <!-- Waste bars -->
      <div class="panel">
        <div class="ph"><div class="ph-title">Waste $/day by Detector</div></div>
        <div class="pb"><div id="waste-bars"></div></div>
      </div>
      <!-- Live stream -->
      <div class="panel">
        <div class="ph">
          <div class="ph-title">Live Feed</div>
          <div class="ph-right">
            <span id="stream-badge" style="font-size:10px;font-family:var(--mono);color:var(--t3)">0 findings</span>
          </div>
        </div>
        <div class="pb"><div class="stream" id="stream">
          <div class="empty"><div class="empty-icon">📡</div><div class="empty-text">No data yet.<br><br><button class="nav-btn" style="margin:0 auto" onclick="loadSample()">⚡ Load sample data</button><br><span style="font-size:11px;color:var(--t4)">or drop .jsonl trace files into traces/</span></div></div>
        </div></div>
      </div>
    </div>

  </div>

  <!-- ══ FINDINGS ══════════════════════════════════════════════════════════ -->
  <div class="tab-pane" id="tab-findings">

    <div class="sec-label">Top Findings & Actionable Fixes</div>
    <div class="panel" style="margin-bottom:16px">
      <div class="tbl">
        <table>
          <thead><tr>
            <th>Detector</th><th>Severity</th><th>Waste/Day</th>
            <th>What it catches</th><th>Fix</th>
          </tr></thead>
          <tbody id="findings-tbody">
            <tr><td colspan="5" style="text-align:center;padding:32px;color:var(--t3)">Loading…</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="sec-label">Recent Findings Detail</div>
    <div class="panel">
      <div class="pb" style="padding-top:14px">
        <div class="stream" style="max-height:500px" id="findings-stream">
          <div class="empty"><div class="empty-icon">🔍</div><div class="empty-text">No findings yet.</div></div>
        </div>
      </div>
    </div>

  </div>

  <!-- ══ DATASETS ══════════════════════════════════════════════════════════ -->
  <div class="tab-pane" id="tab-datasets">

    <div class="sec-label">Available Datasets</div>
    <p style="font-size:13px;color:var(--t2);margin-bottom:18px;line-height:1.6">
      Download real-world agent trace data from
      <a href="https://huggingface.co/datasets/Blueconomy/monk-benchmarks" target="_blank" style="color:var(--or);text-decoration:none">Blueconomy/monk-benchmarks ↗</a>.
      Files are saved to your <span style="font-family:var(--mono);color:var(--t1)">traces/</span> folder and analyzed automatically within 30 seconds.
    </p>
    <div class="ds-grid" id="ds-grid">
      <div class="empty" style="grid-column:1/-1"><div class="empty-icon">⏳</div><div class="empty-text">Loading…</div></div>
    </div>

  </div>

</div>

<footer>
  <div class="left">🕵️ monk v__VERSION__ &nbsp;·&nbsp; Blueconomy AI &nbsp;·&nbsp; Techstars '25 &nbsp;·&nbsp; MIT</div>
  <div class="right">
    <a href="/findings">findings JSON</a>
    <a href="/metrics">prometheus</a>
    <a href="/datasets">datasets API</a>
    <a href="https://github.com/Blueconomy/monk" target="_blank">GitHub ↗</a>
  </div>
</footer>

<script>
// ── Charts ──────────────────────────────────────────────────────────────────
Chart.defaults.color='#555';Chart.defaults.font.family="Inter,-apple-system,sans-serif";Chart.defaults.font.size=11;
let detChart=null,sevChart=null;

// ── Helpers ──────────────────────────────────────────────────────────────────
const fmt=n=>n>=1e6?(n/1e6).toFixed(1)+'M':n>=1000?(n/1000).toFixed(1)+'k':String(n??'—');
const fmtUSD=v=>v>=1000?'$'+(v/1000).toFixed(1)+'k':'$'+Math.round(v);

const FIXES={
  agent_loop:{what:'Agent cycling A→B→A→B with no progress',fix:'Add a visited-state set; break when (tool,args) recurs within a window.'},
  retry_loop:{what:'Same tool called 3+ consecutive times',fix:'Result-cache keyed on (tool,args). Eliminate recomputation.'},
  empty_return:{what:'Tool returns null → agent retries anyway',fix:'Guard: if result is empty, inject "no data found" before next LLM call.'},
  context_bloat:{what:'System prompt >55% of context budget',fix:'Compress system prompt; sliding-window or summarize every N turns.'},
  token_bloat:{what:'Single tool injected 5× the session median',fix:'Truncate tool outputs to 1,000–2,000 tokens before injecting into context.'},
  error_cascade:{what:'Tool fails silently → 6–8 wasted downstream calls',fix:'Guard every tool call — if error, short-circuit before next LLM call.'},
  cross_turn_memory:{what:'Same tool+args re-fetched across turns',fix:'Session-level result cache with 1-turn TTL.'},
  model_overkill:{what:'Flagship model doing formatting/classification',fix:'Route sub-tasks to gpt-4o-mini/haiku; reserve Opus for reasoning.'},
  text_io:{what:'Low output compression or unbounded input growth',fix:'Truncate inputs; validate task clarity; rolling context window.'},
  latency_spike:{what:'Single call outlier vs session median',fix:'Timeout + retry with exponential backoff.'},
  output_format:{what:'Model violates format rules in system prompt',fix:'Format validator after each LLM call; retry on violation.'},
  plan_execution:{what:'Model planned steps then never executed them',fix:'Assert each planned step is followed by a matching tool call.'},
  span_consistency:{what:'Model asserts facts with no supporting tool call',fix:'Require a grounding tool call before asserting external facts.'},
  tool_dependency:{what:'Cycles or deep chains in tool call graph',fix:'Flatten tool dependencies; eliminate circular tool calls.'},
};

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(id){
  document.querySelectorAll('.tab').forEach((t,i)=>{
    const ids=['overview','findings','datasets'];
    t.className='tab'+(ids[i]===id?' active':'');
  });
  document.querySelectorAll('.tab-pane').forEach(p=>{
    p.className='tab-pane'+(p.id==='tab-'+id?' active':'');
  });
  if(id==='datasets')loadDatasets();
}

// ── Findings data ─────────────────────────────────────────────────────────────
async function loadFindings(){
  try{
    const r=await fetch('/findings',{signal:AbortSignal.timeout(3000)});
    if(!r.ok)throw new Error(r.status);
    const d=await r.json();
    renderFindings(d);
    setLive(true);
    document.getElementById('ts').textContent=new Date().toLocaleTimeString();
  }catch(e){setLive(false)}
}

function setLive(ok){
  const b=document.getElementById('live-badge');
  const t=document.getElementById('badge-text');
  if(ok){b.className='live';t.textContent='live'}
  else{b.className='error';t.textContent='disconnected'}
}

function renderFindings(d){
  const s=d.summary;
  document.getElementById('k-total').textContent=fmt(s.total_findings);
  document.getElementById('k-total-sub').textContent=`${fmt(s.high)} high · ${fmt(s.medium)} med · ${s.low} low`;
  document.getElementById('k-day').textContent=fmtUSD(s.waste_usd_per_day);
  document.getElementById('k-day-sub').textContent='$'+s.waste_usd_per_day.toFixed(2)+'/day estimated';
  document.getElementById('k-month').textContent=fmtUSD(s.waste_usd_per_month);
  document.getElementById('k-calls').textContent=fmt(s.calls_analyzed);
  document.getElementById('k-calls-sub').textContent=fmt(s.sessions_analyzed)+' sessions';
  if(s.calls_analyzed>0)document.getElementById('nav-files').textContent=fmt(s.calls_analyzed)+' calls';
  document.getElementById('s-h').textContent=fmt(s.high);
  document.getElementById('s-m').textContent=fmt(s.medium);
  document.getElementById('s-l').textContent=fmt(s.low);

  // Detector chart
  const dets=Object.keys(d.by_detector);
  const counts={};
  (d.recent_findings||[]).forEach(f=>counts[f.detector]=(counts[f.detector]||0)+1);
  const vals=dets.map(k=>counts[k]||Math.ceil((d.by_detector[k].waste_per_day||0)*10));
  const bg=vals.map(v=>v>500?'#f97316cc':v>100?'#f9731680':'#f9731640');
  if(detChart){detChart.data.labels=dets;detChart.data.datasets[0].data=vals;detChart.data.datasets[0].backgroundColor=bg;detChart.update('active');}
  else detChart=new Chart(document.getElementById('det-chart'),{type:'bar',data:{labels:dets,datasets:[{data:vals,backgroundColor:bg,borderRadius:3,borderSkipped:false}]},options:{indexAxis:'y',responsive:true,plugins:{legend:{display:false},tooltip:{backgroundColor:'#161616',borderColor:'#1f1f1f',borderWidth:1,callbacks:{label:c=>'  '+c.parsed.x+' findings'}}},scales:{x:{grid:{color:'#1a1a1a'},ticks:{callback:v=>v>=1000?(v/1000).toFixed(0)+'k':v}},y:{grid:{display:false},ticks:{font:{family:'JetBrains Mono,monospace',size:10},color:'#666'}}}}});

  // Severity donut
  if(sevChart){sevChart.data.datasets[0].data=[s.high,s.medium,s.low];sevChart.update('active');}
  else sevChart=new Chart(document.getElementById('sev-chart'),{type:'doughnut',data:{labels:['High','Medium','Low'],datasets:[{data:[s.high,s.medium,s.low],backgroundColor:['#ff444488','#f9731688','#22c55e88'],borderColor:['#ff4444','#f97316','#22c55e'],borderWidth:1.5,hoverOffset:4}]},options:{responsive:true,cutout:'68%',plugins:{legend:{position:'bottom',labels:{color:'#555',padding:10,usePointStyle:true,pointStyleWidth:8}},tooltip:{backgroundColor:'#161616',borderColor:'#1f1f1f',borderWidth:1}}}});

  // Waste bars
  const maxW=Math.max(...Object.values(d.by_detector).map(v=>v.waste_per_day||0),0.01);
  document.getElementById('waste-bars').innerHTML=Object.entries(d.by_detector)
    .sort((a,b)=>(b[1].waste_per_day||0)-(a[1].waste_per_day||0))
    .filter(([,v])=>(v.waste_per_day||0)>0)
    .map(([n,v])=>{
      const pct=((v.waste_per_day/maxW)*100).toFixed(1);
      return`<div class="wrow"><div class="wrow-top"><span class="wrow-name">${n}</span><span class="wrow-val${v.waste_per_day<1?' dim':''}">${'$'+v.waste_per_day.toFixed(2)}</span></div><div class="wtrack"><div class="wfill" style="width:${pct}%"></div></div></div>`;
    }).join('')||'<div style="color:var(--t3);font-size:12px">No waste data yet.</div>';

  // Stream (overview)
  const recent=(d.recent_findings||[]).slice(-30).reverse();
  document.getElementById('stream-badge').textContent=recent.length+' findings';
  const streamHTML=recent.map(f=>`<div class="stream-item ${f.severity}">
    <div class="si-top"><span class="si-det">${f.detector}</span><span class="si-sev">${f.severity}</span></div>
    <div class="si-title">${f.title||''}</div>
    ${f.fix?`<div class="si-fix">${f.fix.slice(0,120)}${f.fix.length>120?'…':''}</div>`:''}
  </div>`).join('');
  document.getElementById('stream').innerHTML=streamHTML||'<div class="empty"><div class="empty-icon">📡</div><div class="empty-text">No findings yet.<br>Drop trace files into traces/ or go to Datasets to download.</div></div>';

  // Findings tab — table
  const rows=Object.entries(d.by_detector)
    .sort((a,b)=>(b[1].waste_per_day||0)-(a[1].waste_per_day||0))
    .slice(0,12)
    .map(([name,v])=>{
      const fx=FIXES[name]||{what:'—',fix:'—'};
      const w=v.waste_per_day||0;
      const wStr=w>0?`<span class="mn" style="color:${w>5?'var(--or)':'var(--t2)'}">${'$'+w.toFixed(2)}</span>`:`<span class="mn" style="color:var(--t3)">—</span>`;
      const sev=w>5?`<div class="sev-badge sev-h"><div class="dot"></div>High</div>`:`<div class="sev-badge sev-m"><div class="dot"></div>Medium</div>`;
      return`<tr><td><span class="chip ${w>1?'chip-or':'chip-w'}">${name}</span></td><td>${sev}</td><td>${wStr}</td><td style="color:var(--t3);font-size:11px;max-width:160px">${fx.what}</td><td style="font-size:11px;color:var(--t3);max-width:240px;line-height:1.5">${fx.fix}</td></tr>`;
    }).join('');
  document.getElementById('findings-tbody').innerHTML=rows||'<tr><td colspan="5" style="text-align:center;padding:28px;color:var(--t3)">No findings yet.</td></tr>';

  // Findings tab — detail stream
  document.getElementById('findings-stream').innerHTML=recent.map(f=>`<div class="stream-item ${f.severity}">
    <div class="si-top"><span class="si-det">${f.detector}</span><span class="si-sev">${f.severity}</span></div>
    <div class="si-title">${f.title||''}</div>
    <div style="font-size:11px;color:var(--t3);margin-top:4px;line-height:1.5">${f.detail||''}</div>
    ${f.fix?`<div class="si-fix">${f.fix}</div>`:''}
    ${(f.sessions||[]).length?`<div style="font-family:var(--mono);font-size:10px;color:var(--t4);margin-top:5px">sessions: ${f.sessions.slice(0,2).join(', ')}</div>`:''}
  </div>`).join('')||'<div class="empty"><div class="empty-icon">🔍</div><div class="empty-text">No findings yet.</div></div>';
}

// ── Datasets tab ──────────────────────────────────────────────────────────────
async function loadDatasets(){
  try{
    const r=await fetch('/datasets',{signal:AbortSignal.timeout(4000)});
    const datasets=await r.json();
    renderDatasets(datasets);
  }catch(e){
    document.getElementById('ds-grid').innerHTML='<div class="empty" style="grid-column:1/-1"><div class="empty-icon">⚠️</div><div class="empty-text">Could not load dataset list.</div></div>';
  }
}

function renderDatasets(datasets){
  const maxSize=Math.max(...datasets.map(d=>d.size_mb));
  document.getElementById('ds-grid').innerHTML=datasets.map(ds=>{
    const loaded=ds.loaded;
    const status=ds.status;
    const isDownloading=status==='downloading';
    const isError=status==='error';
    const pct=loaded?100:0;
    const btnText=isDownloading?'Downloading…':loaded?'✓ Loaded':'↓ Download';
    const cardClass='ds-card'+(loaded?' loaded':'');
    const statusClass='ds-status '+status;
    const statusLabel=isDownloading?'downloading…':loaded?'loaded':isError?'error':'available';
    const barPct=((ds.size_mb/maxSize)*100).toFixed(0);
    return`<div class="${cardClass}" id="ds-${ds.id}">
      <div class="ds-top">
        <div class="ds-name">${ds.label}</div>
        <div class="${statusClass}">${statusLabel}</div>
      </div>
      <div class="ds-desc">${ds.desc}</div>
      <div class="ds-meta">
        <span class="ds-tag">${ds.calls} calls</span>
        <span class="ds-tag ${ds.format==='OTEL'?'otel':''}">${ds.format}</span>
        <span class="ds-tag">${ds.size_mb}MB</span>
      </div>
      <div class="ds-bar-wrap">
        <div class="ds-bar-label"><span>size</span><span>${ds.size_mb}MB</span></div>
        <div class="ds-bar-track"><div class="ds-bar-fill" style="width:${barPct}%"></div></div>
      </div>
      <button class="ds-btn${loaded?' loaded':''}" onclick="downloadDataset('${ds.id}')" ${isDownloading?'disabled':''}>${btnText}</button>
    </div>`;
  }).join('');
}

async function downloadDataset(id){
  // Optimistically update UI
  const card=document.getElementById('ds-'+id);
  if(!card)return;
  card.querySelector('.ds-btn').disabled=true;
  card.querySelector('.ds-btn').textContent='Downloading…';
  card.querySelector('.ds-status').className='ds-status downloading';
  card.querySelector('.ds-status').textContent='downloading…';

  try{
    const r=await fetch('/download',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:id})
    });
    const result=await r.json();
    if(!result.ok){
      card.querySelector('.ds-btn').textContent='Error — retry';
      card.querySelector('.ds-btn').disabled=false;
      card.querySelector('.ds-status').className='ds-status error';
      card.querySelector('.ds-status').textContent='error';
    }else{
      // Poll until done
      pollDataset(id);
    }
  }catch(e){
    card.querySelector('.ds-btn').textContent='Error — retry';
    card.querySelector('.ds-btn').disabled=false;
  }
}

function pollDataset(id,attempts=0){
  if(attempts>60)return; // 5 min timeout
  setTimeout(async()=>{
    try{
      const r=await fetch('/datasets');
      const datasets=await r.json();
      const ds=datasets.find(d=>d.id===id);
      if(ds&&ds.status==='ready'&&ds.loaded){
        renderDatasets(datasets);
        // Also refresh findings since new data was loaded
        setTimeout(loadFindings,2000);
      }else if(ds&&ds.status==='downloading'){
        pollDataset(id,attempts+1);
      }else{
        loadDatasets();
      }
    }catch(e){pollDataset(id,attempts+1)}
  },5000);
}

function loadAll(){loadFindings();if(document.getElementById('tab-datasets').classList.contains('active'))loadDatasets()}

async function loadSample(){
  try{
    await fetch('/load-sample',{method:'POST'});
    setTimeout(loadFindings,1500);
    setTimeout(loadFindings,4000);
  }catch(e){console.warn('load-sample failed',e)}
}

// ── Boot ──────────────────────────────────────────────────────────────────────
loadFindings();
setInterval(loadFindings,15000);
</script>
</body>
</html>"""


class _Handler(BaseHTTPRequestHandler):
    store: MetricsStore
    dlmgr: "DownloadManager"
    traces_dir: Path
    rescan_fn: "callable"
    version: str = __version__

    def do_GET(self):
        if self.path in ("/", "/dashboard"):
            html = DASHBOARD_HTML.replace("__VERSION__", self.version).encode()
            self._respond(200, "text/html; charset=utf-8", html)
        elif self.path == "/findings":
            self._respond(200, "application/json", self.store.to_json().encode())
        elif self.path == "/metrics":
            self._respond(200, "text/plain; version=0.0.4", self.store.to_prometheus().encode())
        elif self.path == "/datasets":
            body = json.dumps(self.dlmgr.status()).encode()
            self._respond(200, "application/json", body)
        elif self.path == "/health":
            self._respond(200, "text/plain", b"ok")
        else:
            self._respond(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path == "/download":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length)) if length else {}
            name   = body.get("name", "")
            ok     = self.dlmgr.start_download(name)
            self._respond(200, "application/json", json.dumps({"ok": ok, "name": name}).encode())
        elif self.path == "/scan":
            # Trigger immediate rescan in background
            threading.Thread(target=self.rescan_fn, daemon=True).start()
            self._respond(200, "application/json", b'{"ok":true}')
        elif self.path == "/load-sample":
            # Write built-in demo traces to traces dir and trigger rescan
            try:
                dest = self.traces_dir / "demo_traces.jsonl"
                if not dest.exists():
                    dest.write_text(DEMO_TRACES_JSONL)
                threading.Thread(target=self.rescan_fn, daemon=True).start()
                self._respond(200, "application/json", json.dumps({"ok": True, "file": str(dest)}).encode())
            except Exception as e:
                self._respond(500, "application/json", json.dumps({"ok": False, "error": str(e)}).encode())
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _scan_all(path: Path) -> tuple:
    from monk.parsers.otel import is_otel_format, parse_spans, spans_to_trace_calls

    fresh = MetricsStore()
    files = list(path.rglob("*.jsonl")) if path.is_dir() else [path]
    total_calls = total_findings = 0

    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            if is_otel_format(text):
                roots    = parse_spans(text)
                calls    = spans_to_trace_calls(roots)
                findings = []
                for det in ALL_DETECTORS:
                    findings.extend(det.run_spans(roots) if det.requires_spans else det.run(calls))
            else:
                calls    = parse_traces(str(f))
                findings = []
                for det in ALL_DETECTORS:
                    if not det.requires_spans:
                        findings.extend(det.run(calls))

            sessions = len({c.session_id for c in calls})
            fresh.update(findings, calls, sessions)
            total_calls    += len(calls)
            total_findings += len(findings)
        except Exception as e:
            print(f"[monk] error scanning {f.name}: {e}")

    return fresh, total_calls, total_findings, len(files)


def serve(path: str, port: int = 9090, interval: int = 30) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    store = MetricsStore()
    dlmgr = DownloadManager(target)
    _lock = threading.Lock()

    # ── Initial scan ─────────────────────────────────────────────────────────
    fresh, calls, findings, nfiles = _scan_all(target)
    with _lock:
        store.__dict__.update(fresh.__dict__)

    print(f"")
    print(f"  🕵️  monk serve  v{_Handler.version}")
    print(f"")
    if calls:
        print(f"  initial scan   {nfiles} file(s) · {calls:,} calls · {findings:,} findings")
    else:
        print(f"  no trace files yet — download datasets from the dashboard")
    print(f"")
    print(f"  dashboard  →  http://localhost:{port}/")
    print(f"  findings   →  http://localhost:{port}/findings")
    print(f"  metrics    →  http://localhost:{port}/metrics")
    print(f"  watching   →  {target}  (every {interval}s)")
    print(f"")

    # ── Rescan helper (also called by /scan and /load-sample endpoints) ─────
    def _rescan():
        try:
            fresh, calls, findings, nfiles = _scan_all(target)
            with _lock:
                store.__dict__.update(fresh.__dict__)
            if calls:
                ts = time.strftime("%H:%M:%S")
                print(f"[monk] {ts}  {nfiles} file(s) · {calls:,} calls · {findings:,} findings")
        except Exception as e:
            print(f"[monk] rescan error: {e}")

    # ── Background watcher ───────────────────────────────────────────────────
    def _watch():
        while True:
            time.sleep(interval)
            _rescan()

    threading.Thread(target=_watch, daemon=True).start()

    class Handler(_Handler):
        pass
    Handler.store      = store
    Handler.dlmgr      = dlmgr
    Handler.traces_dir = target
    Handler.rescan_fn  = staticmethod(_rescan)
    Handler.version    = __version__

    httpd = HTTPServer(("0.0.0.0", port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[monk] stopped.")
