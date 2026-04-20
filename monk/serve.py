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
<title>monk — Agent Cost Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f8f8f6;--s1:#fff;--s2:#fafaf8;--s3:#f4f3f0;
  --b1:#e8e6e1;--b2:#d4d0c8;--b3:#c0bbb0;
  --t1:#1a1814;--t2:#4a4540;--t3:#8a8078;--t4:#b0a898;
  --or:#f97316;--or2:#fff7ed;--or3:#fed7aa;--or4:#fdba74;
  --red:#dc2626;--red2:#fef2f2;--grn:#16a34a;--grn2:#f0fdf4;--blue:#2563eb;
  --mono:'JetBrains Mono',monospace;
  --radius:10px;--shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.05);
  --shadow-md:0 4px 6px rgba(0,0,0,.07),0 2px 4px rgba(0,0,0,.05);
}
html{font-size:14px}
body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--t1);-webkit-font-smoothing:antialiased}

/* NAV */
nav{position:sticky;top:0;z-index:100;background:rgba(255,255,255,.96);backdrop-filter:blur(20px);border-bottom:1px solid var(--b1);height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 32px;box-shadow:0 1px 0 var(--b1)}
.nav-l{display:flex;align-items:center;gap:16px}
.logo{font-size:16px;font-weight:800;letter-spacing:-.5px;display:flex;align-items:center;gap:9px;color:var(--t1)}
.logo-k{color:var(--or)}
.version-pill{font-size:10px;font-weight:600;padding:3px 8px;border-radius:99px;background:var(--or2);color:var(--or);border:1px solid var(--or3);font-family:var(--mono)}
.nav-r{display:flex;align-items:center;gap:10px}
#live-badge{display:flex;align-items:center;gap:5px;font-size:11px;padding:4px 10px;border-radius:99px;font-family:var(--mono);font-weight:600;transition:all .3s;background:var(--s3);color:var(--t3);border:1px solid var(--b1)}
#live-badge.live{color:var(--grn);background:var(--grn2);border-color:#bbf7d0}
#live-badge.error{color:var(--red);background:var(--red2);border-color:#fecaca}
.badge-dot{width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0}
#live-badge.live .badge-dot{animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.ts{font-size:11px;color:var(--t4);font-family:var(--mono)}
.nav-sep{width:1px;height:22px;background:var(--b1)}
.btn{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:500;padding:6px 14px;border-radius:7px;border:1px solid var(--b2);background:var(--s1);color:var(--t2);cursor:pointer;transition:all .15s;font-family:inherit}
.btn:hover{border-color:var(--or4);color:var(--or);background:var(--or2)}
.btn-primary{background:var(--or);color:#fff;border-color:var(--or);font-weight:600}
.btn-primary:hover{background:#ea6c0a;border-color:#ea6c0a;color:#fff}
#nav-calls{font-size:11px;color:var(--t3);font-family:var(--mono)}

/* TABS */
.tabs{display:flex;background:var(--s1);border-bottom:1px solid var(--b1);padding:0 32px;gap:0}
.tab{font-size:13px;font-weight:500;padding:12px 18px;cursor:pointer;color:var(--t3);border-bottom:2px solid transparent;transition:all .15s;user-select:none}
.tab:hover{color:var(--t1)}
.tab.active{color:var(--or);border-bottom-color:var(--or);font-weight:600}

/* PAGE */
.page{max-width:1400px;margin:0 auto;padding:28px 32px 80px}
.tab-pane{display:none}.tab-pane.active{display:block}

/* SECTION HEADER */
.sh{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.sh-title{font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:var(--t3)}

/* KPI ROW */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
.kpi{background:var(--s1);border:1px solid var(--b1);border-radius:var(--radius);padding:20px 22px;box-shadow:var(--shadow);transition:box-shadow .2s,transform .2s;cursor:default;position:relative;overflow:hidden}
.kpi:hover{box-shadow:var(--shadow-md);transform:translateY(-1px)}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--or);opacity:0;transition:opacity .2s}
.kpi:hover::before{opacity:1}
.kpi-label{font-size:11px;font-weight:600;color:var(--t3);margin-bottom:10px;letter-spacing:.3px}
.kpi-val{font-size:32px;font-weight:800;letter-spacing:-1.2px;line-height:1;color:var(--t1);margin-bottom:6px}
.kpi-val.orange{color:var(--or)}
.kpi-val.red{color:var(--red)}
.kpi-sub{font-size:11px;color:var(--t4);line-height:1.5}
.kpi.highlight{border-color:var(--or3);background:var(--or2)}
.kpi.highlight .kpi-label{color:var(--or)}
.kpi.highlight::before{opacity:1}

/* SEV ROW */
.sev-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
.sev-card{border-radius:var(--radius);padding:14px 16px;display:flex;align-items:center;gap:12px;border:1px solid}
.sev-card.high{background:#fef2f2;border-color:#fecaca}
.sev-card.med{background:#fff7ed;border-color:var(--or3)}
.sev-card.low{background:#f0fdf4;border-color:#bbf7d0}
.sev-num{font-size:28px;font-weight:800;letter-spacing:-1px;line-height:1}
.sev-card.high .sev-num{color:var(--red)}
.sev-card.med .sev-num{color:var(--or)}
.sev-card.low .sev-num{color:var(--grn)}
.sev-lbl{font-size:11px;font-weight:600;color:var(--t3);letter-spacing:.3px}
.sev-desc{font-size:10px;color:var(--t4);margin-top:2px}

/* GRID LAYOUTS */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.g31{display:grid;grid-template-columns:3fr 1fr;gap:16px;margin-bottom:16px}
.g13{display:grid;grid-template-columns:1fr 3fr;gap:16px;margin-bottom:16px}

/* CARD */
.card{background:var(--s1);border:1px solid var(--b1);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow)}
.card-head{padding:14px 18px 12px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--b1)}
.card-title{font-size:12px;font-weight:600;color:var(--t1);letter-spacing:.2px}
.card-sub{font-size:11px;color:var(--t3);font-family:var(--mono)}
.card-body{padding:16px 18px}
.card-body.flush{padding:0}

/* FINDINGS LIST */
.f-list{display:flex;flex-direction:column;gap:8px;max-height:380px;overflow-y:auto}
.f-list::-webkit-scrollbar{width:4px}
.f-list::-webkit-scrollbar-thumb{background:var(--b2);border-radius:2px}
.f-item{padding:12px 14px;border-radius:8px;border:1px solid var(--b1);background:var(--s2);border-left:3px solid var(--b2);transition:background .12s,box-shadow .12s;cursor:default}
.f-item:hover{background:var(--s1);box-shadow:var(--shadow)}
.f-item.high{border-left-color:var(--red)}
.f-item.medium{border-left-color:var(--or)}
.f-item.low{border-left-color:var(--grn)}
.f-top{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.f-det{font-family:var(--mono);font-size:10px;font-weight:600;padding:2px 6px;border-radius:4px;background:var(--or2);color:var(--or);border:1px solid var(--or3)}
.f-sev{font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;letter-spacing:.3px}
.f-sev.high{background:#fef2f2;color:var(--red);border:1px solid #fecaca}
.f-sev.medium{background:#fff7ed;color:var(--or);border:1px solid var(--or3)}
.f-sev.low{background:#f0fdf4;color:var(--grn);border:1px solid #bbf7d0}
.f-waste{margin-left:auto;font-family:var(--mono);font-size:11px;font-weight:700;color:var(--or)}
.f-title{font-size:12px;color:var(--t1);font-weight:500;margin-bottom:3px;line-height:1.45}
.f-fix{font-size:11px;color:var(--t3);line-height:1.45}
.f-fix::before{content:'Fix → ';color:var(--or);font-weight:600}
.f-detail{font-size:11px;color:var(--t3);margin-top:4px;line-height:1.5}
.f-sessions{font-family:var(--mono);font-size:10px;color:var(--t4);margin-top:5px}

/* WASTE BARS */
.w-list{display:flex;flex-direction:column;gap:12px}
.w-row-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}
.w-name{font-size:12px;color:var(--t2);font-weight:500}
.w-val{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--or)}
.w-val.dim{color:var(--t4)}
.w-track{height:6px;background:var(--b1);border-radius:3px;overflow:hidden}
.w-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--or4),var(--or));transition:width .7s ease}

/* TABLE */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse}
thead th{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--t3);padding:10px 16px;text-align:left;border-bottom:2px solid var(--b1);background:var(--s2)}
thead th:first-child{padding-left:20px}
thead th:last-child{padding-right:20px}
tbody tr{border-bottom:1px solid var(--b1);transition:background .1s}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--or2)}
tbody td{padding:12px 16px;font-size:12px;color:var(--t2);vertical-align:middle}
tbody td:first-child{padding-left:20px}
tbody td:last-child{padding-right:20px}
.chip{display:inline-block;font-family:var(--mono);font-size:10px;font-weight:700;padding:3px 8px;border-radius:5px;background:var(--or2);color:var(--or);border:1px solid var(--or3)}
.sev-dot{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700}
.dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.sev-h{color:var(--red)}.sev-h .dot{background:var(--red)}
.sev-m{color:var(--or)}.sev-m .dot{background:var(--or)}
.sev-l{color:var(--grn)}.sev-l .dot{background:var(--grn)}
.mn{font-family:var(--mono);font-size:11px;font-weight:600}

/* DATASET CARDS */
.ds-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.ds-card{background:var(--s1);border:1px solid var(--b1);border-radius:var(--radius);padding:16px 18px;transition:all .15s;box-shadow:var(--shadow)}
.ds-card:hover{box-shadow:var(--shadow-md);transform:translateY(-1px)}
.ds-card.loaded{border-color:var(--or3);background:var(--or2)}
.ds-top{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:8px}
.ds-name{font-size:13px;font-weight:700;color:var(--t1)}
.ds-badge{font-size:10px;font-weight:700;padding:2px 8px;border-radius:99px;font-family:var(--mono)}
.ds-badge.loaded{background:#dcfce7;color:var(--grn);border:1px solid #bbf7d0}
.ds-badge.available{background:var(--s3);color:var(--t3);border:1px solid var(--b1)}
.ds-badge.downloading{background:#fff7ed;color:var(--or);border:1px solid var(--or3);animation:blink 1s infinite}
.ds-badge.error{background:#fef2f2;color:var(--red);border:1px solid #fecaca}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.5}}
.ds-desc{font-size:12px;color:var(--t3);margin-bottom:10px;line-height:1.5}
.ds-meta{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}
.ds-tag{font-size:10px;font-family:var(--mono);padding:2px 7px;border-radius:4px;color:var(--t3);background:var(--s3);border:1px solid var(--b1)}
.ds-tag.otel{color:var(--blue);background:#eff6ff;border-color:#bfdbfe}
.ds-btn{width:100%;padding:8px 12px;border-radius:7px;border:1px solid var(--b2);background:var(--s1);color:var(--t2);font-size:12px;font-weight:500;cursor:pointer;transition:all .15s;font-family:inherit}
.ds-btn:hover:not(:disabled){border-color:var(--or);color:var(--or);background:var(--or2)}
.ds-btn:disabled{opacity:.45;cursor:not-allowed}
.ds-btn.loaded{border-color:#bbf7d0;color:var(--grn);background:#dcfce7}

/* EMPTY STATE */
.empty{text-align:center;padding:48px 24px;color:var(--t3)}
.empty-icon{font-size:36px;margin-bottom:14px}
.empty-title{font-size:14px;font-weight:600;color:var(--t2);margin-bottom:6px}
.empty-body{font-size:12px;color:var(--t3);line-height:1.6;margin-bottom:18px}

/* FOOTER */
footer{border-top:1px solid var(--b1);padding:18px 32px;display:flex;align-items:center;justify-content:space-between;background:var(--s1)}
footer .left{font-size:11px;color:var(--t3)}
footer .right{display:flex;gap:18px}
footer a{font-size:11px;color:var(--t3);text-decoration:none;transition:color .15s}
footer a:hover{color:var(--or)}

/* SCAN BANNER */
#scan-banner{display:none;align-items:center;gap:12px;padding:10px 32px;background:var(--or2);border-bottom:1px solid var(--or3)}
#scan-banner.show{display:flex}
.scan-text{font-size:12px;color:var(--or);font-weight:500}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="nav-l">
    <div class="logo">🕵️ mon<span class="logo-k">k</span></div>
    <span class="version-pill">v__VERSION__</span>
  </div>
  <div class="nav-r">
    <span class="ts" id="ts"></span>
    <span id="nav-calls"></span>
    <div class="nav-sep"></div>
    <button class="btn" onclick="triggerScan()">↻ Scan now</button>
    <button class="btn btn-primary" onclick="loadSample()" id="sample-btn">⚡ Load sample</button>
    <div id="live-badge"><div class="badge-dot"></div><span id="badge-text">connecting</span></div>
  </div>
</nav>

<!-- SCAN BANNER -->
<div id="scan-banner">
  <span class="scan-text">⚡ Scanning traces folder…</span>
</div>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('overview')">Overview</div>
  <div class="tab" onclick="switchTab('findings')">Findings</div>
  <div class="tab" onclick="switchTab('datasets')">Datasets</div>
</div>

<div class="page">

  <!-- ══ OVERVIEW ══════════════════════════════════════════════════════════ -->
  <div class="tab-pane active" id="tab-overview">

    <!-- KPI CARDS -->
    <div class="kpi-row">
      <div class="kpi highlight">
        <div class="kpi-label">Waste / Day</div>
        <div class="kpi-val orange" id="k-day">—</div>
        <div class="kpi-sub" id="k-day-sub">estimated avoidable spend</div>
      </div>
      <div class="kpi highlight">
        <div class="kpi-label">Projected / Month</div>
        <div class="kpi-val orange" id="k-month">—</div>
        <div class="kpi-sub">if patterns continue unchanged</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Total Findings</div>
        <div class="kpi-val" id="k-total">—</div>
        <div class="kpi-sub" id="k-total-sub">across all detectors</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">LLM Calls Analyzed</div>
        <div class="kpi-val" id="k-calls">—</div>
        <div class="kpi-sub" id="k-calls-sub">sessions analyzed</div>
      </div>
    </div>

    <!-- SEVERITY ROW -->
    <div class="sev-row">
      <div class="sev-card high">
        <div>
          <div class="sev-num" id="s-h">—</div>
        </div>
        <div>
          <div class="sev-lbl">High Severity</div>
          <div class="sev-desc">Fix immediately</div>
        </div>
      </div>
      <div class="sev-card med">
        <div>
          <div class="sev-num" id="s-m">—</div>
        </div>
        <div>
          <div class="sev-lbl">Medium Severity</div>
          <div class="sev-desc">Address this sprint</div>
        </div>
      </div>
      <div class="sev-card low">
        <div>
          <div class="sev-num" id="s-l">—</div>
        </div>
        <div>
          <div class="sev-lbl">Low Severity</div>
          <div class="sev-desc">Backlog candidates</div>
        </div>
      </div>
    </div>

    <!-- CHARTS + FINDINGS FEED -->
    <div class="g31">
      <div class="card">
        <div class="card-head">
          <span class="card-title">Cost Waste by Detector</span>
          <span class="card-sub" id="waste-total-label"></span>
        </div>
        <div class="card-body">
          <div class="w-list" id="waste-bars"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><span class="card-title">Severity</span></div>
        <div class="card-body" style="display:flex;align-items:center;justify-content:center">
          <canvas id="sev-chart" style="max-height:200px"></canvas>
        </div>
      </div>
    </div>

    <!-- RECENT FINDINGS FEED -->
    <div class="card">
      <div class="card-head">
        <span class="card-title">Recent Findings</span>
        <span class="card-sub" id="stream-count"></span>
      </div>
      <div class="card-body flush">
        <div id="stream" style="padding:12px 16px">
          <div class="empty">
            <div class="empty-icon">📡</div>
            <div class="empty-title">No data yet</div>
            <div class="empty-body">Drop .jsonl trace files into your traces/ folder,<br>or load the built-in sample to see monk in action.</div>
            <button class="btn btn-primary" onclick="loadSample()">⚡ Load sample data</button>
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- ══ FINDINGS ══════════════════════════════════════════════════════════ -->
  <div class="tab-pane" id="tab-findings">

    <div class="sh" style="margin-bottom:16px">
      <span class="sh-title">Actionable Findings</span>
      <button class="btn" onclick="triggerScan()">↻ Re-scan</button>
    </div>

    <div class="card" style="margin-bottom:20px">
      <div class="card-body flush">
        <div class="tbl-wrap">
          <table>
            <thead><tr>
              <th>Detector</th><th>Severity</th><th>Waste / Day</th>
              <th>What it catches</th><th>Recommended fix</th>
            </tr></thead>
            <tbody id="findings-tbody">
              <tr><td colspan="5" style="text-align:center;padding:36px;color:var(--t3)">No findings yet — load data first.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="sh"><span class="sh-title">Finding Details</span></div>
    <div class="card">
      <div class="card-body">
        <div class="f-list" style="max-height:520px" id="findings-stream">
          <div class="empty"><div class="empty-icon">🔍</div><div class="empty-title">No findings yet</div></div>
        </div>
      </div>
    </div>

  </div>

  <!-- ══ DATASETS ══════════════════════════════════════════════════════════ -->
  <div class="tab-pane" id="tab-datasets">

    <div class="sh">
      <span class="sh-title">Available Datasets</span>
      <a href="https://huggingface.co/datasets/Blueconomy/monk-benchmarks" target="_blank" style="font-size:12px;color:var(--or);text-decoration:none;font-weight:500">Blueconomy/monk-benchmarks ↗</a>
    </div>
    <p style="font-size:13px;color:var(--t3);margin-bottom:20px;line-height:1.6">
      Download real-world agent traces. Files land in your <code style="font-family:var(--mono);background:var(--s3);padding:1px 5px;border-radius:4px;font-size:11px">traces/</code> folder and are picked up automatically within 30 seconds.
    </p>
    <div class="ds-grid" id="ds-grid">
      <div class="empty" style="grid-column:1/-1"><div class="empty-icon">⏳</div><div class="empty-title">Loading…</div></div>
    </div>

  </div>

</div>

<footer>
  <div class="left">🕵️ monk v__VERSION__ &nbsp;·&nbsp; Blueconomy AI &nbsp;·&nbsp; Techstars '25 &nbsp;·&nbsp; MIT</div>
  <div class="right">
    <a href="/findings">findings JSON</a>
    <a href="/metrics">prometheus</a>
    <a href="https://github.com/Blueconomy/monk" target="_blank">GitHub ↗</a>
  </div>
</footer>

<script>
Chart.defaults.color='#8a8078';
Chart.defaults.font.family="Inter,-apple-system,sans-serif";
Chart.defaults.font.size=11;
let sevChart=null;

const fmt=n=>n>=1e6?(n/1e6).toFixed(1)+'M':n>=1000?(n/1000).toFixed(1)+'k':String(n??'—');
const fmtUSD=v=>v===0?'$0':v<0.01?'<$0.01':v>=1000?'$'+(v/1000).toFixed(1)+'k':'$'+v.toFixed(2);

const FIXES={
  agent_loop:{what:'Agent cycles A→B→A→B with no progress',fix:'Add visited-state guard; break when (tool,args) recurs.'},
  retry_loop:{what:'Same tool called 3+ consecutive times',fix:'Result-cache on (tool,args); eliminate recomputation.'},
  empty_return:{what:'Tool returns null/empty → agent retries',fix:'Inject "no data found" message; skip next LLM call.'},
  context_bloat:{what:'System prompt > 55% of context budget',fix:'Compress prompt; use sliding-window summarization.'},
  token_bloat:{what:'Tool output 5× session median tokens',fix:'Truncate tool outputs to 1–2k tokens before injection.'},
  error_cascade:{what:'Tool error ignored → wasted downstream calls',fix:'Short-circuit on tool error before next LLM call.'},
  cross_turn_memory:{what:'Same tool+args re-fetched across turns',fix:'Session-level result cache with 1-turn TTL.'},
  model_overkill:{what:'Flagship model on trivial tasks',fix:'Route formatting/classification to mini models.'},
  text_io:{what:'Low output compression / unbounded input',fix:'Truncate inputs; rolling context window.'},
  latency_spike:{what:'Single call outlier vs session median',fix:'Add timeout + retry with exponential backoff.'},
  output_format:{what:'Model violates format rules in system prompt',fix:'Validate format after each call; retry on violation.'},
  plan_execution:{what:'Model planned steps then never executed',fix:'Assert each planned step has a matching tool call.'},
  span_consistency:{what:'Model asserts facts with no tool grounding',fix:'Require tool call before asserting external facts.'},
  tool_dependency:{what:'Cycles/deep chains in tool call graph',fix:'Flatten dependencies; eliminate circular calls.'},
};

function switchTab(id){
  document.querySelectorAll('.tab').forEach((t,i)=>{
    t.className='tab'+((['overview','findings','datasets'])[i]===id?' active':'');
  });
  document.querySelectorAll('.tab-pane').forEach(p=>{
    p.className='tab-pane'+(p.id==='tab-'+id?' active':'');
  });
  if(id==='datasets')loadDatasets();
}

async function loadFindings(){
  try{
    const r=await fetch('/findings',{signal:AbortSignal.timeout(4000)});
    if(!r.ok)throw new Error(r.status);
    const d=await r.json();
    renderFindings(d);
    setLive(true);
    document.getElementById('ts').textContent='Updated '+new Date().toLocaleTimeString();
  }catch(e){setLive(false)}
}

function setLive(ok){
  const b=document.getElementById('live-badge');
  const t=document.getElementById('badge-text');
  b.className=ok?'live':'error';
  t.textContent=ok?'live':'offline';
}

function renderFindings(d){
  const s=d.summary;
  // KPIs
  document.getElementById('k-day').textContent=fmtUSD(s.waste_usd_per_day);
  document.getElementById('k-day-sub').textContent='$'+s.waste_usd_per_day.toFixed(4)+' per day';
  document.getElementById('k-month').textContent=fmtUSD(s.waste_usd_per_month);
  document.getElementById('k-total').textContent=fmt(s.total_findings);
  document.getElementById('k-total-sub').textContent=`${s.high||0} high · ${s.medium||0} med · ${s.low||0} low`;
  document.getElementById('k-calls').textContent=fmt(s.calls_analyzed);
  document.getElementById('k-calls-sub').textContent=fmt(s.sessions_analyzed)+' sessions';
  // Nav
  if(s.calls_analyzed>0){
    document.getElementById('nav-calls').textContent=fmt(s.calls_analyzed)+' calls · '+fmt(s.sessions_analyzed)+' sessions';
    document.getElementById('sample-btn').style.display='none';
  }
  // Severity
  document.getElementById('s-h').textContent=fmt(s.high||0);
  document.getElementById('s-m').textContent=fmt(s.medium||0);
  document.getElementById('s-l').textContent=fmt(s.low||0);

  // Donut chart
  if(sevChart){sevChart.data.datasets[0].data=[s.high||0,s.medium||0,s.low||0];sevChart.update();}
  else{
    const ctx=document.getElementById('sev-chart');
    if(ctx)sevChart=new Chart(ctx,{type:'doughnut',data:{labels:['High','Medium','Low'],datasets:[{data:[s.high||0,s.medium||0,s.low||0],backgroundColor:['#dc2626','#f97316','#16a34a'],borderColor:['#fff','#fff','#fff'],borderWidth:3,hoverOffset:6}]},options:{responsive:true,cutout:'72%',plugins:{legend:{position:'bottom',labels:{color:'#4a4540',padding:12,usePointStyle:true,pointStyleWidth:8,font:{size:11}}},tooltip:{backgroundColor:'#1a1814',callbacks:{label:c=>'  '+c.parsed+' findings'}}}}});
  }

  // Waste bars (improved — thicker, gradient)
  const dets=Object.entries(d.by_detector).sort((a,b)=>(b[1].waste_per_day||0)-(a[1].waste_per_day||0)).filter(([,v])=>(v.waste_per_day||0)>0);
  const maxW=Math.max(...dets.map(([,v])=>v.waste_per_day||0),0.001);
  const totalW=dets.reduce((s,[,v])=>s+(v.waste_per_day||0),0);
  document.getElementById('waste-total-label').textContent=totalW>0?fmtUSD(totalW)+'/day total':'';
  document.getElementById('waste-bars').innerHTML=dets.slice(0,8).map(([n,v])=>{
    const pct=((v.waste_per_day/maxW)*100).toFixed(1);
    return`<div><div class="w-row-top"><span class="w-name">${n}</span><span class="w-val">${fmtUSD(v.waste_per_day)}</span></div><div class="w-track"><div class="w-fill" style="width:${pct}%"></div></div></div>`;
  }).join('')||'<div style="color:var(--t4);font-size:12px;padding:8px 0">No waste data yet — run analysis on trace files.</div>';

  // Recent findings feed
  const recent=(d.recent_findings||[]).slice(-25).reverse();
  document.getElementById('stream-count').textContent=recent.length?recent.length+' findings':'';
  document.getElementById('stream').innerHTML=recent.length?
    `<div class="f-list">`+recent.map(f=>`<div class="f-item ${f.severity}">
      <div class="f-top"><span class="f-det">${f.detector}</span><span class="f-sev ${f.severity}">${f.severity}</span>${f.waste_per_day>0?`<span class="f-waste">${fmtUSD(f.waste_per_day)}/day</span>`:''}</div>
      <div class="f-title">${f.title||''}</div>
      ${f.fix?`<div class="f-fix">${f.fix.slice(0,140)}${f.fix.length>140?'…':''}</div>`:''}
    </div>`).join('')+'</div>'
    :`<div class="empty"><div class="empty-icon">📡</div><div class="empty-title">No data yet</div><div class="empty-body">Drop .jsonl trace files into traces/ or load sample data.</div><button class="btn btn-primary" onclick="loadSample()">⚡ Load sample data</button></div>`;

  // Findings tab — table
  const rows=Object.entries(d.by_detector).sort((a,b)=>(b[1].waste_per_day||0)-(a[1].waste_per_day||0)).slice(0,12).map(([name,v])=>{
    const fx=FIXES[name]||{what:'—',fix:'—'};
    const w=v.waste_per_day||0;
    const sc=w>5?'sev-h':w>1?'sev-m':'sev-l';
    const sl=w>5?'High':w>1?'Medium':'Low';
    return`<tr><td><span class="chip">${name}</span></td>
      <td><span class="sev-dot ${sc}"><span class="dot"></span>${sl}</span></td>
      <td><span class="mn" style="color:${w>1?'var(--or)':'var(--t3)'}">${w>0?fmtUSD(w):'—'}</span></td>
      <td style="font-size:11px;color:var(--t3);max-width:200px">${fx.what}</td>
      <td style="font-size:11px;color:var(--t3);max-width:260px;line-height:1.5">${fx.fix}</td></tr>`;
  }).join('');
  document.getElementById('findings-tbody').innerHTML=rows||'<tr><td colspan="5" style="text-align:center;padding:36px;color:var(--t3)">No findings — drop traces into your folder.</td></tr>';

  // Findings tab — detail list
  document.getElementById('findings-stream').innerHTML=recent.length?
    recent.map(f=>`<div class="f-item ${f.severity}">
      <div class="f-top"><span class="f-det">${f.detector}</span><span class="f-sev ${f.severity}">${f.severity}</span>${f.waste_per_day>0?`<span class="f-waste">${fmtUSD(f.waste_per_day)}/day</span>`:''}</div>
      <div class="f-title">${f.title||''}</div>
      ${f.detail?`<div class="f-detail">${f.detail}</div>`:''}
      ${f.fix?`<div class="f-fix">${f.fix}</div>`:''}
      ${(f.sessions||[]).length?`<div class="f-sessions">sessions: ${f.sessions.slice(0,3).join(', ')}</div>`:''}
    </div>`).join('')
    :'<div class="empty"><div class="empty-icon">🔍</div><div class="empty-title">No findings yet</div></div>';
}

async function loadDatasets(){
  try{
    const r=await fetch('/datasets',{signal:AbortSignal.timeout(5000)});
    renderDatasets(await r.json());
  }catch(e){
    document.getElementById('ds-grid').innerHTML='<div class="empty" style="grid-column:1/-1"><div class="empty-icon">⚠️</div><div class="empty-title">Could not load datasets</div></div>';
  }
}

function renderDatasets(datasets){
  document.getElementById('ds-grid').innerHTML=datasets.map(ds=>{
    const loaded=ds.loaded,st=ds.status,dling=st==='downloading',err=st==='error';
    const btnTxt=dling?'Downloading…':loaded?'✓ Already loaded':'↓ Download';
    return`<div class="ds-card${loaded?' loaded':''}" id="ds-${ds.id}">
      <div class="ds-top">
        <div class="ds-name">${ds.label}</div>
        <span class="ds-badge ${st}">${dling?'downloading…':loaded?'loaded':err?'error':'available'}</span>
      </div>
      <div class="ds-desc">${ds.desc}</div>
      <div class="ds-meta">
        <span class="ds-tag">${ds.calls}</span>
        <span class="ds-tag${ds.format==='OTEL'?' otel':''}">${ds.format}</span>
        <span class="ds-tag">${ds.size_mb} MB</span>
      </div>
      <button class="ds-btn${loaded?' loaded':''}" onclick="downloadDataset('${ds.id}')" ${dling?'disabled':''}>${btnTxt}</button>
    </div>`;
  }).join('');
}

async function downloadDataset(id){
  const card=document.getElementById('ds-'+id);
  if(!card)return;
  card.querySelector('.ds-btn').disabled=true;
  card.querySelector('.ds-btn').textContent='Downloading…';
  card.querySelector('.ds-badge').className='ds-badge downloading';
  card.querySelector('.ds-badge').textContent='downloading…';
  try{
    const r=await fetch('/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:id})});
    if((await r.json()).ok)pollDataset(id);
    else{card.querySelector('.ds-btn').textContent='Error — retry';card.querySelector('.ds-btn').disabled=false;}
  }catch(e){card.querySelector('.ds-btn').textContent='Error — retry';card.querySelector('.ds-btn').disabled=false;}
}

function pollDataset(id,n=0){
  if(n>60)return;
  setTimeout(async()=>{
    try{
      const datasets=await(await fetch('/datasets')).json();
      const ds=datasets.find(d=>d.id===id);
      if(ds?.status==='ready'&&ds.loaded){renderDatasets(datasets);setTimeout(loadFindings,2000);}
      else if(ds?.status==='downloading')pollDataset(id,n+1);
      else loadDatasets();
    }catch(e){pollDataset(id,n+1);}
  },5000);
}

async function triggerScan(){
  document.getElementById('scan-banner').className='show';
  try{await fetch('/scan',{method:'POST'});}catch(e){}
  setTimeout(()=>{document.getElementById('scan-banner').className='';loadFindings();},2500);
}

async function loadSample(){
  document.getElementById('scan-banner').className='show';
  try{await fetch('/load-sample',{method:'POST'});}catch(e){}
  setTimeout(()=>{document.getElementById('scan-banner').className='';loadFindings();},2000);
  setTimeout(loadFindings,5000);
}

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
