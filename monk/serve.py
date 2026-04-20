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

/* SIMULATE TAB */
.sim-bar{display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.sim-main{display:grid;grid-template-columns:196px 1fr 196px;gap:12px;height:400px;margin-bottom:16px}
.sim-sidebar{display:flex;flex-direction:column;gap:10px}
.sim-sec{background:var(--s1);border:1px solid var(--b1);border-radius:var(--radius);padding:12px 14px;box-shadow:var(--shadow)}
.sim-add{width:100%;padding:8px 10px;border-radius:7px;border:1.5px solid;font-size:11px;font-weight:600;cursor:pointer;text-align:left;margin-bottom:6px;font-family:inherit;transition:all .12s}
.sim-add.llm{border-color:#fed7aa;color:#c2410c;background:#fff7ed}.sim-add.llm:hover{background:#fed7aa}
.sim-add.tool{border-color:#bfdbfe;color:#1d4ed8;background:#eff6ff}.sim-add.tool:hover{background:#bfdbfe}
#sim-wrap{position:relative;border:1.5px solid var(--b1);border-radius:var(--radius);overflow:hidden;background:var(--bg);background-image:radial-gradient(var(--b1) 1px,transparent 1px);background-size:22px 22px}
#sim-svg{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible;z-index:1}
#sim-canvas{position:absolute;top:0;left:0;width:100%;height:100%;z-index:2}
.sim-empty-cv{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none}
.sim-run{background:var(--s1);border:1px solid var(--b1);border-radius:var(--radius);padding:14px;box-shadow:var(--shadow);display:flex;flex-direction:column}
.cfg-f{margin-bottom:10px}
.cfg-lbl{display:block;font-size:10px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--t3);margin-bottom:4px}
.cfg-inp{width:100%;padding:5px 8px;border-radius:6px;border:1px solid var(--b2);background:var(--s1);font-size:12px;color:var(--t1);font-family:inherit;box-sizing:border-box}
.cfg-inp:focus{outline:none;border-color:var(--or3)}
input[type=range].cfg-inp{padding:2px 0;accent-color:var(--or)}
/* nodes */
.sn{position:absolute;width:160px;height:52px;border-radius:9px;border:1.5px solid;display:flex;align-items:center;user-select:none;z-index:10;box-shadow:var(--shadow);transition:box-shadow .12s}
.sn:hover{box-shadow:var(--shadow-md)}
.sn.sel{outline:2px solid var(--or);outline-offset:2px}
.sn.llm{background:#fff7ed;border-color:#fed7aa}
.sn.tool{background:#eff6ff;border-color:#bfdbfe}
.sn-body{flex:1;display:flex;align-items:center;gap:7px;padding:0 8px;min-width:0;cursor:grab}
.sn-body:active{cursor:grabbing}
.sn-icon{font-size:15px;flex-shrink:0}
.sn-lbl{font-size:11px;font-weight:600;color:var(--t1);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sn-port{width:13px;height:13px;border-radius:50%;flex-shrink:0;cursor:crosshair;transition:transform .1s;z-index:11}
.sn-port:hover{transform:scale(1.4)}
.sn-in{background:var(--b2);border:2px solid var(--b3)}
.sn-out{background:var(--or);border:2px solid #c2410c}
.sn.conn-src .sn-out{background:#16a34a;border-color:#14532d;animation:pulse 1s infinite}
#sim-status{font-size:11px;color:var(--or);font-weight:500;min-height:16px;margin-bottom:8px}
.sim-res-item{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--b1);font-size:12px}
.sim-res-item:last-child{border-bottom:none}
.sim-res-lbl{color:var(--t3)}
.sim-res-val{font-weight:700;font-family:var(--mono);color:var(--t1)}
.sim-res-val.or{color:var(--or)}
.sim-res-val.red{color:var(--red)}

/* SCAN BANNER */
#scan-banner{display:none;align-items:center;gap:12px;padding:10px 32px;background:var(--or2);border-bottom:1px solid var(--or3)}
#scan-banner.show{display:flex}
.scan-text{font-size:12px;color:var(--or);font-weight:500}

/* USAGE TAB */
.drop-zone{border:2px dashed var(--b2);border-radius:12px;padding:56px 24px;text-align:center;cursor:pointer;transition:all .2s;background:var(--s1)}
.drop-zone:hover,.drop-zone.drag-over{border-color:var(--or);background:var(--or2)}
.drop-zone-icon{font-size:42px;margin-bottom:14px}
.drop-zone-title{font-size:15px;font-weight:700;color:var(--t1);margin-bottom:6px}
.drop-zone-sub{font-size:12px;color:var(--t3);line-height:1.6}
.drop-zone .btn{margin-top:16px}
#u-report{display:none}
.u-kpi-row{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px}
.u-kpi{background:var(--s1);border:1px solid var(--b1);border-radius:var(--radius);padding:16px 18px;box-shadow:var(--shadow)}
.u-kpi-label{font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;color:var(--t3);margin-bottom:8px}
.u-kpi-val{font-size:26px;font-weight:800;letter-spacing:-1px;color:var(--t1);line-height:1}
.u-kpi-val.orange{color:var(--or)}
.u-kpi-val.red{color:var(--red)}
.u-kpi-val.green{color:var(--grn)}
.u-kpi-sub{font-size:10px;color:var(--t4);margin-top:4px}
.u-kpi.savings{border-color:var(--or3);background:var(--or2)}
.u-kpi.savings .u-kpi-label{color:var(--or)}
.u-users{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-bottom:20px}
.u-card{background:var(--s1);border:1px solid var(--b1);border-radius:var(--radius);padding:16px 18px;box-shadow:var(--shadow)}
.u-card-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.u-card-label{font-size:13px;font-weight:700;color:var(--t1)}
.u-card-cost{font-family:var(--mono);font-size:14px;font-weight:800;color:var(--or)}
.u-card-rows{display:flex;flex-direction:column;gap:7px}
.u-card-row{display:flex;justify-content:space-between;align-items:center;font-size:12px}
.u-card-row-label{color:var(--t3)}
.u-card-row-val{font-family:var(--mono);font-weight:600;color:var(--t1)}
.u-bar-wrap{height:4px;background:var(--b1);border-radius:2px;margin-top:6px;overflow:hidden}
.u-bar-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--or4),var(--or))}
.u-finding{padding:14px 16px;border-radius:8px;border:1px solid var(--b1);background:var(--s2);border-left:4px solid var(--b2);margin-bottom:8px;transition:background .12s}
.u-finding:hover{background:var(--s1)}
.u-finding.high{border-left-color:var(--red)}
.u-finding.medium{border-left-color:var(--or)}
.u-finding.low{border-left-color:var(--grn)}
.u-f-top{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.u-f-pattern{font-family:var(--mono);font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;background:var(--or2);color:var(--or);border:1px solid var(--or3)}
.u-f-sev{font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;letter-spacing:.3px}
.u-f-sev.high{background:#fef2f2;color:var(--red);border:1px solid #fecaca}
.u-f-sev.medium{background:#fff7ed;color:var(--or);border:1px solid var(--or3)}
.u-f-sev.low{background:#f0fdf4;color:var(--grn);border:1px solid #bbf7d0}
.u-f-savings{margin-left:auto;font-family:var(--mono);font-size:12px;font-weight:800;color:var(--grn)}
.u-f-title{font-size:13px;font-weight:600;color:var(--t1);margin-bottom:4px}
.u-f-detail{font-size:12px;color:var(--t2);margin-bottom:6px;line-height:1.5}
.u-f-fix{font-size:11px;color:var(--t3);line-height:1.5}
.u-f-fix::before{content:'Fix → ';color:var(--or);font-weight:700}
.u-models{display:flex;flex-direction:column;gap:10px}
.u-model-row{display:flex;align-items:center;gap:12px}
.u-model-name{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--t2);width:200px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.u-model-bar-wrap{flex:1;height:8px;background:var(--b1);border-radius:4px;overflow:hidden}
.u-model-bar{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--or4),var(--or))}
.u-model-cost{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--or);width:60px;text-align:right}
.u-cs{background:linear-gradient(135deg,#1a1814 0%,#2a2520 100%);border-radius:12px;padding:24px 28px;margin-bottom:20px;color:#f8f8f6}
.u-cs-title{font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#f97316;margin-bottom:4px}
.u-cs-sub{font-size:18px;font-weight:800;color:#fff;margin-bottom:14px;letter-spacing:-.3px}
.u-cs-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:16px}
.u-cs-stat{text-align:center}
.u-cs-stat-val{font-size:22px;font-weight:800;color:#f97316;letter-spacing:-1px;line-height:1}
.u-cs-stat-label{font-size:10px;color:#b0a898;margin-top:3px}
.u-cs-body{font-size:12px;color:#c0bbb0;line-height:1.7}
.u-cs-body strong{color:#f8f8f6}
.u-platform-badge{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;padding:4px 10px;border-radius:99px;background:var(--or2);color:var(--or);border:1px solid var(--or3);font-family:var(--mono);margin-bottom:12px}
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
  <div class="tab" onclick="switchTab('simulate')">Simulate ✦</div>
  <div class="tab" onclick="switchTab('usage')">Usage ◈</div>
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

  <!-- ══ SIMULATE ══════════════════════════════════════════════════════════ -->
  <div class="tab-pane" id="tab-simulate">

    <div class="sim-bar">
      <span class="sh-title" style="margin-right:4px">PRESETS</span>
      <button class="btn" onclick="simLoad('retry_loop')">🔄 Retry Loop</button>
      <button class="btn" onclick="simLoad('agent_loop')">🔁 Agent Loop</button>
      <button class="btn" onclick="simLoad('empty_return')">🕳 Empty Return</button>
      <button class="btn" onclick="simLoad('context_bloat')">📈 Context Bloat</button>
      <button class="btn" onclick="simLoad('healthy')">✅ Healthy</button>
      <button class="btn" onclick="simLoad('supervisor')">🏢 Supervisor</button>
      <button class="btn" onclick="simLoad('swarm')">🐝 Swarm</button>
      <button class="btn" style="margin-left:auto" onclick="simClear()">✕ Clear</button>
    </div>

    <div id="sim-status"></div>

    <div class="sim-main">
      <!-- left sidebar -->
      <div class="sim-sidebar">
        <div class="sim-sec">
          <div class="cfg-lbl" style="margin-bottom:8px">ADD NODE</div>
          <button class="sim-add llm" onclick="simAddNode('llm')">🤖 LLM Call</button>
          <button class="sim-add tool" onclick="simAddNode('tool')">🔧 Tool Call</button>
          <div style="height:1px;background:var(--b1);margin:8px 0"></div>
          <div style="font-size:10px;color:var(--t4);line-height:1.5">
            Click <span style="color:var(--or);font-weight:600">●</span> right port<br>
            then a left port to connect nodes
          </div>
        </div>
        <div class="sim-sec" style="flex:1;overflow-y:auto">
          <div id="sim-cfg">
            <div style="text-align:center;padding:16px 0">
              <div style="font-size:22px">👆</div>
              <div style="font-size:11px;color:var(--t3);margin-top:6px">Click a node to configure</div>
            </div>
          </div>
        </div>
      </div>

      <!-- graph canvas -->
      <div id="sim-wrap">
        <svg id="sim-svg"><defs>
          <marker id="ah-or" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#f97316"/></marker>
          <marker id="ah-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#dc2626"/></marker>
        </defs></svg>
        <div id="sim-canvas">
          <div class="sim-empty-cv" id="sim-empty">
            <div style="font-size:36px">🕸️</div>
            <div style="font-size:13px;font-weight:600;color:var(--t2);margin-top:10px">Build your workflow</div>
            <div style="font-size:11px;color:var(--t3);margin-top:4px">Load a preset above or add nodes from the left</div>
          </div>
        </div>
      </div>

      <!-- run panel -->
      <div class="sim-run">
        <div class="cfg-lbl" style="margin-bottom:12px">RUN CONFIG</div>
        <div class="cfg-f">
          <label class="cfg-lbl">Sessions</label>
          <input type="number" id="sim-sessions" value="5" min="1" max="30" class="cfg-inp">
        </div>
        <div class="cfg-f">
          <label class="cfg-lbl">Default model</label>
          <select id="sim-model" class="cfg-inp">
            <option value="gpt-4o">gpt-4o</option>
            <option value="gpt-4o-mini">gpt-4o-mini</option>
            <option value="claude-sonnet-4-6">claude-sonnet-4-6</option>
            <option value="claude-opus-4-6">claude-opus-4-6</option>
          </select>
        </div>
        <button class="btn btn-primary" id="sim-run-btn"
          style="width:100%;padding:10px;font-size:13px;margin-top:8px" onclick="simRun()">
          ▶ Run Simulation
        </button>
        <div id="sim-res" style="display:none;margin-top:14px">
          <div style="height:1px;background:var(--b1);margin-bottom:12px"></div>
          <div class="cfg-lbl" style="margin-bottom:8px">RESULTS</div>
          <div id="sim-res-body"></div>
          <button class="btn" style="width:100%;margin-top:10px;font-size:11px"
            onclick="switchTab('overview');setTimeout(loadFindings,500)">
            View in Overview ↑
          </button>
        </div>
      </div>
    </div>

  </div>

  <!-- ══ USAGE ════════════════════════════════════════════════════════════ -->
  <div class="tab-pane" id="tab-usage">

    <!-- DROP ZONE -->
    <div id="u-drop-wrap">
      <div class="drop-zone" id="u-drop"
        ondragover="event.preventDefault();this.classList.add('drag-over')"
        ondragleave="this.classList.remove('drag-over')"
        ondrop="uDrop(event)">
        <div class="drop-zone-icon">📊</div>
        <div class="drop-zone-title">Drop your team billing CSV here</div>
        <div class="drop-zone-sub">
          Supported: <strong>Claude.ai team usage export</strong> · Cursor billing export · OpenAI usage export<br>
          All user data is anonymised in the browser — nothing is sent to any external server.
        </div>
        <button class="btn btn-primary" style="margin-top:18px" onclick="document.getElementById('u-file-input').click()">
          📂 Choose CSV file
        </button>
        <input type="file" id="u-file-input" accept=".csv" style="display:none" onchange="uFileChosen(this)">
      </div>
      <p style="font-size:11px;color:var(--t3);text-align:center;margin-top:12px">
        Need sample data? <a href="#" style="color:var(--or)" onclick="uLoadDemo();return false">Load anonymised case study →</a>
      </p>
    </div>

    <!-- REPORT (hidden until file uploaded) -->
    <div id="u-report">

      <!-- platform badge + re-upload -->
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <div id="u-platform-badge" class="u-platform-badge"></div>
        <button class="btn" onclick="uReset()">↑ Upload different file</button>
      </div>

      <!-- KPI ROW -->
      <div class="u-kpi-row">
        <div class="u-kpi">
          <div class="u-kpi-label">Period</div>
          <div class="u-kpi-val" id="u-period" style="font-size:16px;letter-spacing:0">—</div>
          <div class="u-kpi-sub" id="u-period-sub"></div>
        </div>
        <div class="u-kpi">
          <div class="u-kpi-label">Total Spend</div>
          <div class="u-kpi-val orange" id="u-total-cost">—</div>
          <div class="u-kpi-sub" id="u-total-calls"></div>
        </div>
        <div class="u-kpi">
          <div class="u-kpi-label">Monthly Run Rate</div>
          <div class="u-kpi-val orange" id="u-monthly">—</div>
          <div class="u-kpi-sub">if usage continues unchanged</div>
        </div>
        <div class="u-kpi">
          <div class="u-kpi-label">Total Tokens</div>
          <div class="u-kpi-val" id="u-tokens">—</div>
          <div class="u-kpi-sub" id="u-users-count"></div>
        </div>
        <div class="u-kpi savings">
          <div class="u-kpi-label">Savings Potential / mo</div>
          <div class="u-kpi-val green" id="u-savings">—</div>
          <div class="u-kpi-sub" id="u-savings-pct"></div>
        </div>
      </div>

      <!-- CASE STUDY PANEL -->
      <div id="u-case-study" class="u-cs" style="display:none">
        <div class="u-cs-title">Real-World Case Study · Blueconomy AI Internal</div>
        <div class="u-cs-sub">69 days of Claude.ai team usage — anonymised benchmark</div>
        <div class="u-cs-grid">
          <div class="u-cs-stat"><div class="u-cs-stat-val">$2,117</div><div class="u-cs-stat-label">Total Spend</div></div>
          <div class="u-cs-stat"><div class="u-cs-stat-val">$920</div><div class="u-cs-stat-label">Monthly Run Rate</div></div>
          <div class="u-cs-stat"><div class="u-cs-stat-val">93%</div><div class="u-cs-stat-label">Cache-Read Tokens</div></div>
          <div class="u-cs-stat"><div class="u-cs-stat-val">$873/mo</div><div class="u-cs-stat-label">Savings Identified</div></div>
        </div>
        <div class="u-cs-body">
          <strong>Finding:</strong> One power user (User A) drove 95% of all spend. Conversations were never reset — cache tokens compounded over weeks.
          Worst single call: $47.05. <strong>Fix applied:</strong> Added session-reset policy + context window cap.
          <strong>Result:</strong> Projected ~$10,400/yr savings with zero capability loss.
        </div>
      </div>

      <!-- LAYOUT: findings left, models right -->
      <div class="g2" style="margin-bottom:20px">

        <!-- FINDINGS -->
        <div>
          <div class="sh" style="margin-bottom:12px">
            <span class="sh-title">Patterns Detected</span>
            <span class="card-sub" id="u-findings-count"></span>
          </div>
          <div id="u-findings-list">
            <div class="empty"><div class="empty-icon">✅</div><div class="empty-title">No issues found</div></div>
          </div>
        </div>

        <!-- RIGHT COLUMN: models + users -->
        <div>
          <div class="sh" style="margin-bottom:12px"><span class="sh-title">Spend by Model</span></div>
          <div class="card" style="margin-bottom:16px">
            <div class="card-body">
              <div class="u-models" id="u-models-list"></div>
            </div>
          </div>

          <div class="sh" style="margin-bottom:12px"><span class="sh-title">Users (anonymised)</span></div>
          <div class="u-users" id="u-users-list"></div>
        </div>

      </div>

      <!-- RECOMMENDATIONS FOOTER -->
      <div class="card" id="u-recs-card" style="display:none">
        <div class="card-head">
          <span class="card-title">🗺️ Recommended Workflow</span>
          <span class="card-sub">Based on detected patterns</span>
        </div>
        <div class="card-body" id="u-recs-body" style="font-size:12px;line-height:1.8;color:var(--t2)"></div>
      </div>

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
    t.className='tab'+((['overview','findings','datasets','simulate','usage'])[i]===id?' active':'');
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

// ── SIMULATE TAB ──────────────────────────────────────────────────────────
const NW=160,NH=52;
let G={nodes:[],edges:[]};
let simSel=null,simConn=null,simDrag=null,simNC=0;
const SIM_PRESETS={
  retry_loop:{name:'Retry Loop',desc:'Same tool called 4× — timeout pattern',
    nodes:[
      {id:'n1',type:'llm',label:'LLM Call',x:80,y:130,config:{model:'gpt-4o',tokens_in:2100,tokens_out:95,latency_ms:800}},
      {id:'n2',type:'tool',label:'database_query',x:310,y:130,config:{tool_name:'database_query',failure:'timeout',failure_rate:0.9,failure_result:'Connection timeout',success_result:'[{id:1}]',latency_ms:1200,max_retries:4}},
    ],
    edges:[{src:'n1',dst:'n2'},{src:'n2',dst:'n2'}]},
  agent_loop:{name:'Agent Loop',desc:'search→rank→search→rank with no progress',
    nodes:[
      {id:'n1',type:'llm',label:'LLM Plan',x:60,y:150,config:{model:'gpt-4o',tokens_in:1900,tokens_out:140,latency_ms:700}},
      {id:'n2',type:'tool',label:'search_docs',x:270,y:60,config:{tool_name:'search_documents',failure:'none',failure_rate:0,success_result:'results',latency_ms:400}},
      {id:'n3',type:'llm',label:'LLM Rank',x:480,y:60,config:{model:'gpt-4o',tokens_in:2100,tokens_out:120,latency_ms:700}},
      {id:'n4',type:'tool',label:'rank_results',x:480,y:230,config:{tool_name:'rank_results',failure:'none',failure_rate:0,success_result:'ranked',latency_ms:300}},
    ],
    edges:[{src:'n1',dst:'n2'},{src:'n2',dst:'n3'},{src:'n3',dst:'n4'},{src:'n4',dst:'n1'}]},
  empty_return:{name:'Empty Return',desc:'Tool returns empty 80% — agent retries',
    nodes:[
      {id:'n1',type:'llm',label:'LLM Call',x:80,y:130,config:{model:'gpt-4o',tokens_in:1800,tokens_out:110,latency_ms:600}},
      {id:'n2',type:'tool',label:'web_search',x:310,y:130,config:{tool_name:'web_search',failure:'empty',failure_rate:0.8,failure_result:'',success_result:'Search results...',latency_ms:600,max_retries:5}},
    ],
    edges:[{src:'n1',dst:'n2'},{src:'n2',dst:'n2'}]},
  context_bloat:{name:'Context Bloat',desc:'System prompt > 65% of token budget',
    nodes:[
      {id:'n1',type:'llm',label:'LLM (bloated)',x:80,y:130,config:{model:'gpt-4o',tokens_in:14000,tokens_out:200,system_tokens:9100,latency_ms:1800}},
      {id:'n2',type:'tool',label:'lookup',x:310,y:130,config:{tool_name:'lookup',failure:'none',failure_rate:0,success_result:'data',latency_ms:300}},
    ],
    edges:[{src:'n1',dst:'n2'},{src:'n2',dst:'n1'}]},
  healthy:{name:'Healthy Agent',desc:'Clean agent — expect zero findings',
    nodes:[
      {id:'n1',type:'llm',label:'LLM Call',x:80,y:130,config:{model:'gpt-4o-mini',tokens_in:900,tokens_out:250,latency_ms:400}},
      {id:'n2',type:'tool',label:'fetch_data',x:310,y:70,config:{tool_name:'fetch_data',failure:'none',failure_rate:0,success_result:'data',latency_ms:200}},
      {id:'n3',type:'tool',label:'write_output',x:310,y:200,config:{tool_name:'write_output',failure:'none',failure_rate:0,success_result:'written',latency_ms:150}},
    ],
    edges:[{src:'n1',dst:'n2'},{src:'n1',dst:'n3'}]},
  supervisor:{name:'Supervisor (Model Overkill)',desc:'gpt-4o routes to gpt-4o-mini specialist — expensive routing',
    nodes:[
      {id:'n1',type:'llm',label:'Supervisor LLM',x:60,y:140,config:{model:'gpt-4o',tokens_in:1200,tokens_out:60,latency_ms:700}},
      {id:'n2',type:'tool',label:'transfer_to_specialist',x:260,y:140,config:{tool_name:'transfer_to_specialist',failure:'none',failure_rate:0,success_result:'Transferred',latency_ms:50}},
      {id:'n3',type:'llm',label:'Specialist LLM',x:440,y:140,config:{model:'gpt-4o-mini',tokens_in:800,tokens_out:200,latency_ms:400}},
      {id:'n4',type:'tool',label:'compute',x:600,y:70,config:{tool_name:'compute',failure:'none',failure_rate:0,success_result:'result',latency_ms:180}},
      {id:'n5',type:'tool',label:'transfer_back_to_supervisor',x:600,y:210,config:{tool_name:'transfer_back_to_supervisor',failure:'none',failure_rate:0,success_result:'Back',latency_ms:50}},
    ],
    edges:[{src:'n1',dst:'n2'},{src:'n2',dst:'n3'},{src:'n3',dst:'n4'},{src:'n3',dst:'n5'}]},
  swarm:{name:'Swarm (Handoff Loop)',desc:'Peer agents bouncing back and forth without resolving',
    nodes:[
      {id:'n1',type:'llm',label:'agent_A',x:60,y:140,config:{model:'gpt-4o',tokens_in:1400,tokens_out:80,latency_ms:600}},
      {id:'n2',type:'tool',label:'transfer_to_agent_B',x:250,y:140,config:{tool_name:'transfer_to_agent_B',failure:'none',failure_rate:0,success_result:'Transferred',latency_ms:40}},
      {id:'n3',type:'llm',label:'agent_B',x:430,y:140,config:{model:'gpt-4o',tokens_in:1600,tokens_out:90,latency_ms:650}},
      {id:'n4',type:'tool',label:'partial_work',x:580,y:60,config:{tool_name:'partial_work',failure:'empty',failure_rate:0.7,failure_result:'',success_result:'partial result',latency_ms:300}},
      {id:'n5',type:'tool',label:'transfer_back_to_agent_A',x:430,y:240,config:{tool_name:'transfer_back_to_agent_A',failure:'none',failure_rate:0,success_result:'Back',latency_ms:40}},
    ],
    edges:[{src:'n1',dst:'n2'},{src:'n2',dst:'n3'},{src:'n3',dst:'n4'},{src:'n3',dst:'n5'},{src:'n5',dst:'n1'}]},
};

function simLoad(name){
  const p=SIM_PRESETS[name];if(!p)return;
  G=JSON.parse(JSON.stringify({nodes:p.nodes,edges:p.edges}));
  simNC=G.nodes.length;simSel=null;simConn=null;
  document.getElementById('sim-status').textContent='Loaded: '+p.name+' — '+p.desc;
  simRender();
}
function simClear(){G={nodes:[],edges:[]};simSel=null;simConn=null;simNC=0;simRender();document.getElementById('sim-status').textContent='';}
function simAddNode(type){
  const id='n'+(++simNC);
  G.nodes.push({id,type,label:type==='llm'?'LLM Call':'tool_call',
    x:120+Math.random()*180,y:80+Math.random()*140,
    config:type==='llm'?{model:'gpt-4o',tokens_in:1200,tokens_out:100,latency_ms:600}
      :{tool_name:'my_tool',failure:'none',failure_rate:0,success_result:'data',latency_ms:400}});
  simSel=id;simRender();simShowCfg(id);
}
function simDelNode(nid){
  G.nodes=G.nodes.filter(n=>n.id!==nid);
  G.edges=G.edges.filter(e=>e.src!==nid&&e.dst!==nid);
  simSel=null;simRender();document.getElementById('sim-cfg').innerHTML='<div style="text-align:center;padding:16px 0"><div style="font-size:22px">👆</div><div style="font-size:11px;color:var(--t3);margin-top:6px">Click a node to configure</div></div>';
}
function simDelEdge(i){G.edges.splice(i,1);simRender();}

function _isBack(src,dst){
  if(src===dst)return true;
  const si=G.nodes.findIndex(n=>n.id===src);
  const di=G.nodes.findIndex(n=>n.id===dst);
  return di<=si;
}

function simRenderEdges(){
  const svg=document.getElementById('sim-svg');if(!svg)return;
  const paths=G.edges.map((e,i)=>{
    const sn=G.nodes.find(n=>n.id===e.src),dn=G.nodes.find(n=>n.id===e.dst);
    if(!sn||!dn)return'';
    const back=_isBack(e.src,e.dst);
    const col=back?'#dc2626':'#f97316';
    const mk=back?'url(#ah-red)':'url(#ah-or)';
    let d;
    if(e.src===e.dst){
      const x=sn.x+NW-16,y=sn.y;
      d=`M${x} ${y+12} C${x+65} ${y-35} ${x+65} ${y+NH+15} ${x} ${y+NH-12}`;
    }else{
      const sx=sn.x+NW,sy=sn.y+NH/2,dx=dn.x,dy=dn.y+NH/2;
      const cx=Math.max(50,Math.abs(dx-sx)*0.55);
      d=`M${sx} ${sy} C${sx+cx} ${sy} ${dx-cx} ${dy} ${dx} ${dy}`;
    }
    return`<path d="${d}" stroke="${col}" stroke-width="2" fill="none"
      stroke-dasharray="${back||e.src===e.dst?'6,3':'none'}"
      marker-end="${mk}" style="cursor:pointer" onclick="simDelEdge(${i})" title="Click to delete edge"/>`;
  }).join('');
  // preserve defs
  const defs=svg.querySelector('defs');
  svg.innerHTML='';
  if(defs)svg.appendChild(defs);
  svg.insertAdjacentHTML('beforeend',paths);
}

function simRenderNodes(){
  const canvas=document.getElementById('sim-canvas');if(!canvas)return;
  const empty=document.getElementById('sim-empty');
  if(empty)empty.style.display=G.nodes.length?'none':'flex';
  // remove old node divs
  canvas.querySelectorAll('.sn').forEach(el=>el.remove());
  G.nodes.forEach(n=>{
    const icon=n.type==='llm'?'🤖':'🔧';
    const isSel=simSel===n.id,isConn=simConn===n.id;
    const div=document.createElement('div');
    div.className=`sn ${n.type}${isSel?' sel':''}${isConn?' conn-src':''}`;
    div.id='sn-'+n.id;
    div.style.cssText=`left:${n.x}px;top:${n.y}px;width:${NW}px;height:${NH}px`;
    div.innerHTML=`<div class="sn-port sn-in" data-nid="${n.id}" data-side="in"></div>
      <div class="sn-body" data-nid="${n.id}">
        <span class="sn-icon">${icon}</span>
        <span class="sn-lbl">${n.label||n.type}</span>
      </div>
      <div class="sn-port sn-out" data-nid="${n.id}" data-side="out"></div>`;
    canvas.appendChild(div);
  });
}

function simRender(){simRenderNodes();simRenderEdges();}

function simShowCfg(nid){
  const node=G.nodes.find(n=>n.id===nid);
  if(!node){document.getElementById('sim-cfg').innerHTML='';return;}
  const cfg=node.config||{};
  let html=`<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span class="cfg-lbl" style="margin:0">NODE CONFIG</span>
    <button class="btn" style="color:var(--red);border-color:#fecaca;font-size:10px;padding:3px 8px" onclick="simDelNode('${nid}')">✕ Delete</button>
  </div>
  <div class="cfg-f"><label class="cfg-lbl">Label</label>
    <input type="text" class="cfg-inp" value="${node.label}" oninput="simUpd('${nid}','label',this.value)">
  </div>`;
  if(node.type==='llm'){
    html+=`<div class="cfg-f"><label class="cfg-lbl">Model</label>
      <select class="cfg-inp" onchange="simUpdCfg('${nid}','model',this.value)">
        ${['gpt-4o','gpt-4o-mini','claude-sonnet-4-6','claude-opus-4-6'].map(m=>`<option value="${m}"${cfg.model===m?' selected':''}>${m}</option>`).join('')}
      </select></div>
    <div class="cfg-f"><label class="cfg-lbl">Input tokens</label>
      <input type="number" class="cfg-inp" value="${cfg.tokens_in||1200}" oninput="simUpdCfg('${nid}','tokens_in',+this.value)"></div>
    <div class="cfg-f"><label class="cfg-lbl">System tokens</label>
      <input type="number" class="cfg-inp" value="${cfg.system_tokens||0}" oninput="simUpdCfg('${nid}','system_tokens',+this.value)"></div>`;
  }else{
    html+=`<div class="cfg-f"><label class="cfg-lbl">Tool name</label>
      <input type="text" class="cfg-inp" value="${cfg.tool_name||node.label}" oninput="simUpdCfg('${nid}','tool_name',this.value)"></div>
    <div class="cfg-f"><label class="cfg-lbl">Failure mode</label>
      <select class="cfg-inp" onchange="simUpdCfg('${nid}','failure',this.value)">
        ${[['none','None'],['timeout','Timeout / 503'],['empty','Empty result'],['error','Hard error']].map(([v,l])=>`<option value="${v}"${cfg.failure===v?' selected':''}>${l}</option>`).join('')}
      </select></div>
    <div class="cfg-f"><label class="cfg-lbl">Failure rate: <span id="fr-${nid}">${Math.round((cfg.failure_rate||0)*100)}%</span></label>
      <input type="range" class="cfg-inp" min="0" max="100" value="${Math.round((cfg.failure_rate||0)*100)}"
        oninput="simUpdCfg('${nid}','failure_rate',this.value/100);document.getElementById('fr-${nid}').textContent=this.value+'%'"></div>
    <div class="cfg-f"><label class="cfg-lbl">Max retries (self-loop)</label>
      <input type="number" class="cfg-inp" value="${cfg.max_retries||4}" min="1" max="10" oninput="simUpdCfg('${nid}','max_retries',+this.value)"></div>`;
  }
  document.getElementById('sim-cfg').innerHTML=html;
}

function simUpd(nid,key,val){const n=G.nodes.find(n=>n.id===nid);if(n)n[key]=val;simRenderNodes();}
function simUpdCfg(nid,key,val){const n=G.nodes.find(n=>n.id===nid);if(n){if(!n.config)n.config={};n.config[key]=val;}}

// ── drag + port click via event delegation ────────────────────────────────
document.addEventListener('mousedown',e=>{
  const body=e.target.closest('.sn-body');
  if(body&&body.dataset.nid){
    const nid=body.dataset.nid;
    const wrap=document.getElementById('sim-wrap');if(!wrap)return;
    const rect=wrap.getBoundingClientRect();
    const node=G.nodes.find(n=>n.id===nid);if(!node)return;
    simDrag={nid,ox:e.clientX-rect.left-node.x,oy:e.clientY-rect.top-node.y,moved:false};
    e.preventDefault();return;
  }
  const port=e.target.closest('.sn-port');
  if(port&&port.dataset.nid){
    const {nid,side}=port.dataset;
    if(side==='out'){
      simConn=nid;simRenderNodes();
      document.getElementById('sim-status').textContent='Now click any node\'s left (input) port to connect →';
    }else if(side==='in'&&simConn){
      if(!G.edges.some(ed=>ed.src===simConn&&ed.dst===nid))
        G.edges.push({src:simConn,dst:nid});
      simConn=null;document.getElementById('sim-status').textContent='';simRender();
    }
    e.stopPropagation();return;
  }
});

document.addEventListener('mousemove',e=>{
  if(!simDrag)return;
  const wrap=document.getElementById('sim-wrap');if(!wrap)return;
  const rect=wrap.getBoundingClientRect();
  const node=G.nodes.find(n=>n.id===simDrag.nid);if(!node)return;
  node.x=Math.max(0,Math.min(e.clientX-rect.left-simDrag.ox,rect.width-NW));
  node.y=Math.max(0,Math.min(e.clientY-rect.top-simDrag.oy,rect.height-NH));
  simDrag.moved=true;
  const el=document.getElementById('sn-'+simDrag.nid);
  if(el){el.style.left=node.x+'px';el.style.top=node.y+'px';}
  simRenderEdges();
});

document.addEventListener('mouseup',e=>{
  if(simDrag){
    if(!simDrag.moved){simSel=simDrag.nid;simRenderNodes();simShowCfg(simDrag.nid);}
    simDrag=null;
  }
});

async function simRun(){
  if(!G.nodes.length){alert('Add nodes first or load a preset.');return;}
  const btn=document.getElementById('sim-run-btn');
  btn.textContent='⏳ Running…';btn.disabled=true;
  document.getElementById('sim-res').style.display='none';
  try{
    const r=await fetch('/simulate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({nodes:G.nodes,edges:G.edges,
        sessions:parseInt(document.getElementById('sim-sessions').value)||5,
        model:document.getElementById('sim-model').value})});
    const d=await r.json();
    if(d.ok){
      document.getElementById('sim-status').textContent=`✓ Generated ${d.spans} spans across ${d.sessions} sessions → ${d.file}`;
      setTimeout(()=>{loadFindings();simShowRes(d);},2500);
    }else{
      document.getElementById('sim-status').textContent='Error: '+(d.error||'unknown');
    }
  }catch(ex){document.getElementById('sim-status').textContent='Request failed: '+ex.message;}
  btn.textContent='▶ Run Simulation';btn.disabled=false;
}

function simShowRes(d){
  const el=document.getElementById('sim-res');
  el.style.display='block';
  document.getElementById('sim-res-body').innerHTML=`
    <div class="sim-res-item"><span class="sim-res-lbl">Spans generated</span><span class="sim-res-val">${d.spans}</span></div>
    <div class="sim-res-item"><span class="sim-res-lbl">Sessions</span><span class="sim-res-val">${d.sessions}</span></div>
    <div class="sim-res-item"><span class="sim-res-lbl">File saved</span><span class="sim-res-val" style="font-size:10px">${d.file}</span></div>
    <div class="sim-res-item"><span class="sim-res-lbl">Status</span><span class="sim-res-val" style="color:#16a34a">Analyzed ✓</span></div>`;
}

// load retry_loop preset by default when tab opens
document.querySelector('.tab[onclick="switchTab(\'simulate\')"]')
  ?.addEventListener('click',()=>{if(!G.nodes.length)simLoad('retry_loop');});

// ── Usage tab ─────────────────────────────────────────────────────────────────

const fmtM=n=>n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(0)+'k':String(n??0);
const fmtU=v=>v===0?'$0':v>=10000?'$'+(v/1000).toFixed(0)+'k':v>=1000?'$'+(v/1000).toFixed(1)+'k':'$'+v.toFixed(2);
const sevIcon={high:'🔴',medium:'🟡',low:'🟢'};
const platformLabel={'claude_ai':'Claude.ai','cursor':'Cursor','openai':'OpenAI'};

function uDrop(e){
  e.preventDefault();
  document.getElementById('u-drop').classList.remove('drag-over');
  const file=e.dataTransfer.files[0];
  if(file)uReadFile(file);
}
function uFileChosen(input){
  const file=input.files[0];
  if(file)uReadFile(file);
}
function uReadFile(file){
  const reader=new FileReader();
  reader.onload=ev=>uAnalyse(ev.target.result,false);
  reader.readAsText(file);
}
function uReset(){
  document.getElementById('u-report').style.display='none';
  document.getElementById('u-drop-wrap').style.display='block';
  document.getElementById('u-file-input').value='';
}
async function uLoadDemo(){
  // Load anonymised case study demo CSV
  const resp=await fetch('/usage-demo');
  if(!resp.ok){alert('Demo data not available.');return;}
  const csv=await resp.text();
  uAnalyse(csv,true);
}

async function uAnalyse(csvText,isDemo){
  const drop=document.getElementById('u-drop');
  drop.innerHTML='<div class="drop-zone-icon">⏳</div><div class="drop-zone-title">Analysing…</div>';

  try{
    const resp=await fetch('/analyze-usage',{
      method:'POST',
      headers:{'Content-Type':'text/plain'},
      body:csvText,
    });
    const d=await resp.json();
    if(!d.ok){
      drop.innerHTML=`<div class="drop-zone-icon">❌</div><div class="drop-zone-title">Parse error</div><div class="drop-zone-sub">${d.error}</div><button class="btn" onclick="uReset()" style="margin-top:16px">Try again</button>`;
      return;
    }
    uRenderReport(d.report,isDemo);
  }catch(ex){
    drop.innerHTML=`<div class="drop-zone-icon">❌</div><div class="drop-zone-title">Network error</div><div class="drop-zone-sub">${ex.message}</div><button class="btn" onclick="uReset()" style="margin-top:16px">Try again</button>`;
  }
}

function uRenderReport(r,isDemo){
  document.getElementById('u-drop-wrap').style.display='none';
  document.getElementById('u-report').style.display='block';

  // Platform badge
  document.getElementById('u-platform-badge').innerHTML=
    '📊 '+( platformLabel[r.platform]||r.platform)+' export';

  // KPIs
  const s=r.period_start.slice(0,10),e=r.period_end.slice(0,10);
  document.getElementById('u-period').textContent=s+' → '+e;
  document.getElementById('u-period-sub').textContent=r.period_days+' days';
  document.getElementById('u-total-cost').textContent=fmtU(r.total_cost_usd);
  document.getElementById('u-total-calls').textContent=r.total_calls.toLocaleString()+' calls';
  document.getElementById('u-monthly').textContent=fmtU(r.monthly_run_rate_usd);
  document.getElementById('u-tokens').textContent=fmtM(r.total_tokens);
  document.getElementById('u-users-count').textContent=r.users.length+' users';
  document.getElementById('u-savings').textContent=fmtU(r.total_savings_potential_usd);
  const pct=r.monthly_run_rate_usd>0?Math.round(r.total_savings_potential_usd/r.monthly_run_rate_usd*100):0;
  document.getElementById('u-savings-pct').textContent=pct+'% of monthly run rate';

  // Case study
  const csEl=document.getElementById('u-case-study');
  csEl.style.display=isDemo?'block':'none';

  // Findings
  const fList=document.getElementById('u-findings-list');
  document.getElementById('u-findings-count').textContent=r.findings.length+' issue'+(r.findings.length!==1?'s':'');
  if(r.findings.length===0){
    fList.innerHTML='<div class="empty"><div class="empty-icon">✅</div><div class="empty-title">No issues found</div></div>';
  }else{
    fList.innerHTML=r.findings.map(f=>`
      <div class="u-finding ${f.severity}">
        <div class="u-f-top">
          <span class="u-f-pattern">${f.pattern}</span>
          <span class="u-f-sev ${f.severity}">${sevIcon[f.severity]} ${f.severity}</span>
          ${f.savings_per_month_usd>0?`<span class="u-f-savings">save ${fmtU(f.savings_per_month_usd)}/mo</span>`:''}
        </div>
        <div class="u-f-title">${f.title}</div>
        <div class="u-f-detail">${f.detail}</div>
        <div class="u-f-fix">${f.fix}</div>
      </div>`).join('');
  }

  // Models
  const mList=document.getElementById('u-models-list');
  const maxModelCost=Math.max(...r.models.map(m=>m.cost),0.01);
  mList.innerHTML=r.models.slice(0,8).map(m=>`
    <div class="u-model-row">
      <div class="u-model-name">${m.model}</div>
      <div class="u-model-bar-wrap"><div class="u-model-bar" style="width:${Math.round(m.cost/maxModelCost*100)}%"></div></div>
      <div class="u-model-cost">${fmtU(m.cost)}</div>
    </div>`).join('');

  // Users
  const uList=document.getElementById('u-users-list');
  uList.innerHTML=r.users.map(u=>`
    <div class="u-card">
      <div class="u-card-head">
        <div class="u-card-label">${u.label}</div>
        <div class="u-card-cost">${fmtU(u.cost)}</div>
      </div>
      <div class="u-card-rows">
        <div class="u-card-row"><span class="u-card-row-label">API calls</span><span class="u-card-row-val">${u.calls.toLocaleString()}</span></div>
        <div class="u-card-row"><span class="u-card-row-label">Cache read %</span><span class="u-card-row-val">${u.cache_pct}%</span></div>
        <div class="u-card-row"><span class="u-card-row-label">Top model</span><span class="u-card-row-val" style="font-size:10px">${u.top_model||'—'}</span></div>
      </div>
      <div class="u-bar-wrap" style="margin-top:10px">
        <div class="u-bar-fill" style="width:${r.total_cost_usd>0?Math.round(u.cost/r.total_cost_usd*100):0}%"></div>
      </div>
      <div style="font-size:10px;color:var(--t4);margin-top:3px">${r.total_cost_usd>0?Math.round(u.cost/r.total_cost_usd*100):0}% of team spend</div>
    </div>`).join('');

  // Recommendations
  const patterns=new Set(r.findings.map(f=>f.pattern));
  const recs=[];
  if(patterns.has('context_bloat'))
    recs.push('<strong>Context bloat (highest priority):</strong> Implement session-reset triggers — start a new context after each major task completion. Cap history to last 20 turns using a sliding window. This alone typically saves 60–70% on cache-read costs.');
  if(patterns.has('user_concentration'))
    recs.push('<strong>Usage concentration:</strong> One power user is driving most spend. Review their workflows for automation opportunities. Consider dedicated API keys per project for cost attribution.');
  if(patterns.has('model_overkill'))
    recs.push('<strong>Model routing:</strong> Route classification, formatting, and short-answer tasks to claude-haiku or gpt-4o-mini. Reserve opus/sonnet for complex reasoning and code generation. Typical savings: 40–60% on those call types.');
  if(patterns.has('usage_spike'))
    recs.push('<strong>Cost guardrails:</strong> Add per-session token limits and cost circuit-breakers. A single runaway conversation can cost as much as a full day of normal usage.');
  if(patterns.has('zero_output'))
    recs.push('<strong>Aborted calls:</strong> Calls with no output tokens indicate timeouts or cancellations. Add retry logic with exponential backoff and surface errors to users immediately to avoid silent waste.');
  if(patterns.has('peak_day_spend'))
    recs.push('<strong>Peak spend anomalies:</strong> Set up daily spend alerts at 1.5× your average daily budget. Use monk in CI to detect runaway patterns before they compound.');
  if(recs.length===0)
    recs.push('No critical issues detected. Continue monitoring with monk and review again after major workflow changes.');

  const recCard=document.getElementById('u-recs-card');
  recCard.style.display='block';
  document.getElementById('u-recs-body').innerHTML='<ol style="padding-left:18px;display:flex;flex-direction:column;gap:10px">'+
    recs.map(r=>`<li>${r}</li>`).join('')+'</ol>';
}
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
        elif self.path == "/usage-demo":
            # Serve the anonymised demo CSV fixture if it exists
            demo_paths = [
                Path(__file__).parent.parent / "tests" / "fixtures" / "demo_usage.csv",
                Path("tests/fixtures/demo_usage.csv"),
            ]
            for p in demo_paths:
                if p.exists():
                    self._respond(200, "text/csv", p.read_bytes())
                    return
            self._respond(404, "text/plain", b"demo fixture not found")
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
        elif self.path == "/analyze-usage":
            length = int(self.headers.get("Content-Length", 0))
            csv_text = self.rfile.read(length).decode("utf-8-sig", errors="replace") if length else ""
            try:
                from monk.parsers.usage_csv import parse_usage_csv, FormatError
                from monk.usage_analyzer import analyze
                records, warnings, platform = parse_usage_csv(csv_text, anonymise=True)
                report = analyze(records, warnings=warnings, platform=platform)
                self._respond(200, "application/json",
                    json.dumps({"ok": True, "report": report.to_dict()}).encode())
            except Exception as exc:
                self._respond(200, "application/json",
                    json.dumps({"ok": False, "error": str(exc)}).encode())
        elif self.path == "/simulate":
            length = int(self.headers.get("Content-Length", 0))
            body_data = json.loads(self.rfile.read(length)) if length else {}
            try:
                from monk.simulate import simulate_workflow_otel, write_jsonl
                import time as _t
                nodes  = body_data.get("nodes", [])
                edges  = body_data.get("edges", [])
                sessions = int(body_data.get("sessions", 5))
                model  = body_data.get("model", "gpt-4o")
                if not nodes:
                    self._respond(400, "application/json", b'{"ok":false,"error":"No nodes defined"}')
                    return
                spans = simulate_workflow_otel(nodes, edges, sessions=sessions, model=model, seed=None)
                fname = f"sim_{int(_t.time())}.jsonl"
                write_jsonl(spans, self.traces_dir / fname)
                threading.Thread(target=self.rescan_fn, daemon=True).start()
                self._respond(200, "application/json", json.dumps({
                    "ok": True, "file": fname,
                    "spans": len(spans), "sessions": sessions,
                }).encode())
            except Exception as exc:
                self._respond(500, "application/json",
                    json.dumps({"ok": False, "error": str(exc)}).encode())
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
    _last_scan: list[float] = [0.0]   # mutable container so closure can mutate it
    _DEBOUNCE = 8                      # minimum seconds between scans

    # ── Initial scan ─────────────────────────────────────────────────────────
    fresh, calls, findings, nfiles = _scan_all(target)
    with _lock:
        store.__dict__.update(fresh.__dict__)
        _last_scan[0] = time.time()

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
        now = time.time()
        with _lock:
            if now - _last_scan[0] < _DEBOUNCE:
                return          # debounce: skip if scanned too recently
            _last_scan[0] = now
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
