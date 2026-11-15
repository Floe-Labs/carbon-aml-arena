from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

ROOT = Path(__file__).resolve().parent
SCENARIOS: dict[str, dict[str, Any]] = {s["id"]: s for s in json.loads((ROOT / "scenarios.json").read_text("utf-8"))}
RUNS_PATH = ROOT / "runs.json"
STATE: dict[str, Any] = json.loads(RUNS_PATH.read_text("utf-8"))
_RATE_LOG: list[tuple[float, str]] = []

SYSTEM_PROMPT = (
    "You are an autonomous financial-crime compliance investigator operating inside a stateful multi-day sandbox. "
    "Gather all required evidence using tools, advance the clock with wait_days when records or interbank notices are pending, "
    "trace multi-hop beneficial ownership where applicable, and conclude by calling commit_verdict."
)

_ENT_PARAMS = {"type": "object", "properties": {"entity_id": {"type": "string"}}, "required": ["entity_id"]}
_VERDICT_PARAMS = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["clear", "escalate", "block"]}, "cited_evidence": {"type": "array", "items": {"type": "string"}}, "rationale": {"type": "string"}},
    "required": ["verdict", "cited_evidence", "rationale"],
}
TOOLS: list[dict[str, Any]] = [
    {"name": "inspect_dossier", "description": "Inspect entity KYC, wires, and notices settled up to current simulated day.", "parameters": _ENT_PARAMS},
    {"name": "trace_ownership", "description": "Return direct 1-hop owners of entity_id. Call on parent entities to trace multi-hop shells.", "parameters": _ENT_PARAMS},
    {"name": "wait_days", "description": "Advance simulated compliance clock by 1 to 14 days so pending notices/batches settle.", "parameters": {"type": "object", "properties": {"days": {"type": "integer"}}, "required": ["days"]}},
    {"name": "commit_verdict", "description": "Commit final audited disposition ('clear', 'escalate', or 'block') with cited evidence IDs and rationale.", "parameters": _VERDICT_PARAMS},
]


class SandboxEpisode:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec, self.sim_day, self.verdict, self.rationale, self.observed_blob = spec, 0, None, "", ""
        self.cited: list[str] = []
        self.steps: list[dict[str, Any]] = []

    def state_hash(self) -> str:
        raw = json.dumps({"id": self.spec["id"], "day": self.sim_day, "verdict": self.verdict, "n": len(self.steps)}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def step(self, name: str, args: dict[str, Any], thought_sig: bool = False) -> dict[str, Any]:
        if name == "inspect_dossier":
            records = {k: v for k, v in self.spec["dossier_by_day"].items() if int(k) <= self.sim_day}
            res: dict[str, Any] = {"sim_day": self.sim_day, "records_by_day": records}
            summary = f"Inspected {args.get('entity_id', '')} at Day {self.sim_day} (settled days: {','.join(sorted(records))})"
        elif name == "trace_ownership":
            eid = str(args.get("entity_id", ""))
            edges = self.spec["ownership_graph"].get(eid, [])
            res, summary = {"entity_id": eid, "direct_owners": edges}, f"Traced {eid} -> {len(edges)} direct owner(s)"
        elif name == "wait_days":
            delta = max(1, min(int(args.get("days", 1)), 14))
            self.sim_day += delta
            unlocked = [k for k in self.spec["dossier_by_day"] if self.sim_day - delta < int(k) <= self.sim_day]
            res, summary = {"sim_day": self.sim_day, "newly_settled_days": unlocked}, f"Advanced clock +{delta}d to Day {self.sim_day} (settled: {unlocked or 'none'})"
        elif name == "commit_verdict":
            self.verdict = str(args.get("verdict", "")).lower().strip()
            self.cited = [str(x) for x in (args.get("cited_evidence") or [])]
            self.rationale = str(args.get("rationale", "")).strip()
            res, summary = {"committed": True, "verdict": self.verdict, "sim_day": self.sim_day}, f"Committed verdict={self.verdict} on Day {self.sim_day}"
        else:
            res, summary = {"error": f"unknown tool {name}"}, f"Unknown tool {name}"
        self.observed_blob += " " + json.dumps(res)
        clean_args = {k: v for k, v in args.items() if k != "rationale"}
        self.steps.append({"step": len(self.steps) + 1, "day": self.sim_day, "tool": name, "args": clean_args, "thought_sig": thought_sig, "state_hash": self.state_hash(), "summary": summary})
        return res

    def grade(self, model: str, thinking: str, tokens: int) -> dict[str, Any]:
        req_ev = self.spec["required_evidence"]
        passed = self.verdict == self.spec["expected_verdict"] and self.sim_day >= self.spec["required_min_day"] and (req_ev in self.observed_blob or req_ev in " ".join(self.cited))
        return {"model": model, "thinking": thinking, "verdict": self.verdict or "none", "cited_evidence": self.cited, "rationale": self.rationale or "No verdict committed before turn cap.", "sim_day_final": self.sim_day, "tokens_used": tokens, "passed": passed, "liability_incurred_usd": 0 if passed else int(self.spec["liability_usd"]), "steps": self.steps}


def run_live_agent(spec: dict[str, Any], model: str, high_thinking: bool, api_key: str = "", max_turns: int = 8) -> dict[str, Any]:
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ValueError("No Gemini API key configured; supply a key in the form or mount GEMINI_API_KEY on Cloud Run.")
    ep = SandboxEpisode(spec)
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": spec["alert_summary"]}]}]
    gen_cfg: dict[str, Any] = {"maxOutputTokens": 8192}
    if high_thinking:
        gen_cfg["thinkingConfig"] = {"thinkingLevel": "HIGH"}
        if "carbon" not in model.lower():
            gen_cfg["temperature"] = 0.0
    else:
        gen_cfg.update({"temperature": 0.0, "thinkingConfig": {"thinkingLevel": "LOW"}})
    total_tokens, active_model = 0, model
    fallbacks = {"gemini-4-carbon": "gemini-4-carbon-preview", "gemini-4-hydrogen": "gemini-3.8-flash"}
    for _ in range(max_turns):
        payload = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents, "tools": [{"functionDeclarations": TOOLS}], "generationConfig": gen_cfg}
        resp = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent", headers={"x-goog-api-key": key}, json=payload, timeout=90.0)
        if resp.status_code == 404 and active_model in fallbacks:
            active_model = fallbacks[active_model]
            resp = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent", headers={"x-goog-api-key": key}, json=payload, timeout=90.0)
        if resp.status_code == 400 and not high_thinking and "thinkingConfig" in gen_cfg:
            gen_cfg.pop("thinkingConfig", None)
            resp = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent", headers={"x-goog-api-key": key}, json=payload, timeout=90.0)
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini API ({active_model}) returned HTTP {resp.status_code}")
        data = resp.json()
        total_tokens += int((data.get("usageMetadata") or {}).get("totalTokenCount", 0))
        parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
        fcalls = [p for p in parts if "functionCall" in p]
        if not fcalls:
            break
        model_parts, user_parts = [], []
        for p in fcalls:
            fc, sig = p["functionCall"], p.get("thoughtSignature")
            mp: dict[str, Any] = {"functionCall": {"name": fc["name"], "args": dict(fc.get("args") or {})}}
            if sig:
                mp["thoughtSignature"] = sig
            model_parts.append(mp)
            out = ep.step(fc["name"], dict(fc.get("args") or {}), thought_sig=bool(sig) or high_thinking)
            user_parts.append({"functionResponse": {"name": fc["name"], "response": out}})
        contents.extend([{"role": "model", "parts": model_parts}, {"role": "user", "parts": user_parts}])
        if ep.verdict is not None:
            break
    return ep.grade(active_model, "HIGH" if high_thinking else "LOW", total_tokens)


app = FastAPI(title="FinTrap Arena — Gemini Carbon Stateful AML Sandbox")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "arena": "fintrap-carbon"}


@app.get("/api/runs")
def api_runs() -> dict[str, Any]:
    return STATE


@app.post("/run-live", response_model=None)
def trigger_live_run(request: Request, case_id: str = Form("TRAP-01"), api_key: str = Form("")) -> JSONResponse | RedirectResponse:
    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        return JSONResponse({"error": "cross-site POST blocked"}, status_code=403)
    cid = case_id if case_id in SCENARIOS else "TRAP-01"
    user_key = api_key.strip()
    now, client_ip = time.time(), (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "local")).split(",")[0].strip()
    _RATE_LOG[:] = [(t, ip) for t, ip in _RATE_LOG if now - t < 3600]
    if not user_key:
        if len(_RATE_LOG) >= 20 or sum(1 for t, ip in _RATE_LOG if ip == client_ip and now - t < 900) >= 5:
            return RedirectResponse(f"/?case={cid}&err={quote('Public server-key rate limit reached; paste your own Gemini API key to run immediately.')}", status_code=303)
        _RATE_LOG.append((now, client_ip))
    c_mod, b_mod = os.getenv("CARBON_MODEL", "gemini-4-carbon"), os.getenv("BASELINE_MODEL", "gemini-4-hydrogen")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fc = pool.submit(run_live_agent, SCENARIOS[cid], c_mod, True, user_key)
            fb = pool.submit(run_live_agent, SCENARIOS[cid], b_mod, False, user_key)
            STATE["episodes"][cid] = {"carbon": fc.result(), "baseline": fb.result()}
        STATE["mode"] = f"live_cloud_run ({datetime.now(UTC).strftime('%H:%M:%S')} UTC)"
        return RedirectResponse(f"/?case={cid}&msg={quote(f'Executed live {c_mod} vs {b_mod} on {cid}')}", status_code=303)
    except Exception as exc:
        return RedirectResponse(f"/?case={cid}&err={quote(str(exc))}", status_code=303)


@app.get("/", response_class=HTMLResponse)
def arena_ui(case: str = Query("TRAP-01"), msg: str = Query(""), err: str = Query("")) -> HTMLResponse:
    esc, cid = html.escape, (case if case in SCENARIOS else "TRAP-01")
    spec, pair, eps = SCENARIOS[cid], STATE["episodes"][cid], STATE["episodes"]
    c_pass, b_pass = sum(1 for v in eps.values() if v["carbon"]["passed"]), sum(1 for v in eps.values() if v["baseline"]["passed"])
    b_loss = sum(int(v["baseline"]["liability_incurred_usd"]) for v in eps.values())
    c_loss = sum(int(v["carbon"]["liability_incurred_usd"]) for v in eps.values())
    has_srv_key = bool((os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip())
    tabs = "".join(f"<a class='tab {'active' if k == cid else ''}' href='/?case={k}'><code>{k}</code> {esc(s['category'])}</a>" for k, s in SCENARIOS.items())
    flash = f"<div class='fl ok'>{esc(msg)}</div>" if msg else (f"<div class='fl err'>{esc(err)}</div>" if err else "")

    def _panel(label: str, run: dict[str, Any]) -> str:
        ok = run["passed"]
        badge = "<span class='b ok'>PASS &middot; $0 Liability</span>" if ok else f"<span class='b err'>FAIL &middot; ${run['liability_incurred_usd']:,} Liability</span>"
        rows = "".join(
            f"<li><div class='st'><span><strong>Step {s['step']}</strong> (Day {s['day']}) <code>{esc(s['tool'])}({esc(json.dumps(s['args']))})</code></span>"
            f"<span>{'<em class=sig>thoughtSig</em> ' if s.get('thought_sig') else ''}<code>#{esc(s['state_hash'][:8])}</code></span></div><div class='muted'>{esc(s['summary'])}</div></li>"
            for s in run["steps"]
        )
        return (
            f"<div class='card {'ok-b' if ok else 'err-b'}'><div class='hdr'><div><h3>{esc(label)}</h3>"
            f"<div class='muted'>Model: <code>{esc(run['model'])}</code> &middot; Thinking: <code>{esc(run['thinking'])}</code> &middot; {run['tokens_used']:,} tok</div></div>{badge}</div>"
            f"<p><strong>Verdict:</strong> <code>{esc(run['verdict'].upper())}</code> (Day {run['sim_day_final']}) &middot; <strong>Cited:</strong> <code>{esc(', '.join(run['cited_evidence']) or 'none')}</code></p>"
            f"<p class='rat'>{esc(run['rationale'])}</p><ol class='tl'>{rows}</ol></div>"
        )

    css = (
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0b1020;color:#e2e8f0;margin:0;line-height:1.45}"
        ".wrap{max-width:1240px;margin:0 auto;padding:22px}code{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;background:#090d16;padding:2px 5px;border-radius:4px}"
        ".kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}.kpi{background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px}"
        ".kv{font-size:22px;font-weight:700;margin-top:4px}.muted{color:#94a3b8;font-size:12px}.tabs{display:flex;gap:8px;margin:14px 0;flex-wrap:wrap;align-items:center}"
        ".tab{padding:8px 14px;border-radius:6px;background:#111827;border:1px solid #1f2937;color:#cbd5e1;text-decoration:none;font-size:13px}.tab.active{border-color:#38bdf8;color:#fff;background:#0c4a6e}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:#111827;border:1px solid #1f2937;border-radius:8px;padding:16px}"
        ".ok-b{border-top:3px solid #10b981}.err-b{border-top:3px solid #ef4444}.hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}h1,h2,h3{margin:0 0 4px}"
        ".b{padding:4px 9px;border-radius:999px;font-size:11px;font-weight:600;white-space:nowrap}.ok{background:#064e3b;color:#6ee7b7}.err{background:#7f1d1d;color:#fca5a5}"
        ".tl{list-style:none;padding:0;margin:12px 0 0;display:grid;gap:8px}.tl li{background:#090d16;border:1px solid #1e293b;border-radius:6px;padding:8px 10px}.st{display:flex;justify-content:space-between;gap:8px;font-size:12px}"
        ".sig{color:#38bdf8;font-style:normal;font-size:11px}.rat{background:#0f172a;padding:10px;border-radius:6px;font-size:13px}.btn{background:#0284c7;color:#fff;border:0;padding:8px 14px;border-radius:6px;cursor:pointer;font-weight:600;font-size:13px}"
        ".inp{background:#090d16;border:1px solid #334155;color:#f8fafc;padding:7px 10px;border-radius:6px;font-size:12px;width:250px}.fl{padding:10px 14px;border-radius:6px;margin:10px 0;font-size:13px}"
    )
    key_ph = "Optional override API key (server key active)" if has_srv_key else "Paste Gemini API key to run live"
    return HTMLResponse(
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>FinTrap Arena — Gemini Carbon</title><style>{css}</style></head><body><main class='wrap'>"
        f"<div class='hdr'><div><h1>FinTrap Arena <span style='color:#38bdf8;font-size:15px'>Stateful Financial-Crime Stress Test</span></h1>"
        f"<div class='muted'>Built with Antigravity CLI + Google Gemini Carbon &middot; Public-Domain Clean-Room Sandbox by Floe Labs (<code>floelabs.io</code>)</div></div>"
        f"<span class='b ok'>Mode: {esc(STATE['mode'].split('(')[0].strip())}</span></div>{flash}"
        f"<div class='kpis'><div class='kpi'><div class='muted'>Gemini Carbon Pass Rate</div><div class='kv' style='color:#6ee7b7'>{c_pass} / {len(SCENARIOS)}</div></div>"
        f"<div class='kpi'><div class='muted'>Fast Baseline Pass Rate</div><div class='kv' style='color:#fca5a5'>{b_pass} / {len(SCENARIOS)}</div></div>"
        f"<div class='kpi'><div class='muted'>Baseline Regulatory/SLA Loss</div><div class='kv' style='color:#fca5a5'>${b_loss:,}</div></div>"
        f"<div class='kpi'><div class='muted'>Liability Prevented by Carbon</div><div class='kv' style='color:#38bdf8'>${max(0, b_loss - c_loss):,}</div></div></div>"
        f"<div class='tabs'>{tabs}<form method='post' action='/run-live' style='margin-left:auto;display:flex;gap:8px;align-items:center'>"
        f"<input type='hidden' name='case_id' value='{cid}'><input class='inp' type='password' name='api_key' placeholder='{key_ph}' autocomplete='off'>"
        f"<button class='btn' type='submit'>&#9654; Run Live ({cid})</button></form></div>"
        f"<div class='card' style='margin-bottom:16px'><h2><code>{cid}</code> — {esc(spec['title'])}</h2><p style='font-size:13px;margin:6px 0'>{esc(spec['alert_summary'])}</p>"
        f"<div class='muted'>Deterministic Grader Gate: Expected Verdict=<code>{esc(spec['expected_verdict'].upper())}</code> &middot; Min Clock=<code>Day {spec['required_min_day']}</code> &middot; Required Evidence=<code>{esc(spec['required_evidence'])}</code></div></div>"
        f"<div class='grid'>{_panel('Gemini Carbon (HIGH Thinking + thoughtSignature)', pair['carbon'])}{_panel('Fast Reactive Baseline (No-Thinking)', pair['baseline'])}</div></main></body></html>"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture", action="store_true")
    p.add_argument("--carbon-model", default=os.getenv("CARBON_MODEL", "gemini-4-carbon"))
    p.add_argument("--baseline-model", default=os.getenv("BASELINE_MODEL", "gemini-4-hydrogen"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    args = p.parse_args()
    if args.capture:
        for cid, s in SCENARIOS.items():
            STATE["episodes"][cid] = {"carbon": run_live_agent(s, args.carbon_model, True), "baseline": run_live_agent(s, args.baseline_model, False)}
        STATE.update({"captured_at": datetime.now(UTC).isoformat(), "mode": "live_gemini_api"})
        RUNS_PATH.write_text(json.dumps(STATE, indent=2) + "\n", "utf-8")
        return
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
