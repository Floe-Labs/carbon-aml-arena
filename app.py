from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
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

ROOT, RUNS_PATH = Path(__file__).resolve().parent, Path(__file__).resolve().parent / "runs.json"
SCENARIO_LIST: list[dict[str, Any]] = json.loads((ROOT / "scenarios.json").read_text("utf-8"))
SCENARIOS, STATE = {s["id"]: s for s in SCENARIO_LIST}, json.loads(RUNS_PATH.read_text("utf-8"))
CSS_TEXT, IC, _RATE_LOG = (ROOT / "arena.css").read_text("utf-8"), json.loads((ROOT / "icons.json").read_text("utf-8")), []
_CODE_RE = re.compile(r"\b(?:TRAP|INV|ENT|PER|POL|KYC|CTR|SDN|CA-SOS|SY|TM|SAN|PUB|NOTICE)-[A-Za-z0-9-]+\b")
SYSTEM_PROMPT = "You are an autonomous financial-crime compliance investigator operating inside a stateful multi-day sandbox. Gather all required evidence using tools, advance the clock with wait_days when records or interbank notices are pending, trace multi-hop beneficial ownership where applicable, and conclude by calling commit_verdict."
_ENT_P, _VER_P = {"type": "object", "properties": {"entity_id": {"type": "string"}}, "required": ["entity_id"]}, {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["clear", "escalate", "block"]}, "cited_evidence": {"type": "array", "items": {"type": "string"}}, "rationale": {"type": "string"}}, "required": ["verdict", "cited_evidence", "rationale"]}
TOOLS: list[dict[str, Any]] = [{"name": "inspect_dossier", "description": "Inspect entity KYC, wires, and notices settled up to current simulated day.", "parameters": _ENT_P}, {"name": "trace_ownership", "description": "Return direct 1-hop owners of entity_id. Call on parent entities to trace multi-hop shells.", "parameters": _ENT_P}, {"name": "wait_days", "description": "Advance simulated compliance clock by 1 to 14 days so pending notices/batches settle.", "parameters": {"type": "object", "properties": {"days": {"type": "integer"}}, "required": ["days"]}}, {"name": "commit_verdict", "description": "Commit final audited disposition ('clear', 'escalate', or 'block') with cited evidence IDs and rationale.", "parameters": _VER_P}]


def _canon_eid(raw: str, spec: dict[str, Any]) -> str:
    val, legal = raw.strip(), spec["dossier_by_day"]["0"].get("legal_name", "")
    if val in spec["ownership_graph"] or val in spec["entity_labels"]:
        return spec["subject_id"] if val == legal else val
    return spec["subject_id"] if legal and legal.lower() in val.lower() else val


def _fmt_model(raw: str) -> str:
    return " ".join(p.capitalize() for p in re.split(r"[-_\s]+", raw.strip().removeprefix("models/")) if p) or "Gemini Model"


class SandboxEpisode:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec, self.sim_day, self.verdict, self.rationale, self.observed_blob, self.cited, self.steps = spec, 0, None, "", "", [], []

    def step(self, name: str, args: dict[str, Any], *, thought_sig: bool = False) -> dict[str, Any]:
        if self.verdict is not None:
            return {"error": "verdict already committed"}
        if name == "inspect_dossier":
            res: dict[str, Any] = {"sim_day": self.sim_day, "records_by_day": {k: v for k, v in self.spec["dossier_by_day"].items() if int(k) <= self.sim_day}}
        elif name == "trace_ownership":
            eid = _canon_eid(str(args.get("entity_id", "")), self.spec)
            res = {"entity_id": eid, "direct_owners": self.spec["ownership_graph"].get(eid, [])}
        elif name == "wait_days":
            delta = max(1, min(int(args.get("days", 1)), 14))
            self.sim_day += delta
            res = {"sim_day": self.sim_day, "newly_settled_days": [k for k in self.spec["dossier_by_day"] if self.sim_day - delta < int(k) <= self.sim_day]}
        elif name == "commit_verdict":
            self.verdict = str(args.get("verdict", "")).lower().strip()
            self.cited, self.rationale = [str(x) for x in (args.get("cited_evidence") or [])], str(args.get("rationale", "")).strip()
            res = {"committed": True, "verdict": self.verdict, "sim_day": self.sim_day}
        else:
            res = {"error": f"unknown tool {name}"}
        self.observed_blob += " " + json.dumps(res)
        h_in = json.dumps({"id": self.spec["id"], "day": self.sim_day, "verdict": self.verdict, "n": len(self.steps)}, sort_keys=True)
        self.steps.append({"step": len(self.steps) + 1, "day": self.sim_day, "tool": name, "args": {k: v for k, v in args.items() if k != "rationale"}, "thought_sig": thought_sig, "state_hash": hashlib.sha256(h_in.encode("utf-8")).hexdigest()[:16]})
        return res

    def grade(self, model: str, thinking: str, tokens: int) -> dict[str, Any]:
        req_ev = self.spec["required_evidence"]
        passed = self.verdict == self.spec["expected_verdict"] and self.sim_day >= self.spec["required_min_day"] and (req_ev in self.observed_blob or req_ev in " ".join(self.cited))
        return {"model": model, "thinking": thinking, "verdict": self.verdict or "none", "cited_evidence": self.cited, "rationale": self.rationale, "sim_day_final": self.sim_day, "tokens_used": tokens, "passed": passed, "liability_incurred_usd": 0 if passed else int(self.spec["liability_usd"]), "steps": self.steps}


def run_live_agent(spec: dict[str, Any], model: str, *, high_thinking: bool, api_key: str = "", max_turns: int = 8) -> dict[str, Any]:
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ValueError("No Gemini API key configured; paste an API key above to run live.")
    ep, contents = SandboxEpisode(spec), [{"role": "user", "parts": [{"text": spec["alert_summary"]}]}]
    gen_cfg: dict[str, Any] = {"maxOutputTokens": 8192, "thinkingConfig": {"thinkingLevel": "HIGH" if high_thinking else "LOW"}}
    if not high_thinking or "carbon" not in model.lower():
        gen_cfg["temperature"] = 0.0
    total_tokens, active_model, fallbacks = 0, model, {"gemini-4-carbon": "gemini-4-carbon-preview", "gemini-4-hydrogen": "gemini-3.8-flash"}
    base, hdr = "https://generativelanguage.googleapis.com/v1beta/models", {"x-goog-api-key": key}
    for _ in range(max_turns):
        payload = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents, "tools": [{"functionDeclarations": TOOLS}], "generationConfig": gen_cfg}
        resp = httpx.post(f"{base}/{active_model}:generateContent", headers=hdr, json=payload, timeout=90.0)
        if resp.status_code == 404 and active_model in fallbacks:
            active_model = fallbacks[active_model]
            resp = httpx.post(f"{base}/{active_model}:generateContent", headers=hdr, json=payload, timeout=90.0)
        if resp.status_code == 400 and not high_thinking and "thinkingConfig" in gen_cfg:
            gen_cfg.pop("thinkingConfig", None)
            resp = httpx.post(f"{base}/{active_model}:generateContent", headers=hdr, json=payload, timeout=90.0)
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini API ({active_model}) returned HTTP {resp.status_code}")
        data = resp.json()
        total_tokens += int((data.get("usageMetadata") or {}).get("totalTokenCount", 0))
        fcalls = [p for p in ((((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []) if "functionCall" in p]
        if not fcalls:
            break
        model_parts, user_parts = [], []
        for p in fcalls:
            fc, sig = p["functionCall"], p.get("thoughtSignature")
            f_args = dict(fc.get("args") or {})
            model_parts.append({"functionCall": {"name": fc["name"], "args": f_args}, **({"thoughtSignature": sig} if sig else {})})
            user_parts.append({"functionResponse": {"name": fc["name"], "response": ep.step(fc["name"], f_args, thought_sig=bool(sig) or high_thinking)}})
            if ep.verdict is not None:
                break
        if ep.verdict is not None:
            break
        contents.extend([{"role": "model", "parts": model_parts}, {"role": "user", "parts": user_parts}])
    return ep.grade(active_model, str((gen_cfg.get("thinkingConfig") or {}).get("thinkingLevel", "STANDARD")), total_tokens)


def resolve_case(raw: str) -> dict[str, Any]:
    key, by_slug = raw.strip(), {s["slug"]: s for s in SCENARIO_LIST}
    return by_slug[key] if key in by_slug else (SCENARIOS[key] if key in SCENARIOS else (SCENARIO_LIST[int(key) - 1] if key in {"1", "2", "3"} else SCENARIO_LIST[0]))


def audit_run(spec: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    steps, min_d = run.get("steps") or [], int(spec["required_min_day"])
    v_idx = next((i for i, s in enumerate(steps) if s.get("tool") == "commit_verdict"), -1)
    pre, c_day = (steps[:v_idx], int(steps[v_idx].get("day", 0))) if v_idx >= 0 else (steps, -1)
    d_steps, owner_steps = [s for s in pre if s.get("tool") == "inspect_dossier"], [s for s in pre if s.get("tool") == "trace_ownership"]
    traced = {_canon_eid(str((s.get("args") or {}).get("entity_id", "")), spec) for s in owner_steps}
    dossier_ok, owner_ok = bool(d_steps), all(req in traced for req in spec["owner_required_targets"])
    wait_ok = c_day >= min_d and any(int(s.get("day", 0)) >= min_d for s in d_steps)
    v_val = str((steps[v_idx].get("args") or {}).get("verdict") or run.get("verdict", "")).lower() if v_idx >= 0 else ""
    label_ok = v_idx >= 0 and v_val == spec["expected_verdict"]
    verdict_ok = label_ok and bool(run.get("passed")) and c_day >= min_d
    return {"dossier_ok": dossier_ok, "d_day": int(d_steps[0].get("day", 0)) if d_steps else 0, "owner_ok": owner_ok, "owner_n": len(owner_steps), "wait_ok": wait_ok, "c_day": c_day, "label_ok": label_ok, "verdict_ok": verdict_ok, "full_ok": dossier_ok and owner_ok and wait_ok and verdict_ok}


def _verdict_note(spec: dict[str, Any], run: dict[str, Any], aud: dict[str, Any]) -> tuple[str, str]:
    if aud["full_ok"]:
        return ("", spec["verdict_ok_sub"])
    if not aud["label_ok"]:
        return ("No verdict committed", "Stopped before committing a disposition") if aud["c_day"] < 0 else ("Wrong disposition", f'Committed {str(run.get("verdict", "")).lower()} instead of {spec["expected_verdict"]}')
    failed = [k for k in ("dossier_ok", "owner_ok", "wait_ok") if not aud[k]]
    if failed == ["owner_ok"] and aud["verdict_ok"]:
        return ("Skipped owner gate", spec["verdict_unverified_sub"])
    if failed in (["wait_ok"], ["dossier_ok"]):
        return ("Premature verdict", spec["wait_skip_sub"]) if failed == ["wait_ok"] else ("Skipped file review", "Committed label without entity file check")
    return ("Missing proof citation", "Committed label without required evidence") if not failed else ("Incomplete audit", "Committed disposition with multiple gates skipped")


def _human_step(spec: dict[str, Any], step: dict[str, Any], v_sub: str) -> tuple[str, str]:
    tool, args, day = step.get("tool", ""), step.get("args") or {}, int(step.get("day", 0))
    eid = _canon_eid(str(args.get("entity_id", "")), spec)
    label = spec["entity_labels"].get(eid) or _CODE_RE.sub("subject file", eid).strip() or "subject file"
    if tool == "inspect_dossier":
        return ("Settled records", spec["settled_step_label"]) if eid == spec["subject_id"] and day >= int(spec["required_min_day"]) > 0 else ("Owner file" if eid != spec["subject_id"] else "Entity file", label)
    if tool == "trace_ownership":
        return ("Owner trace", spec["owner_edge_labels"].get(eid, f"Checked upstream owners for {label}"))
    return ("Settlement wait", spec["wait_step_label"]) if tool == "wait_days" else ("Final decision", v_sub)


app = FastAPI(title="Floe Stateful Compliance Sandbox")


@app.get("/healthz")
@app.get("/status")
def status() -> dict[str, str]:
    return {"status": "ok", "arena": "floe-carbon-sandbox"}


@app.get("/api/runs")
def api_runs() -> dict[str, Any]:
    return STATE


@app.post("/run-live", response_model=None)
def trigger_live_run(request: Request, case_slug: str = Form("bakery"), api_key: str = Form("")) -> JSONResponse | RedirectResponse:
    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        return JSONResponse({"error": "cross-site POST blocked"}, status_code=403)
    spec, user_key = resolve_case(case_slug), api_key.strip()
    slug, cid, lead = spec["slug"], spec["id"], spec["context_lead"]
    now, ip = time.time(), (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "local")).split(",")[0].strip()
    _RATE_LOG[:] = [(t, a) for t, a in _RATE_LOG if now - t < 3600]
    if not user_key:
        if len(_RATE_LOG) >= 20 or sum(1 for t, a in _RATE_LOG if a == ip and now - t < 900) >= 5:
            return RedirectResponse(f"/?case={slug}&err={quote('Public rate limit reached; paste your own Gemini API key to run immediately.')}", status_code=303)
        _RATE_LOG.append((now, ip))
    c_mod, b_mod = os.getenv("CARBON_MODEL", "gemini-4-carbon"), os.getenv("BASELINE_MODEL", "gemini-4-hydrogen")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fc, fb = pool.submit(run_live_agent, spec, c_mod, high_thinking=True, api_key=user_key), pool.submit(run_live_agent, spec, b_mod, high_thinking=False, api_key=user_key)
            STATE["episodes"][cid] = {"carbon": fc.result(), "baseline": fb.result()}
        STATE["mode"] = f"live_cloud_run ({datetime.now(UTC).strftime('%H:%M:%S')} UTC)"
        return RedirectResponse(f"/?case={slug}&msg={quote(f'Live audit finished on {lead}')}", status_code=303)
    except Exception as exc:
        return RedirectResponse(f"/?case={slug}&err={quote(_CODE_RE.sub('alert', str(exc)))}", status_code=303)


def _stage_li(icon_svg: str, title: str, sub: str, tag: str, *, ok: bool, focus: bool = False) -> str:
    esc = html.escape
    row_cls = "stage-row stage-row--focus" if (ok and focus) else ("stage-row" if ok else "stage-row stage-row--fail")
    ic_cls, (t_cls, tag_cls) = ("stage-icon stage-icon--ok" if ok else "stage-icon stage-icon--fail"), (("stage-title", "stage-tag") if ok else ("stage-title stage-title--fail", "stage-tag stage-tag--fail"))
    return f'<li class="{row_cls}"><div class="stage-left"><span class="{ic_cls}">{icon_svg}</span><div><div class="{t_cls}">{esc(title)}</div><div class="stage-sub">{esc(sub)}</div></div></div><div class="stage-right">{IC["check"] if ok else ""}<span class="{tag_cls}">{esc(tag)}</span></div></li>'


@app.get("/", response_class=HTMLResponse)
def arena_ui(case: str = Query("bakery"), msg: str = Query(""), err: str = Query("")) -> HTMLResponse:
    esc, spec, eps, n_tot = html.escape, resolve_case(case), dict(STATE["episodes"]), len(SCENARIO_LIST)
    cid, slug = spec["id"], spec["slug"]
    audits = {s["id"]: {"c": audit_run(s, eps[s["id"]]["carbon"]), "b": audit_run(s, eps[s["id"]]["baseline"])} for s in SCENARIO_LIST}
    b_skip = sum(1 for a in audits.values() if not a["b"]["owner_ok"])
    c_lbl, b_lbl = sum(1 for a in audits.values() if a["c"]["label_ok"]), sum(1 for a in audits.values() if a["b"]["label_ok"])
    c_full, b_full = sum(1 for a in audits.values() if a["c"]["full_ok"]), sum(1 for a in audits.values() if a["b"]["full_ok"])
    c_steps, b_steps = sum(len(eps[s["id"]]["carbon"]["steps"]) for s in SCENARIO_LIST), sum(len(eps[s["id"]]["baseline"]["steps"]) for s in SCENARIO_LIST)
    all_ch = all("carbon" in str(eps[s["id"]]["carbon"].get("model", "")).lower() and "hydrogen" in str(eps[s["id"]]["baseline"].get("model", "")).lower() for s in SCENARIO_LIST)
    bk_aud, b_role = audits["TRAP-03"]["b"], "Hydrogen" if "hydrogen" in str(eps["TRAP-03"]["baseline"].get("model", "")).lower() else "Baseline"
    skip_sub = f"{b_role} cleared an 89% hit blind" if (not bk_aud["owner_ok"] and bk_aud["label_ok"]) else (f"{b_role} skipped owner on {b_skip} of {n_tot}" if b_skip else "All owner checks verified")
    lbl_op, lbl_sub = ("=", "Carbon (High) ties Hydrogen (Low)" if all_ch else "Both runs tie on final labels") if c_lbl == b_lbl else ("vs", f"Carbon {c_lbl}/{n_tot} vs baseline {b_lbl}/{n_tot} labels")
    tab_parts = [f'<a class="case-tab{" case-tab--last" if i == n_tot - 1 else ""}{" case-tab--active" if s["slug"] == slug else ""}" href="/?case={s["slug"]}"><div class="tab-left"><span class="tab-icon">{IC[s["tab_icon"]]}</span><div><div class="tab-name">{esc(s["tab_name"])}</div><div class="tab-hint">{esc(s["tab_hint"])}</div></div></div><span class="tab-flag{" tab-flag--ok" if audits[s["id"]]["c"]["full_ok"] and audits[s["id"]]["b"]["full_ok"] else ""}">{f"Verified ({len(eps[s["id"]]["carbon"]["steps"])} vs {len(eps[s["id"]]["baseline"]["steps"])})" if audits[s["id"]]["c"]["full_ok"] and audits[s["id"]]["b"]["full_ok"] else ("Owner skipped" if not audits[s["id"]]["b"]["owner_ok"] else "Audit failed")}</span></a>' for i, s in enumerate(SCENARIO_LIST)]
    flash = f'<div class="flash flash--ok">{esc(msg)}</div>' if msg else (f'<div class="flash flash--err">{esc(err)}</div>' if err else "")

    def _pane(run: dict[str, Any], aud: dict[str, Any], *, right: bool = False) -> str:
        err_txt, v_sub = _verdict_note(spec, run, aud)
        pill = f'<span class="verdict-pill verdict-pill--ok">{IC["check"]}All 4 gates verified</span>' if aud["full_ok"] else f'<span class="verdict-pill verdict-pill--err">{IC["cross"]}{esc(err_txt)}</span>'
        w_tag = f'Day {aud["c_day"]}' if aud["wait_ok"] else ("Incomplete" if aud["c_day"] < 0 or int(spec["required_min_day"]) == 0 else "Premature")
        stages = _stage_li(IC["doc"], "Entity dossier", spec["dossier_sub"], f'Day {aud["d_day"]}' if aud["dossier_ok"] else "Missed", ok=aud["dossier_ok"]) + _stage_li(IC["owner"] if aud["owner_ok"] else IC["cross"], "Beneficial owner", spec["owner_ok_sub"] if aud["owner_ok"] else spec["owner_skip_sub"], f'Verified ({aud["owner_n"]}x)' if aud["owner_ok"] else "Skipped", ok=aud["owner_ok"], focus=not right) + _stage_li(IC["clock"], "Settlement wait", spec["wait_ok_sub"] if aud["wait_ok"] else spec["wait_skip_sub"], w_tag, ok=aud["wait_ok"]) + _stage_li(IC["shield"] if aud["full_ok"] else IC["warn_raw"], "Audited verdict", v_sub, str(run.get("verdict", "none")).capitalize() if aud["full_ok"] else "Unverified", ok=aud["full_ok"])
        step_lis = "".join(f'<li class="step-row"><span><strong>{s["step"]}. {esc(t)}</strong> · {esc(d)}</span><span class="step-day">Day {s["day"]}</span></li>' for s in run["steps"] for t, d in [_human_step(spec, s, v_sub)])
        n_st, tok, m_lbl, thk = len(run["steps"]), int(run.get("tokens_used", 0)), _fmt_model(str(run.get("model", ""))), str(run.get("thinking") or "low").strip().capitalize()
        return f'<div class="model-pane{" model-pane--right" if right else ""}"><div class="pane-top"><div><h2>{esc(m_lbl)}</h2><div class="model-meta">{esc(thk)} reasoning · <span class="tab-num">{n_st}</span> steps · <span class="tab-num">{tok:,}</span> tokens</div></div>{pill}</div><ul class="stage-list">{stages}</ul><details class="trace-drawer"><summary class="trace-summary"><span>Inspect all {n_st} verification steps</span><span class="trace-toggle">Trace {IC["chev"]}</span></summary><ol class="step-list">{step_lis}</ol></details></div>'

    fonts = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
    return HTMLResponse(f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Floe — Process-level compliance agent audit</title>{fonts}<style>{CSS_TEXT}</style></head><body><main class="page"><header class="top-bar"><div class="brand-cluster">{IC["mark"]}<div class="brand-title"><span class="eyebrow"><span class="wordmark">Floe</span> stateful compliance sandbox · Antigravity CLI + Gemini Carbon</span><h1>Standard outcome evals call these agents tied</h1></div></div><form class="live-form" method="post" action="/run-live"><input type="hidden" name="case_slug" value="{esc(slug)}"><input class="key-input" type="password" name="api_key" placeholder="Optional Gemini API key" autocomplete="off"><button class="btn btn--primary" type="submit">{IC["play"]}Run live audit</button></form></header>{flash}<section class="ledger-sheet"><div class="summary-band"><div class="summary-cell{" summary-cell--focal" if b_skip else ""}"><div class="cell-head">{IC["warn"] if b_skip else IC["shield_ok"]}<span>Skipped owner check</span></div><div class="{"focal-val" if b_skip else "stat-val stat-val--ok"}">{b_skip} of {n_tot}</div><div class="cell-sub">{esc(skip_sub)}</div></div><div class="summary-cell"><div class="cell-head">{IC["eq"]}<span>Final verdict labels</span></div><div class="stat-val">{c_lbl}/{n_tot} {lbl_op} {b_lbl}/{n_tot}</div><div class="cell-sub">{esc(lbl_sub)}</div></div><div class="summary-cell"><div class="cell-head">{IC["shield_ok"]}<span>Stateful gate audit</span></div><div class="stat-val stat-val--ok">{c_full}/{n_tot} vs {b_full}/{n_tot}</div><div class="cell-sub">Owner trace + Day 1 / Day 3 waits</div></div><div class="summary-cell summary-cell--last"><div class="cell-head">{IC["nodes"]}<span>Verification actions</span></div><div class="stat-val">{c_steps} vs {b_steps}</div><div class="depth-meter" aria-hidden="true"><div class="depth-carbon"></div><div class="depth-base"></div></div><div class="cell-sub">Carbon {c_steps} checks · {"Hydrogen" if all_ch else "Baseline"} {b_steps}</div></div></div><nav class="case-tabs" aria-label="Compliance scenarios">{"".join(tab_parts)}</nav><div class="context-bar"><div class="context-main">{IC["info"]}<span><strong>{esc(spec["context_lead"])}</strong> · {esc(spec["context_body"])}</span></div><div class="stakes-label">{esc(spec["stakes_text"])}</div></div><div class="compare-grid">{_pane(eps[cid]["carbon"], audits[cid]["c"])}{_pane(eps[cid]["baseline"], audits[cid]["b"], right=True)}</div><footer class="sheet-foot"><span>Public sandbox · Private uncontaminated multi-day bank environments for institutions</span><a class="foot-link" href="https://floelabs.io">Request institutional evaluation at floelabs.io →</a></footer></section></main></body></html>')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture", action="store_true")
    p.add_argument("--carbon-model", default=os.getenv("CARBON_MODEL", "gemini-4-carbon"))
    p.add_argument("--baseline-model", default=os.getenv("BASELINE_MODEL", "gemini-4-hydrogen"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    args = p.parse_args()
    if args.capture:
        for s in SCENARIO_LIST:
            STATE["episodes"][s["id"]] = {"carbon": run_live_agent(s, args.carbon_model, high_thinking=True), "baseline": run_live_agent(s, args.baseline_model, high_thinking=False)}
        STATE.update({"captured_at": datetime.now(UTC).isoformat(), "mode": "live_gemini_api"})
        RUNS_PATH.write_text(json.dumps(STATE, indent=2) + "\n", "utf-8")
        return
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
