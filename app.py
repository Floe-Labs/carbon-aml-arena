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
    return (spec["subject_id"] if val == legal else val) if (val in spec["ownership_graph"] or val in spec["entity_labels"]) else (spec["subject_id"] if legal and legal.lower() in val.lower() else val)


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
            self.verdict, self.cited, self.rationale = str(args.get("verdict", "")).lower().strip(), [str(x) for x in (args.get("cited_evidence") or [])], str(args.get("rationale", "")).strip()
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
        raise ValueError("Server Gemini API key is not configured.")
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


app = FastAPI(title="Floe FinCrime Compliance Gym")


@app.get("/healthz")
@app.get("/status")
def status() -> dict[str, str]:
    return {"status": "ok", "arena": "floe-carbon-sandbox"}


@app.get("/api/runs")
def api_runs() -> dict[str, Any]:
    return STATE


@app.post("/run-live", response_model=None)
def trigger_live_run(request: Request, case_slug: str = Form("bakery")) -> JSONResponse | RedirectResponse:
    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        return JSONResponse({"error": "cross-site POST blocked"}, status_code=403)
    spec = resolve_case(case_slug)
    slug, cid, lead = spec["slug"], spec["id"], spec["context_lead"]
    now, ip = time.time(), (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "local")).split(",")[0].strip()
    _RATE_LOG[:] = [(t, a) for t, a in _RATE_LOG if now - t < 3600]
    if len(_RATE_LOG) >= 20 or sum(1 for t, a in _RATE_LOG if a == ip and now - t < 900) >= 5:
        return RedirectResponse(f"/?case={slug}&err={quote('Public live-run rate limit reached; please retry shortly.')}", status_code=303)
    _RATE_LOG.append((now, ip))
    c_mod, b_mod = os.getenv("CARBON_MODEL", "gemini-4-carbon"), os.getenv("BASELINE_MODEL", "gemini-4-hydrogen")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fc, fb = pool.submit(run_live_agent, spec, c_mod, high_thinking=True), pool.submit(run_live_agent, spec, b_mod, high_thinking=False)
            STATE["episodes"][cid] = {"carbon": fc.result(), "baseline": fb.result()}
        STATE["mode"] = f"live_cloud_run ({datetime.now(UTC).strftime('%H:%M:%S')} UTC)"
        return RedirectResponse(f"/?case={slug}&msg={quote(f'Live audit finished on {lead}')}", status_code=303)
    except Exception as exc:
        return RedirectResponse(f"/?case={slug}&err={quote(_CODE_RE.sub('alert', str(exc)))}", status_code=303)


@app.get("/", response_class=HTMLResponse)
def arena_ui(case: str = Query("bakery"), msg: str = Query(""), err: str = Query("")) -> HTMLResponse:
    esc, spec, eps, n_tot = html.escape, resolve_case(case), dict(STATE["episodes"]), len(SCENARIO_LIST)
    cid, slug = spec["id"], spec["slug"]
    audits = {s["id"]: {"c": audit_run(s, eps[s["id"]]["carbon"]), "b": audit_run(s, eps[s["id"]]["baseline"])} for s in SCENARIO_LIST}
    b_skip, c_lbl, b_lbl = sum(1 for a in audits.values() if not a["b"]["owner_ok"]), sum(1 for a in audits.values() if a["c"]["label_ok"]), sum(1 for a in audits.values() if a["b"]["label_ok"])
    c_full, b_full, c_steps, b_steps = sum(1 for a in audits.values() if a["c"]["full_ok"]), sum(1 for a in audits.values() if a["b"]["full_ok"]), sum(len(eps[s["id"]]["carbon"]["steps"]) for s in SCENARIO_LIST), sum(len(eps[s["id"]]["baseline"]["steps"]) for s in SCENARIO_LIST)
    all_ch = all("carbon" in str(eps[s["id"]]["carbon"].get("model", "")).lower() and "hydrogen" in str(eps[s["id"]]["baseline"].get("model", "")).lower() for s in SCENARIO_LIST)
    bk_aud, b_role, c_run, b_run = audits["TRAP-03"]["b"], ("Hydrogen" if "hydrogen" in str(eps["TRAP-03"]["baseline"].get("model", "")).lower() else "Baseline"), eps[cid]["carbon"], eps[cid]["baseline"]
    c_aud, b_aud = audits[cid]["c"], audits[cid]["b"]
    skip_sub = f"{b_role} cleared an 89% hit blind" if (cid == "TRAP-03" and not bk_aud["owner_ok"] and bk_aud["label_ok"]) else (f"{b_role} decided {spec['tab_name']} blind" if (not b_aud["owner_ok"] and b_aud["label_ok"]) else (f"{b_role} bypassed Bakery + Meridian" if (not bk_aud["owner_ok"] and not audits["TRAP-01"]["b"]["owner_ok"] and bk_aud["label_ok"]) else (f"{b_role} skipped owner on {b_skip} of {n_tot}" if b_skip else "All owner checks verified")))
    lbl_op, lbl_sub = ("=", "Carbon (High) ties Hydrogen (Low)" if all_ch else "Both runs tie on final labels") if c_lbl == b_lbl else ("vs", f"Carbon {c_lbl}/{n_tot} vs baseline {b_lbl}/{n_tot} labels")
    pips_owner = "".join(f'<span class="pip-box {"pip-box--ok" if audits[s["id"]]["b"]["owner_ok"] else "pip-box--err"}">{IC["check"] if audits[s["id"]]["b"]["owner_ok"] else IC["cross"]}</span>' for s in SCENARIO_LIST)
    pips_c, pips_b = "".join(f'<span class="pip-seg{" pip-seg--ok" if audits[s["id"]]["c"]["label_ok"] else ""}"></span>' for s in SCENARIO_LIST), "".join(f'<span class="pip-seg{" pip-seg--p2" if audits[s["id"]]["b"]["label_ok"] else ""}"></span>' for s in SCENARIO_LIST)
    tabs = "".join(f'<a class="mission-card{" mission-card--last" if i == n_tot - 1 else ""}{" mission-card--active" if s["slug"] == slug else ""}" href="/?case={s["slug"]}"><div class="mission-left"><span class="mission-icon">{IC[s["tab_icon"]]}</span><div><div class="mission-name">{esc(s["tab_name"])}</div><div class="mission-hint">{esc(s["tab_hint"])}</div></div></div><span class="mission-flag{" mission-flag--ok" if audits[s["id"]]["c"]["full_ok"] and audits[s["id"]]["b"]["full_ok"] else ""}">{(IC["check"] + f" Verified {len(eps[s["id"]]["carbon"]["steps"])}/{len(eps[s["id"]]["baseline"]["steps"])}") if audits[s["id"]]["c"]["full_ok"] and audits[s["id"]]["b"]["full_ok"] else (IC["cross"] + (" Bypass" if not audits[s["id"]]["b"]["owner_ok"] else " Failed"))}</span></a>' for i, s in enumerate(SCENARIO_LIST))
    flash = f'<div class="flash flash--ok">{esc(msg)}</div>' if msg else (f'<div class="flash flash--err">{esc(err)}</div>' if err else "")
    (_, c_vsub), (_, b_vsub), c_name, b_name = _verdict_note(spec, c_run, c_aud), _verdict_note(spec, b_run, b_aud), _fmt_model(str(c_run.get("model", ""))), _fmt_model(str(b_run.get("model", "")))

    def _lane(run: dict[str, Any], aud: dict[str, Any], *, p2: bool = False) -> str:
        def _nd(i: int, svg: str, t: str, tag: str, m: str) -> str:
            return f'<label for="st-{i}" class="wf-node wf-node--s{i} wf-node--{m}"><span class="wf-main"><span class="wf-icon">{svg}</span><span class="wf-copy"><span class="wf-title">{esc(t)}</span><span class="wf-tag">{esc(tag)}</span></span></span><span class="wf-cue">0{i}{IC["chev_r"]}</span></label>'
        err_txt, _ = _verdict_note(spec, run, aud)
        pill = f'<span class="verdict-pill verdict-pill--ok">{IC["check"]}All 4 gates verified</span>' if aud["full_ok"] else f'<span class="verdict-pill verdict-pill--err">{IC["cross"]}{esc(err_txt)}</span>'
        arc_cls, arc_txt = ("bypass-arc--1-3", "Bypassed owner") if (aud["dossier_ok"] and not aud["owner_ok"] and aud["wait_ok"]) else (("bypass-arc--2-4", "Bypassed settle") if (aud["dossier_ok"] and aud["owner_ok"] and not aud["wait_ok"] and aud["c_day"] >= 0) else (("bypass-arc--1-4", "Skipped gates") if (aud["dossier_ok"] and not aud["owner_ok"] and not aud["wait_ok"] and aud["c_day"] >= 0) else ("", "")))
        bypass = f'<div class="bypass-grid"><div class="bypass-arc {arc_cls}"><span class="bypass-pill">{IC["bolt_err"]}{arc_txt}</span><span class="bypass-head">{IC["arr_d"]}</span></div></div>' if arc_cls else ""
        w_tag, v_lbl = (f'Day {aud["c_day"]}' if aud["wait_ok"] else ("Missed" if aud["c_day"] < 0 or int(spec["required_min_day"]) == 0 else "Skipped")), (str(run.get("verdict", "none")).capitalize() if aud["c_day"] >= 0 else "None")
        nodes = [_nd(1, IC["doc"], "Dossier", f'Day {aud["d_day"]}' if aud["dossier_ok"] else "Missed", "ok" if aud["dossier_ok"] else "fog"), _nd(2, IC["owner"] if aud["owner_ok"] else IC["lock_fog"], "Owner", f'Traced {aud["owner_n"]}x' if aud["owner_ok"] else "Fog of war", "ok" if aud["owner_ok"] else "fog"), _nd(3, IC["clock"], "Settle", w_tag, "ok" if aud["wait_ok"] else "fog"), _nd(4, IC["shield"] if aud["full_ok"] else IC["warn_raw"], v_lbl, "Audited" if aud["full_ok"] else "Unverified", "ok" if aud["full_ok"] else "fail")]
        links = [f'<span class="wf-link{"" if ok else " wf-link--cut"}">{IC["arr_r"] if ok else IC["cross"]}</span>' for ok in (aud["dossier_ok"] and aud["owner_ok"], aud["owner_ok"] and aud["wait_ok"], aud["wait_ok"] and aud["c_day"] >= 0)]
        n_st, tok, m_lbl, thk = len(run["steps"]), int(run.get("tokens_used", 0)), _fmt_model(str(run.get("model", ""))), str(run.get("thinking") or "low").strip().capitalize()
        ptag = f'<span class="player-tag{" player-tag--p2" if p2 else ""}">{"Agent 2" if p2 else "Agent 1"}</span>'
        return f'<div class="agent-lane{"" if aud["full_ok"] else " agent-lane--alert"}"><div class="lane-head"><div class="lane-id">{ptag}<h2>{esc(m_lbl)}</h2><span class="lane-meta">{esc(thk)} reasoning · <span class="tab-num">{n_st}</span> steps · <span class="tab-num">{tok:,}</span> tok</span></div>{pill}</div>{bypass}<div class="wf-row">{nodes[0]}{links[0]}{nodes[1]}{links[1]}{nodes[2]}{links[2]}{nodes[3]}</div></div>'

    intel_defs = [(1, "Station 1 · Entity dossier", c_aud["dossier_ok"], spec["dossier_sub"] if c_aud["dossier_ok"] else "Locked in fog of war", b_aud["dossier_ok"], spec["dossier_sub"] if b_aud["dossier_ok"] else "Locked in fog of war"), (2, "Station 2 · Beneficial ownership", c_aud["owner_ok"], spec["owner_ok_sub"] if c_aud["owner_ok"] else spec["owner_skip_sub"], b_aud["owner_ok"], spec["owner_ok_sub"] if b_aud["owner_ok"] else spec["owner_skip_sub"]), (3, "Station 3 · Settlement clock", c_aud["wait_ok"], spec["wait_ok_sub"] if c_aud["wait_ok"] else spec["wait_skip_sub"], b_aud["wait_ok"], spec["wait_ok_sub"] if b_aud["wait_ok"] else spec["wait_skip_sub"]), (4, "Station 4 · Audited verdict", c_aud["full_ok"], c_vsub, b_aud["full_ok"], b_vsub)]
    panels = "".join(f'<div class="intel-panel intel-panel--{k}"><div class="intel-head"><span>{esc(st_t)} <span class="intel-brief">· {esc(spec["context_body"])}</span></span></div><div class="intel-grid"><div class="intel-card{"" if cok else " intel-card--err"}"><div class="intel-who">{esc(c_name)} · {("Audited" if k == 4 else "Unlocked") if cok else ("Unverified" if k == 4 else "Fog of war")}</div><div class="intel-body">{esc(ctxt)}</div></div><div class="intel-card{"" if bok else " intel-card--err"}"><div class="intel-who">{esc(b_name)} · {("Audited" if k == 4 else "Unlocked") if bok else ("Unverified" if k == 4 else "Fog of war")}</div><div class="intel-body">{esc(btxt)}</div></div></div></div>' for k, st_t, cok, ctxt, bok, btxt in intel_defs)
    c_lis, b_lis = "".join(f'<li class="step-row"><span><strong>{s["step"]}. {esc(t)}</strong> · {esc(d)}</span><span class="step-day">Day {s["day"]}</span></li>' for s in c_run["steps"] for t, d in [_human_step(spec, s, c_vsub)]), "".join(f'<li class="step-row"><span><strong>{s["step"]}. {esc(t)}</strong> · {esc(d)}</span><span class="step-day">Day {s["day"]}</span></li>' for s in b_run["steps"] for t, d in [_human_step(spec, s, b_vsub)])
    radios = "".join(f'<input class="st-rad" type="radio" name="st" id="st-{i}"{" checked" if i == 0 else ""}>' for i in range(5))
    fonts = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
    return HTMLResponse(f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Floe — Financial crime compliance training sandbox</title>{fonts}<style>{CSS_TEXT}\n.depth-meter{{grid-template-columns:{max(c_steps, 1)}fr {max(b_steps, 1)}fr}}</style></head><body><main class="page"><header class="top-bar"><div class="brand-cluster">{IC["mark"]}<div class="brand-title"><span class="eyebrow"><span class="wordmark">Floe</span> financial crime compliance training sandbox using Antigravity + Gemini</span><h1>Standard evals result in failed models/agents evaluation (as tied)</h1></div></div><form class="live-form" method="post" action="/run-live"><input type="hidden" name="case_slug" value="{esc(slug)}"><button class="btn btn--primary" type="submit">{IC["play"]}Run live</button></form></header>{flash}<section class="ops-board"><div class="hud-strip"><div class="hud-cell{" hud-cell--alert" if b_skip else ""}"><div class="hud-head">{IC["lock_open"]}<span>Skipped owner gate</span></div><div class="hud-main"><span class="{"focal-val" if b_skip else "stat-val stat-val--ok"}">{b_skip} of {n_tot}</span><span class="pip-group">{pips_owner}</span></div><div class="hud-sub">{esc(skip_sub)}</div></div><div class="hud-cell"><div class="hud-head">{IC["eq"]}<span>Final verdict labels</span></div><div class="hud-main"><span class="stat-val">{c_lbl}/{n_tot} {lbl_op} {b_lbl}/{n_tot}</span><span class="pip-bars"><span class="pip-row">{pips_c}</span><span class="pip-row">{pips_b}</span></span></div><div class="hud-sub">{esc(lbl_sub)}</div></div><div class="hud-cell hud-cell--last"><div class="hud-head">{IC["steps"]}<span>Gym actions &amp; gates</span></div><div class="hud-main"><span class="stat-val"><span class="ink-ok">{c_steps}</span> vs {b_steps}</span><span class="depth-meter" aria-hidden="true"><span class="depth-carbon"></span><span class="depth-base"></span></span></div><div class="hud-sub">Full audit {c_full}/{n_tot} vs {b_full}/{n_tot}</div></div></div><nav class="mission-strip" aria-label="Crime world missions">{tabs}</nav><div class="arena-stage">{radios}<div class="mission-hud"><div class="mission-target">{IC["target"]}<strong>{esc(spec["context_lead"])}</strong><span class="stake-pill">{esc(spec["stake_badge"])}</span></div><span class="tap-hint">Click any station node to unlock intel</span><label for="st-0" class="close-intel">{IC["cross"]}Close intel</label></div><div class="tracks-stack">{_lane(c_run, c_aud)}{_lane(b_run, b_aud, p2=True)}</div>{panels}<details class="replay-drawer"><summary class="replay-summary"><span class="replay-left">{IC["log"]}<span>Gym replay log · {len(c_run["steps"])} vs {len(b_run["steps"])} actions</span></span><span class="replay-toggle"><span class="lbl-open">Expand</span><span class="lbl-close">Collapse</span> {IC["chev"]}</span></summary><div class="replay-cols"><div class="step-col"><div class="step-col-head">{esc(c_name)} ({len(c_run["steps"])} steps)</div><ol class="step-list">{c_lis}</ol></div><div class="step-col step-col--right"><div class="step-col-head">{esc(b_name)} ({len(b_run["steps"])} steps)</div><ol class="step-list">{b_lis}</ol></div></div></details></div></section></main></body></html>')


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
