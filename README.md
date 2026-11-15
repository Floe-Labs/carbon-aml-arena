# FinTrap Arena — Stateful Financial-Crime Stress Test for Gemini Carbon

Built from scratch with **Antigravity CLI (AGY)** and **Google Gemini Carbon**, deployed on **Google Cloud Run**.

## Why Single-Turn Compliance Benchmarks Fail

Banks and fintechs spend tens of billions annually investigating transaction-monitoring alerts, sanctions hits, and customer onboarding files. Standard single-turn QA benchmarks hide catastrophic operational failures because real financial-crime investigations are **stateful, multi-day processes with irreversible legal and financial consequences**:

1. **Delayed Adverse Evidence (`TRAP-01`):** Local Day-0 invoices look clean while a clearing-bank recall inquiry is still in transit (`Day 3` settlement). Fast reactive agents close the alert prematurely, allowing illicit funds to clear.
2. **Split-Shell Indirect Sanctions (`TRAP-02`):** Neither direct holding company exceeds 50% ownership alone (`28%` + `26%`), but tracing both branches 2 hops upstream reveals the same OFAC SDN individual controls both shells (`54% >= 50%` aggregate blocked property under the public OFAC 50% rule).
3. **Phonetic False-Positive Collisions (`TRAP-03`):** An innocent US small business triggers an 89% fuzzy SDN name score alongside a legitimate `>$10,000` weekend catering cash deposit whose POS batch and FinCEN CTR settle on `Day 1`. Over-triggering a block freezes employee payroll.

## Gemini Carbon Technical Integration

`app.py` (`<250` lines, zero external agent frameworks) runs **Gemini Carbon** (`thinkingLevel: "HIGH"` with multi-turn `thoughtSignature` preservation across `functionCall` / `functionResponse` turns) head-to-head against a fast low-thinking baseline (`gemini-4-hydrogen` with `thinkingLevel: "LOW"`) inside an isolated, deterministic 4-tool state machine:

- `inspect_dossier(entity_id)` — reads records settled up to `sim_day`
- `trace_ownership(entity_id)` — traverses 1-hop shareholder edges per call
- `wait_days(days)` — advances the simulated compliance clock (`Day 0 -> Day N`)
- `commit_verdict(verdict, cited_evidence, rationale)` — records final disposition (`clear | escalate | block`) with deterministic SHA-256 `state_hash` verification and zero-LLM grading

## Quickstart

```bash
pip install -r requirements.txt

# 1. (Optional) Capture fresh live API traces with Gemini Carbon + Baseline into runs.json
GEMINI_API_KEY="your_key" python app.py --capture

# 2. Run interactive replay & comparison server locally (http://localhost:8080)
python app.py
```

## Deploy to Google Cloud Run

The container loads instant deterministic baseline/Carbon traces on startup (`<50ms` cold render) **and** exposes an interactive **`▶ Run Live`** button (`POST /run-live` with per-IP/hourly rate limiting, cross-site POST protection, and optional BYO API key input) for live head-to-head execution on Cloud Run:

```bash
gcloud run deploy carbon-aml-arena \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --max-instances=2
```

## Anti-Contamination & Commercial Note

Because open-source repositories contaminate LLM pretraining corpora and invalidate held-out evaluation benchmarks, **FinTrap Arena** was built in a clean room using strictly public-domain regulatory rules and synthetic showcase scenarios. At **Floe Labs** (`floelabs.io`), we operate our closed-source institutional platform providing uncontaminated financial-crime RL environments, expert-adjudicated decision trajectories, and held-out agent validation for frontier AI labs and regulated financial institutions.
