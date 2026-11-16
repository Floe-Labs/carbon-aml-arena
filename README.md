# Floe Stateful Compliance Sandbox — Gemini Carbon vs. Hydrogen Audit

Built with **Antigravity CLI (AGY)** and **Google Gemini Carbon**, deployed on **Google Cloud Run** (`https://carbon-aml-arena-715925646198.us-central1.run.app`).

## Key Finding: Standard Outcome Evals Call These Agents Tied

On our stateful multi-day compliance sandbox with inline policy rules, both **Gemini Carbon** (`HIGH` reasoning + multi-turn `thoughtSignature`) and **Gemini Hydrogen** (`LOW` reasoning) hit **`3/3` on final verdict labels** (`Clear` / `Escalate` / `Block`). A standard single-label outcome benchmark calls them tied.

Deterministic step-trace auditing exposes a critical operational failure:

- **Skipped beneficial ownership (`2 of 3` alerts):** Hydrogen skipped `trace_ownership` completely on 2 of the 3 alerts—including clearing an **89% sanctions fuzzy name hit** without checking who owns the company.
- **Stateful process audit (`3/3` vs `1/3`):** Carbon verified every required ownership gate and multi-day settlement wait (`Day 1` card/cash batch and `Day 3` interbank recall) across all 3 alerts.
- **Verification depth (`27` vs `17` tool steps):** Carbon traced upstream beneficial owners, parent holding shells, and terminal nominees before committing dispositions.

## Three Stateful Compliance Traps

1. **Sanctions 89% Name Hit (`bakery`):** Local bakery triggers an 89% phonetic sanctions match alongside an `\$11,400` Friday catering cash deposit before payroll (`Day 1` batch settlement). Clearing without checking beneficial ownership creates unverified sanctions exposure; freezing wrongfully harms payroll.
2. **Interbank Wire Recall (`recall`):** Day-0 supplier invoices appear clean while a correspondent bank recall inquiry is still in transit (`Day 3` settlement confirming `\$68,600` split-bank structuring).
3. **Split Offshore Shells (`shells`):** Direct name screening returns zero hits, while tracing two offshore holding firms two hops upstream reveals the same sanctioned individual controls `28% + 26% = 54% >= 50%` aggregate blocked property.

## Technical Integration (`< 250` Lines of Python)

`app.py` runs `gemini-4-carbon` (`thinkingLevel: "HIGH"` with `thoughtSignature` preserved across `functionCall` / `functionResponse` turns) head-to-head against `gemini-4-hydrogen` (`thinkingLevel: "LOW"`) inside a zero-framework, deterministic 4-tool state machine (`inspect_dossier`, `trace_ownership`, `wait_days`, `commit_verdict`) with deterministic gate auditing.

```bash
pip install -r requirements.txt
GEMINI_API_KEY="your_key" python app.py --capture   # optional fresh capture
python app.py                                       # http://localhost:8080
```

## Anti-Contamination & Institutional Platform

Because public repositories enter pretraining corpora and invalidate held-out evaluations, this showcase is a clean-room slice built strictly from public regulatory principles and synthetic scenarios. **Floe Labs** (`https://floelabs.io`) operates private, uncontaminated multi-day bank environments, expert-adjudicated decision trajectories, and held-out agent validation for frontier AI labs and financial institutions.
