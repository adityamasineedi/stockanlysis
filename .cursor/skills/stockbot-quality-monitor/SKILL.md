---
name: stockbot-quality-monitor
description: >-
  Audits stockbot for Telegram bot quality regressions, LLM token waste, and
  cost leaks using SQLite and logs. Use when the user asks to monitor bot
  quality, find token leaks, cost leaks, wasted spend, retry loops, or run a
  health check on stock analysis pipeline.
---

# stockbot Quality Monitor

Deterministic audit — **no LLM spend**. Run after `/analyze` sessions, weekly, or when spend looks wrong.

## Run the audit

```powershell
cd E:\stockanlysis
uv run stockbot-monitor
uv run stockbot-monitor --days 7 --out logs/health_audit.md
uv run stockbot-monitor --json --fail-on warning
```

Telegram: `/health` (same audit, compact summary + `.md` attachment when warnings/critical exist).

## What it checks

| Category | Examples |
|----------|----------|
| **cost_leak** | Budget >80%, abandoned sessions (calls without saved analysis), retry spend > saved cost, per-run cost near ₹80 cap |
| **token_waste** | Stage 1 input >50k, Stage 2 thinking >55% of output, cache writes without reads, truncated fixtures |
| **quality** | Invalid verdict JSON, missing `expected_return`, log patterns (validation fail, render fail) |

## Data sources

- `data/db/analyses.sqlite3` — completed runs + cache
- `data/db/analyses.sqlite3` → `llm_calls` — every billed call (now includes `stage`, `ticker`)
- `logs/stockbot.log` — ERROR/WARNING patterns
- `data/llm_fixtures/` — truncation (`stop_reason=max_tokens`)

## Agent workflow

1. Run `uv run stockbot-monitor --days 14 --out logs/health_audit_latest.md`
2. Read critical + warning findings first
3. For each **cost_leak / abandoned session**: check same-day `llm_calls` grouped by ticker; confirm bot restart or double `/analyze`
4. For **token_waste / Stage 1 huge**: verify `extract.py` trim + annual report cap
5. For **thinking ratio**: check if FULL path should have been LITE (`verdict_json.stage2_mode`)
6. Propose minimal code fixes — do not disable cost caps

## Fix priority

1. Abandoned sessions → prescan gate, `/refresh` discipline, don't restart bot mid-run
2. Retry excess → validation auto-fix (`validate.py`) or narrow retry
3. Stage 1 bloat → annual report / news trim
4. FULL vs LITE routing → `analysis_routing.py`

## Exit codes

- `--fail-on critical` (default): exit 1 if any critical finding — use in CI/cron
- `--fail-on warning`: stricter
- `--fail-on none`: report only
