# stockbot

Read-only Telegram bot that analyzes NSE stocks: fetch market data → two-stage Anthropic analysis → validated verdict + markdown report.

**Not multi-user production yet.** Fine for personal use. Anyone with the bot link can trigger paid analyses unless you keep the token private (allowlist is still outstanding).

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Telegram bot token ([BotFather](https://t.me/BotFather))
- Anthropic API key

## Setup

```powershell
cd E:\stockanlysis
uv sync --dev
copy .env.example .env
# Edit .env with real values
```

### Environment (`.env`)

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `TELEGRAM_BOT_TOKEN` | yes (bot) | — | Telegram polling |
| `ANTHROPIC_API_KEY` | yes (paid runs) | — | Stage 1 + Stage 2 |
| `MONTHLY_BUDGET_INR` | no | `1400` | Hard monthly spend block |
| `USD_INR_RATE` | no | `95.5` | Cost conversion (update with spot USD/INR) |
| `MAX_CONCURRENT_ANALYSES` | no | `1` | Paid analyses in parallel |
| `FORCE_STAGE2_FULL` | no | `false` | Always use Sonnet full Stage 2 (skip Haiku lite path) |
| `OPENAI_API_KEY` | no | — | Pre-screener ranking (gpt-4o-mini, preferred) |
| `DEEPSEEK_API_KEY` | no | — | Pre-screener fallback + A/B tooling |

### Data directories (created on demand)

| Path | Role |
|------|------|
| `data/symbols/` | Cached NSE equity CSV (auto-downloaded) |
| `data/cache/screener/` | Screener HTML cache |
| `data/cache/annual_reports/` | Annual-report PDFs |
| `data/db/analyses.sqlite3` | Analyses + `llm_calls` cost log |
| `reports/` | Delivered `.md` report copies |
| `logs/stockbot.log` | Application log |

First ticker resolve downloads the NSE symbol table into `data/symbols/`. Needs network.

## Run the bot

```powershell
uv run stockbot-bot
```

Commands:

- `/analyze <name or symbol>` or plain text — full analysis (often **5–15 minutes**); blocked unless prescan eligibility passes
- `/analyze force SYMBOL` — bypass eligibility gate (not recommended)
- `/spend` — month-to-date LLM spend vs monthly budget
- `/health` — cost/token/quality audit (no LLM spend; attaches report if issues found)
- `/health clear` — verify first, clear baseline only if clean
- `/help` — short usage

**Do not restart the bot mid-analysis.** In-flight Anthropic calls are still billed; results are not recovered after a process kill.

## Portfolio pre-screener

Reduces a 40+ stock watchlist to ~10–18 candidates before expensive deep analysis.

```powershell
# Edit data/portfolio/watchlist.txt then:
uv run stockbot-prescreen --dry-run          # quant only, no AI, no deep analysis
uv run stockbot-prescreen --skip-ai          # quant + diversification, no AI ranker
uv run stockbot-prescreen                    # quant + cheap AI ranking (auto: openai > deepseek > haiku)
uv run stockbot-prescreen --ai-provider deepseek
uv run stockbot-prescreen --run-deep         # then run_full_analysis on survivors
```

Watchlist: `data/portfolio/watchlist.txt` (one NSE symbol per line). Audit JSON lands in `logs/portfolio_screen_*.json`.

**Single-ticker (Telegram / CLI):** ask whether one stock is worth deep analysis — does **not** scan the whole list.

```powershell
uv run stockbot-prescreen BEL
```

Telegram (restart bot to load new command):

- `/prescan BEL` or plain text `prescan BEL`
- `/health` — audit spend, token waste, and quality issues
- If suitable → then `/analyze BEL` for the expensive master-prompt run (gate enforced; use `/analyze force BEL` to override)

## Cost caps

- **Monthly:** `MONTHLY_BUDGET_INR` — new paid runs refused when hit
- **Per analysis:** ₹80 mid-flight kill switch (Stage 1 + Stage 2 + retries)
- **Concurrency:** only `MAX_CONCURRENT_ANALYSES` paid runs at once (default 1)

Cache hits (≤7 days, price move ≤10%) do not call the LLM.

## Ops / QA CLIs

```powershell
uv run stockbot-verify SYMBOL      # fetch-layer check
uv run stockbot-dry-run SYMBOL     # brief + Stage 1/2 payloads, no full spend path as designed in that script
uv run stockbot-smoke-test SYMBOL  # live full pipeline (spends money)
uv run stockbot-monitor             # cost/token/quality audit (no LLM spend)
uv run stockbot-monitor --verify-and-clear   # verify, clear only if clean (deploy path)
uv run pytest                      # unit tests (no network / no keys)
uv run ruff check src tests
```

## Process supervision

Polling is long-lived. Use a supervisor so crashes restart cleanly **between** analyses (never force-kill during a run if you can avoid it).

### Windows (NSSM)

1. Install [NSSM](https://nssm.cc/).
2. Point the service at your uv/python entry, for example:

```text
Application: E:\stockanlysis\.venv\Scripts\stockbot-bot.exe
AppDirectory: E:\stockanlysis
```

Or `uv.exe` with arguments `run stockbot-bot` and the same AppDirectory. Ensure `.env` is readable from that directory.

### Linux (systemd)

```ini
[Unit]
Description=stockbot Telegram bot
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/stockbot
ExecStart=/usr/local/bin/uv run stockbot-bot
Restart=on-failure
RestartSec=10
EnvironmentFile=/opt/stockbot/.env

[Install]
WantedBy=multi-user.target
```

Rotate or truncate `logs/stockbot.log` periodically; the process opens it in append mode.

## Deploy (Railway + GitHub Actions)

Production runs on [Railway](https://railway.com) (`stockanlysis` service, `production` environment). Pushes to `main` run tests then `railway up` via GitHub Actions.

**One-time CI setup** (if the `deploy` job fails with “RAILWAY_TOKEN secret is missing”):

1. Railway → project **stockanlysis** → **Settings** → **Tokens** → create a **Project Token** for **production**
2. GitHub → repo **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
3. Name: `RAILWAY_TOKEN` — paste the project token (not an account/workspace token)
4. Re-run the failed workflow: **Actions** → latest **ci** run → **Re-run failed jobs**, or push any commit to `main`

Optional Telegram deploy ping: set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_DEPLOY_CHAT_ID` repo secrets (notify step skips if unset).

Manual deploy from a machine with `railway login`:

```powershell
railway up --service stockanlysis --detach
```

## Architecture (short)

1. Resolve ticker → optional SQLite cache  
2. Budget + concurrency gate → assemble brief (prices, technicals, fundamentals, shareholding, news, annual report)  
3. Stage 1 extract → Stage 2 verdict (Sonnet) → validate → render placeholders → store  
4. Telegram HTML verdict + `.md` attachment  

Details: `PROJECT.md` (data contracts), `src/stockbot/pipeline.py` (orchestration).

## Disclaimer

Outputs are educational research, **not** investment advice. Verdicts can be wrong, stale (cached), or based on incomplete data (`MISSING: …` lines). You are responsible for any trading decisions.
