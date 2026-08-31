"""Runtime configuration, filesystem layout, and logging setup."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Everything under DATA_DIR is state that must survive a redeploy: the SQLite
# analysis cache, the prescan outcome log, and the Screener/annual-report file
# caches. On a container host (Railway et al.) the image filesystem is
# ephemeral, so point STOCKBOT_DATA_DIR at a mounted persistent volume —
# otherwise every deploy silently resets the bot's entire history.
DATA_DIR = Path(os.environ.get("STOCKBOT_DATA_DIR") or (PROJECT_ROOT / "data"))
SYMBOLS_DIR = DATA_DIR / "symbols"
PORTFOLIO_DIR = DATA_DIR / "portfolio"
WATCHLIST_PATH = PORTFOLIO_DIR / "watchlist.txt"
# Bundled in the Docker image at config/portfolio/ — outside the Railway volume
# mount (/app/data), which otherwise hides repo files copied under data/.
SIP_PORTFOLIOS_BUNDLED_PATH = PROJECT_ROOT / "config" / "portfolio" / "sip_portfolios.json"
SIP_PORTFOLIOS_VOLUME_PATH = PORTFOLIO_DIR / "sip_portfolios.json"
SIP_PORTFOLIOS_PATH = SIP_PORTFOLIOS_BUNDLED_PATH
CACHE_DIR = DATA_DIR / "cache"
SCREENER_CACHE_DIR = CACHE_DIR / "screener"
ANNUAL_REPORT_CACHE_DIR = CACHE_DIR / "annual_reports"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "analyses.sqlite3"

REPORTS_DIR = PROJECT_ROOT / "reports"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MASTER_PROMPT_PATH = PROMPTS_DIR / "master-stock-analysis-prompt-v3.md"
LOGS_DIR = PROJECT_ROOT / "logs"

# Shared browser UA for NSE / Screener / Google News fetches. Keep near
# current Chrome stable — ancient versions (e.g. Chrome/120) get blocked
# more often by WAF cookie gates.
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    telegram_bot_token: str = ""
    anthropic_api_key: str = ""
    # 17D: DeepSeek A/B test + portfolio pre-screener ranking.
    deepseek_api_key: str = ""
    # Optional — portfolio pre-screener ranking (gpt-4o-mini default when set).
    openai_api_key: str = ""
    monthly_budget_inr: float = 1400.0
    # Spot USD/INR for LLM cost conversion. Refresh periodically — do not
    # leave a years-old rate here (was 88 for a long stretch while spot ~95).
    usd_inr_rate: float = 95.5
    # Paid analyses only (cache hits bypass the semaphore). Default 1 so two
    # overlapping runs cannot both pass check_budget() then double-bill past
    # the monthly cap.
    max_concurrent_analyses: int = 1
    # Debug / quality check: always run Sonnet full 16-section Stage 2 even when
    # routing would pick the cheaper Haiku lite path.
    force_stage2_full: bool = False
    # Block /analyze unless prescan eligibility passes (AUTO_DEEP or SECTOR_REVIEW).
    require_prescan_for_analyze: bool = True
    # Comma-separated Telegram chat IDs allowed to use the bot. Empty = allow all
    # (private single-user bots). Set to your numeric chat id to block strangers.
    telegram_allowed_chat_ids: str = ""


settings = Settings()


def resolve_sip_portfolios_path(explicit: Path | None = None) -> Path:
    """Return SIP config path — volume override, else bundled image copy."""
    if explicit is not None:
        return explicit
    if SIP_PORTFOLIOS_VOLUME_PATH.exists():
        return SIP_PORTFOLIOS_VOLUME_PATH
    return SIP_PORTFOLIOS_BUNDLED_PATH


def parse_telegram_allowed_chat_ids(raw: str | None = None) -> frozenset[int]:
    """Parse TELEGRAM_ALLOWED_CHAT_IDS — empty set means no restriction."""
    text = (raw if raw is not None else settings.telegram_allowed_chat_ids).strip()
    if not text:
        return frozenset()
    ids: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            logging.getLogger(__name__).warning(
                "Ignoring invalid TELEGRAM_ALLOWED_CHAT_IDS entry: %r", part
            )
    return frozenset(ids)


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(LOGS_DIR / "stockbot.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
