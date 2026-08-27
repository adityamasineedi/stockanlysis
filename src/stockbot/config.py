"""Runtime configuration, filesystem layout, and logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SYMBOLS_DIR = DATA_DIR / "symbols"
CACHE_DIR = DATA_DIR / "cache"
SCREENER_CACHE_DIR = CACHE_DIR / "screener"
ANNUAL_REPORT_CACHE_DIR = CACHE_DIR / "annual_reports"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "analyses.sqlite3"

REPORTS_DIR = PROJECT_ROOT / "reports"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MASTER_PROMPT_PATH = PROMPTS_DIR / "master-stock-analysis-prompt-v3.md"
LOGS_DIR = PROJECT_ROOT / "logs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    telegram_bot_token: str = ""
    anthropic_api_key: str = ""
    # 17D: DeepSeek A/B test only — not used by the production pipeline.
    deepseek_api_key: str = ""
    monthly_budget_inr: float = 1400.0
    usd_inr_rate: float = 88.0


settings = Settings()


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
