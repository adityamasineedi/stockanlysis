"""Module 11 — storage and cache.

The `analyses` table is the persistent record of every completed run.
get_cached deliberately does NOT refresh the price into the cached
verdict on a hit — v1's design did, and it's unsafe: a 6-day-old
"BUY BELOW ₹355" shown against today's ₹340 close reads as a live
trigger from an analysis that never actually saw ₹340. Instead:
  - within max_age_days, the cache is served with its ORIGINAL price and
    date, untouched;
  - today's price is fetched only to decide whether the cache is even
    safe to serve at all — if it moved more than PRICE_MOVE_REFUSE_PCT,
    get_cached refuses the cache entirely (returns None) so the caller
    re-runs fresh, rather than serving a verdict whose buy zone no longer
    means anything;
  - if the live price check itself fails (network), the cache is also
    refused rather than served un-verified — consistent with this
    project's bias toward failing loudly over serving a possibly-wrong
    number.
build_staleness_banner is a separate, pure function for the caller
(bot.py) to render when it does choose to serve a cache hit.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from stockbot.config import DB_PATH
from stockbot.fetch.prices import fetch_price_data
from stockbot.models import Analysis, ValidationResult

PRICE_MOVE_REFUSE_PCT = 0.10


@dataclass(frozen=True)
class CacheHit:
    """A served cache row plus the live price used to decide it was safe."""

    analysis: Analysis
    current_price_abs: float


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            verdict_json TEXT NOT NULL,
            report_md TEXT NOT NULL,
            brief_text TEXT NOT NULL,
            stage1_tokens INTEGER NOT NULL,
            stage2_tokens INTEGER NOT NULL,
            cost_inr REAL NOT NULL,
            validation_passed INTEGER NOT NULL,
            missing TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_analyses_ticker_created ON analyses(ticker, created_at)"
    )
    return conn


def save_analysis(
    ticker: str,
    verdict_json: dict,
    report_md: str,
    brief_text: str,
    stage1_tokens: int,
    stage2_tokens: int,
    cost_inr: float,
    validation_passed: bool,
    missing: list[str] | None = None,
) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses
                (ticker, verdict_json, report_md, brief_text, stage1_tokens,
                 stage2_tokens, cost_inr, validation_passed, missing, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                json.dumps(verdict_json),
                report_md,
                brief_text,
                stage1_tokens,
                stage2_tokens,
                cost_inr,
                int(validation_passed),
                json.dumps(missing or []),
                datetime.now(UTC).isoformat(),
            ),
        )
        return int(cursor.lastrowid)


def _row_to_analysis(row: sqlite3.Row) -> Analysis:
    verdict_json = json.loads(row["verdict_json"])
    return Analysis(
        ticker=row["ticker"],
        run_date=datetime.fromisoformat(row["created_at"]).date(),
        verdict_json=verdict_json,
        report_md=row["report_md"],
        costs=row["cost_inr"],
        validation=ValidationResult(passed=bool(row["validation_passed"]), failures=[]),
        missing=json.loads(row["missing"]),
    )


def get_cached(ticker: str, max_age_days: int = 7) -> CacheHit | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE ticker = ? ORDER BY created_at DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    if row is None:
        return None

    created_at = datetime.fromisoformat(row["created_at"])
    age_days = (datetime.now(UTC) - created_at).total_seconds() / 86400
    if age_days > max_age_days:
        return None

    verdict_json = json.loads(row["verdict_json"])
    original_price = verdict_json.get("current_price_abs")
    try:
        current_price = fetch_price_data(ticker).current_price_abs
    except Exception:  # noqa: BLE001 - can't verify safety, so refuse below rather than guess
        return None
    if original_price and abs(current_price - original_price) / original_price > PRICE_MOVE_REFUSE_PCT:
        return None

    return CacheHit(analysis=_row_to_analysis(row), current_price_abs=current_price)


def build_staleness_banner(analysis: Analysis, current_price_abs: float) -> str:
    original_price = analysis.verdict_json.get("current_price_abs")
    original_date = analysis.verdict_json.get("price_date", analysis.run_date.isoformat())
    if not original_price:
        return ""

    change_pct = (current_price_abs - original_price) / original_price * 100
    sign = "+" if change_pct >= 0 else ""
    return (
        f"⚠️ Analysis from {original_date} at ₹{original_price:.2f}. "
        f"Price today: ₹{current_price_abs:.2f} ({sign}{change_pct:.1f}%). "
        f"Buy zone below assumes conditions as of {original_date}. "
        f"Send /analyze {analysis.ticker} fresh for a current view."
    )
