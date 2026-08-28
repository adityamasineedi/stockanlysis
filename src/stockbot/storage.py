"""Module 11 — storage and cache.

The `analyses` table is the persistent record of every completed run.
On a cache hit within max_age_days (and price move ≤ PRICE_MOVE_REFUSE_PCT),
the stored report is reused and only the live price is synced from
yfinance (NSE/BSE last close — same source as a fresh run). Constitution
gates (anti-chase, valuation tension, buy-zone blocks) are recomputed
against the live price; the qualitative report prose is unchanged.

If price moved more than PRICE_MOVE_REFUSE_PCT, get_cached returns None
so the caller re-runs a full analysis. If the live price fetch fails,
the cache is also refused rather than served un-verified.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime

from stockbot.config import DB_PATH
from stockbot.fetch.prices import fetch_price_data
from stockbot.models import Analysis, ValidationResult

PRICE_MOVE_REFUSE_PCT = 0.10


@dataclass(frozen=True)
class CacheHit:
    """A served cache row plus the live price used to refresh the card."""

    analysis: Analysis
    current_price_abs: float
    price_date: date


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


def invalidate_cached_analyses(ticker: str) -> int:
    """Delete stored analyses for a ticker so the next /analyze is fresh."""
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM analyses WHERE ticker = ?", (ticker.upper(),))
        return int(cursor.rowcount)


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
    original_price = verdict_json.get("analysis_price_abs") or verdict_json.get(
        "current_price_abs"
    )
    try:
        live = fetch_price_data(ticker)
    except Exception:  # noqa: BLE001 - can't verify safety, so refuse below rather than guess
        return None
    if original_price and abs(live.current_price_abs - original_price) / original_price > PRICE_MOVE_REFUSE_PCT:
        return None

    return CacheHit(
        analysis=_row_to_analysis(row),
        current_price_abs=live.current_price_abs,
        price_date=live.price_date,
    )


def build_staleness_banner(analysis: Analysis, current_price_abs: float) -> str:
    """Banner when the cached report date differs from the live synced price."""
    verdict = analysis.verdict_json
    analysis_price = verdict.get("analysis_price_abs") or verdict.get("current_price_abs")
    analysis_date = verdict.get("analysis_price_date") or verdict.get(
        "price_date", analysis.run_date.isoformat()
    )
    if not analysis_price:
        return ""

    if abs(current_price_abs - float(analysis_price)) / float(analysis_price) < 0.001:
        return (
            f"Report from {analysis_date} · price unchanged at ₹{current_price_abs:.2f} "
            f"(live sync, no new LLM run)."
        )

    change_pct = (current_price_abs - float(analysis_price)) / float(analysis_price) * 100
    sign = "+" if change_pct >= 0 else ""
    return (
        f"Report from {analysis_date} at ₹{float(analysis_price):.2f} · "
        f"live price ₹{current_price_abs:.2f} ({sign}{change_pct:.1f}%). "
        f"Qualitative analysis unchanged; gates recomputed at live price. "
        f"Send /analyze {analysis.ticker} fresh after new results/filings."
    )
