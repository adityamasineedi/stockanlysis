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
class BackfillResult:
    rows_scanned: int
    rows_updated: int
    rows_skipped: int


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
    # One active plan per chat — the plan is mutable (amount, step-up, pause),
    # so it lives in a row that gets updated rather than an append-only log.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sip_plans (
            chat_id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            monthly_amount REAL NOT NULL,
            risk_profile TEXT NOT NULL,
            step_up_pct REAL NOT NULL DEFAULT 0,
            horizon_years INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    # Append-only: a contribution is a historical fact and is never rewritten,
    # which is also what makes realised-vs-projected measurable later.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sip_contributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            amount REAL NOT NULL,
            price_at_contribution REAL,
            was_topup INTEGER NOT NULL DEFAULT 0,
            contributed_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sip_contrib_chat ON sip_contributions(chat_id, contributed_at)"
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


def update_analysis_verdict(row_id: int, verdict_json: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE analyses SET verdict_json = ? WHERE id = ?",
            (json.dumps(verdict_json), row_id),
        )


def backfill_cached_verdicts() -> BackfillResult:
    """Recompute constitution gates and expected_return on all stored analyses."""
    from stockbot.constitution_gates import refresh_constitution_fields

    with _connect() as conn:
        rows = conn.execute("SELECT id, verdict_json FROM analyses ORDER BY id").fetchall()

    updated = 0
    skipped = 0
    for row in rows:
        original = json.loads(row["verdict_json"])
        refreshed = refresh_constitution_fields(original)
        if json.dumps(original, sort_keys=True) == json.dumps(refreshed, sort_keys=True):
            skipped += 1
            continue
        update_analysis_verdict(int(row["id"]), refreshed)
        updated += 1

    return BackfillResult(rows_scanned=len(rows), rows_updated=updated, rows_skipped=skipped)


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


@dataclass(frozen=True)
class SipPlan:
    chat_id: int
    ticker: str
    monthly_amount: float
    risk_profile: str
    step_up_pct: float
    horizon_years: int
    started_at: datetime
    active: bool


def _row_to_sip_plan(row: sqlite3.Row) -> SipPlan:
    return SipPlan(
        chat_id=int(row["chat_id"]),
        ticker=str(row["ticker"]),
        monthly_amount=float(row["monthly_amount"]),
        risk_profile=str(row["risk_profile"]),
        step_up_pct=float(row["step_up_pct"]),
        horizon_years=int(row["horizon_years"]),
        started_at=datetime.fromisoformat(row["started_at"]),
        active=bool(row["active"]),
    )


def save_sip_plan(
    chat_id: int,
    ticker: str,
    monthly_amount: float,
    *,
    risk_profile: str = "moderate",
    step_up_pct: float = 0.0,
    horizon_years: int = 20,
) -> SipPlan:
    """Create or replace this chat's plan. Re-running /sip re-plans, it does
    not stack a second plan on the same chat."""
    started = datetime.now(UTC).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sip_plans
                (chat_id, ticker, monthly_amount, risk_profile, step_up_pct,
                 horizon_years, started_at, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(chat_id) DO UPDATE SET
                ticker = excluded.ticker,
                monthly_amount = excluded.monthly_amount,
                risk_profile = excluded.risk_profile,
                step_up_pct = excluded.step_up_pct,
                horizon_years = excluded.horizon_years,
                active = 1
            """,
            (
                chat_id,
                ticker.upper(),
                float(monthly_amount),
                risk_profile,
                float(step_up_pct),
                int(horizon_years),
                started,
            ),
        )
    plan = get_sip_plan(chat_id)
    assert plan is not None  # just written
    return plan


def get_sip_plan(chat_id: int) -> SipPlan | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sip_plans WHERE chat_id = ?", (chat_id,)
        ).fetchone()
    return _row_to_sip_plan(row) if row else None


def list_active_sip_plans() -> list[SipPlan]:
    """Every plan the monthly job should message."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM sip_plans WHERE active = 1").fetchall()
    return [_row_to_sip_plan(r) for r in rows]


def set_sip_plan_active(chat_id: int, active: bool) -> bool:
    """Pause or resume. Returns False when the chat has no plan.

    The spec says never to *suggest* stopping; the user must still be able to
    actually stop, so pausing flips a flag and keeps the plan and its ledger.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE sip_plans SET active = ? WHERE chat_id = ?", (int(active), chat_id)
        )
        return int(cursor.rowcount) > 0


def record_sip_contribution(
    chat_id: int,
    ticker: str,
    amount: float,
    *,
    price_at_contribution: float | None = None,
    was_topup: bool = False,
) -> int:
    """Append one contribution. Never updates an earlier row."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sip_contributions
                (chat_id, ticker, amount, price_at_contribution, was_topup, contributed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                ticker.upper(),
                float(amount),
                price_at_contribution,
                int(was_topup),
                datetime.now(UTC).isoformat(),
            ),
        )
        return int(cursor.lastrowid)


@dataclass(frozen=True)
class SipLedgerSummary:
    contributions: int
    total_invested: float
    units_estimate: float | None


def summarize_sip_contributions(chat_id: int) -> SipLedgerSummary:
    """Invested total and accumulated units.

    Units are summed per contribution at that contribution's own price, which
    is the whole point of rupee-cost averaging — a single average price applied
    to the total would misstate it. Rows logged without a price cannot
    contribute units, so units go None rather than silently undercounting.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT amount, price_at_contribution FROM sip_contributions WHERE chat_id = ?",
            (chat_id,),
        ).fetchall()

    total = sum(float(r["amount"]) for r in rows)
    priced = [r for r in rows if r["price_at_contribution"]]
    units: float | None = None
    if rows and len(priced) == len(rows):
        units = sum(float(r["amount"]) / float(r["price_at_contribution"]) for r in rows)
    return SipLedgerSummary(
        contributions=len(rows),
        total_invested=round(total, 2),
        units_estimate=round(units, 4) if units is not None else None,
    )


def summarize_sip_contributions_by_symbol_for_month(
    chat_id: int,
    *,
    year: int,
    month: int,
) -> dict[str, float]:
    """Sum logged amounts per ticker for a calendar month (portfolio SIP track)."""
    prefix = f"{year:04d}-{month:02d}"
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT ticker, amount FROM sip_contributions
            WHERE chat_id = ? AND contributed_at LIKE ?
            """,
            (chat_id, f"{prefix}%"),
        ).fetchall()
    totals: dict[str, float] = {}
    for row in rows:
        sym = str(row["ticker"]).upper()
        totals[sym] = round(totals.get(sym, 0.0) + float(row["amount"]), 2)
    return totals


def summarize_average_cost_by_symbol(chat_id: int) -> dict[str, float]:
    """Weighted average buy price per ticker from logged SIP contributions."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT ticker, amount, price_at_contribution
            FROM sip_contributions
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchall()
    invested: dict[str, float] = {}
    units: dict[str, float] = {}
    for row in rows:
        price = row["price_at_contribution"]
        if price is None:
            continue
        sym = str(row["ticker"]).upper()
        amount = float(row["amount"])
        px = float(price)
        invested[sym] = invested.get(sym, 0.0) + amount
        units[sym] = units.get(sym, 0.0) + amount / px
    return {
        sym: round(invested[sym] / units[sym], 2)
        for sym in invested
        if units.get(sym, 0.0) > 0
    }


def get_latest_verdict_json(ticker: str) -> tuple[dict, datetime] | None:
    """Latest stored verdict for a ticker, with when it was produced.

    Deliberately not ``get_cached``: that one answers "is this report still
    servable as a fresh answer", so it fetches the live price and refuses on a
    >10% move. Callers who only want the numbers a past analysis computed (SIP
    scenario CAGRs, say) must not pay for that price fetch, and must not be
    denied a perfectly good three-year-old scenario just because the stock has
    since moved — over a SIP's life it always will. The timestamp comes back so
    callers can say how old the figures are.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT verdict_json, created_at FROM analyses WHERE ticker = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (ticker.upper(),),
        ).fetchone()
    if row is None:
        return None
    try:
        verdict = json.loads(row["verdict_json"])
    except (TypeError, ValueError):
        return None
    return verdict, datetime.fromisoformat(row["created_at"])
