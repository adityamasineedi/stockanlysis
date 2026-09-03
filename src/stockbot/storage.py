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

from stockbot.config import DB_PATH, settings
from stockbot.fetch.prices import fetch_price_data
from stockbot.models import Analysis, ValidationResult

PRICE_MOVE_REFUSE_PCT = 0.10  # default; overridden by settings.cache_price_refuse_pct in lookup


@dataclass(frozen=True)
class BackfillResult:
    rows_scanned: int
    rows_updated: int
    rows_skipped: int


@dataclass(frozen=True)
class CacheLookup:
    hit: CacheHit | None
    miss_reason: str | None = None


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
    # The holder's own limits. Without these the constitution's
    # "maximum_intended_position_pct" stays null and every concentration rule
    # in it is unenforceable — it declines to invent the number, correctly.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_policy (
            chat_id INTEGER PRIMARY KEY,
            total_capital_inr REAL NOT NULL,
            max_position_pct REAL NOT NULL DEFAULT 10,
            max_sector_pct REAL NOT NULL DEFAULT 25,
            emergency_fund_months REAL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # The goal and the cashflow behind it, kept apart from risk_policy on
    # purpose: that table is the holder's *limits*, this one is where they are
    # trying to get to. Separate tables also mean no migration — a new table is
    # created correctly by CREATE TABLE IF NOT EXISTS, whereas adding columns to
    # risk_policy would be silently skipped on the deployed database and then
    # fail on every read.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS financial_plan (
            chat_id INTEGER PRIMARY KEY,
            current_age REAL NOT NULL,
            target_age REAL NOT NULL,
            monthly_income_inr REAL,
            monthly_expenses_inr REAL,
            monthly_investment_inr REAL NOT NULL,
            desired_monthly_spend_inr REAL NOT NULL,
            post_retirement_income_inr REAL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Current state, so rows are updated in place — unlike sip_contributions,
    # which is the append-only historical record and stays that way.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdings (
            chat_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            quantity REAL NOT NULL,
            avg_cost REAL NOT NULL,
            opened_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (chat_id, ticker)
        )
        """
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


def lookup_cached(
    ticker: str,
    max_age_days: int | None = None,
) -> CacheLookup:
    """Return cache hit or a human-readable miss reason."""
    max_days = max_age_days if max_age_days is not None else settings.analysis_cache_max_age_days
    refuse_pct = settings.cache_price_refuse_pct

    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE ticker = ? ORDER BY created_at DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    if row is None:
        return CacheLookup(hit=None, miss_reason=None)

    created_at = datetime.fromisoformat(row["created_at"])
    age_days = (datetime.now(UTC) - created_at).total_seconds() / 86400
    if age_days > max_days:
        return CacheLookup(
            hit=None,
            miss_reason=(
                f"Cached report is {age_days:.0f} days old (max {max_days}d) — running fresh analysis."
            ),
        )

    verdict_json = json.loads(row["verdict_json"])
    original_price = verdict_json.get("analysis_price_abs") or verdict_json.get(
        "current_price_abs"
    )
    try:
        live = fetch_price_data(ticker)
    except Exception:  # noqa: BLE001 - can't verify safety, so refuse below rather than guess
        return CacheLookup(
            hit=None,
            miss_reason="Could not verify live price for cached report — running fresh analysis.",
        )
    if original_price and abs(live.current_price_abs - original_price) / original_price > refuse_pct:
        move_pct = abs(live.current_price_abs - original_price) / original_price * 100
        return CacheLookup(
            hit=None,
            miss_reason=(
                f"Price moved {move_pct:.1f}% since cached report (>{refuse_pct * 100:.0f}% limit) "
                f"— running fresh analysis."
            ),
        )

    return CacheLookup(
        hit=CacheHit(
            analysis=_row_to_analysis(row),
            current_price_abs=live.current_price_abs,
            price_date=live.price_date,
        )
    )


def get_cached(ticker: str, max_age_days: int = 7) -> CacheHit | None:
    return lookup_cached(ticker, max_age_days=max_age_days).hit


def build_staleness_banner(analysis: Analysis, current_price_abs: float) -> str:
    """Banner when the cached report date differs from the live synced price."""
    verdict = analysis.verdict_json
    analysis_price = verdict.get("analysis_price_abs") or verdict.get("current_price_abs")
    analysis_date = verdict.get("analysis_price_date") or verdict.get(
        "price_date", analysis.run_date.isoformat()
    )
    if not analysis_price:
        return ""

    age_days = (datetime.now(UTC).date() - analysis.run_date).days
    age_note = f"{age_days}d old" if age_days > 0 else "same day"
    synced = verdict.get("price_synced_at")
    sync_note = f" · synced {str(synced)[:19]}" if synced else ""

    if abs(current_price_abs - float(analysis_price)) / float(analysis_price) < 0.001:
        return (
            f"📋 Cached report ({age_note}) from {analysis_date} · "
            f"price unchanged at ₹{current_price_abs:.2f}{sync_note} · gates refreshed, no LLM cost."
        )

    change_pct = (current_price_abs - float(analysis_price)) / float(analysis_price) * 100
    sign = "+" if change_pct >= 0 else ""
    return (
        f"📋 Cached report ({age_note}) from {analysis_date} at ₹{float(analysis_price):.2f} · "
        f"live ₹{current_price_abs:.2f} ({sign}{change_pct:.1f}%){sync_note}. "
        f"Prose unchanged; gates recomputed. Use /analyze fresh SYMBOL after results/filings."
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


def summarize_sip_contributions(chat_id: int, ticker: str) -> SipLedgerSummary:
    """Invested total and accumulated units for one chat's holding in ``ticker``.

    Units are summed per contribution at that contribution's own price, which
    is the whole point of rupee-cost averaging — a single average price applied
    to the total would misstate it. Rows logged without a price cannot
    contribute units, so units go None rather than silently undercounting.

    Filtering by ticker is not optional. ``save_sip_plan`` lets a chat re-point
    its plan (BEL → CRISIL); without this filter the totals would merge both
    stocks and the caller would multiply BEL-derived units by the live CRISIL
    price, printing a gain that never happened. The abandoned rows stay in the
    append-only ledger, they just stop counting toward the new holding.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT amount, price_at_contribution FROM sip_contributions "
            "WHERE chat_id = ? AND ticker = ?",
            (chat_id, ticker.upper()),
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


def list_sip_contribution_chat_ids() -> frozenset[int]:
    """Distinct chat IDs that have logged at least one SIP contribution."""
    with _connect() as conn:
        rows = conn.execute("SELECT DISTINCT chat_id FROM sip_contributions").fetchall()
    return frozenset(int(row["chat_id"]) for row in rows)


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


def list_latest_analyses() -> list[tuple[str, dict, datetime]]:
    """Latest analysis row per ticker (for /rank). Oldest cache first is fine."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT ticker, verdict_json, created_at
            FROM analyses
            WHERE id IN (
                SELECT MAX(id) FROM analyses GROUP BY ticker
            )
            ORDER BY ticker
            """
        ).fetchall()
    out: list[tuple[str, dict, datetime]] = []
    for row in rows:
        try:
            verdict = json.loads(row["verdict_json"])
        except (TypeError, ValueError):
            continue
        ticker = str(row["ticker"] or "").upper()
        if not ticker:
            continue
        out.append((ticker, verdict, datetime.fromisoformat(row["created_at"])))
    return out


@dataclass(frozen=True)
class RiskPolicy:
    """The holder's limits. The bot proposes; this decides."""

    chat_id: int
    total_capital_inr: float
    max_position_pct: float
    max_sector_pct: float
    emergency_fund_months: float | None
    updated_at: datetime


@dataclass(frozen=True)
class FinancialPlan:
    """Where the holder is trying to get to, and what they are putting in.

    Amounts in today's rupees; ``post_retirement_income_inr`` is annual and
    already in target-year terms, because a pension or rental figure is quoted
    as what it will actually pay.
    """

    chat_id: int
    current_age: float
    target_age: float
    monthly_income_inr: float | None
    monthly_expenses_inr: float | None
    monthly_investment_inr: float
    desired_monthly_spend_inr: float
    post_retirement_income_inr: float | None
    updated_at: datetime

    @property
    def years_to_target(self) -> int:
        """Whole years left. Zero once the target age is reached."""
        return max(int(self.target_age - self.current_age), 0)

    @property
    def monthly_surplus_inr(self) -> float | None:
        """Income less expenses, or None when either is undeclared.

        Not the same as ``monthly_investment_inr``: the gap between what is
        left over and what is actually invested is the headroom the savings
        lever has to work with.
        """
        if self.monthly_income_inr is None or self.monthly_expenses_inr is None:
            return None
        return round(self.monthly_income_inr - self.monthly_expenses_inr, 2)


@dataclass(frozen=True)
class Holding:
    chat_id: int
    ticker: str
    quantity: float
    avg_cost: float
    opened_at: datetime
    updated_at: datetime

    @property
    def cost_basis_inr(self) -> float:
        return round(self.quantity * self.avg_cost, 2)


def save_risk_policy(
    chat_id: int,
    total_capital_inr: float,
    *,
    max_position_pct: float = 10.0,
    max_sector_pct: float = 25.0,
    emergency_fund_months: float | None = None,
) -> RiskPolicy:
    """Create or update this chat's policy."""
    now = datetime.now(UTC).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO risk_policy
                (chat_id, total_capital_inr, max_position_pct, max_sector_pct,
                 emergency_fund_months, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                total_capital_inr = excluded.total_capital_inr,
                max_position_pct = excluded.max_position_pct,
                max_sector_pct = excluded.max_sector_pct,
                emergency_fund_months = excluded.emergency_fund_months,
                updated_at = excluded.updated_at
            """,
            (
                chat_id,
                float(total_capital_inr),
                float(max_position_pct),
                float(max_sector_pct),
                emergency_fund_months,
                now,
            ),
        )
    policy = get_risk_policy(chat_id)
    assert policy is not None  # just written
    return policy


def get_risk_policy(chat_id: int) -> RiskPolicy | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM risk_policy WHERE chat_id = ?", (chat_id,)
        ).fetchone()
    if row is None:
        return None
    return RiskPolicy(
        chat_id=int(row["chat_id"]),
        total_capital_inr=float(row["total_capital_inr"]),
        max_position_pct=float(row["max_position_pct"]),
        max_sector_pct=float(row["max_sector_pct"]),
        emergency_fund_months=(
            float(row["emergency_fund_months"])
            if row["emergency_fund_months"] is not None
            else None
        ),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def save_financial_plan(
    chat_id: int,
    *,
    current_age: float,
    target_age: float,
    monthly_investment_inr: float,
    desired_monthly_spend_inr: float,
    monthly_income_inr: float | None = None,
    monthly_expenses_inr: float | None = None,
    post_retirement_income_inr: float | None = None,
) -> FinancialPlan:
    """Create or update this chat's plan."""
    now = datetime.now(UTC).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO financial_plan
                (chat_id, current_age, target_age, monthly_income_inr,
                 monthly_expenses_inr, monthly_investment_inr,
                 desired_monthly_spend_inr, post_retirement_income_inr, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                current_age = excluded.current_age,
                target_age = excluded.target_age,
                monthly_income_inr = excluded.monthly_income_inr,
                monthly_expenses_inr = excluded.monthly_expenses_inr,
                monthly_investment_inr = excluded.monthly_investment_inr,
                desired_monthly_spend_inr = excluded.desired_monthly_spend_inr,
                post_retirement_income_inr = excluded.post_retirement_income_inr,
                updated_at = excluded.updated_at
            """,
            (
                chat_id,
                float(current_age),
                float(target_age),
                monthly_income_inr,
                monthly_expenses_inr,
                float(monthly_investment_inr),
                float(desired_monthly_spend_inr),
                post_retirement_income_inr,
                now,
            ),
        )
    plan = get_financial_plan(chat_id)
    assert plan is not None  # just written
    return plan


def get_financial_plan(chat_id: int) -> FinancialPlan | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM financial_plan WHERE chat_id = ?", (chat_id,)
        ).fetchone()
    if row is None:
        return None

    def optional(key: str) -> float | None:
        value = row[key]
        return float(value) if value is not None else None

    return FinancialPlan(
        chat_id=int(row["chat_id"]),
        current_age=float(row["current_age"]),
        target_age=float(row["target_age"]),
        monthly_income_inr=optional("monthly_income_inr"),
        monthly_expenses_inr=optional("monthly_expenses_inr"),
        monthly_investment_inr=float(row["monthly_investment_inr"]),
        desired_monthly_spend_inr=float(row["desired_monthly_spend_inr"]),
        post_retirement_income_inr=optional("post_retirement_income_inr"),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_holding(row: sqlite3.Row) -> Holding:
    return Holding(
        chat_id=int(row["chat_id"]),
        ticker=str(row["ticker"]),
        quantity=float(row["quantity"]),
        avg_cost=float(row["avg_cost"]),
        opened_at=datetime.fromisoformat(row["opened_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def save_holding(chat_id: int, ticker: str, quantity: float, avg_cost: float) -> Holding:
    """Record a position, replacing any existing one for that ticker.

    ``opened_at`` survives an update so the holding period stays measurable;
    only the quantity, cost and ``updated_at`` move.
    """
    now = datetime.now(UTC).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO holdings
                (chat_id, ticker, quantity, avg_cost, opened_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, ticker) DO UPDATE SET
                quantity = excluded.quantity,
                avg_cost = excluded.avg_cost,
                updated_at = excluded.updated_at
            """,
            (chat_id, ticker.upper(), float(quantity), float(avg_cost), now, now),
        )
    holding = get_holding(chat_id, ticker)
    assert holding is not None  # just written
    return holding


def get_holding(chat_id: int, ticker: str) -> Holding | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM holdings WHERE chat_id = ? AND ticker = ?",
            (chat_id, ticker.upper()),
        ).fetchone()
    return _row_to_holding(row) if row else None


def list_holdings(chat_id: int) -> list[Holding]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM holdings WHERE chat_id = ? ORDER BY ticker", (chat_id,)
        ).fetchall()
    return [_row_to_holding(r) for r in rows]


def delete_holding(chat_id: int, ticker: str) -> bool:
    """Remove a position. Returns False when there was nothing to remove."""
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM holdings WHERE chat_id = ? AND ticker = ?",
            (chat_id, ticker.upper()),
        )
        return int(cursor.rowcount) > 0


def seed_holding_from_sip(chat_id: int, ticker: str) -> Holding | None:
    """Build a holding from logged SIP contributions.

    The ledger already stores each contribution at its own price, so it knows
    both the units accumulated and what they cost — exactly a position. Returns
    None when the ledger cannot say (no rows, or any row logged without a
    price), rather than seeding a holding that understates the quantity.
    """
    summary = summarize_sip_contributions(chat_id, ticker)
    if summary.units_estimate is None or summary.units_estimate <= 0:
        return None
    avg_cost = summary.total_invested / summary.units_estimate
    return save_holding(chat_id, ticker, summary.units_estimate, round(avg_cost, 2))
