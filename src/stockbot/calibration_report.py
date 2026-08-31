"""Assemble the calibration report from stored history — does I/O.

Reads two sources of dated calls, resolves each against a current price, and
hands the result to ``calibration`` for the statistics. Kept apart from that
module so the maths stays testable without a database or a network.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from stockbot.calibration import (
    MIN_SAMPLE,
    CalibrationBucket,
    ScoredCall,
    TierSpread,
    bucket_by_age,
    bucket_by_label,
    score_calls,
    tier_spread,
)

logger = logging.getLogger(__name__)

# The prescan verdicts the bot liked, versus the ones it turned down. The
# spread between them is the headline because both rode the same market.
PRESCAN_POSITIVE = {"AUTO_DEEP_ANALYSIS"}
PRESCAN_NEGATIVE = {"NOT_SUITABLE_FOR_3Y_RESEARCH", "HOLDING_MONITOR_ONLY"}

# /analyze verdicts, per the v3 prompt's schema.
ANALYZE_POSITIVE = {"BUY", "BUY ON CORRECTION"}
ANALYZE_NEGATIVE = {"SKIP"}

# A broad-market ETF. fetch_price_data tries .NS/.BO directly and never
# consults EQUITY_L.csv, so this resolves even though ETFs are absent from the
# universe. Absolute returns without it are not evidence of skill.
BENCHMARK_SYMBOL = "NIFTYBEES"


@dataclass(frozen=True)
class CalibrationReport:
    source: str
    total_rows: int
    scored: int
    spread: TierSpread | None
    by_label: list[CalibrationBucket]
    by_age: list[CalibrationBucket]

    @property
    def has_enough_history(self) -> bool:
        return self.scored >= MIN_SAMPLE


def _current_prices(tickers: set[str]) -> dict[str, float | None]:
    """One fetch per distinct ticker, never per row.

    A year of scans repeats names heavily and yfinance rate-limits, so the
    dedupe is load-bearing rather than tidiness. A failed fetch drops that
    ticker from the sample instead of killing the report — the same posture
    the SIP reminder takes when a price is unavailable.
    """
    from stockbot.fetch.prices import fetch_price_data

    prices: dict[str, float | None] = {}
    for ticker in sorted(tickers):
        try:
            prices[ticker] = fetch_price_data(ticker).current_price_abs
        except Exception:  # noqa: BLE001 - one bad symbol must not sink the report
            logger.warning("Calibration: no current price for %s", ticker)
            prices[ticker] = None
    return prices


def _load_prescan_rows() -> list[dict]:
    """Every logged prescan call, not just the newest per ticker.

    ``load_prescan_outcomes`` defaults to ``latest_per_ticker=True``, which
    would discard most of the history this report exists to measure.
    """
    from stockbot.portfolio_screener.outcome_log import load_prescan_outcomes

    return load_prescan_outcomes(latest_per_ticker=False)


def _load_analyze_rows() -> list[dict]:
    """Dated /analyze verdicts with the price the analysis saw."""
    from stockbot.storage import _connect

    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT ticker, verdict_json, created_at FROM analyses ORDER BY created_at"
            ).fetchall()
    except sqlite3.Error:
        logger.exception("Calibration: could not read stored analyses")
        return []

    out: list[dict] = []
    for row in rows:
        try:
            verdict_json = json.loads(row["verdict_json"])
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "ticker": row["ticker"],
                "verdict": verdict_json.get("verdict"),
                "price_at_scan": verdict_json.get("analysis_price_abs")
                or verdict_json.get("current_price_abs"),
                "logged_at": row["created_at"],
            }
        )
    return out


def _build(
    source: str,
    rows: list[dict],
    positive: set[str],
    negative: set[str],
    *,
    now: datetime | None = None,
    prices: dict[str, float | None] | None = None,
) -> CalibrationReport:
    tickers = {
        str(r.get("ticker") or "").strip().upper() for r in rows if r.get("ticker")
    }
    resolved = prices if prices is not None else _current_prices(tickers)
    calls: list[ScoredCall] = score_calls(rows, resolved, now=now)
    return CalibrationReport(
        source=source,
        total_rows=len(rows),
        scored=len(calls),
        spread=tier_spread(calls, positive, negative),
        by_label=bucket_by_label(calls),
        by_age=bucket_by_age(calls),
    )


def build_prescan_calibration(
    *, now: datetime | None = None, prices: dict[str, float | None] | None = None
) -> CalibrationReport:
    return _build(
        "prescan", _load_prescan_rows(), PRESCAN_POSITIVE, PRESCAN_NEGATIVE,
        now=now, prices=prices,
    )


def build_analyze_calibration(
    *, now: datetime | None = None, prices: dict[str, float | None] | None = None
) -> CalibrationReport:
    return _build(
        "analyze", _load_analyze_rows(), ANALYZE_POSITIVE, ANALYZE_NEGATIVE,
        now=now, prices=prices,
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1f}%"


def format_calibration_report(report: CalibrationReport) -> str:
    """Telegram HTML. Says plainly when there is not enough history yet."""
    title = (
        "🎯 <b>Track record — /prescan calls</b>"
        if report.source == "prescan"
        else "🎯 <b>Track record — /analyze verdicts</b>"
    )
    if not report.has_enough_history:
        return (
            f"{title}\n"
            f"{report.scored} scoreable call(s) from {report.total_rows} logged.\n\n"
            f"Not enough history yet — at least {MIN_SAMPLE} are needed before any "
            "median is worth printing. This fills in on its own as you keep "
            "using the bot; nothing to do."
        )

    lines = [title, f"{report.scored} scoreable of {report.total_rows} logged.", ""]

    if report.spread is not None:
        spread = report.spread
        verdict = (
            "the score is discriminating"
            if spread.discriminates
            else "<b>the score is not discriminating</b>"
        )
        lines.extend(
            [
                "<b>Does the ranking work?</b>",
                f"Liked ({spread.n_positive}): {_pct(spread.positive_median_pct)} median",
                f"Passed ({spread.n_negative}): {_pct(spread.negative_median_pct)} median",
                f"Spread: <b>{spread.spread_pct:+.1f} points</b> — {verdict}.",
                (
                    "<i>Both groups rode the same market, so this comparison "
                    "needs no benchmark.</i>"
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "<i>Not enough calls on both sides yet to compare the tiers — "
                    "that comparison is the only one that means anything without a "
                    "benchmark.</i>"
                ),
                "",
            ]
        )

    lines.append("<b>By verdict</b>")
    for bucket in report.by_label:
        if not bucket.is_reportable:
            lines.append(f"{bucket.label} — {bucket.n} call(s), too few to summarise")
            continue
        annual = bucket.annualized_median_pct
        tail = f" · {_pct(annual)}/yr" if annual is not None else ""
        lines.append(
            f"{bucket.label} — n={bucket.n} · median {_pct(bucket.median_return_pct)} "
            f"(worst {_pct(bucket.worst_pct)}, best {_pct(bucket.best_pct)})"
            f" · ~{bucket.median_days}d{tail}"
        )

    lines.extend(
        [
            "",
            (
                "<i>Returns are price-only, measured from the price the call saw. "
                "Not benchmarked against the market unless stated, so treat the "
                "absolute numbers as context and the spread as the signal.</i>"
            ),
        ]
    )
    return "\n".join(lines)
