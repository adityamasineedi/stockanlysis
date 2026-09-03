"""Calibration report assembly — I/O boundaries and rendering."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from stockbot.calibration_report import (
    PRESCAN_NEGATIVE,
    PRESCAN_POSITIVE,
    _build,
    _current_prices,
    format_calibration_report,
)

NOW = datetime(2026, 8, 31, tzinfo=UTC)


def _row(ticker: str, verdict: str, entry: float, days_ago: int) -> dict:
    return {
        "ticker": ticker,
        "verdict": verdict,
        "price_at_scan": entry,
        "logged_at": (NOW - timedelta(days=days_ago)).isoformat(),
    }


def _plain(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def test_repeated_tickers_across_rows_are_fetched_once(monkeypatch):
    """A year of scans repeats names heavily and yfinance rate-limits, so
    collapsing many rows to one fetch per ticker is load-bearing, not tidiness.
    Twelve rows covering two names must cost two fetches."""
    fetched: list[str] = []

    def fake_fetch(symbol):
        fetched.append(symbol)
        return type("P", (), {"current_price_abs": 118.0})()

    import stockbot.fetch.prices as prices_mod

    monkeypatch.setattr(prices_mod, "fetch_price_data", fake_fetch)

    rows = [_row("BEL", "AUTO_DEEP_ANALYSIS", 100, 40 + 30 * i) for i in range(6)]
    rows += [_row("CRISIL", "AUTO_DEEP_ANALYSIS", 100, 40 + 30 * i) for i in range(6)]
    report = _build("prescan", rows, PRESCAN_POSITIVE, PRESCAN_NEGATIVE, now=NOW)

    assert sorted(fetched) == ["BEL", "CRISIL"]  # 12 rows, 2 fetches
    assert report.scored == 12


def test_one_failing_fetch_drops_only_that_ticker(monkeypatch):
    import stockbot.fetch.prices as prices_mod

    def flaky(symbol):
        if symbol == "BROKEN":
            raise RuntimeError("yfinance said no")
        return type("P", (), {"current_price_abs": 120.0})()

    monkeypatch.setattr(prices_mod, "fetch_price_data", flaky)

    resolved = _current_prices({"GOOD", "BROKEN"})
    assert resolved["GOOD"] == 120.0
    assert resolved["BROKEN"] is None  # dropped downstream, not counted flat


def test_report_reports_a_working_ranking():
    rows = [_row(f"GOOD{i}", "AUTO_DEEP_ANALYSIS", 100, 200) for i in range(7)]
    rows += [_row(f"BAD{i}", "NOT_SUITABLE_FOR_3Y_RESEARCH", 100, 200) for i in range(6)]
    prices = {f"GOOD{i}": 118.0 for i in range(7)} | {f"BAD{i}": 103.0 for i in range(6)}

    report = _build(
        "prescan", rows, PRESCAN_POSITIVE, PRESCAN_NEGATIVE, now=NOW, prices=prices
    )
    assert report.scored == 13
    assert report.spread.spread_pct == 15.0  # 18 - 3
    assert report.spread.discriminates is True

    text = _plain(format_calibration_report(report))
    assert "the score is discriminating" in text
    assert "Spread: +15.0 points" in text
    assert "needs no benchmark" in text


def test_report_states_plainly_when_the_ranking_does_not_work():
    """The score failing to discriminate is the honest finding, and the report
    must be willing to print it."""
    rows = [_row(f"GOOD{i}", "AUTO_DEEP_ANALYSIS", 100, 200) for i in range(6)]
    rows += [_row(f"BAD{i}", "NOT_SUITABLE_FOR_3Y_RESEARCH", 100, 200) for i in range(6)]
    prices = {f"GOOD{i}": 102.0 for i in range(6)} | {f"BAD{i}": 115.0 for i in range(6)}

    report = _build(
        "prescan", rows, PRESCAN_POSITIVE, PRESCAN_NEGATIVE, now=NOW, prices=prices
    )
    assert report.spread.spread_pct == -13.0
    text = _plain(format_calibration_report(report))
    assert "inverted" in text


def test_thin_history_gets_a_message_not_a_table():
    rows = [_row("A", "AUTO_DEEP_ANALYSIS", 100, 30)]
    report = _build(
        "prescan", rows, PRESCAN_POSITIVE, PRESCAN_NEGATIVE, now=NOW, prices={"A": 110.0}
    )
    assert report.has_enough_history is False

    text = _plain(format_calibration_report(report))
    assert "Not enough history yet" in text
    assert "By verdict" not in text
    assert "fills in on its own" in text


def test_empty_history_does_not_crash():
    report = _build("prescan", [], PRESCAN_POSITIVE, PRESCAN_NEGATIVE, now=NOW, prices={})
    assert report.total_rows == 0 and report.scored == 0
    assert "Not enough history yet" in _plain(format_calibration_report(report))


def test_unscoreable_rows_count_as_logged_but_not_scored():
    """The gap between the two numbers is itself information — it says how
    much history is unusable."""
    rows = [_row(f"GOOD{i}", "AUTO_DEEP_ANALYSIS", 100, 200) for i in range(6)]
    rows.append({"ticker": "NOPRICE", "verdict": "AUTO_DEEP_ANALYSIS", "logged_at": NOW.isoformat()})
    prices = {f"GOOD{i}": 118.0 for i in range(6)} | {"NOPRICE": None}

    report = _build(
        "prescan", rows, PRESCAN_POSITIVE, PRESCAN_NEGATIVE, now=NOW, prices=prices
    )
    assert report.total_rows == 7
    assert report.scored == 6
    assert "7 logged · 6 scoreable" in _plain(format_calibration_report(report))


def test_buckets_below_min_sample_are_named_not_summarised():
    rows = [_row(f"GOOD{i}", "AUTO_DEEP_ANALYSIS", 100, 200) for i in range(6)]
    rows += [_row("LONE", "SECTOR_SPECIFIC_REVIEW", 100, 200)]
    prices = {f"GOOD{i}": 118.0 for i in range(6)} | {"LONE": 300.0}

    report = _build(
        "prescan", rows, PRESCAN_POSITIVE, PRESCAN_NEGATIVE, now=NOW, prices=prices
    )
    text = _plain(format_calibration_report(report))
    # The 200% outlier must not be presented as a median.
    assert "SECTOR_SPECIFIC_REVIEW — 1 call(s), too few to summarise" in text
    assert "+200.0%" not in text


def test_header_shows_where_the_missing_rows_went():
    """A report that silently shrinks its sample reads as confidently on twelve
    rows as on twelve hundred."""
    rows = [_row(f"OK{i}", "AUTO_DEEP_ANALYSIS", 100, 200) for i in range(6)]
    rows += [_row(f"NEW{i}", "AUTO_DEEP_ANALYSIS", 100, 2) for i in range(4)]
    rows += [_row("NOJUDGE", "DATA_UNAVAILABLE_RETRY", 100, 200)]
    prices = (
        {f"OK{i}": 118.0 for i in range(6)}
        | {f"NEW{i}": 118.0 for i in range(4)}
        | {"NOJUDGE": 118.0}
    )

    report = _build(
        "prescan", rows, PRESCAN_POSITIVE, PRESCAN_NEGATIVE, now=NOW, prices=prices
    )
    assert report.total_rows == 11
    assert report.scored == 6
    assert report.too_recent == 4
    assert report.not_a_judgment == 1

    text = _plain(format_calibration_report(report))
    assert "11 logged · 6 scoreable · 4 too recent to score · 1 not judgements" in text
    # The failed-fetch rows must not appear as a verdict with a return.
    assert "DATA_UNAVAILABLE_RETRY" not in text


def test_hairline_spread_renders_as_no_difference():
    """The live report said "+0.0 points — the score is discriminating"."""
    rows = [_row(f"GOOD{i}", "AUTO_DEEP_ANALYSIS", 100, 200) for i in range(6)]
    rows += [_row(f"BAD{i}", "NOT_SUITABLE_FOR_3Y_RESEARCH", 100, 200) for i in range(6)]
    # 1.24% vs 1.20% — both render as +1.2%, a 0.04-point gap.
    prices = {f"GOOD{i}": 101.24 for i in range(6)} | {f"BAD{i}": 101.20 for i in range(6)}

    report = _build(
        "prescan", rows, PRESCAN_POSITIVE, PRESCAN_NEGATIVE, now=NOW, prices=prices
    )
    assert report.spread.spread_pct == 0.04
    text = _plain(format_calibration_report(report))
    assert "no discernible difference between the tiers" in text
    assert "the score is discriminating" not in text
