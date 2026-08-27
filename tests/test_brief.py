"""Module 7 (brief assembly) unit tests. Pure-logic pieces (red-flag
capping, MISSING formatting) are tested directly; assemble_brief's
degrade-on-failure orchestration is tested by monkeypatching the fetch
functions it calls — no network. Live behaviour (a full real brief for
RELIANCE and JYOTHYLAB, correct section content, timing) was verified by
hand during development."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stockbot import brief as brief_module
from stockbot.brief import (
    _fmt_or_missing,
    _format_annual_report_section,
    _format_news_section,
    _pct_or_missing,
    assemble_brief,
    cap_red_flags_per_query,
    format_financials_section,
    format_shareholding_section,
)
from stockbot.models import (
    Financials,
    NewsItems,
    PriceData,
    RedFlag,
    ReportText,
    Shareholding,
    Technicals,
    TickerInfo,
)

NOW = datetime.now(UTC)


def _ticker() -> TickerInfo:
    return TickerInfo(symbol="TEST", exchange="NSE", company_name="Test Co Limited", isin=None)


def _price() -> PriceData:
    df = pd.DataFrame({"Close": [100.0]})
    return PriceData(
        current_price_abs=100.0,
        price_date=date(2026, 8, 25),
        ohlcv_adjusted=df,
        ohlcv_unadjusted=df,
        week52_high_abs=120.0,
        week52_low_abs=80.0,
        source="yfinance:TEST.NS",
        fetched_at=NOW,
    )


def _technicals() -> Technicals:
    return Technicals(
        sma50=95.0,
        sma200=90.0,
        rsi14=55.0,
        support_abs=[85.0],
        resistance_abs=[110.0],
        as_of_date=date(2026, 8, 25),
        source="computed",
        fetched_at=NOW,
    )


def _financials() -> Financials:
    df = pd.DataFrame({"Mar 2025": [100.0]}, index=["Sales"])
    return Financials(
        pnl=df,
        balance_sheet=df,
        cash_flow=df,
        ratios=df,
        quarterly=df,
        basis="consolidated",
        years_available=1,
        source="screener:consolidated",
        fetched_at=NOW,
    )


def _empty_report_text(source_url: str | None = None) -> ReportText:
    return ReportText(
        sections={},
        report_year=None,
        source_url=source_url,
        truncated=False,
        dropped_sections=[],
        source="nse_annual_reports",
        fetched_at=NOW,
    )


def test_fmt_or_missing():
    assert _fmt_or_missing(1.5, "x") == "1.50"
    assert _fmt_or_missing(None, "insufficient history") == "MISSING: insufficient history"


def test_pct_or_missing():
    assert _pct_or_missing(12.345) == "12.35%"
    assert _pct_or_missing(None) == "MISSING: not available from this source"


def test_cap_red_flags_per_query_respects_limit():
    items = [
        RedFlag(f"Headline {i}", f"url-{i}", date(2026, 1, i % 28 + 1), "Q1") for i in range(20)
    ]
    capped = cap_red_flags_per_query(items, ["Q1"])
    assert len(capped) == brief_module.RED_FLAGS_PER_QUERY_LIMIT


def test_cap_red_flags_per_query_gives_each_query_its_own_budget():
    q1_items = [RedFlag(f"Q1 item {i}", f"u{i}", date(2026, 1, 1), "Q1") for i in range(20)]
    q2_items = [RedFlag("Q2 item", "u-q2", date(2026, 1, 1), "Q2")]
    capped = cap_red_flags_per_query(q1_items + q2_items, ["Q1", "Q2"])
    assert any(item.headline == "Q2 item" for item in capped)


def test_cap_red_flags_per_query_dedups_merged_query_labels():
    items = [RedFlag("Shared headline", "same-url", date(2026, 1, 1), "Q1, Q2")]
    capped = cap_red_flags_per_query(items, ["Q1", "Q2"])
    assert len(capped) == 1


def test_format_financials_section_missing_when_none():
    text = format_financials_section(None)
    assert "MISSING: financials" in text


def test_format_financials_section_states_basis():
    text = format_financials_section(_financials())
    assert "CONSOLIDATED" in text


def test_format_shareholding_section_missing_when_none():
    assert "MISSING: shareholding" in format_shareholding_section(None)


def test_format_shareholding_section_pledge_none_is_unconfirmed_not_zero():
    sh = Shareholding(
        promoter_pct=50.0,
        pledge_pct_of_promoter_holding=None,
        fii_pct=None,
        dii_pct=None,
        quarter="Q1",
        source="NSE",
        fetched_at=NOW,
    )
    text = format_shareholding_section(sh)
    pledge_line = next(line for line in text.splitlines() if "pledge" in line.lower())
    assert "unconfirmed" in pledge_line.lower()
    assert "0.00%" not in pledge_line


def test_format_news_section_missing_when_none():
    assert "MISSING: news" in _format_news_section(None)


def test_format_annual_report_missing_distinguishes_not_found_vs_no_content():
    not_found = _format_annual_report_section(_empty_report_text(source_url=None))
    assert "not found on NSE" in not_found

    found_but_empty = _format_annual_report_section(
        _empty_report_text(source_url="https://example.com/ar.pdf")
    )
    assert "no usable text extracted" in found_but_empty


def test_assemble_brief_raises_when_price_fails(monkeypatch):
    def _raise(symbol):
        raise ValueError("no price data")

    monkeypatch.setattr(brief_module, "fetch_price_data", _raise)
    with pytest.raises(ValueError):
        assemble_brief(_ticker())


def test_assemble_brief_degrades_financials_and_caps_confidence(monkeypatch):
    monkeypatch.setattr(brief_module, "fetch_price_data", lambda symbol: _price())
    monkeypatch.setattr(brief_module, "compute_technicals", lambda price: _technicals())

    def _raise_financials(symbol):
        raise RuntimeError("Screener unavailable")

    monkeypatch.setattr(brief_module, "fetch_fundamentals", _raise_financials)
    monkeypatch.setattr(
        brief_module,
        "fetch_shareholding",
        lambda symbol: Shareholding(50.0, None, None, None, "Q1", "NSE", NOW),
    )
    monkeypatch.setattr(
        brief_module,
        "fetch_news",
        lambda company: NewsItems([], [], [], [], "google_news_rss", NOW),
    )
    monkeypatch.setattr(brief_module, "fetch_annual_report", lambda symbol: _empty_report_text())

    result = assemble_brief(_ticker())

    assert result.financials is None
    assert result.confidence_ceiling == 4
    assert any("financials" in m for m in result.missing)
    assert result.token_count > 0


def test_assemble_brief_caps_confidence_when_annual_report_missing(monkeypatch):
    monkeypatch.setattr(brief_module, "fetch_price_data", lambda symbol: _price())
    monkeypatch.setattr(brief_module, "compute_technicals", lambda price: _technicals())
    monkeypatch.setattr(brief_module, "fetch_fundamentals", lambda symbol: _financials())
    monkeypatch.setattr(
        brief_module,
        "fetch_shareholding",
        lambda symbol: Shareholding(50.0, None, None, None, "Q1", "NSE", NOW),
    )
    monkeypatch.setattr(
        brief_module,
        "fetch_news",
        lambda company: NewsItems([], [], [], [], "google_news_rss", NOW),
    )
    monkeypatch.setattr(brief_module, "fetch_annual_report", lambda symbol: _empty_report_text())

    result = assemble_brief(_ticker())

    assert result.financials is not None
    assert result.confidence_ceiling == 5
    assert any("annual report" in m for m in result.missing)


def test_assemble_brief_full_success_has_no_missing_entries(monkeypatch):
    monkeypatch.setattr(brief_module, "fetch_price_data", lambda symbol: _price())
    monkeypatch.setattr(brief_module, "compute_technicals", lambda price: _technicals())
    monkeypatch.setattr(brief_module, "fetch_fundamentals", lambda symbol: _financials())
    monkeypatch.setattr(
        brief_module,
        "fetch_shareholding",
        lambda symbol: Shareholding(50.0, None, None, None, "Q1", "NSE", NOW),
    )
    monkeypatch.setattr(
        brief_module,
        "fetch_news",
        lambda company: NewsItems([], [], [], [], "google_news_rss", NOW),
    )
    report = ReportText(
        sections={"Key Audit Matters": "some text"},
        report_year=2026,
        source_url="https://example.com/ar.pdf",
        truncated=False,
        dropped_sections=[],
        source="nse_annual_reports",
        fetched_at=NOW,
    )
    monkeypatch.setattr(brief_module, "fetch_annual_report", lambda symbol: report)

    result = assemble_brief(_ticker())

    assert result.confidence_ceiling == 10
    assert result.missing == []
