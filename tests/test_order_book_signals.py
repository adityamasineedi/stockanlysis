"""Tests for structured order-book signal extraction."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from stockbot.models import (
    Brief,
    Financials,
    NewsItems,
    PriceData,
    RedFlag,
    ReportText,
    Technicals,
    TickerInfo,
)
from stockbot.order_book_signals import (
    _parse_amount_cr,
    collect_order_book_signals,
    extract_order_book_annual_report_signals,
    order_book_wc_billing_hint,
)

NOW = datetime.now(UTC)


def _brief(
    *,
    news: NewsItems | None = None,
    annual_report: ReportText | None = None,
    revenue_cr: float | None = None,
) -> Brief:
    pnl = pd.DataFrame({"TTM": [10.0]}, index=["EPS in Rs"])
    if revenue_cr is not None:
        pnl.loc["Sales"] = [revenue_cr]
    fin = Financials(
        pnl=pnl,
        balance_sheet=pd.DataFrame(),
        cash_flow=pd.DataFrame(),
        ratios=pd.DataFrame(),
        quarterly=pd.DataFrame(),
        basis="consolidated",
        years_available=1,
        source="test",
        fetched_at=NOW,
    )
    df = pd.DataFrame({"Close": [100.0]})
    return Brief(
        ticker=TickerInfo(symbol="TEST", exchange="NSE", company_name="Test", isin=None),
        price=PriceData(100.0, date(2026, 8, 26), df, df, 120.0, 80.0, "yfinance", NOW),
        technicals=Technicals(95.0, 90.0, 55.0, [85.0], [110.0], date(2026, 8, 26), "computed", NOW),
        financials=fin,
        shareholding=None,
        news=news,
        annual_report=annual_report
        or ReportText({}, None, None, False, [], "nse_annual_reports", NOW),
        missing=[],
        token_count=0,
        confidence_ceiling=7,
        generated_at=NOW,
    )


def test_parse_amount_cr_from_headline():
    assert _parse_amount_cr("order book at Rs 20,535 crore") == 20535.0
    assert _parse_amount_cr("backlog of 500 lakh") == 5.0


def test_annual_report_order_book_snippet():
    report = ReportText(
        sections={
            "Management Discussion": (
                "The unexecuted order book stood at Rs 18,200 crore as on March 31."
            )
        },
        report_year=2025,
        source_url=None,
        truncated=False,
        dropped_sections=[],
        source="nse_annual_reports",
        fetched_at=NOW,
    )
    signals = extract_order_book_annual_report_signals(_brief(annual_report=report))
    assert len(signals) == 1
    assert signals[0].amount_cr == 18200.0
    assert "order book" in signals[0].text.lower()


def test_wc_billing_hint_when_backlog_exceeds_revenue_multiple():
    news = NewsItems(
        general=[
            RedFlag(
                headline="Co order book Rs 20,535 crore on strong pipeline",
                url="https://example.com",
                published_date=date(2026, 5, 1),
                found_by_query="q",
            )
        ],
        red_flags=[],
        queries_run=["q"],
        queries_empty=[],
        source="google",
        fetched_at=NOW,
    )
    brief = _brief(news=news, revenue_cr=5000.0)
    signals = collect_order_book_signals(brief)
    hint = order_book_wc_billing_hint(brief, signals)
    assert hint is not None
    assert "TEMPORARY_BILLING_CYCLE" in hint
