"""Tests for prescan metric extraction helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from stockbot.models import Financials, TickerInfo
from stockbot.portfolio_screener.issuer_routing import classify_issuer
from stockbot.portfolio_screener.metrics import count_derived_key_ratios, extract_metrics


def test_sector_override_for_vguard() -> None:
    ticker = TickerInfo(symbol="VGUARD", exchange="NSE", company_name="V-Guard", isin=None)
    m = extract_metrics(
        ticker,
        financials=None,
        price=None,
        shareholding=None,
        market_meta={"sector": "Utilities", "industry": "Utilities - Regulated Electric"},
    )
    assert m.sector_source == "override"
    assert m.sector == "Consumer Cyclical"
    assert classify_issuer(m) == "NON_FINANCIAL"


def test_financials_basis_recorded() -> None:
    import pandas as pd

    ticker = TickerInfo(symbol="TCS", exchange="NSE", company_name="TCS", isin=None)
    pnl = pd.DataFrame({"Mar 2024": [100.0]}, index=["Sales"])
    bs = pd.DataFrame({"Mar 2024": [500.0]}, index=["Total Assets"])
    cf = pd.DataFrame({"Mar 2024": [10.0]}, index=["Net Cash Flow"])
    ratios = pd.DataFrame({"Mar 2024": [20.0]}, index=["ROE %"])
    fin = Financials(
        pnl=pnl,
        balance_sheet=bs,
        cash_flow=cf,
        ratios=ratios,
        quarterly=pd.DataFrame(),
        basis="consolidated",
        years_available=1,
        source="test",
        fetched_at=datetime.now(UTC),
    )
    m = extract_metrics(ticker, financials=fin, price=None, shareholding=None)
    assert m.financials_basis == "consolidated"


def test_ttm_column_stripped_for_cumulative_ocf_pat_alignment() -> None:
    """Screener's P&L often carries a trailing TTM column that its cash-flow
    statement does not. Positionally pairing the raw "last 3" of each series
    would silently sum mismatched fiscal periods (e.g. PAT's FY25/FY26/TTM
    against OCF's FY24/FY25/FY26). net_income_series_fy_only must drop TTM so
    the cumulative 3y ratio is computed over the same three fiscal years."""
    import pandas as pd

    from stockbot.portfolio_screener.issuer_routing import assess_cash_conversion

    ticker = TickerInfo(symbol="HEROMOTOCO", exchange="NSE", company_name="Hero MotoCorp", isin=None)
    pnl = pd.DataFrame(
        {"Mar 2024": [100.0], "Mar 2025": [120.0], "Mar 2026": [140.0], "TTM": [150.0]},
        index=["Net Profit"],
    )
    bs = pd.DataFrame({"Mar 2026": [500.0]}, index=["Total Assets"])
    cf = pd.DataFrame(
        {"Mar 2024": [110.0], "Mar 2025": [140.0], "Mar 2026": [168.0]},
        index=["Cash from Operating Activity"],
    )
    ratios = pd.DataFrame({"Mar 2026": [20.0]}, index=["ROE %"])
    fin = Financials(
        pnl=pnl,
        balance_sheet=bs,
        cash_flow=cf,
        ratios=ratios,
        quarterly=pd.DataFrame(),
        basis="consolidated",
        years_available=3,
        source="test",
        fetched_at=datetime.now(UTC),
    )
    m = extract_metrics(ticker, financials=fin, price=None, shareholding=None)

    # Raw series keep TTM (used for "latest" figures elsewhere).
    assert m.net_income_series == [100.0, 120.0, 140.0, 150.0]
    # FY-only series used for cumulative multi-year ratios drops it.
    assert m.net_income_series_fy_only == [100.0, 120.0, 140.0]
    assert m.ocf_series_fy_only == [110.0, 140.0, 168.0]

    cash = assess_cash_conversion(m, "NON_FINANCIAL")
    # Correct: ΣOCF(FY24-26)=418 / ΣPAT(FY24-26)=360 ≈ 1.16 — not the
    # TTM-misaligned ΣOCF=418 / ΣPAT(FY25,FY26,TTM=120+140+150=410) ≈ 1.02.
    assert cash.ocf_pat_3y is not None
    assert abs(cash.ocf_pat_3y - (418.0 / 360.0)) < 0.001


def test_count_derived_key_ratios() -> None:
    from stockbot.portfolio_screener.models import StockMetrics

    metrics = StockMetrics(
        ticker="X",
        metric_sources={
            "roe": "computed",
            "debt_equity": "computed",
            "ocf_to_pat": "computed",
            "interest_coverage": "fetched",
        },
    )
    assert count_derived_key_ratios(metrics) == 3
