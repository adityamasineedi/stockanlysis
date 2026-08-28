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
