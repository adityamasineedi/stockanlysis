"""Missing single-stock snapshot metrics: quarterly P/E, TTM sales/PAT, OPM, order book."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stockbot.models import Financials, PriceData, TickerInfo
from stockbot.portfolio_screener.metrics import (
    compute_quarterly_pe,
    extract_metrics,
)
from stockbot.portfolio_screener.prescan_display import format_metric_lines

NOW = datetime.now(UTC)


def test_compute_quarterly_pe_annualizes_latest_quarter_eps() -> None:
    # Screener convention: price / (Q EPS × 4)
    assert compute_quarterly_pe(248.0, 1.49) == pytest.approx(248.0 / (1.49 * 4.0))
    assert compute_quarterly_pe(100.0, None) is None
    assert compute_quarterly_pe(100.0, 0.0) is None


def test_extract_metrics_fills_ttm_opm_quarterly_pe_and_order_book() -> None:
    ticker = TickerInfo(symbol="TEST", exchange="NSE", company_name="Test", isin=None)
    pnl = pd.DataFrame(
        {
            "Mar 2024": [1000.0, 150.0, 80.0, 10.0],
            "Mar 2025": [1200.0, 200.0, 100.0, 12.0],
            "TTM": [5993.0, 1145.0, 573.0, 20.0],
        },
        index=["Sales", "Operating Profit", "Net Profit", "EPS in Rs"],
    )
    quarterly = pd.DataFrame(
        {
            "Dec 2025": [1400.0, 1.20],
            "Mar 2026": [1500.0, 1.49],
        },
        index=["Sales", "EPS in Rs"],
    )
    fin = Financials(
        pnl=pnl,
        balance_sheet=pd.DataFrame({"Mar 2025": [500.0]}, index=["Total Assets"]),
        cash_flow=pd.DataFrame(
            {"Mar 2025": [100.0]}, index=["Cash from Operating Activity"]
        ),
        ratios=pd.DataFrame({"Mar 2025": [15.0]}, index=["ROE %"]),
        quarterly=quarterly,
        basis="consolidated",
        years_available=2,
        source="test",
        fetched_at=NOW,
    )
    price = PriceData(
        248.0,
        date(2026, 9, 4),
        pd.DataFrame({"Close": [248.0]}),
        pd.DataFrame({"Close": [248.0]}),
        300.0,
        200.0,
        "yfinance",
        NOW,
    )
    m = extract_metrics(
        ticker,
        financials=fin,
        price=price,
        shareholding=None,
        market_meta={"market_cap_cr": 37982.0, "trailing_pe": 66.3},
        order_book_cr=26665.0,
    )
    assert m.sales_ttm_cr == 5993.0
    assert m.pat_ttm_cr == 573.0
    assert m.opm_pct == pytest.approx((1145.0 / 5993.0) * 100.0, abs=0.05)
    assert m.quarterly_pe == pytest.approx(248.0 / (1.49 * 4.0), abs=0.05)
    assert m.order_book_cr == 26665.0
    assert m.pe == 66.3
    assert m.market_cap_cr == 37982.0


def test_format_metric_lines_includes_new_snapshot_fields() -> None:
    lines = format_metric_lines(
        market_cap_cr=37982.0,
        pe=66.3,
        quarterly_pe=41.6,
        opm_pct=19.1,
        sales_ttm_cr=5993.0,
        pat_ttm_cr=573.0,
        order_book_cr=26665.0,
        roe=6.95,
    )
    joined = "\n".join(lines)
    assert "Market Cap ₹37,982 Cr" in joined
    assert "Stock P/E 66.30×" in joined
    assert "Quarterly P/E 41.60×" in joined
    assert "OPM 19.1%" in joined
    assert "Sales TTM ₹5,993 Cr" in joined
    assert "PAT TTM ₹573 Cr" in joined
    assert "Order Book ₹26,665 Cr" in joined
    assert "ROE" in joined
