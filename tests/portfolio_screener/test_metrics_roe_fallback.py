"""ROE / key-trio fallback extraction — prevents false DATA_INSUFFICIENT
when Screener omits ROE % but P&L + BS still have PAT and book equity
(found live on BBOX)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from stockbot.models import Financials, TickerInfo
from stockbot.portfolio_screener.data_validator import validate_stock_data
from stockbot.portfolio_screener.hard_filters import apply_hard_filters
from stockbot.portfolio_screener.metrics import extract_metrics
from stockbot.portfolio_screener.scoring_config import HardFilterThresholds


def _financials_without_roe_ratio() -> Financials:
    years = ["Mar 2024", "Mar 2025", "Mar 2026"]
    pnl = pd.DataFrame(
        {
            "Mar 2024": [1000.0, 120.0, 80.0, 5.0],
            "Mar 2025": [1100.0, 140.0, 100.0, 6.0],
            "Mar 2026": [1200.0, 160.0, 218.0, 8.0],
        },
        index=["Sales", "Operating Profit", "Net Profit", "EPS in Rs"],
    )
    bs = pd.DataFrame(
        {
            "Mar 2024": [30.0, 500.0, 200.0, 50.0, 800.0],
            "Mar 2025": [34.0, 725.0, 300.0, 60.0, 1100.0],
            "Mar 2026": [36.0, 1251.0, 400.0, 70.0, 1800.0],
        },
        index=["Equity Capital", "Reserves", "Borrowings", "Cash Equivalents", "Total Assets"],
    )
    cf = pd.DataFrame(
        {
            "Mar 2024": [90.0, -40.0],
            "Mar 2025": [110.0, -50.0],
            "Mar 2026": [200.0, -60.0],
        },
        index=["Cash from Operating Activity", "Cash from Investing Activity"],
    )
    # Ratios deliberately omit ROE % (BBOX-style) but include ROCE
    ratios = pd.DataFrame(
        {"Mar 2024": [18.0], "Mar 2025": [19.0], "Mar 2026": [20.0]},
        index=["ROCE %"],
    )
    return Financials(
        pnl=pnl,
        balance_sheet=bs,
        cash_flow=cf,
        ratios=ratios,
        quarterly=pd.DataFrame(),
        basis="consolidated",
        years_available=3,
        business_description=None,
        source="test",
        fetched_at=datetime.now(UTC),
    )


def test_roe_computed_from_pat_and_book_equity_when_ratio_missing():
    ticker = TickerInfo(symbol="BBOX", exchange="NSE", company_name="Black Box", isin=None)
    m = extract_metrics(
        ticker,
        financials=_financials_without_roe_ratio(),
        price=None,
        shareholding=None,
        market_meta={"sector": "Technology", "market_cap_cr": 13000.0},
    )
    # PAT 218 / equity (36+1251)=1287 → ~16.94%
    assert m.roe is not None
    assert abs(m.roe - (218.0 / 1287.0) * 100.0) < 0.05
    assert m.metric_sources.get("roe") == "computed"
    assert "roe" not in m.missing


def test_single_missing_roe_without_compute_does_not_force_data_insufficient():
    # If PAT/equity also missing, roe stays null — but alone it must not
    # flip critical_ok (key trio needs ≥2 gaps).
    from stockbot.portfolio_screener.models import StockMetrics

    m = StockMetrics(
        ticker="X",
        current_price_abs=100.0,
        revenue=1000.0,
        net_income=50.0,
        eps=2.0,
        operating_cash_flow=60.0,
        debt_equity=0.4,
        ocf_to_pat=1.2,
        years_available=5,
        missing={"roe": "ROE % missing from ratios; cannot compute from P&L+BS"},
    )
    validation = validate_stock_data(m, HardFilterThresholds())
    assert validation.critical_ok is True
    hard = apply_hard_filters(m, validation, HardFilterThresholds())
    assert hard.status == "PASS"


def test_two_key_trio_gaps_force_data_insufficient():
    from stockbot.portfolio_screener.models import StockMetrics

    m = StockMetrics(
        ticker="X",
        current_price_abs=100.0,
        revenue=1000.0,
        net_income=50.0,
        eps=2.0,
        operating_cash_flow=60.0,
        years_available=5,
        missing={
            "roe": "missing",
            "debt_equity": "missing",
            "ocf_to_pat": "present elsewhere",
        },
    )
    # ocf_to_pat present via attribute
    m.ocf_to_pat = 1.0
    validation = validate_stock_data(m, HardFilterThresholds())
    assert validation.critical_ok is False
    hard = apply_hard_filters(m, validation, HardFilterThresholds())
    assert hard.status == "DATA_INSUFFICIENT"


def test_roce_computed_when_ratio_missing():
    ticker = TickerInfo(symbol="BBOX", exchange="NSE", company_name="Black Box", isin=None)
    fin = _financials_without_roe_ratio()
    # Drop ROCE from ratios too
    fin = Financials(
        pnl=fin.pnl,
        balance_sheet=fin.balance_sheet,
        cash_flow=fin.cash_flow,
        ratios=pd.DataFrame({"Mar 2026": [40.0]}, index=["Debtor Days"]),
        quarterly=fin.quarterly,
        basis=fin.basis,
        years_available=fin.years_available,
        business_description=None,
        source="test",
        fetched_at=fin.fetched_at,
    )
    m = extract_metrics(
        ticker,
        financials=fin,
        price=None,
        shareholding=None,
        market_meta={"sector": "Technology"},
    )
    # OP 160 / (equity 1287 + debt 400) = 160/1687 ≈ 9.48%
    assert m.roce is not None
    assert m.metric_sources.get("roce") == "computed"
    assert m.metric_sources.get("interest_coverage") == "computed" or m.interest_coverage is None


def test_format_computed_metric_warnings_are_user_facing():
    from stockbot.portfolio_screener.outcome_log import format_computed_metric_warnings

    lines = format_computed_metric_warnings(
        {"roe": "computed", "debt_equity": "computed", "roce": "fetched"},
        {"roe": 16.9, "debt_equity": 0.9},
    )
    assert any("ROE" in x and "Screener.in" in x for x in lines)
    assert any("D/E" in x and "not a Screener ratio row" in x for x in lines)
    assert not any(x.startswith("ROCE") for x in lines)


def test_roe_from_total_equity_row_when_ratios_omit_roe():
    from datetime import UTC, datetime

    import pandas as pd

    from stockbot.models import Financials, TickerInfo
    from stockbot.portfolio_screener.metrics import extract_metrics

    pnl = pd.DataFrame(
        {"Mar 2024": [1000.0, 80.0], "Mar 2025": [1200.0, 200.0]},
        index=["Sales", "Net Profit"],
    )
    bs = pd.DataFrame(
        {"Mar 2024": [800.0], "Mar 2025": [1000.0]},
        index=["Total Equity"],
    )
    cf = pd.DataFrame(
        {"Mar 2024": [90.0], "Mar 2025": [110.0]},
        index=["Cash from Operating Activity"],
    )
    ratios = pd.DataFrame({"Mar 2024": [10.0], "Mar 2025": [11.0]}, index=["ROCE %"])
    fin = Financials(
        pnl=pnl,
        balance_sheet=bs,
        cash_flow=cf,
        ratios=ratios,
        quarterly=pd.DataFrame(),
        basis="consolidated",
        years_available=2,
        business_description=None,
        source="test",
        fetched_at=datetime.now(UTC),
    )
    ticker = TickerInfo(symbol="X", exchange="NSE", company_name="X", isin=None)
    m = extract_metrics(
        ticker,
        financials=fin,
        price=None,
        shareholding=None,
        market_meta={"sector": "Utilities", "market_cap_cr": 50000.0},
    )
    assert m.roe is not None
    assert m.metric_sources.get("roe") == "computed"
    assert abs(m.roe - 20.0) < 0.05
