"""Tests for deterministic constitution gates."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from stockbot.constitution_gates import (
    apply_constitution_overrides,
    compute_valuation_tension,
    compute_valuation_tension_from_dict,
    refresh_constitution_fields,
    should_anti_chase,
    should_anti_chase_from_dict,
    sync_live_price_into_verdict,
)
from stockbot.llm.verdict import (
    VerdictJSON,
    compute_valuation,
)
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
from stockbot.news_claims import extract_order_book_news_claims

NOW = datetime.now(UTC)


def _brief_with_eps(eps: float) -> Brief:
    pnl = pd.DataFrame({"TTM": [eps]}, index=["EPS in Rs"])
    empty = pd.DataFrame()
    fin = Financials(
        pnl=pnl,
        balance_sheet=empty,
        cash_flow=empty,
        ratios=empty,
        quarterly=empty,
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
        news=None,
        annual_report=ReportText({}, None, None, False, [], "nse_annual_reports", NOW),
        missing=[],
        token_count=0,
        confidence_ceiling=7,
        generated_at=NOW,
    )


def _verdict(**overrides) -> VerdictJSON:
    base = {
        "verdict": "WATCH",
        "current_price_abs": 400.0,
        "price_date": date(2026, 8, 26),
        "buy_zone_abs": None,
        "valuation_inputs": {
            "eps_bear": 8.0,
            "eps_base": 10.0,
            "eps_bull": 12.0,
            "multiple_bear": [30.0, 32.0],
            "multiple_base": [38.0, 40.0],
            "multiple_bull": [42.0, 45.0],
        },
        "confidence": 5,
        "risk": "MEDIUM",
        "business_quality": 7,
        "financial_health": 7,
        "management_quality": 7,
        "earnings_quality": "MEDIUM",
        "holding_period": "6-12 months",
        "reasons_buy": [],
        "reasons_avoid": [],
        "biggest_watch": "cash",
        "missing_data_impact": "none",
        "gates_failed": [],
    }
    base.update(overrides)
    return VerdictJSON.model_validate(base)


def test_should_anti_chase_when_price_at_base_fv_top():
    verdict = _verdict(current_price_abs=400.0)
    valuation = compute_valuation(verdict.valuation_inputs)
    assert valuation.fair_value_base_abs[1] == 400.0
    flag, reason = should_anti_chase(verdict, valuation, _brief_with_eps(10.0))
    assert flag is True
    assert "base fair-value top" in reason


def test_should_anti_chase_when_pe_rich_and_earnings_not_high():
    verdict = _verdict(current_price_abs=360.0, earnings_quality="MEDIUM")
    valuation = compute_valuation(verdict.valuation_inputs)
    flag, reason = should_anti_chase(verdict, valuation, _brief_with_eps(10.0))
    assert flag is True
    assert "P/E" in reason


def test_apply_overrides_clears_buy_zone_when_anti_chase():
    verdict = _verdict(
        current_price_abs=400.0,
        buy_zone_abs=(350.0, 380.0),
        buy_range_allowed=True,
        anti_chase_flag=False,
    )
    valuation = compute_valuation(verdict.valuation_inputs)
    updated = apply_constitution_overrides(verdict, valuation, _brief_with_eps(10.0))
    assert updated.anti_chase_flag is True
    assert updated.buy_range_allowed is False
    assert updated.buy_zone_abs is None
    assert updated.external_valuation_tension == "HIGH"


def test_should_anti_chase_from_dict_mazdock_style():
    verdict = {
        "current_price_abs": 2625.0,
        "earnings_quality": "MEDIUM",
        "anti_chase_flag": False,
        "fair_value_base_abs": [2280.0, 2584.0],
        "valuation_inputs": {
            "eps_bear": 55.0,
            "eps_base": 76.0,
            "eps_bull": 88.0,
            "multiple_bear": [16.0, 20.0],
            "multiple_base": [30.0, 34.0],
            "multiple_bull": [36.0, 40.0],
        },
    }
    flag, reason = should_anti_chase_from_dict(verdict)
    assert flag is True
    assert "2584" in reason

    news = NewsItems(
        general=[
            RedFlag(
                headline="Mazagon Dock order book at Rs 20,535 crore",
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
    claims = extract_order_book_news_claims(news)
    assert len(claims) == 1
    assert "UNVERIFIED NEWS" in claims[0]
    assert "20,535" in claims[0]


def test_valuation_tension_high_at_base_fv_top():
    verdict = _verdict(current_price_abs=400.0)
    valuation = compute_valuation(verdict.valuation_inputs)
    assert compute_valuation_tension(400.0, valuation) == "HIGH"


def test_refresh_constitution_fields_sets_anti_chase_on_cache_style_dict():
    verdict = {
        "current_price_abs": 2625.0,
        "earnings_quality": "MEDIUM",
        "anti_chase_flag": False,
        "buy_range_allowed": True,
        "buy_zone_abs": [2000.0, 2400.0],
        "fair_value_base_abs": [2280.0, 2584.0],
        "valuation_inputs": {
            "eps_bear": 55.0,
            "eps_base": 76.0,
            "eps_bull": 88.0,
            "multiple_bear": [16.0, 20.0],
            "multiple_base": [30.0, 34.0],
            "multiple_bull": [36.0, 40.0],
        },
    }
    refreshed = refresh_constitution_fields(verdict)
    assert refreshed["anti_chase_flag"] is True
    assert refreshed["buy_zone_abs"] is None
    assert refreshed["external_valuation_tension"] == "HIGH"
    assert compute_valuation_tension_from_dict(verdict) == "HIGH"


def test_sync_live_price_updates_display_and_gates():
    verdict = {
        "current_price_abs": 2625.0,
        "price_date": "2026-08-26",
        "earnings_quality": "MEDIUM",
        "fair_value_base_abs": [2280.0, 2584.0],
        "valuation_inputs": {
            "eps_bear": 55.0,
            "eps_base": 76.0,
            "eps_bull": 88.0,
            "multiple_bear": [16.0, 20.0],
            "multiple_base": [30.0, 34.0],
            "multiple_bull": [36.0, 40.0],
        },
    }
    synced = sync_live_price_into_verdict(
        verdict, live_price_abs=408.54998779296875, live_price_date=date(2026, 8, 28)
    )
    assert synced["analysis_price_abs"] == 2625.0
    assert synced["analysis_price_date"] == "2026-08-26"
    assert synced["current_price_abs"] == 408.55
    assert synced["price_date"] == "2026-08-28"
    assert synced["anti_chase_flag"] is False
