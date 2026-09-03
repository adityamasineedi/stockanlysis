"""Recency-weighted five-year horizon: boom-then-fade, short history, holes."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from stockbot.constitution_gates import apply_constitution_overrides
from stockbot.five_year_policy import assess_five_year_horizon
from stockbot.llm.verdict import VerdictJSON, compute_valuation
from stockbot.models import (
    Brief,
    Financials,
    PriceData,
    ReportText,
    Technicals,
    TickerInfo,
)
from stockbot.trade_policy import five_year_allows_buy_zone

NOW = datetime.now(UTC)


def _financials(
    sales: list[float | None],
    *,
    years: int | None = None,
    extra_columns: dict[str, list[float | None]] | None = None,
    index: list[str] | None = None,
) -> Financials:
    cols = [f"FY{21 + i}" for i in range(len(sales))]
    data = {col: [value] for col, value in zip(cols, sales, strict=True)}
    if extra_columns:
        data.update(extra_columns)
    pnl = pd.DataFrame(data, index=index or ["Sales"])
    empty = pd.DataFrame()
    return Financials(
        pnl=pnl,
        balance_sheet=empty,
        cash_flow=empty,
        ratios=empty,
        quarterly=empty,
        basis="consolidated",
        years_available=years if years is not None else len(sales),
        source="test",
        fetched_at=NOW,
    )


def _brief_from_financials(financials: Financials | None) -> Brief:
    df = pd.DataFrame({"Close": [100.0]})
    return Brief(
        ticker=TickerInfo(symbol="TEST", exchange="NSE", company_name="Test", isin=None),
        price=PriceData(100.0, date(2026, 8, 26), df, df, 120.0, 80.0, "yfinance", NOW),
        technicals=Technicals(95.0, 90.0, 55.0, [85.0], [110.0], date(2026, 8, 26), "computed", NOW),
        financials=financials,
        shareholding=None,
        news=None,
        annual_report=ReportText({}, None, None, False, [], "nse_annual_reports", NOW),
        missing=[],
        token_count=0,
        confidence_ceiling=7,
        generated_at=NOW,
    )


def _verdict(**overrides: object) -> VerdictJSON:
    base: dict[str, object] = {
        "verdict": "WATCH",
        "current_price_abs": 300.0,
        "price_date": date(2026, 8, 26),
        "buy_zone_abs": [250.0, 280.0],
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
        "buy_range_allowed": True,
        "add_range_allowed": True,
        "five_year_business_test": {
            "answer": "YES",
            "confidence": "HIGH",
            "evidence_for": ["capacity added", "ROCE durable"],
            "evidence_against": [],
        },
    }
    base.update(overrides)
    return VerdictJSON.model_validate(base)


def _apply(brief: Brief, **verdict_overrides: object) -> VerdictJSON:
    verdict = _verdict(**verdict_overrides)
    valuation = compute_valuation(verdict.valuation_inputs)
    return apply_constitution_overrides(verdict, valuation, brief)


def test_boom_then_fade_is_decelerating():
    # Early doubling, then a stall — 5y CAGR stays high, 3y collapses.
    brief = _brief_from_financials(_financials([10.0, 25.0, 50.0, 55.0, 56.0, 57.0], years=6))
    assessment = assess_five_year_horizon(brief)
    assert assessment.growth_trend == "DECELERATING"
    assert assessment.withhold_buy_zone is True
    assert assessment.force_uncertain is True
    assert assessment.revenue_cagr_3y is not None
    assert assessment.revenue_cagr_5y is not None
    assert assessment.revenue_cagr_3y < assessment.revenue_cagr_5y - 0.03


def test_deceleration_overrides_yes_and_clears_buy_zone():
    brief = _brief_from_financials(_financials([10.0, 25.0, 50.0, 55.0, 56.0, 57.0], years=6))
    updated = _apply(brief)
    assert updated.five_year_business_test is not None
    assert updated.five_year_business_test.answer == "UNCERTAIN"
    assert updated.five_year_business_test.confidence == "LOW"
    assert updated.buy_zone_abs is None
    assert updated.buy_range_allowed is False
    assert updated.add_range_allowed is False
    assert any("five_year_horizon:decelerating" in g for g in updated.gates_failed)
    assert any("DECELERATING" in item for item in updated.five_year_business_test.evidence_against)
    assert five_year_allows_buy_zone(updated.five_year_business_test) is False


def test_deceleration_does_not_upgrade_no():
    brief = _brief_from_financials(_financials([10.0, 25.0, 50.0, 55.0, 56.0, 57.0], years=6))
    updated = _apply(
        brief,
        five_year_business_test={
            "answer": "NO",
            "confidence": "HIGH",
            "evidence_for": [],
            "evidence_against": ["leverage"],
        },
    )
    assert updated.five_year_business_test is not None
    assert updated.five_year_business_test.answer == "NO"
    assert updated.buy_zone_abs is None


def test_short_history_allows_research_not_ideal_buy():
    brief = _brief_from_financials(_financials([100.0, 140.0], years=2))
    assessment = assess_five_year_horizon(brief)
    assert assessment.short_history is True
    assert assessment.withhold_buy_zone is True
    updated = _apply(brief)
    assert updated.five_year_business_test is not None
    assert updated.five_year_business_test.answer == "UNCERTAIN"
    assert updated.buy_zone_abs is None
    assert any("five_year_horizon:short_history" in g for g in updated.gates_failed)
    assert any("research is allowed" in item.lower() for item in updated.five_year_business_test.evidence_against)


def test_one_year_growth_is_research_only():
    brief = _brief_from_financials(_financials([100.0], years=1))
    assessment = assess_five_year_horizon(brief)
    assert assessment.short_history is True
    assert assessment.withhold_buy_zone is True
    updated = _apply(brief)
    assert updated.buy_range_allowed is False
    assert updated.five_year_business_test is not None
    assert updated.five_year_business_test.answer != "YES"


def test_missing_latest_year_is_data_review_not_reject():
    brief = _brief_from_financials(
        _financials([100.0, 110.0, 121.0, 133.0, None], years=5)
    )
    assessment = assess_five_year_horizon(brief)
    assert assessment.latest_fiscal_incomplete is True
    assert assessment.withhold_buy_zone is True
    updated = _apply(brief)
    assert updated.five_year_business_test is not None
    assert updated.five_year_business_test.answer == "UNCERTAIN"
    assert "DATA REVIEW" in updated.missing_data_impact
    assert updated.buy_zone_abs is None
    # Incomplete is not a NO / bad-business auto-reject.
    assert updated.five_year_business_test.answer != "NO"
    assert any("incomplete" in item.lower() for item in updated.five_year_business_test.evidence_against)


def test_ttm_fills_latest_year_hole():
    sales = [100.0, 110.0, 121.0, 133.0, None]
    brief = _brief_from_financials(
        _financials(
            sales,
            years=5,
            extra_columns={"TTM": [146.0]},
        )
    )
    assessment = assess_five_year_horizon(brief)
    assert assessment.latest_fiscal_incomplete is False


def test_accelerating_path_keeps_yes_buy_zone():
    # 3y CAGR above 5y — recency supports durability.
    brief = _brief_from_financials(_financials([100.0, 110.0, 121.0, 140.0, 170.0, 220.0], years=6))
    assessment = assess_five_year_horizon(brief)
    assert assessment.growth_trend == "ACCELERATING"
    assert assessment.withhold_buy_zone is False
    updated = _apply(brief)
    assert updated.five_year_business_test is not None
    assert updated.five_year_business_test.answer == "YES"
    assert updated.buy_zone_abs == (250.0, 280.0)
    assert updated.buy_range_allowed is True


def test_negative_recent_cagr_withholds_yes():
    brief = _brief_from_financials(_financials([200.0, 180.0, 150.0, 120.0, 100.0, 80.0], years=6))
    assessment = assess_five_year_horizon(brief)
    assert assessment.growth_trend == "NEGATIVE"
    updated = _apply(brief)
    assert updated.five_year_business_test is not None
    assert updated.five_year_business_test.answer == "UNCERTAIN"
    assert updated.buy_zone_abs is None
