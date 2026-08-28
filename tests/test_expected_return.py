"""Tests for scenario CAGR ranges (not yearly return ladders)."""

from datetime import date

import pytest

from stockbot.expected_return import (
    compute_expected_return,
    format_expected_return_telegram,
    merge_expected_return_into_verdict_json,
    report_contains_yearly_return_ladder,
    resolve_display_mode,
)
from stockbot.llm.verdict import (
    ExpectedReturnInputs,
    FiveYearBusinessTest,
    ValuationComputed,
    ValuationInputs,
    VerdictJSON,
)
from stockbot.validate import validate_report
from stockbot.models import Brief, PriceData, ReportText, Technicals, TickerInfo
from datetime import UTC, datetime

import pandas as pd

NOW = datetime.now(UTC)


def _mazdock_verdict(*, buy_allowed: bool = False) -> VerdictJSON:
    return VerdictJSON(
        verdict="WATCH",
        current_price_abs=2625.0,
        price_date=date(2026, 8, 26),
        buy_zone_abs=None,
        valuation_inputs=ValuationInputs(
            eps_bear=55.0,
            eps_base=76.0,
            eps_bull=88.0,
            multiple_bear=(16.0, 20.0),
            multiple_base=(30.0, 34.0),
            multiple_bull=(36.0, 40.0),
        ),
        confidence=5,
        risk="MEDIUM",
        business_quality=7,
        financial_health=6,
        management_quality=6,
        earnings_quality="MEDIUM",
        holding_period="6-12 months (monitoring)",
        reasons_buy=["growth"],
        reasons_avoid=["weak CFO"],
        biggest_watch="CFO recovery",
        missing_data_impact="order book missing",
        gates_failed=["wc_gap_classification_inconclusive"],
        five_year_business_test=FiveYearBusinessTest(
            answer="UNCERTAIN",
            confidence="MEDIUM",
            evidence_for=[],
            evidence_against=[],
        ),
        buy_range_allowed=buy_allowed,
        add_range_allowed=False,
        thesis_status="THESIS_UNDER_REVIEW",
        wc_gap_classification="INCONCLUSIVE",
        expected_return=ExpectedReturnInputs(
            horizon_years=3,
            assumptions=[
                "Revenue CAGR ~10–15% if orders execute",
                "P/E mid-30s base, compression in bear",
            ],
            confidence="MEDIUM",
            note="Probabilistic ranges — not guaranteed yearly returns.",
        ),
    )


def _valuation() -> ValuationComputed:
    return ValuationComputed(
        fair_value_bear_abs=(880.0, 1100.0),
        fair_value_base_abs=(2280.0, 2584.0),
        fair_value_bull_abs=(3168.0, 3520.0),
    )


def test_compute_expected_return_cagr_from_fair_values():
    computed = compute_expected_return(_mazdock_verdict(), _valuation())
    assert computed.horizon_years == 3
    assert computed.bear_cagr_range_pct[1] < 0
    assert computed.base_cagr_range_pct[0] < computed.bull_cagr_range_pct[1]
    assert computed.display_mode == "EDUCATIONAL_ONLY"


def test_mazdock_style_blocks_scenario_actionable_mode():
    assert resolve_display_mode(_mazdock_verdict()) == "EDUCATIONAL_ONLY"


def test_buy_allowed_and_five_year_yes_allows_scenario_ranges():
    verdict = _mazdock_verdict(buy_allowed=True)
    verdict = verdict.model_copy(
        update={
            "five_year_business_test": FiveYearBusinessTest(
                answer="YES",
                confidence="HIGH",
                evidence_for=["moat"],
                evidence_against=[],
            ),
            "wc_gap_classification": "TEMPORARY_BILLING_CYCLE",
            "thesis_status": "THESIS_CONFIRMING",
            "anti_chase_flag": False,
        }
    )
    assert resolve_display_mode(verdict) == "SCENARIO_RANGES"


def test_merge_into_verdict_json():
    verdict = _mazdock_verdict()
    valuation = _valuation()
    payload = {
        **verdict.model_dump(mode="json"),
        **valuation.model_dump(mode="json"),
    }
    merged = merge_expected_return_into_verdict_json(payload)
    er = merged["expected_return"]
    assert "bear_cagr_range_pct" in er
    assert er["display_mode"] == "EDUCATIONAL_ONLY"
    assert er["assumptions"]


def test_telegram_format_educational_disclaimer():
    lines = format_expected_return_telegram(
        {
            "horizon_years": 3,
            "bear_cagr_range_pct": [-12.0, -2.0],
            "base_cagr_range_pct": [4.0, 9.0],
            "bull_cagr_range_pct": [12.0, 18.0],
            "display_mode": "EDUCATIONAL_ONLY",
            "note": "Not guaranteed.",
        }
    )
    assert any("Expected 3y CAGR" in line for line in lines)
    assert any("Educational scenario ranges only" in line for line in lines)


def test_yearly_ladder_detection():
    assert report_contains_yearly_return_ladder("Year 1 = 12% then year 2 = 14%")
    assert not report_contains_yearly_return_ladder("Base-case 3y CAGR 8–12%")


def test_validate_rejects_yearly_ladder_in_prose():
    brief = Brief(
        ticker=TickerInfo("T", "NSE", "Test", None),
        price=PriceData(100.0, date(2026, 8, 1), pd.DataFrame(), pd.DataFrame(), 110.0, 90.0, "y", NOW),
        technicals=Technicals(None, None, None, [], [], date(2026, 8, 1), "c", NOW),
        financials=None,
        shareholding=None,
        news=None,
        annual_report=ReportText({}, None, None, False, [], "nse", NOW),
        missing=[],
        token_count=1,
        confidence_ceiling=10,
        generated_at=NOW,
    )
    report = (
        "Year 1 = 12% expected.\n\n**SHOULD I BUY?**\nNo.\n\n"
        '```json\n{"verdict":"WATCH","current_price_abs":100,"price_date":"2026-08-26",'
        '"buy_zone_abs":null,'
        '"valuation_inputs":{"eps_bear":10,"eps_base":12,"eps_bull":14,'
        '"multiple_bear":[10,12],"multiple_base":[14,16],"multiple_bull":[18,20]},'
        '"confidence":3,"risk":"LOW","business_quality":5,"financial_health":5,'
        '"management_quality":5,"earnings_quality":"MEDIUM","holding_period":"3y",'
        '"reasons_buy":["a"],"reasons_avoid":["b"],"biggest_watch":"c",'
        '"missing_data_impact":"none","gates_failed":[]}\n```\n'
        "Research and education, not investment advice.\n"
    )
    result = validate_report(report, brief)
    assert not result.passed
    assert any("no_yearly_return_ladder" in f for f in result.failures)
