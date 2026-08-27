"""Module 10 (validation) unit tests. Pure Python, no LLM, no network —
every check is hand-crafted against fixture verdict_json blocks: one
fully-passing case, and one case failing each individual check."""

import json
from datetime import UTC, date, datetime

import pandas as pd

from stockbot.models import (
    Brief,
    Financials,
    PriceData,
    ReportText,
    Shareholding,
    Technicals,
    TickerInfo,
)
from stockbot.validate import validate_report

NOW = datetime.now(UTC)
TODAY = datetime.now(UTC).date()

BASE_VERDICT = {
    "verdict": "WATCH",
    "current_price_abs": 400.0,
    "price_date": TODAY.isoformat(),
    "buy_zone_abs": [370.0, 380.0],  # ~10.6-12.9% below fv_mid 425 -> inside LOW band (10-15%)
    "valuation_inputs": {
        # eps_base * multiple_base -> [400, 450], midpoint 425 (was a
        # direct fair_value_abs field pre-v3; now Python-computed from
        # these, see llm.verdict.compute_valuation)
        "eps_base": 25.0,
        "multiple_base": [16.0, 18.0],
        # eps_bear * multiple_bear -> [300, 340]
        "eps_bear": 20.0,
        "multiple_bear": [15.0, 17.0],
        # eps_bull * multiple_bull -> [540, 600]
        "eps_bull": 30.0,
        "multiple_bull": [18.0, 20.0],
    },
    "confidence": 6,
    "risk": "LOW",
    "business_quality": 7,
    "financial_health": 7,
    "management_quality": 7,
    "earnings_quality": "HIGH",
    "holding_period": "3-5 years",
    "reasons_buy": ["a"],
    "reasons_avoid": ["b"],
    "biggest_watch": "c",
    "missing_data_impact": "no meaningful impact",
    "gates_failed": [],
}


def _report(overrides: dict | None = None, prose: str = "") -> str:
    verdict = {**BASE_VERDICT, **(overrides or {})}
    return f"{prose}\n\n## FINAL BEGINNER SUMMARY\n\n```json\n{json.dumps(verdict)}\n```\n"


def _brief(
    *,
    shareholding: Shareholding | None = "default",
    financials: Financials | None = None,
    missing: list[str] | None = None,
    confidence_ceiling: int = 10,
) -> Brief:
    df = pd.DataFrame({"Close": [100.0]})
    if shareholding == "default":
        shareholding = Shareholding(50.0, 20.0, None, None, "Q1", "NSE", NOW)  # pledge confirmed
    return Brief(
        ticker=TickerInfo(symbol="TEST", exchange="NSE", company_name="Test Co Limited", isin=None),
        price=PriceData(100.0, TODAY, df, df, 120.0, 80.0, "yfinance", NOW),
        technicals=Technicals(95.0, 90.0, 55.0, [85.0], [110.0], TODAY, "computed", NOW),
        financials=financials,
        shareholding=shareholding,
        news=None,
        annual_report=ReportText({}, None, None, False, [], "nse_annual_reports", NOW),
        missing=missing or [],
        token_count=0,
        confidence_ceiling=confidence_ceiling,
        generated_at=NOW,
    )


def test_clean_report_passes_all_checks():
    result = validate_report(_report(), _brief())
    assert result.passed is True
    assert result.failures == []


def test_fails_on_unparseable_json():
    result = validate_report("no json block anywhere in this text", _brief())
    assert result.passed is False
    assert any("verdict_json_parse" in f for f in result.failures)


def test_fails_when_confidence_exceeds_cap():
    result = validate_report(_report({"confidence": 8}), _brief())
    assert result.passed is False
    assert any("confidence_cap" in f for f in result.failures)


def test_fails_when_confidence_exceeds_brief_ceiling():
    # Financials-failed briefs set ceiling 4; pipeline must not allow 6–7.
    result = validate_report(_report({"confidence": 6}), _brief(confidence_ceiling=4))
    assert result.passed is False
    assert any("confidence_cap" in f and "cap=4" in f for f in result.failures)


def test_passes_when_confidence_at_brief_ceiling():
    result = validate_report(_report({"confidence": 4}), _brief(confidence_ceiling=4))
    assert result.passed is True


def test_fails_when_buy_verdict_violates_quality_gate():
    result = validate_report(
        _report({"verdict": "BUY", "management_quality": 5}), _brief()
    )
    assert result.passed is False
    assert any("verdict_gate" in f for f in result.failures)


def test_passes_when_buy_verdict_meets_quality_gate():
    result = validate_report(
        _report(
            {
                "verdict": "BUY",
                "business_quality": 8,
                "management_quality": 8,
                "earnings_quality": "HIGH",
            }
        ),
        _brief(),
    )
    assert result.passed is True


def test_fails_when_ranges_out_of_order():
    result = validate_report(_report({"buy_zone_abs": [360.0, 340.0]}), _brief())
    assert result.passed is False
    assert any("ranges_ordered" in f for f in result.failures)


def test_fails_when_buy_zone_discount_too_shallow_for_risk():
    # LOW risk should be 10-15% below fair value midpoint (425); this is ~2%
    result = validate_report(
        _report({"risk": "LOW", "buy_zone_abs": [415.0, 420.0]}),
        _brief(),
    )
    assert result.passed is False
    assert any("buy_zone_discount" in f for f in result.failures)


def test_fails_when_buy_zone_discount_too_deep_for_risk():
    # LOW risk band is 10-15%; 50%+ below midpoint (425) is way beyond it
    result = validate_report(
        _report({"risk": "LOW", "buy_zone_abs": [200.0, 210.0]}),
        _brief(),
    )
    assert result.passed is False
    assert any("buy_zone_discount" in f for f in result.failures)


def test_fails_medium_risk_buy_zone_just_short_of_band_floor():
    # Regression for a real KPITTECH report: MEDIUM band floor is 20%, and a
    # buy-zone top only 18% below fv_mid (425) — 2 points short — passed
    # under the old 0.03 (3-point) DISCOUNT_TOLERANCE. Tightened to 0.01 so
    # a miss this size is caught rather than waved through.
    result = validate_report(
        _report({"risk": "MEDIUM", "buy_zone_abs": [330.0, 348.5]}),
        _brief(),
    )
    assert result.passed is False
    assert any("buy_zone_discount" in f for f in result.failures)


def test_passes_high_risk_deep_discount():
    result = validate_report(
        _report({"risk": "HIGH", "buy_zone_abs": [200.0, 250.0]}),
        _brief(),
    )
    assert result.passed is True


def test_fails_when_price_date_stale():
    stale_date = date(2020, 1, 1)
    result = validate_report(_report({"price_date": stale_date.isoformat()}), _brief())
    assert result.passed is False
    assert any("price_date_fresh" in f for f in result.failures)


def test_fails_when_pledge_stated_but_never_confirmed():
    unconfirmed_shareholding = Shareholding(50.0, None, None, None, "Q1", "NSE", NOW)
    report = _report(prose="Promoter pledge stands at 12.5% currently.")
    result = validate_report(report, _brief(shareholding=unconfirmed_shareholding))
    assert result.passed is False
    assert any("pledge_not_invented" in f for f in result.failures)


def test_passes_when_pledge_confirmed_and_stated():
    confirmed_shareholding = Shareholding(50.0, 20.0, None, None, "Q1", "NSE", NOW)
    report = _report(prose="Promoter pledge stands at 20.0% currently.")
    result = validate_report(report, _brief(shareholding=confirmed_shareholding))
    assert result.passed is True


def test_passes_when_pledge_unconfirmed_and_not_mentioned():
    unconfirmed_shareholding = Shareholding(50.0, None, None, None, "Q1", "NSE", NOW)
    result = validate_report(_report(), _brief(shareholding=unconfirmed_shareholding))
    assert result.passed is True


def test_passes_when_pledge_only_appears_as_hypothetical_threshold():
    # Regression test: a real Sonnet 5 report on JYOTHYLAB (pledge genuinely
    # unconfirmed) correctly wrote "unconfirmed" everywhere, but this check
    # still failed because master-prompt section 16 ("What would change the
    # verdict") requires stating a threshold like "pledge confirmed above
    # 40%" — a hypothetical trigger, not a claim about the current pledge.
    unconfirmed_shareholding = Shareholding(50.0, None, None, None, "Q1", "NSE", NOW)
    prose = (
        "Pledge status: unconfirmed — cannot be stated as zero or otherwise.\n\n"
        "# 16. WHAT WOULD CHANGE THE VERDICT?\n\n"
        "- Downgrade to SKIP: promoter pledge confirmed above 40%.\n\n"
        "# SHOULD I BUY?\n\nWatch, don't buy yet."
    )
    result = validate_report(_report(prose=prose), _brief(shareholding=unconfirmed_shareholding))
    assert result.passed is True


def test_fails_when_pledge_stated_outside_hypothetical_section():
    # The exclusion must not swallow a real, present-tense invented claim
    # that happens to sit elsewhere in the report.
    unconfirmed_shareholding = Shareholding(50.0, None, None, None, "Q1", "NSE", NOW)
    prose = (
        "Promoter pledge stands at 12.5% currently.\n\n"
        "# 16. WHAT WOULD CHANGE THE VERDICT?\n\n"
        "- Downgrade to SKIP: promoter pledge confirmed above 40%.\n"
    )
    result = validate_report(_report(prose=prose), _brief(shareholding=unconfirmed_shareholding))
    assert result.passed is False
    assert any("pledge_not_invented" in f for f in result.failures)


def _standalone_financials() -> Financials:
    df = pd.DataFrame({"Mar 2025": [100.0]}, index=["Sales"])
    return Financials(df, df, df, df, df, "standalone", 1, "screener:standalone", NOW)


def test_fails_when_standalone_basis_not_disclosed():
    result = validate_report(_report(), _brief(financials=_standalone_financials()))
    assert result.passed is False
    assert any("standalone_disclosed" in f for f in result.failures)


def test_passes_when_standalone_basis_disclosed():
    report = _report(prose="Note: figures below are on a STANDALONE basis.")
    result = validate_report(report, _brief(financials=_standalone_financials()))
    assert result.passed is True


def test_passes_when_financials_are_consolidated_no_disclosure_needed():
    consolidated = Financials(
        pd.DataFrame({"Mar 2025": [1.0]}, index=["Sales"]),
        pd.DataFrame({"Mar 2025": [1.0]}, index=["Total Assets"]),
        pd.DataFrame({"Mar 2025": [1.0]}, index=["Net Cash Flow"]),
        pd.DataFrame({"Mar 2025": [1.0]}, index=["ROCE %"]),
        pd.DataFrame({"Mar 2025": [1.0]}, index=["Sales"]),
        "consolidated",
        1,
        "screener:consolidated",
        NOW,
    )
    result = validate_report(_report(), _brief(financials=consolidated))
    assert result.passed is True


def test_fails_when_report_states_wrong_rsi():
    # Hand-written broken report, ₹0 to test: the fixture's real RSI is
    # 55.0 (see _brief()'s Technicals), the report claims 72 instead —
    # exactly the "model recomputed/guessed instead of using [FACT]"
    # failure mode hard injection #1 exists to prevent.
    report = _report(prose="Momentum looks strong with RSI at 72, well above neutral.")
    result = validate_report(report, _brief())
    assert result.passed is False
    assert any("technical_figures_not_recomputed" in f for f in result.failures)


def test_passes_when_report_states_correct_rsi():
    report = _report(prose="RSI currently sits at 55.0, a neutral reading.")
    result = validate_report(report, _brief())
    assert result.passed is True


def test_fails_when_report_states_wrong_sma():
    # fixture's real SMA50 is 95.0
    report = _report(prose="The stock trades above its 50 DMA of 150, a bullish signal.")
    result = validate_report(report, _brief())
    assert result.passed is False
    assert any("technical_figures_not_recomputed" in f for f in result.failures)


def test_passes_when_rsi_stated_via_placeholder_token():
    # Regression test: a real Sonnet 5 v3 report correctly wrote
    # "RSI14 of {{rsi14}}" (using the mandated placeholder token exactly as
    # instructed), but the check ran on the raw unrendered text — and the
    # token's own NAME contains digits ("rsi14" -> "14"), so the regex
    # matched that "14" as if it were a stated value, flagging a correctly
    # token-using report as wrong. Placeholder tokens must be stripped
    # before scanning for literal figures.
    report = _report(prose="Momentum is roughly flat. RSI14 of {{rsi14}} is near the neutral midpoint.")
    result = validate_report(report, _brief())
    assert result.passed is True


def test_passes_when_sma_stated_via_placeholder_token():
    # Same collision for SMA: {{sma50}} contains "50", {{sma200}} contains "200".
    report = _report(prose="Price sits above its 50-day average of {{sma50}} and above {{sma200}}.")
    result = validate_report(report, _brief())
    assert result.passed is True


def test_passes_when_rsi_correctly_restated_with_period_label():
    # Regression test: a real Sonnet 5 report wrote "RSI(14): 59.82" (the
    # real computed value, fixture RSI=55.0 here for a different figure —
    # see below), but the old regex captured "14" from the label itself as
    # if it were the reading, flagging a mismatch that didn't exist.
    report = _report(prose="RSI(14): 55.0 — neutral, no overbought/oversold extreme.")
    result = validate_report(report, _brief())
    assert result.passed is True


def test_passes_when_rsi_correctly_restated_despite_earlier_substring_collision():
    # Regression test: a real Opus 5 report on BEL correctly wrote
    # "RSI(14) 54.71", but earlier in the same report the word "reversion"
    # (contains "rsi" as a bare substring, no word boundary) matched first
    # via .search(), capturing an unrelated number from "24-26%" nearby and
    # flagging a mismatch that didn't exist — burning one real, wasted
    # Opus validation retry live.
    report = _report(
        prose=(
            "Mean reversion toward 24-26% would cut earnings sharply if it occurs. "
            "RSI(14) 54.71 — neutral, neither overbought nor oversold."
        )
    )
    result = validate_report(report, _brief())
    assert result.passed is True


def test_fails_when_rsi_wrong_even_with_period_label():
    report = _report(prose="RSI(14): 72.0 — overbought.")
    result = validate_report(report, _brief())
    assert result.passed is False
    assert any("technical_figures_not_recomputed" in f for f in result.failures)


def test_passes_when_technical_figures_not_mentioned_at_all():
    report = _report(prose="No technical commentary in this report.")
    result = validate_report(report, _brief())
    assert result.passed is True


def test_fails_when_confidence_too_high_for_missing_data():
    # Regression test: the real BEL run had 7 MISSING items — including the
    # order book, which the report itself called "the single most
    # important number for this company" — and still claimed confidence 7,
    # the pipeline maximum. v3's own text says every MISSING item should
    # visibly lower confidence; this makes that non-optional.
    seven_missing = [f"item {i}" for i in range(7)]
    result = validate_report(_report({"confidence": 7}), _brief(missing=seven_missing))
    assert result.passed is False
    assert any("confidence_vs_missing_data" in f for f in result.failures)


def test_passes_when_confidence_low_enough_for_missing_data():
    seven_missing = [f"item {i}" for i in range(7)]
    result = validate_report(_report({"confidence": 4}), _brief(missing=seven_missing))
    assert result.passed is True


def test_confidence_cap_of_5_between_five_and_six_missing_items():
    five_missing = [f"item {i}" for i in range(5)]
    result = validate_report(_report({"confidence": 6}), _brief(missing=five_missing))
    assert result.passed is False
    assert any("confidence_vs_missing_data" in f for f in result.failures)

    result = validate_report(_report({"confidence": 5}), _brief(missing=five_missing))
    assert result.passed is True


def test_confidence_vs_missing_data_not_triggered_below_threshold():
    four_missing = [f"item {i}" for i in range(4)]
    result = validate_report(_report({"confidence": 7}), _brief(missing=four_missing))
    assert result.passed is True


def _financials_with_eps(ttm_eps: float, fy_eps: float = 0.0) -> Financials:
    pnl = pd.DataFrame(
        {"Mar 2025": [fy_eps], "TTM": [ttm_eps]},
        index=["EPS in Rs"],
    )
    return Financials(pnl, pnl, pnl, pnl, pnl, "consolidated", 5, "screener:consolidated", NOW)


def test_fails_when_bear_eps_exceeds_trailing_without_justification():
    # Regression test: a real Sonnet 5 report on VMM (a real, recently-
    # listed retailer) stated a bear-case EPS of Rs2.00 against a real
    # trailing EPS of Rs1.91 -- +11% growth mislabeled as the bear case, on
    # a 58x stock. Reconstructed here in the v3 schema (the real fixture
    # predates the schema migration) using the actual verified numbers.
    overrides = {
        "current_price_abs": 111.17,
        "valuation_inputs": {
            "eps_bear": 2.00, "multiple_bear": [30.0, 35.0],
            "eps_base": 2.20, "multiple_base": [45.0, 50.0],
            "eps_bull": 2.35, "multiple_bull": [55.0, 60.0],
        },
    }
    report = _report(overrides)
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert result.passed is False
    assert any("bear_eps_sanity" in f for f in result.failures)


def test_passes_when_bear_eps_exceeds_trailing_with_justification():
    overrides = {
        "current_price_abs": 111.17,
        "valuation_inputs": {
            "eps_bear": 2.00, "multiple_bear": [30.0, 35.0],
            "eps_base": 2.20, "multiple_base": [45.0, 50.0],
            "eps_bull": 2.35, "multiple_bull": [55.0, 60.0],
        },
        "bear_growth_justification": (
            "Order book already contracted through FY27 covers this revenue with no new bookings assumed"
        ),
    }
    report = _report(overrides)
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert not any("bear_eps_sanity" in f for f in result.failures)


def test_passes_when_bear_eps_at_or_below_trailing():
    overrides = {
        "current_price_abs": 111.17,
        "valuation_inputs": {
            "eps_bear": 1.50, "multiple_bear": [30.0, 35.0],
            "eps_base": 2.20, "multiple_base": [45.0, 50.0],
            "eps_bull": 2.35, "multiple_bull": [55.0, 60.0],
        },
    }
    report = _report(overrides)
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert not any("bear_eps_sanity" in f for f in result.failures)


def test_fails_when_bear_case_insufficiently_adverse_for_high_multiple_stock():
    # Same real VMM numbers: current_price 111.17 / trailing EPS 1.91 =
    # ~58x, comfortably above the 40x threshold. Bear fair value (30-35x *
    # Rs2.00 = Rs60-70, mid Rs65) gives downside of only ~41% -- actually
    # sufficient. Use a shallower bear case to prove the check fires when
    # it should: bear mid Rs90 against price 111.17 is only ~19% downside.
    overrides = {
        "current_price_abs": 111.17,
        "valuation_inputs": {
            "eps_bear": 1.80, "multiple_bear": [48.0, 52.0],  # mid ~90
            "eps_base": 2.20, "multiple_base": [45.0, 50.0],
            "eps_bull": 2.35, "multiple_bull": [55.0, 60.0],
        },
    }
    report = _report(overrides)
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert result.passed is False
    assert any("bear_adequacy_high_multiple" in f for f in result.failures)


def test_passes_when_bear_case_adequately_adverse_for_high_multiple_stock():
    # Real VMM numbers as actually reported: 30-35x * Rs2.00 -> mid Rs65,
    # ~41.5% downside from Rs111.17 -- clears the 30% bar.
    overrides = {
        "current_price_abs": 111.17,
        "valuation_inputs": {
            "eps_bear": 2.00, "multiple_bear": [30.0, 35.0],
            "eps_base": 2.20, "multiple_base": [45.0, 50.0],
            "eps_bull": 2.35, "multiple_bull": [55.0, 60.0],
        },
        "bear_growth_justification": "placeholder so bear_eps_sanity doesn't also fire in this check-isolation test",
    }
    report = _report(overrides)
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert not any("bear_adequacy_high_multiple" in f for f in result.failures)


def test_bear_checks_not_applicable_below_40x_multiple():
    overrides = {
        "current_price_abs": 50.0,  # 50/1.91 ~ 26x, below the 40x threshold
        "valuation_inputs": {
            "eps_bear": 2.00, "multiple_bear": [30.0, 35.0],  # exceeds trailing, but PE too low to trigger adequacy check
            "eps_base": 2.20, "multiple_base": [45.0, 50.0],
            "eps_bull": 2.35, "multiple_bull": [55.0, 60.0],
        },
    }
    report = _report(overrides)
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert not any("bear_adequacy_high_multiple" in f for f in result.failures)


def test_bear_checks_not_applicable_when_financials_missing():
    result = validate_report(_report(), _brief(financials=None))
    assert not any("bear_eps_sanity" in f for f in result.failures)
    assert not any("bear_adequacy_high_multiple" in f for f in result.failures)


def test_format_validation_errors_lists_all_failures():
    from stockbot.validate import format_validation_errors

    result = validate_report(_report({"confidence": 9}), _brief())
    formatted = format_validation_errors(result)
    assert "confidence_cap" in formatted
    assert formatted.startswith("The previous attempt failed")
