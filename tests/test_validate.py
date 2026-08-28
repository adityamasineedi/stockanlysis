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
    # Nested valuation_inputs: shallow merge replaces the whole dict when
    # overrides supplies it (existing call sites pass a complete block).
    if "valuation_inputs" in (overrides or {}) and overrides is not None:
        verdict["valuation_inputs"] = overrides["valuation_inputs"]
    return (
        f"{prose}\n\n"
        f"**SHOULD I BUY?**\n"
        f"- **Decision:** {verdict['verdict']}\n"
        f"- **Current Price:** {{{{current_price}}}} ({{{{price_date}}}})\n"
        f"- **Buy Zone:** {{{{buy_zone_low}}}}–{{{{buy_zone_high}}}}\n"
        f"- **Fair Value:** {{{{fair_value_base_low}}}}–{{{{fair_value_base_high}}}}\n\n"
        f"```json\n{json.dumps(verdict)}\n```\n\n"
        f"*Research and education, not investment advice. Verify the numbers before "
        f"acting, and consider a SEBI-registered investment adviser.*\n"
    )


BEAR_DOWNSIDE_PASS_PROSE = (
    "### 11. VALUATION\n"
    "Bear downside check: {{downside_pct}} vs 30% floor for >40x trailing — PASS — "
    "bear multiple compressed with EPS at/below TTM.\n"
)

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


def test_five_year_gate_blocks_buy_zone_when_answer_not_yes():
    result = validate_report(
        _report(
            {
                "verdict": "WATCH",
                "buy_zone_abs": [370.0, 380.0],
                "buy_range_allowed": True,
                "five_year_business_test": {
                    "answer": "NO",
                    "confidence": "MEDIUM",
                    "evidence_for": [],
                    "evidence_against": ["no path to stronger business"],
                },
            }
        ),
        _brief(),
    )
    assert result.passed is False
    assert any("five_year_buy_gate" in f for f in result.failures)


def test_five_year_gate_allows_null_zone_when_uncertain():
    result = validate_report(
        _report(
            {
                "verdict": "WATCH",
                "buy_zone_abs": None,
                "buy_range_allowed": False,
                "add_range_allowed": False,
                "five_year_business_test": {
                    "answer": "UNCERTAIN",
                    "confidence": "LOW",
                    "evidence_for": [],
                    "evidence_against": [],
                },
            }
        ),
        _brief(),
    )
    assert result.passed is True


def _extreme_cash_financials() -> Financials:
    # 3y ΣOCF=10 / ΣPAT=510 → ~0.02 (Mazdock-style escalated weakness)
    empty = pd.DataFrame()
    pnl = pd.DataFrame(
        {"FY24": [160.0], "FY25": [170.0], "FY26": [180.0]},
        index=["Net Profit"],
    )
    cash_flow = pd.DataFrame(
        {"FY24": [50.0], "FY25": [60.0], "FY26": [-100.0]},
        index=["Cash from Operating Activity"],
    )
    return Financials(
        pnl=pnl,
        balance_sheet=empty,
        cash_flow=cash_flow,
        ratios=empty,
        quarterly=empty,
        basis="consolidated",
        years_available=3,
        source="test",
        fetched_at=NOW,
    )


def test_wc_gate_blocks_buy_zone_when_cash_extreme_without_temporary_class():
    result = validate_report(
        _report(
            {
                "verdict": "WATCH",
                "buy_zone_abs": [370.0, 380.0],
                "buy_range_allowed": True,
                "five_year_business_test": {
                    "answer": "YES",
                    "confidence": "MEDIUM",
                    "evidence_for": ["growth"],
                    "evidence_against": ["cash"],
                },
                "wc_gap_classification": "INCONCLUSIVE",
            }
        ),
        _brief(financials=_extreme_cash_financials()),
    )
    assert result.passed is False
    assert any("wc_buy_gate" in f for f in result.failures)


def test_wc_gate_blocks_buy_zone_when_extreme_and_classification_missing():
    result = validate_report(
        _report(
            {
                "verdict": "WATCH",
                "buy_zone_abs": [370.0, 380.0],
                "buy_range_allowed": True,
                "five_year_business_test": {
                    "answer": "YES",
                    "confidence": "HIGH",
                    "evidence_for": ["roce"],
                    "evidence_against": [],
                },
            }
        ),
        _brief(financials=_extreme_cash_financials()),
    )
    assert result.passed is False
    assert any("wc_buy_gate" in f for f in result.failures)


def test_wc_gate_allows_buy_zone_only_for_temporary_billing_cycle():
    result = validate_report(
        _report(
            {
                "verdict": "WATCH",
                "buy_zone_abs": [370.0, 380.0],
                "buy_range_allowed": True,
                "five_year_business_test": {
                    "answer": "YES",
                    "confidence": "HIGH",
                    "evidence_for": ["order book"],
                    "evidence_against": [],
                },
                "wc_gap_classification": "TEMPORARY_BILLING_CYCLE",
            }
        ),
        _brief(financials=_extreme_cash_financials()),
    )
    assert result.passed is True


def test_anti_chase_required_when_price_at_base_fv_top():
    result = validate_report(
        _report(
            {
                "current_price_abs": 400.0,
                "anti_chase_flag": False,
                "earnings_quality": "MEDIUM",
                "buy_zone_abs": None,
                "buy_range_allowed": False,
                "valuation_inputs": {
                    "eps_bear": 8.0,
                    "eps_base": 10.0,
                    "eps_bull": 12.0,
                    "multiple_bear": [30.0, 32.0],
                    "multiple_base": [38.0, 40.0],
                    "multiple_bull": [42.0, 45.0],
                },
            }
        ),
        _brief(financials=_extreme_cash_financials()),
    )
    assert result.passed is False
    assert any("anti_chase_flag" in f for f in result.failures)


def test_holding_period_fails_when_thesis_under_review_but_3_5_years():
    result = validate_report(
        _report(
            {
                "thesis_status": "THESIS_UNDER_REVIEW",
                "holding_period": "3-5 years",
                "buy_range_allowed": False,
                "buy_zone_abs": None,
            }
        ),
        _brief(),
    )
    assert result.passed is False
    assert any("holding_period_vs_thesis" in f for f in result.failures)


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
    report = _report(overrides, prose=BEAR_DOWNSIDE_PASS_PROSE)
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert not any("bear_eps_sanity" in f for f in result.failures)
    assert not any("bear_downside_check_prose" in f for f in result.failures)


def test_passes_when_bear_eps_at_or_below_trailing():
    overrides = {
        "current_price_abs": 111.17,
        "valuation_inputs": {
            "eps_bear": 1.50, "multiple_bear": [30.0, 35.0],
            "eps_base": 2.20, "multiple_base": [45.0, 50.0],
            "eps_bull": 2.35, "multiple_bull": [55.0, 60.0],
        },
    }
    report = _report(overrides, prose=BEAR_DOWNSIDE_PASS_PROSE)
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
    report = _report(overrides, prose=BEAR_DOWNSIDE_PASS_PROSE)
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
    report = _report(overrides, prose=BEAR_DOWNSIDE_PASS_PROSE)
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert not any("bear_adequacy_high_multiple" in f for f in result.failures)
    assert not any("bear_downside_check_prose" in f for f in result.failures)

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


def test_fails_when_confidence_written_as_over_seven():
    # KPITTECH live bug: prose said "Confidence: 5/7" while scale is always /10.
    prose = "1. QUICK VERDICT\nWATCH\nRisk: MEDIUM · Confidence: 5/7\n"
    result = validate_report(_report(prose=prose), _brief())
    assert result.passed is False
    assert any("confidence_scale_over_ten" in f for f in result.failures)


def test_fails_when_rupees_wrapped_in_backticks():
    # KPITTECH live bug: `₹400.00`–`₹430.00` from backtick-wrapped tokens.
    prose = "Buy Zone: `₹370.00`–`₹380.00`\n"
    result = validate_report(_report(prose=prose), _brief())
    assert result.passed is False
    assert any("no_backtick_wrapped_rupees" in f for f in result.failures)


def test_fails_when_headline_fair_value_spans_bear_to_bull():
    # BASE from fixture: 25*[16,18] = 400–450; BEAR 300–340; BULL 540–600.
    # Headline must show BASE, not bear-low–bull-high.
    prose = "Fair Value ₹300.00–₹600.00\n"
    result = validate_report(_report(prose=prose), _brief())
    assert result.passed is False
    assert any("headline_fair_value_is_base" in f for f in result.failures)


def test_passes_when_headline_fair_value_is_base_range():
    prose = "Fair Value ₹400.00–₹450.00\n"
    result = validate_report(_report(prose=prose), _brief())
    assert not any("headline_fair_value_is_base" in f for f in result.failures)


# --- Deployment checklist (master prompt v3) ---------------------------------


def test_deployment_fails_on_invalid_citation_id():
    prose = "Revenue grew 18% [FINANCIALS]. Also saw something [NEWS].\n"
    result = validate_report(_report(prose=prose), _brief())
    assert result.passed is False
    assert any("citation_ids_valid" in f and "NEWS" in f for f in result.failures)


def test_deployment_passes_on_valid_citations_and_evidence_labels():
    prose = (
        "Revenue grew 18% [FINANCIALS]. RSI is {{rsi14}} [PRICE_AND_TECHNICALS]. "
        "Gap noted [MISSING]. This is [FACT] and [ANALYSIS].\n"
    )
    result = validate_report(_report(prose=prose), _brief())
    assert not any("citation_ids_valid" in f for f in result.failures)


def test_deployment_fails_on_unknown_placeholder_token():
    prose = "Price is {{current_price}} but also {{made_up_token}}.\n"
    result = validate_report(_report(prose=prose), _brief())
    assert result.passed is False
    assert any("placeholder_tokens_known" in f and "made_up_token" in f for f in result.failures)


def test_deployment_fails_when_high_pe_missing_bear_downside_check_line():
    overrides = {
        "current_price_abs": 111.17,
        "valuation_inputs": {
            "eps_bear": 1.50,
            "multiple_bear": [30.0, 35.0],
            "eps_base": 2.20,
            "multiple_base": [45.0, 50.0],
            "eps_bull": 2.35,
            "multiple_bull": [55.0, 60.0],
        },
    }
    report = _report(overrides, prose="### 11. VALUATION\nNo sanity check line here.\n")
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert result.passed is False
    assert any("bear_downside_check_prose" in f for f in result.failures)


def test_deployment_fails_when_bear_downside_check_left_on_fail():
    overrides = {
        "current_price_abs": 111.17,
        "valuation_inputs": {
            "eps_bear": 1.50,
            "multiple_bear": [30.0, 35.0],
            "eps_base": 2.20,
            "multiple_base": [45.0, 50.0],
            "eps_bull": 2.35,
            "multiple_bull": [55.0, 60.0],
        },
    }
    prose = (
        "### 11. VALUATION\n"
        "Bear downside check: {{downside_pct}} vs 30% floor for >40x trailing — FAIL — "
        "downside only 22% with no contracted growth evidence.\n"
    )
    report = _report(overrides, prose=prose)
    brief = _brief(financials=_financials_with_eps(ttm_eps=1.91, fy_eps=1.80))
    result = validate_report(report, brief)
    assert result.passed is False
    assert any("bear_downside_check_prose" in f and "FAIL" in f for f in result.failures)


def test_deployment_fails_when_beginner_summary_after_json():
    verdict = {**BASE_VERDICT}
    bad = (
        f"### 1. QUICK VERDICT\nWATCH\n\n"
        f"```json\n{json.dumps(verdict)}\n```\n\n"
        f"**SHOULD I BUY?**\n- **Decision:** WATCH\n\n"
        f"*Research and education, not investment advice. Verify the numbers before "
        f"acting, and consider a SEBI-registered investment adviser.*\n"
    )
    result = validate_report(bad, _brief())
    assert result.passed is False
    assert any("output_order" in f and "SHOULD I BUY" in f for f in result.failures)


def test_deployment_fails_when_footer_missing():
    verdict = {**BASE_VERDICT}
    bad = (
        f"**SHOULD I BUY?**\n- **Decision:** WATCH\n\n"
        f"```json\n{json.dumps(verdict)}\n```\n"
    )
    result = validate_report(bad, _brief())
    assert result.passed is False
    assert any("output_order" in f and "footer" in f.lower() for f in result.failures)


def test_deployment_empty_context_rejects_buy_and_high_confidence():
    # Skeletal brief: no financials, no shareholding — checklist item 1.
    brief = _brief(
        shareholding=None,
        financials=None,
        missing=[
            "MISSING: financials",
            "MISSING: shareholding",
            "MISSING: business description",
            "MISSING: extraction",
            "MISSING: news",
            "MISSING: annual report",
            "MISSING: all fields",
        ],
        confidence_ceiling=2,
    )
    result = validate_report(
        _report({"verdict": "BUY", "confidence": 6, "business_quality": 8, "management_quality": 8}),
        brief,
    )
    assert result.passed is False
    assert any("empty_context_verdict" in f for f in result.failures)


def test_deployment_empty_context_allows_skip_low_confidence():
    brief = _brief(
        shareholding=None,
        financials=None,
        missing=["MISSING: all fields"] * 7,
        confidence_ceiling=2,
    )
    result = validate_report(_report({"verdict": "SKIP", "confidence": 2}), brief)
    assert not any("empty_context_verdict" in f for f in result.failures)


def test_deployment_thin_context_rejects_invented_business_model():
    prose = (
        "### 2. COMPANY IN 60 SECONDS\n"
        "Reliance Industries is a conglomerate spanning oil refining, telecom (Jio), "
        "and retail — a household Indian name with vast operations.\n\n"
        "### 3. WHY COULD THIS STOCK GO UP?\nMomentum.\n"
    )
    brief = _brief(
        financials=None,
        missing=["MISSING: business description — Screener had no About block"],
    )
    result = validate_report(_report(prose=prose), brief)
    assert result.passed is False
    assert any("thin_context_business_model" in f for f in result.failures)


def test_deployment_thin_context_passes_when_section2_cites_missing():
    prose = (
        "### 2. COMPANY IN 60 SECONDS\n"
        "Business description is not available [MISSING]. "
        "This cannot be determined from the supplied evidence.\n\n"
        "### 3. WHY COULD THIS STOCK GO UP?\nNone evidenced.\n"
    )
    brief = _brief(
        financials=None,
        missing=["MISSING: business description — Screener had no About block"],
    )
    result = validate_report(_report(prose=prose), brief)
    assert not any("thin_context_business_model" in f for f in result.failures)
