"""bot.py unit tests for the pure formatting functions — no network, no
Telegram bot token needed. Live testing (a real bot round-trip: HTML
rendering in an actual Telegram client, the ⏳ status-message edit, file
attachment delivery) is still owed once TELEGRAM_BOT_TOKEN is available."""

from copy import deepcopy
from datetime import date

from stockbot.bot import (
    _parse_analyze_command_args,
    _parse_force_analyze_plain_text,
    esc,
    format_ambiguous_reply,
    format_verdict_reply,
)
from stockbot.models import AmbiguousMatch, Analysis, TickerInfo, ValidationResult

VALID_VERDICT_JSON = {
    "verdict": "WATCH",
    "current_price_abs": 412.5,
    "price_date": "2026-08-25",
    "buy_zone_abs": [330.0, 355.0],
    "buy_range_allowed": True,
    "add_range_allowed": False,
    # fair_value_*_abs are Python-computed (compute_valuation) and merged
    # into verdict_json by pipeline.py alongside the model's raw
    # valuation_inputs — this fixture models what pipeline.py actually
    # stores, not what the model's JSON block looks like on its own.
    "valuation_inputs": {
        "eps_base": 25.0, "multiple_base": [16.0, 18.0],
        "eps_bear": 20.0, "multiple_bear": [15.0, 17.0],
        "eps_bull": 30.0, "multiple_bull": [18.0, 20.0],
    },
    "fair_value_base_abs": [420.0, 470.0],
    "fair_value_bear_abs": [300.0, 340.0],
    "fair_value_bull_abs": [540.0, 600.0],
    "confidence": 6,
    "risk": "MEDIUM",
    "business_quality": 7,
    "financial_health": 7,
    "management_quality": 6,
    "earnings_quality": "MEDIUM",
    "holding_period": "3-5 years",
    "reasons_buy": ["Strong moat", "Improving margins"],
    "reasons_avoid": ["Regulatory overhang"],
    "biggest_watch": "Q3 margin trend",
}


def _analysis(missing=None) -> Analysis:
    return Analysis(
        ticker="TEST",
        run_date=date(2026, 8, 25),
        verdict_json=deepcopy(VALID_VERDICT_JSON),
        report_md="# full report",
        costs=39.0,
        validation=ValidationResult(True, []),
        missing=missing or [],
    )


def test_esc_escapes_html_special_characters():
    assert esc("<script>&") == "&lt;script&gt;&amp;"


def test_esc_handles_non_string_input():
    assert esc(6) == "6"


def test_format_verdict_reply_contains_key_fields():
    text = format_verdict_reply(_analysis())
    assert "WATCH" in text
    assert "412.5" in text
    assert "330.00" in text and "355.00" in text
    assert "Buy range: ₹330.00–₹355.00" in text
    assert "Sell range: ₹420.00–₹470.00" in text
    assert "Take-profit targets: ₹540.00–₹600.00" in text
    assert "MEDIUM" in text


def test_format_verdict_reply_action_ranges_before_price():
    text = format_verdict_reply(_analysis())
    headline = text.find("— TEST")
    buy = text.find("Buy range:")
    sell = text.find("Sell range:")
    add_more = text.find("Add-more range:")
    take_profit = text.find("Take-profit targets:")
    price = text.find("Price:")
    assert headline < buy < sell < add_more < take_profit < price


def test_format_verdict_reply_shows_add_more_range_when_allowed():
    analysis = _analysis()
    analysis.verdict_json["add_range_allowed"] = True
    text = format_verdict_reply(analysis)
    assert "Add-more range: ₹300.00–₹330.00 (on-dip · bear FV)" in text


def test_format_verdict_reply_shows_stage2_mode():
    analysis = _analysis()
    analysis.verdict_json["stage2_mode"] = "LITE"
    text = format_verdict_reply(analysis)
    assert "Stage 2:" in text
    assert "LITE" in text
    assert "Haiku compact report" in text


def test_format_verdict_reply_shows_forced_full_override():
    analysis = _analysis()
    analysis.verdict_json["stage2_mode"] = "FULL"
    analysis.verdict_json["stage2_mode_forced"] = True
    text = format_verdict_reply(analysis)
    assert "Stage 2:" in text
    assert "FULL" in text
    assert "config override" in text


def test_format_verdict_reply_omits_stage2_mode_for_legacy_cache():
    text = format_verdict_reply(_analysis())
    assert "Stage 2:" not in text
    assert "Strong moat" in text
    assert "Regulatory overhang" in text
    assert "Q3 margin trend" in text


def test_format_verdict_reply_shows_expected_return_scenarios():
    analysis = _analysis()
    analysis.verdict_json["expected_return"] = {
        "horizon_years": 3,
        "bear_cagr_range_pct": [-10.0, 0.0],
        "base_cagr_range_pct": [8.0, 12.0],
        "bull_cagr_range_pct": [15.0, 20.0],
        "display_mode": "EDUCATIONAL_ONLY",
        "note": "Probabilistic ranges only.",
    }
    text = format_verdict_reply(analysis)
    assert "Expected 3y CAGR" in text
    assert "Educational scenario ranges only" in text
    assert "8.0%" in text or "8.0%–12.0%" in text


def test_format_verdict_reply_hides_buy_zone_when_wc_not_temporary():
    analysis = _analysis()
    analysis.verdict_json["buy_range_allowed"] = True
    analysis.verdict_json["wc_gap_classification"] = "INCONCLUSIVE"
    text = format_verdict_reply(analysis)
    assert "Buy range: not issued" in text
    assert "WC: INCONCLUSIVE" in text
    assert "330.00" not in text


def test_format_verdict_reply_shows_anti_chase_when_price_above_base_fv():
    analysis = _analysis()
    analysis.verdict_json["current_price_abs"] = 2625.0
    analysis.verdict_json["anti_chase_flag"] = False
    analysis.verdict_json["fair_value_base_abs"] = [2280.0, 2584.0]
    analysis.verdict_json["valuation_inputs"] = {
        "eps_base": 76.0, "multiple_base": [30.0, 34.0],
        "eps_bear": 55.0, "multiple_bear": [16.0, 20.0],
        "eps_bull": 88.0, "multiple_bull": [36.0, 40.0],
    }
    text = format_verdict_reply(analysis)
    assert "Anti-chase" in text


def test_card_does_not_suppress_buy_zone_on_descriptive_cash_prose():
    """Prose mentioning cash conversion must not withhold a valid buy zone.

    The card used to fall back to substring-matching report text when
    wc_gap_classification was null, so a report *praising* cash conversion
    tripped the same markers as one reporting a gap — the card announced
    "not issued (WC: RECONCILIATION_REQUIRED)" while the full report printed
    the zone. Suppression is now driven only by the structured field, so the
    two agree. The genuinely dangerous case (extreme reported cash conversion
    with no classification) is caught deterministically at generation time by
    validate._check_wc_buy_gate, which reads FINANCIALS rather than prose —
    see test_validate.test_wc_gate_blocks_buy_zone_when_extreme_and_classification_missing.
    """
    analysis = _analysis()
    analysis.verdict_json["buy_range_allowed"] = True
    analysis.verdict_json["wc_gap_classification"] = None
    analysis.verdict_json["reasons_buy"] = [
        "Cash conversion is strong — 3-year cumulative OCF/PAT of ~1.26x",
    ]
    analysis.verdict_json["five_year_business_test"] = {
        "answer": "YES",
        "confidence": "MEDIUM",
        "evidence_for": ["Cumulative ΣCFO/ΣPAT comfortably above 1.0"],
        "evidence_against": [],
    }
    text = format_verdict_reply(analysis)
    assert "330.00" in text
    assert "RECONCILIATION_REQUIRED" not in text


def test_card_and_report_agree_on_suppressed_buy_zone():
    """The card must not claim a zone the constitution gates already nulled."""
    import dataclasses

    from stockbot.constitution_gates import refresh_constitution_fields

    analysis = _analysis()
    analysis.verdict_json["buy_range_allowed"] = True
    analysis.verdict_json["wc_gap_classification"] = "WORKING_CAPITAL_STRESS"

    # What render.py reads for the full report, after the gates run.
    gated = refresh_constitution_fields(dict(analysis.verdict_json))
    assert gated["buy_zone_abs"] is None
    assert gated["buy_range_allowed"] is False

    # What the Telegram card shows for the same verdict.
    text = format_verdict_reply(dataclasses.replace(analysis, verdict_json=gated))
    assert "Buy range: not issued" in text
    assert "WC: WORKING_CAPITAL_STRESS" in text
    assert "330.00" not in text


def test_format_verdict_reply_recomputes_base_fv_when_abs_missing():
    # Older / partial verdict_json may lack fair_value_base_abs; Telegram
    # must still show base (eps_base × multiple_base), never bear→bull.
    analysis = _analysis()
    del analysis.verdict_json["fair_value_base_abs"]
    del analysis.verdict_json["fair_value_bear_abs"]
    del analysis.verdict_json["fair_value_bull_abs"]
    text = format_verdict_reply(analysis)
    # 25 × [16, 18] = 400–450
    assert "Sell range: ₹400.00–₹450.00" in text
    assert "₹300.00" not in text  # must not fall through to bear


def test_format_verdict_reply_falls_back_to_legacy_fair_value_abs():
    """BEL-style Aug-2026 cache: fair_value_abs only, no valuation_inputs."""
    analysis = _analysis()
    analysis.verdict_json.pop("fair_value_base_abs", None)
    analysis.verdict_json.pop("valuation_inputs", None)
    analysis.verdict_json["fair_value_abs"] = [330.0, 380.0]
    text = format_verdict_reply(analysis)
    assert "Sell range: ₹330.00–₹380.00" in text


def test_format_verdict_reply_rounds_display_price():
    analysis = _analysis()
    analysis.verdict_json["current_price_abs"] = 408.54998779296875
    text = format_verdict_reply(analysis)
    assert "Price: ₹408.55" in text
    assert "54998779296875" not in text


def test_format_verdict_reply_uses_html_tags_not_markdown():
    text = format_verdict_reply(_analysis())
    assert "<b>" in text
    assert "**" not in text  # not markdown bold


def test_format_verdict_reply_escapes_llm_text_content():
    analysis = _analysis()
    analysis.verdict_json["biggest_watch"] = "<script>alert(1)</script>"
    text = format_verdict_reply(analysis)
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text


def test_format_verdict_reply_appends_missing_warnings():
    text = format_verdict_reply(_analysis(missing=["MISSING: shareholding — fetch failed"]))
    assert "⚠️" in text
    assert "MISSING: shareholding" in text


def test_format_verdict_reply_no_missing_no_warning_lines():
    text = format_verdict_reply(_analysis())
    assert "MISSING:" not in text


def test_format_verdict_reply_includes_disclaimer():
    text = format_verdict_reply(_analysis())
    assert "not investment advice" in text.lower()


def test_format_verdict_reply_prepends_staleness_banner():
    text = format_verdict_reply(
        _analysis(),
        staleness_banner="Analysis from 2026-08-19 at ₹400.00. Price today: ₹405.00.",
    )
    assert text.startswith("Analysis from 2026-08-19")
    assert "WATCH" in text


def test_format_verdict_reply_stays_under_telegram_limit():
    analysis = _analysis()
    analysis.verdict_json["reasons_buy"] = ["x" * 500] * 20  # deliberately huge
    text = format_verdict_reply(analysis)
    assert len(text) <= 4096


def test_format_ambiguous_reply_lists_all_candidates():
    candidates = AmbiguousMatch(
        candidates=[
            TickerInfo("HDFCBANK", "NSE", "HDFC Bank Limited", None),
            TickerInfo("HDFCAMC", "NSE", "HDFC Asset Management Company Limited", None),
        ],
        scores=[100.0, 90.0],
    )
    text = format_ambiguous_reply(candidates)
    assert "HDFCBANK" in text
    assert "HDFCAMC" in text
    assert "<code>" in text


def test_parse_analyze_command_args():
    assert _parse_analyze_command_args(["BEL"]) == ("BEL", False)
    assert _parse_analyze_command_args(["force", "MAZDOCK"]) == ("MAZDOCK", True)
    assert _parse_analyze_command_args(["Force", "TCS"]) == ("TCS", True)
    assert _parse_analyze_command_args([]) == ("", False)


def test_parse_force_analyze_plain_text():
    assert _parse_force_analyze_plain_text("force BEL") == ("BEL", True)
    assert _parse_force_analyze_plain_text("Force MAZDOCK") == ("MAZDOCK", True)
    assert _parse_force_analyze_plain_text("BEL") is None
    assert _parse_force_analyze_plain_text("force") is None
