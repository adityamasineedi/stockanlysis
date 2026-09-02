"""bot.py unit tests for the pure formatting functions — no network, no
Telegram bot token needed. Live testing (a real bot round-trip: HTML
rendering in an actual Telegram client, the ⏳ status-message edit, file
attachment delivery) is still owed once TELEGRAM_BOT_TOKEN is available."""

from copy import deepcopy
from datetime import date

from stockbot.bot import (
    AWAITING_PRESCAN_SYMBOL,
    _consume_awaiting,
    _parse_analyze_command_args,
    _parse_analyze_plain_text,
    _parse_prescan_plain_text,
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
    assert "👀" in text
    assert "WATCH" in text
    assert "Plain English: keep on your list" in text
    assert "412.5" in text
    assert "330.00" in text and "355.00" in text
    assert "🛒 Buy range: ₹330.00–₹355.00" in text
    assert "📤 Sell range: ₹420.00–₹470.00" in text
    assert "🎯 Take-profit targets: ₹540.00–₹600.00" in text
    assert "✅ Why this looks good" in text
    assert "⛔ Why to be careful" in text
    assert "👀 Biggest thing to watch" in text
    assert "MEDIUM" in text


def test_verdict_emoji_and_plain_hint():
    from stockbot.bot import _verdict_emoji, _verdict_plain_hint

    assert _verdict_emoji("BUY") == "🟢"
    assert _verdict_emoji("BUY ON CORRECTION") == "🟡"
    assert _verdict_emoji("WATCH") == "👀"
    assert _verdict_emoji("SKIP") == "🔴"
    assert "buy range" in (_verdict_plain_hint("BUY") or "").lower()
    assert "wait" in (_verdict_plain_hint("BUY ON CORRECTION") or "").lower()
    assert "skip" in (_verdict_plain_hint("SKIP") or "").lower()


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
    assert "➕ Add-more range: ₹300.00–₹330.00 (on-dip · bear FV)" in text


def test_format_verdict_reply_compact_limits_reason_bullets():
    analysis = _analysis()
    analysis.verdict_json["reasons_buy"] = ["Reason A", "Reason B", "Reason C"]
    text = format_verdict_reply(analysis, compact=True)
    assert "Reason A" in text
    assert "Reason B" in text
    assert "Reason C" not in text


def test_format_verdict_reply_full_shows_all_reasons():
    analysis = _analysis()
    analysis.verdict_json["reasons_buy"] = ["Reason A", "Reason B", "Reason C"]
    text = format_verdict_reply(analysis, compact=False)
    assert "Reason C" in text


def test_format_verdict_reply_shows_stage2_mode():
    analysis = _analysis()
    analysis.verdict_json["stage2_mode"] = "LITE"
    text = format_verdict_reply(analysis, compact=False)
    assert "Stage 2:" in text
    assert "LITE" in text
    assert "Haiku compact report" in text


def test_format_verdict_reply_shows_forced_full_override():
    analysis = _analysis()
    analysis.verdict_json["stage2_mode"] = "FULL"
    analysis.verdict_json["stage2_mode_forced"] = True
    text = format_verdict_reply(analysis, compact=False)
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
    assert "educational only" in text.lower()


def test_format_verdict_reply_hides_buy_zone_when_wc_not_temporary():
    analysis = _analysis()
    analysis.verdict_json["buy_range_allowed"] = True
    analysis.verdict_json["wc_gap_classification"] = "INCONCLUSIVE"
    text = format_verdict_reply(analysis)
    assert "🛒 Buy range: not issued" in text
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
    assert "anti-chase" in text.lower()


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
    assert "🛒 Buy range: not issued" in text
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


def test_format_verdict_reply_fresh_analysis_footer():
    text = format_verdict_reply(_analysis(), from_cache=False)
    assert "Fresh analysis" in text
    assert "same LLM run" not in text


def test_format_verdict_reply_cached_analysis_footer():
    text = format_verdict_reply(_analysis(), from_cache=True)
    assert "Cached analysis" in text
    assert "gates refreshed" in text
    assert "same LLM run" not in text


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
    assert _parse_analyze_command_args(["BEL"]) == ("BEL", False, False, False, False)
    assert _parse_analyze_command_args(["force", "MAZDOCK"]) == ("MAZDOCK", True, False, False, False)
    assert _parse_analyze_command_args(["Force", "TCS"]) == ("TCS", True, False, False, False)
    assert _parse_analyze_command_args(["full", "CAMS"]) == ("CAMS", False, True, False, False)
    assert _parse_analyze_command_args(["CAMS", "full"]) == ("CAMS", False, True, False, False)
    assert _parse_analyze_command_args(["force", "full", "BSE"]) == ("BSE", True, True, False, False)
    assert _parse_analyze_command_args(["fresh", "BEL"]) == ("BEL", False, False, True, False)
    assert _parse_analyze_command_args(["lite", "BEL"]) == ("BEL", False, False, False, True)
    assert _parse_analyze_command_args(["lite", "fresh", "BEL"]) == ("BEL", False, False, True, True)
    assert _parse_analyze_command_args([]) == ("", False, False, False, False)


def test_parse_analyze_plain_text():
    assert _parse_analyze_plain_text("force BEL") == ("BEL", True, False, False, False)
    assert _parse_analyze_plain_text("full CAMS") == ("CAMS", False, True, False, False)
    assert _parse_analyze_plain_text("fresh BEL") == ("BEL", False, False, True, False)
    assert _parse_analyze_plain_text("lite BEL") == ("BEL", False, False, False, True)
    assert _parse_analyze_plain_text("force full MAZDOCK") == ("MAZDOCK", True, True, False, False)
    assert _parse_analyze_plain_text("BEL") is None
    assert _parse_analyze_plain_text("force") is None


def test_parse_prescan_plain_text():
    assert _parse_prescan_plain_text("prescan BEL") == "BEL"
    assert _parse_prescan_plain_text("Pre-Scan Hero MotoCorp") == "Hero MotoCorp"
    assert _parse_prescan_plain_text("prescan") is None
    assert _parse_prescan_plain_text("BEL") is None


def test_consume_awaiting_symbol():
    class _Ctx:
        user_data: dict[str, bool]

    ctx = _Ctx()
    ctx.user_data = {AWAITING_PRESCAN_SYMBOL: True}
    assert _consume_awaiting(ctx, AWAITING_PRESCAN_SYMBOL) is True
    assert AWAITING_PRESCAN_SYMBOL not in ctx.user_data
    assert _consume_awaiting(ctx, AWAITING_PRESCAN_SYMBOL) is False


def test_analysis_lock_allows_one_run_at_a_time():
    import asyncio

    from stockbot.bot import _active_analysis_query, _end_analysis, _try_begin_analysis

    async def scenario() -> None:
        await _end_analysis()
        assert await _try_begin_analysis("TCS") is True
        assert await _active_analysis_query() == "TCS"
        assert await _try_begin_analysis("INFY") is False
        await _end_analysis()
        assert await _active_analysis_query() is None
        assert await _try_begin_analysis("INFY") is True
        await _end_analysis()

    asyncio.run(scenario())


def test_buy_range_names_the_five_year_gate_like_add_more_does():
    """Live JYOTHYLAB card: add-more said "(five-year: UNCERTAIN)" while the
    buy line — blocked by that same gate — printed a bare "not issued", so the
    card could not explain its own suppression."""
    from stockbot.bot import _format_add_more_range_line, _format_buy_range_line

    verdict = {
        "buy_zone_abs": None,
        "buy_range_allowed": False,
        "add_range_allowed": False,
        "five_year_business_test": {"answer": "UNCERTAIN"},
        "thesis_status": "THESIS_UNDER_REVIEW",
    }
    assert _format_buy_range_line(verdict) == "🛒 Buy range: not issued (five-year: UNCERTAIN)"
    assert _format_add_more_range_line(verdict) == "➕ Add-more range: not issued (five-year: UNCERTAIN)"


def test_buy_range_keeps_existing_gate_wording():
    from stockbot.bot import _format_buy_range_line

    assert _format_buy_range_line({"anti_chase_flag": True}) == (
        "🛒 Buy range: not issued (anti-chase: pause new capital)"
    )
    assert _format_buy_range_line({"wc_gap_classification": "WORKING_CAPITAL_STRESS"}) == (
        "🛒 Buy range: not issued (WC: WORKING_CAPITAL_STRESS)"
    )
    # No identifiable gate still yields the bare line.
    assert _format_buy_range_line({"buy_range_allowed": False}) == "🛒 Buy range: not issued"


def test_buy_range_display_does_not_start_suppressing_on_five_year():
    """This change names the gate; it must not change what gets suppressed.
    A model-issued zone still displays even with five-year UNCERTAIN."""
    from stockbot.bot import _format_buy_range_line

    verdict = {
        "buy_zone_abs": [100.0, 110.0],
        "buy_range_allowed": True,
        "five_year_business_test": {"answer": "UNCERTAIN"},
    }
    assert _format_buy_range_line(verdict) == "🛒 Buy range: ₹100.00–₹110.00"


def test_buy_range_shows_the_price_bar_it_has_to_clear():
    """"Not issued" on a fairly-valued stock looked like a malfunction: the
    card named no gate and never showed that a buy zone is fair value minus a
    risk-scaled margin, so the price simply did not qualify."""
    from stockbot.bot import _format_buy_range_line

    # Live JYOTHYLAB: ₹207.40 against a ₹176.40 ceiling — price is the blocker.
    line = _format_buy_range_line(
        {
            "buy_zone_abs": None,
            "buy_range_allowed": False,
            "risk": "MEDIUM",
            "current_price_abs": 207.40,
            "fair_value_abs": [210.0, 231.0],
            "five_year_business_test": {"answer": "UNCERTAIN"},
        }
    )
    assert line == "🛒 Buy range: not issued (five-year: UNCERTAIN · needs ≤₹176.40 at MEDIUM risk)"


def test_buy_range_price_bar_alone_when_no_gate_fired():
    from stockbot.bot import _format_buy_range_line

    line = _format_buy_range_line(
        {
            "buy_range_allowed": False,
            "risk": "MEDIUM",
            "current_price_abs": 207.40,
            "fair_value_abs": [210.0, 231.0],
        }
    )
    assert line == "🛒 Buy range: not issued (needs ≤₹176.40 at MEDIUM risk)"


def test_buy_range_omits_price_bar_when_price_already_clears_it():
    """A stock at ₹150 told it "needs ≤₹176.40" reads as nonsense — it already
    clears the valuation bar, so the named gate is the whole story."""
    from stockbot.bot import _format_buy_range_line

    verdict = {
        "buy_range_allowed": False,
        "risk": "MEDIUM",
        "current_price_abs": 150.0,
        "fair_value_abs": [210.0, 231.0],
        "five_year_business_test": {"answer": "UNCERTAIN"},
    }
    assert _format_buy_range_line(verdict) == "🛒 Buy range: not issued (five-year: UNCERTAIN)"

    # Unknown price: stay quiet rather than guess which side of the bar it sits.
    no_price = {k: v for k, v in verdict.items() if k != "current_price_abs"}
    assert _format_buy_range_line(no_price) == "🛒 Buy range: not issued (five-year: UNCERTAIN)"


def test_buy_range_omits_price_bar_when_not_computable():
    """No fair value or an unknown risk level must not fabricate a number."""
    from stockbot.bot import _format_buy_range_line

    assert _format_buy_range_line({"buy_range_allowed": False, "risk": "MEDIUM"}) == (
        "🛒 Buy range: not issued"
    )
    assert _format_buy_range_line(
        {"buy_range_allowed": False, "risk": "??", "fair_value_abs": [210.0, 231.0]}
    ) == "🛒 Buy range: not issued"


def test_bot_commands_menu_includes_preflight_and_workflow():
    from stockbot.bot import BOT_COMMANDS

    names = {cmd.command for cmd in BOT_COMMANDS}
    assert {
        "prescan",
        "pick",
        "progress",
        "workflow",
        "rank",
        "analyze",
        "stop",
        "preflight",
        "track",
        "help",
    } <= names
