"""render.py unit tests — placeholder-token substitution, no LLM, no
network. Verification step 2 from the v3 migration: a hand-written report
using {{pledge_pct}} where pledge is unconfirmed must raise PlaceholderError
rather than rendering "None" or silently dropping to a guessed value."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from stockbot.llm.verdict import ValuationComputed, VerdictJSON
from stockbot.models import PriceData, Shareholding, Technicals
from stockbot.render import PlaceholderError, render_report

NOW = datetime.now(UTC)
TODAY = datetime.now(UTC).date()


def _price() -> PriceData:
    df = pd.DataFrame({"Close": [100.0]})
    return PriceData(410.65, TODAY, df, df, 450.0, 320.0, "yfinance", NOW)


def _technicals() -> Technicals:
    return Technicals(
        sma50=395.0,
        sma200=380.0,
        rsi14=54.71,
        support_abs=[380.0, 400.0],
        resistance_abs=[420.0, 440.0],
        as_of_date=TODAY,
        source="computed",
        fetched_at=NOW,
    )


def _verdict(**overrides) -> VerdictJSON:
    base = {
        "verdict": "WATCH",
        "current_price_abs": 410.65,
        "price_date": TODAY,
        "buy_zone_abs": (370.0, 380.0),
        "valuation_inputs": {
            "eps_base": 25.0,
            "multiple_base": [16.0, 18.0],
            "eps_bear": 20.0,
            "multiple_bear": [15.0, 17.0],
            "eps_bull": 30.0,
            "multiple_bull": [18.0, 20.0],
        },
        "confidence": 6,
        "risk": "MEDIUM",
        "business_quality": 7,
        "financial_health": 7,
        "management_quality": 7,
        "earnings_quality": "HIGH",
        "holding_period": "3-5 years",
        "reasons_buy": ["a"],
        "reasons_avoid": ["b"],
        "biggest_watch": "c",
        "missing_data_impact": "none",
        "gates_failed": [],
    }
    base.update(overrides)
    return VerdictJSON.model_validate(base)


def _valuation() -> ValuationComputed:
    return ValuationComputed(
        fair_value_bear_abs=(300.0, 340.0),
        fair_value_base_abs=(400.0, 450.0),
        fair_value_bull_abs=(540.0, 600.0),
    )


def _confirmed_shareholding() -> Shareholding:
    return Shareholding(62.89, 12.5, None, None, "Q1", "NSE", NOW)


def _unconfirmed_pledge_shareholding() -> Shareholding:
    return Shareholding(62.89, None, None, None, "Q1", "NSE", NOW)


def test_add_zone_and_avoid_chase_tokens_substitute():
    verdict = _verdict(
        add_range_allowed=True,
        buy_range_allowed=True,
        five_year_business_test={
            "answer": "YES",
            "confidence": "HIGH",
            "evidence_for": ["a"],
            "evidence_against": [],
        },
    )
    report = (
        "Add more {{add_zone_low}}–{{add_zone_high}}; avoid chasing above {{avoid_chase_above}}."
    )
    rendered = render_report(
        report, _price(), _technicals(), verdict, _valuation(), _confirmed_shareholding()
    )
    assert "₹300.00" in rendered  # bear low
    assert "₹340.00" in rendered  # add high capped at bear high (< buy low 370)
    assert "₹425.00" in rendered  # base midpoint avoid_chase_above


def test_add_zone_tokens_show_not_issued_when_blocked():
    verdict = _verdict(
        add_range_allowed=False,
        five_year_business_test={
            "answer": "UNCERTAIN",
            "confidence": "MEDIUM",
            "evidence_for": [],
            "evidence_against": ["cyclical"],
        },
    )
    report = "Add more {{add_zone_low}}–{{add_zone_high}}."
    rendered = render_report(
        report, _price(), _technicals(), verdict, _valuation(), _confirmed_shareholding()
    )
    assert "not issued" in rendered


def test_substitutes_all_tokens_when_data_is_complete():
    report = (
        "Trades at {{current_price}} ({{price_date}}), between its "
        "{{week52_low}}-{{week52_high}} range. RSI(14) is {{rsi14}}, above "
        "{{sma200}}. Support near {{support}}, resistance near "
        "{{resistance}}. Fair value bear/base/bull: {{fair_value_bear}} / "
        "{{fair_value_base}} / {{fair_value_bull}}. Buy zone "
        "{{buy_zone_low}}-{{buy_zone_high}}, upside {{upside_pct}}, "
        "downside {{downside_pct}}. Promoter holds {{promoter_pct}}, "
        "pledge {{pledge_pct}}."
    )
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert "{{" not in rendered
    assert "₹410.65" in rendered
    assert "54.7" in rendered  # rsi14, 1dp, no ₹ prefix
    assert "₹425.00" in rendered  # fair_value_base midpoint
    assert "12.5%" in rendered  # pledge_pct


def test_raises_when_pledge_token_used_but_pledge_unconfirmed():
    # This is the exact verification case from the v3 migration: a report
    # that writes {{pledge_pct}} when pledge is None must fail loudly, not
    # render "None" or silently substitute 0.
    report = "Promoter pledge stands at {{pledge_pct}}."
    with pytest.raises(PlaceholderError, match=r"pledge_pct"):
        render_report(
            report,
            _price(),
            _technicals(),
            _verdict(),
            _valuation(),
            _unconfirmed_pledge_shareholding(),
        )


def test_raises_when_shareholding_entirely_missing_and_promoter_pct_used():
    report = "Promoter holding is {{promoter_pct}}."
    with pytest.raises(PlaceholderError, match=r"promoter_pct"):
        render_report(report, _price(), _technicals(), _verdict(), _valuation(), None)


def test_does_not_raise_when_pledge_unconfirmed_and_token_not_used():
    report = "Pledge status is unconfirmed from an exchange source."
    rendered = render_report(
        report,
        _price(),
        _technicals(),
        _verdict(),
        _valuation(),
        _unconfirmed_pledge_shareholding(),
    )
    assert rendered == report


def test_raises_on_unknown_token():
    report = "This uses {{made_up_token}} which was never defined."
    with pytest.raises(PlaceholderError, match=r"made_up_token"):
        render_report(report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding())


def test_repairs_spaced_known_placeholder_braces():
    # Spaces inside braces used to fail render after a paid run. Strip-repair
    # known tokens so "{{ current_price }}" still delivers.
    report = "Trades at {{ current_price }} today."
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert "{{" not in rendered
    assert "₹410.65" in rendered


def test_raises_on_truly_unknown_leftover_braces():
    report = "Trades at {{ totally made up }} today."
    with pytest.raises(PlaceholderError, match=r"Unsubstituted"):
        render_report(report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding())


def test_repairs_mangled_sma_rupee_placeholder():
    # Live NATIONALUM batch: model invented {{sma₹365.55}} after validation.
    report = "Price holds above {{sma₹365.55}}."
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert "{{" not in rendered
    assert "₹395.00" in rendered  # fixture sma50


def test_money_tokens_formatted_to_two_decimals():
    report = "SMA200 is {{sma200}}."
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert "₹380.00" in rendered


def test_support_picks_nearest_level_below_current_price():
    # current_price=410.65, support_abs=[380.0, 400.0] -> nearest below is 400.0
    report = "Support at {{support}}."
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert "₹400.00" in rendered


def test_resistance_picks_nearest_level_above_current_price():
    # current_price=410.65, resistance_abs=[420.0, 440.0] -> nearest above is 420.0
    report = "Resistance at {{resistance}}."
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert "₹420.00" in rendered


def test_strips_backticks_wrapped_around_a_token():
    # Regression: the master prompt's own examples show tokens wrapped in
    # backticks ("`{{current_price}}`") as authoring guidance for the model,
    # but a model that copies that literally into its report used to leave
    # the backticks in place around the *substituted* value — literal
    # backtick characters surviving into the delivered report.
    report = "Trades at `{{current_price}}` on `{{price_date}}`."
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert "`" not in rendered
    assert "₹410.65" in rendered


def test_fair_value_base_low_high_tokens_give_the_base_range_not_bear_to_bull():
    # Regression: the headline "Fair Value" figure must be the base case's
    # own range, not a bear-low-to-bull-high span (which mixes two
    # different scenarios into one number and can be 100%+ wide).
    report = "Fair Value: {{fair_value_base_low}}-{{fair_value_base_high}}."
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert "₹400.00" in rendered  # fair_value_base_abs[0]
    assert "₹450.00" in rendered  # fair_value_base_abs[1]


def test_upside_and_downside_pct_share_the_same_sign_convention():
    # Regression: downside_pct used to flip the subtraction order to force
    # a positive number while upside_pct didn't, so the two came out
    # inconsistently signed whenever price was above the base fair value.
    # Both should now be (target - current) / current, sharing one convention.
    report = "Upside {{upside_pct}}, downside {{downside_pct}}."
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    # current_price=410.65, fair_value_base_mid=425 -> positive upside
    # current_price=410.65, fair_value_bear_mid=320 -> negative downside
    # (bear case sits below current price under the shared convention)
    assert "3.5%" in rendered  # upside_pct ~ +3.5%
    assert "-22.1%" in rendered  # downside_pct ~ -22.1%, signed consistently


def test_report_with_no_tokens_passes_through_unchanged():
    report = "Plain prose with no placeholders at all."
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert rendered == report


def test_render_rewrites_bear_to_bull_headline_fair_value_to_base():
    # KPITTECH live bug: model typed ₹323–₹780 (bear→bull) as Fair Value;
    # Telegram showed base ₹400–₹450. Attachment must match the card.
    report = (
        "1. QUICK VERDICT\n"
        "Fair Value ₹300.00–₹600.00 · Upside -10%\n"
        "**Fair Value:** ₹300.00–₹600.00\n"
    )
    rendered = render_report(
        report, _price(), _technicals(), _verdict(), _valuation(), _confirmed_shareholding()
    )
    assert "Fair Value ₹400.00–₹450.00" in rendered
    assert "**Fair Value:** ₹400.00–₹450.00" in rendered
    assert "₹300.00–₹600.00" not in rendered


def test_canonicalize_does_not_touch_single_midpoint_fair_value_mentions():
    from stockbot.render import canonicalize_headline_fair_value

    prose = "about 10% below our base fair value of ₹528.00."
    assert canonicalize_headline_fair_value(prose, _valuation()) == prose
