"""/plan — argument parsing, assembly, and the wording of the verdict."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from stockbot.retirement_report import build_plan_assessment, format_plan_report, money
from stockbot.storage import FinancialPlan


@pytest.fixture
def store(monkeypatch, tmp_path):
    from stockbot import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "plan.db")
    return storage


def _plan(**kw) -> FinancialPlan:
    return FinancialPlan(
        chat_id=1,
        current_age=kw.get("age", 33),
        target_age=kw.get("target", 40),
        monthly_income_inr=kw.get("income"),
        monthly_expenses_inr=kw.get("expenses"),
        monthly_investment_inr=kw.get("invest", 200_000),
        desired_monthly_spend_inr=kw.get("spend", 100_000),
        post_retirement_income_inr=kw.get("pension"),
        updated_at=datetime.now(UTC),
    )


def _plain(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _report(**kw) -> str:
    assessment = build_plan_assessment(_plan(**kw), kw.get("capital", 5_000_000))
    assert assessment is not None
    return _plain(format_plan_report(assessment))


# --- parsing ---------------------------------------------------------------


def test_parses_the_documented_forms():
    from stockbot.bot import _parse_plan_args

    assert _parse_plan_args(["33", "to", "40", "spend", "100000", "invest", "200000"]) == {
        "current_age": 33.0,
        "target_age": 40.0,
        "spend": 100_000.0,
        "invest": 200_000.0,
    }
    # Indian digit grouping and a rupee sign are what people actually type.
    typed = [
        "33", "to", "40", "spend", "1,00,000", "invest", "₹200000",
        "income", "400000", "expenses", "150000", "pension", "600000",
    ]
    full = _parse_plan_args(typed)
    assert full["spend"] == 100_000.0
    assert full["invest"] == 200_000.0
    assert full["pension"] == 600_000.0


def test_a_target_age_that_has_already_passed_is_refused():
    """Without this the horizon is zero or negative and every projection
    silently inverts."""
    from stockbot.bot import _parse_plan_args

    for args in (
        ["40", "to", "33", "spend", "100000", "invest", "200000"],
        ["40", "to", "40", "spend", "100000", "invest", "200000"],
    ):
        message = _parse_plan_args(args)
        assert isinstance(message, str)
        assert "later than" in message


def test_bad_input_explains_itself_rather_than_guessing():
    from stockbot.bot import _parse_plan_args

    for args in (
        [],
        ["33"],
        ["33", "to"],
        ["33", "and", "40"],                                    # missing "to"
        ["abc", "to", "40", "spend", "1", "invest", "1"],
        ["33", "to", "40", "spend", "100000"],                  # no invest
        ["33", "to", "40", "invest", "200000"],                 # no spend
        ["33", "to", "40", "spend", "0", "invest", "200000"],   # zero spend
        ["33", "to", "40", "spend", "-5", "invest", "200000"],
        ["33", "to", "40", "spend", "abc", "invest", "200000"],
        ["33", "to", "40", "spend"],                            # value dropped
        ["33", "to", "40", "wealth", "100000"],                 # unknown keyword
        ["200", "to", "300", "spend", "1", "invest", "1"],      # impossible ages
    ):
        assert isinstance(_parse_plan_args(args), str), args


def test_zero_investment_is_allowed_because_it_is_the_finding():
    """"You are investing nothing" is a legitimate starting point and the gate
    should price it, not reject it."""
    from stockbot.bot import _parse_plan_args

    parsed = _parse_plan_args(["33", "to", "40", "spend", "100000", "invest", "0"])
    assert parsed["invest"] == 0.0


# --- the verdict -----------------------------------------------------------


def test_stretched_plan_names_the_gap_and_the_bigger_lever():
    text = _report()
    assert "HIGHLY STRETCHED" in text or "NOT FEASIBLE" in text
    assert "Target ₹6.01cr" in text
    # The point of the whole gate: over 7 years, picking is not the lever.
    assert "Better stock picking is not your biggest lever" in text
    assert "To close it by investing alone" in text


def test_long_horizon_flips_to_returns_and_drops_the_warning():
    """The savings-first finding must track the horizon rather than being a
    slogan the report always prints."""
    text = _report(target=55)
    assert "ON TRACK" in text
    assert "Better stock picking is not your biggest lever" not in text
    assert "does not depend on finding an exceptional winner" in text


def test_a_plan_needing_a_multibagger_says_so():
    text = _report()
    assert "does not work without an exceptional winner" in text
    assert "Plans that need a multi-bagger usually do not get one" in text


def test_no_margin_is_stated_rather_than_shown_as_zero():
    text = _report()
    assert "Permanent capital loss the plan absorbs: 0%" in text
    assert "one thesis failure moves the retirement date" in text


def test_required_contribution_is_checked_against_actual_surplus():
    """Telling someone to invest ₹4.2L/month when ₹2.5L is all they have spare
    is arithmetic, not advice."""
    text = _report(income=400_000, expenses=150_000)
    assert "more than your ₹2.50L monthly surplus" in text
    assert "the target age or the spend has to move" in text


def test_post_retirement_income_moves_the_target():
    """Often the single biggest lever in the plan, so it must visibly change
    the corpus rather than being recorded and ignored."""
    without = _report()
    with_pension = _report(pension=1_200_000)
    assert "Target ₹6.01cr" in without
    assert "Target ₹2.01cr" in with_pension


def test_every_headline_number_carries_its_assumptions():
    text = _report()
    assert "at 6% inflation, 3% withdrawal, 10% return" in text
    # And the sweep is shown, so no single figure stands alone.
    assert "If the assumptions move" in text
    assert "If returns disappoint" in text
    assert "8% →" in text and "12% →" in text


def test_income_that_covers_the_whole_spend_needs_no_corpus():
    """build returns None — there is nothing to project — and the handler turns
    that into words rather than a zero."""
    assert build_plan_assessment(_plan(pension=50_000_000), 5_000_000) is None


def test_no_horizon_left_produces_no_assessment():
    assert build_plan_assessment(_plan(age=45, target=40), 5_000_000) is None


def test_money_uses_indian_scale():
    assert money(60_100_000) == "₹6.01cr"
    assert money(250_000) == "₹2.50L"
    assert money(4_200) == "₹4,200"


# --- handler paths ---------------------------------------------------------


def test_report_asks_for_capital_before_projecting_from_a_guess(store):
    """Projecting from an assumed starting amount would put a confident number
    on an invented one."""
    from stockbot.bot import _plan_report_text

    store.save_financial_plan(
        7,
        current_age=33,
        target_age=40,
        monthly_investment_inr=200_000,
        desired_monthly_spend_inr=100_000,
    )
    text = _plan_report_text(7)
    assert "/capital" in text
    assert "cr" not in _plain(text)  # no projection was made


def test_report_with_no_plan_shows_usage(store):
    from stockbot.bot import PLAN_USAGE, _plan_report_text

    assert _plan_report_text(7) == PLAN_USAGE


def test_report_end_to_end_once_both_are_set(store):
    from stockbot.bot import _plan_report_text

    store.save_risk_policy(7, 5_000_000)
    store.save_financial_plan(
        7,
        current_age=33,
        target_age=40,
        monthly_investment_inr=200_000,
        desired_monthly_spend_inr=100_000,
    )
    text = _plain(_plan_report_text(7))
    assert "Retire at 40 — 7 year(s) away" in text
    assert "Target ₹6.01cr" in text
    assert "Educational research only" in text  # disclaimer survives


def test_a_plan_whose_target_age_has_passed_is_explained_not_crashed(store):
    from stockbot.bot import _plan_report_text

    store.save_risk_policy(7, 5_000_000)
    store.save_financial_plan(
        7,
        current_age=45,
        target_age=40,
        monthly_investment_inr=200_000,
        desired_monthly_spend_inr=100_000,
    )
    text = _plain(_plan_report_text(7))
    assert "no horizon left" in text
