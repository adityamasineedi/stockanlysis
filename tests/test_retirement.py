"""Retirement feasibility gate — pure maths, no I/O."""

from __future__ import annotations

import pytest

from stockbot.retirement import (
    INFLATION_SCENARIOS_PCT,
    WITHDRAWAL_SCENARIOS_PCT,
    compare_levers,
    corpus_scenarios,
    dominant_lever,
    funding_gap,
    inflate,
    max_survivable_permanent_loss_pct,
    project_wealth,
    required_corpus,
    required_monthly_investment,
    survives_without_outliers,
)


def test_inflate_converts_todays_rupees_to_target_year():
    # 1,00,000 at 6% for 7 years = 100000 * 1.06^7. Rounded to paise, so the
    # tolerance is a paise, not a float epsilon.
    assert inflate(100_000, 6.0, 7) == pytest.approx(100_000 * 1.06**7, abs=0.01)
    # Zero inflation leaves it alone; zero years is a no-op.
    assert inflate(100_000, 0.0, 7) == 100_000.0
    assert inflate(100_000, 6.0, 0) == 100_000.0
    for bad in (None, 0, -5, float("nan"), "100000"):
        assert inflate(bad, 6.0, 7) is None, bad


def test_required_corpus_is_spend_over_withdrawal_rate():
    assert required_corpus(1_800_000, 3.0) == 60_000_000.0
    assert required_corpus(1_800_000, 4.0) == 45_000_000.0
    # A lower withdrawal rate demands a bigger corpus, not a smaller one.
    assert required_corpus(1_800_000, 2.5) > required_corpus(1_800_000, 4.0)
    assert required_corpus(1_800_000, 0) is None
    assert required_corpus(None, 3.0) is None


def test_corpus_grid_sweeps_every_combination():
    scenarios = corpus_scenarios(100_000, 7)
    assert len(scenarios) == len(INFLATION_SCENARIOS_PCT) * len(WITHDRAWAL_SCENARIOS_PCT)

    # 6% inflation, 3% withdrawal — the mid case.
    mid = next(s for s in scenarios if s.inflation_pct == 6.0 and s.withdrawal_pct == 3.0)
    expected_annual = 100_000 * 1.06**7 * 12
    assert mid.annual_spend_at_target == pytest.approx(expected_annual, rel=1e-6)
    assert mid.required_corpus_inr == pytest.approx(expected_annual / 0.03, rel=1e-6)

    # Higher inflation always demands more.
    low = next(s for s in scenarios if s.inflation_pct == 4.0 and s.withdrawal_pct == 3.0)
    high = next(s for s in scenarios if s.inflation_pct == 8.0 and s.withdrawal_pct == 3.0)
    assert low.required_corpus_inr < mid.required_corpus_inr < high.required_corpus_inr


def test_post_retirement_income_reduces_the_corpus_needed():
    """Only the shortfall needs funding from the portfolio — often the single
    biggest lever in the plan."""
    without = corpus_scenarios(100_000, 7)
    with_income = corpus_scenarios(100_000, 7, annual_income_at_target=600_000)

    a = next(s for s in without if s.inflation_pct == 6.0 and s.withdrawal_pct == 3.0)
    b = next(s for s in with_income if s.inflation_pct == 6.0 and s.withdrawal_pct == 3.0)
    assert b.required_corpus_inr < a.required_corpus_inr
    assert b.required_corpus_inr == pytest.approx(
        (a.annual_spend_at_target - 600_000) / 0.03, rel=1e-6
    )


def test_income_covering_the_whole_spend_needs_no_corpus():
    """But the plan then depends entirely on that income continuing, which is
    why callers also run the grid with it at zero."""
    scenarios = corpus_scenarios(10_000, 7, annual_income_at_target=10_000_000)
    assert all(s.required_corpus_inr == 0.0 for s in scenarios)


def test_project_wealth_is_capital_growth_plus_contributions():
    from stockbot.sip import project_corpus

    start, monthly, years, rate = 5_000_000, 200_000, 7, 12.0
    expected = start * 1.12**years + project_corpus(monthly, years, rate).projected_corpus
    assert project_wealth(start, monthly, years, rate) == pytest.approx(expected, rel=1e-9)

    # Capital with no contributions, and contributions with no capital.
    assert project_wealth(start, 0, years, rate) == pytest.approx(start * 1.12**years, rel=1e-9)
    assert project_wealth(0, monthly, years, rate) == pytest.approx(
        project_corpus(monthly, years, rate).projected_corpus, rel=1e-9
    )


def test_funding_gap_and_feasibility_bands():
    gap = funding_gap(60_100_000, 5_000_000, 200_000, 7, 12.0)
    assert gap.projected_corpus_inr == pytest.approx(36_800_000, rel=0.02)
    assert gap.coverage_ratio == pytest.approx(0.61, abs=0.01)
    assert gap.feasibility == "HIGHLY_STRETCHED"
    assert gap.gap_inr == pytest.approx(23_300_000, rel=0.02)

    # A plan that overshoots is feasible and reports no gap.
    covered = funding_gap(10_000_000, 5_000_000, 200_000, 7, 12.0)
    assert covered.feasibility == "FEASIBLE"
    assert covered.gap_inr == 0.0
    assert covered.coverage_ratio > 1.0


def test_feasibility_verdicts_across_the_range():
    def verdict(required: float) -> str:
        return funding_gap(required, 0, 100_000, 10, 10.0).feasibility

    projected = project_wealth(0, 100_000, 10, 10.0)
    assert verdict(projected * 0.9) == "FEASIBLE"           # ratio > 1
    assert verdict(projected / 0.95) == "FEASIBLE_WITH_CONDITIONS"
    assert verdict(projected / 0.7) == "HIGHLY_STRETCHED"
    assert verdict(projected / 0.3) == "NOT_FEASIBLE"


def test_required_monthly_investment_round_trips():
    """Invest exactly what it asks for and the plan should just close."""
    target, start, years, rate = 60_100_000, 5_000_000, 7, 12.0
    monthly = required_monthly_investment(target, start, years, rate)
    assert monthly == pytest.approx(381_000, rel=0.01)

    closed = funding_gap(target, start, monthly, years, rate)
    assert closed.coverage_ratio == pytest.approx(1.0, abs=0.001)
    assert closed.feasibility == "FEASIBLE"

    # Already covered by starting capital alone — nothing more needed.
    assert required_monthly_investment(1_000_000, 5_000_000, 7, 12.0) == 0.0


def test_savings_beats_return_over_a_short_horizon():
    """Step 5's whole point: over 7 years compounding has little time to work,
    so a top-decile improvement in stock picking is worth less than saving
    moderately more. The gate must be able to say so."""
    levers = compare_levers(5_000_000, 200_000, 7, 12.0)
    by_name = {lever.name: lever for lever in levers}

    assert by_name["savings"].gain_inr > by_name["return"].gain_inr
    assert by_name["savings"].gain_pct == pytest.approx(21.0, abs=0.5)
    assert by_name["return"].gain_pct == pytest.approx(14.0, abs=0.5)

    # Sorted worst-first so the biggest lever reads last.
    assert [lever.gain_inr for lever in levers] == sorted(lever.gain_inr for lever in levers)
    assert dominant_lever(levers) is levers[-1]
    assert dominant_lever([]) is None


def test_returns_dominate_once_the_horizon_is_long():
    """The savings-first finding is a property of short horizons, not a law —
    over 30 years compounding wins, and the gate should reflect that rather
    than always giving the same answer."""
    levers = compare_levers(5_000_000, 200_000, 30, 12.0)
    by_name = {lever.name: lever for lever in levers}
    assert by_name["return"].gain_inr > by_name["savings"].gain_inr


def test_plan_that_needs_outliers_is_reported_as_such():
    """If it only closes at a high return, the retirement depends on finding
    exceptional stocks and must be labelled that way."""
    # Stretched plan: does not close even at the optimistic rate.
    assert survives_without_outliers(60_100_000, 5_000_000, 200_000, 7) is False
    # Comfortable plan: closes on a broad-market-like return.
    assert survives_without_outliers(10_000_000, 5_000_000, 200_000, 7) is True
    assert survives_without_outliers(None, 5_000_000, 200_000, 7) is None


def test_permanent_loss_budget_is_the_surplus():
    # 20% more than needed absorbs a 16.7% permanent loss.
    assert max_survivable_permanent_loss_pct(12_000_000, 10_000_000) == pytest.approx(16.7, abs=0.1)
    # A plan with no surplus cannot afford a single permanent loss.
    assert max_survivable_permanent_loss_pct(10_000_000, 10_000_000) == 0.0
    assert max_survivable_permanent_loss_pct(8_000_000, 10_000_000) == 0.0
    assert max_survivable_permanent_loss_pct(None, 10_000_000) is None


def test_nothing_is_computed_from_unusable_inputs():
    assert corpus_scenarios(None, 7) == []
    assert corpus_scenarios(100_000, -1) == []
    assert project_wealth(5_000_000, 200_000, 0, 12.0) is None
    assert project_wealth(5_000_000, 200_000, 7, -100.0) is None
    assert project_wealth(None, 200_000, 7, 12.0) is None
    assert funding_gap(None, 5_000_000, 200_000, 7, 12.0) is None
    assert required_monthly_investment(60_000_000, 5_000_000, 0, 12.0) is None
    assert compare_levers(None, 200_000, 7, 12.0) == []
