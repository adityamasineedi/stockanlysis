"""SIP maths — pure functions, no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.sip import (
    THREE_MONTH_TRADING_DAYS,
    classify_dip,
    dip_pct_from_high,
    next_step_up_amount,
    project_corpus,
    scenario_projections,
    suggest_topup,
    three_month_high,
)


def test_project_corpus_matches_closed_form_annuity_due():
    """Contribute at the start of the month, then compound the balance:
    FV = P * (((1+i)^n - 1) / i) * (1+i), with i from the effective annual rate."""
    monthly, years, rate = 10_000.0, 20, 12.0
    i = (1 + rate / 100) ** (1 / 12) - 1
    n = years * 12
    expected = monthly * (((1 + i) ** n - 1) / i) * (1 + i)

    result = project_corpus(monthly, years, rate)
    assert result.projected_corpus == pytest.approx(expected, rel=1e-9)
    assert result.total_invested == pytest.approx(monthly * n)
    assert result.gain == pytest.approx(result.projected_corpus - result.total_invested)


def test_step_up_raises_the_instalment_on_each_anniversary():
    # A 10% step-up on 5,000 pays 5,500 through year 2 — not retroactively.
    assert next_step_up_amount(5000, 10, 0) == 5000.0
    assert next_step_up_amount(5000, 10, 1) == 5500.0
    assert next_step_up_amount(5000, 10, 2) == 6050.0

    flat = project_corpus(10_000, 20, 12.0)
    stepped = project_corpus(10_000, 20, 12.0, step_up_pct=10)
    assert stepped.total_invested > flat.total_invested
    assert stepped.projected_corpus > flat.projected_corpus
    # First year is identical; only later instalments differ.
    assert project_corpus(10_000, 1, 12.0, step_up_pct=10).total_invested == pytest.approx(
        project_corpus(10_000, 1, 12.0).total_invested
    )


def test_project_corpus_handles_loss_and_rejects_impossible_rates():
    """A plan can lose money; modelling only gains would flatter the output."""
    losing = project_corpus(5_000, 5, -8.0)
    assert losing.projected_corpus < losing.total_invested
    assert losing.gain < 0

    with pytest.raises(ValueError, match="must be > -100"):
        project_corpus(5_000, 5, -100.0)


def test_project_corpus_zero_inputs_are_safe():
    assert project_corpus(0, 20, 12).projected_corpus == 0.0
    assert project_corpus(5000, 0, 12).total_invested == 0.0
    assert project_corpus(-100, 20, 12).projected_corpus == 0.0


def test_classify_dip_boundaries():
    """Spec reads "5-10%" and ">10%", so 10.0 stays in the closed lower band."""
    assert classify_dip(100, 100) is None
    assert classify_dip(96, 100) is None  # 4% — not yet a dip
    assert classify_dip(95, 100) == "MODERATE"  # exactly 5%
    assert classify_dip(90, 100) == "MODERATE"  # exactly 10%
    assert classify_dip(89.9, 100) == "DEEP"  # past 10%
    assert classify_dip(50, 100) == "DEEP"
    # Above the high is not a dip, and never a negative drop.
    assert classify_dip(110, 100) is None
    assert dip_pct_from_high(110, 100) == 0.0


def test_dip_helpers_none_safe():
    assert dip_pct_from_high(100, None) is None
    assert dip_pct_from_high(100, 0) is None
    assert dip_pct_from_high(0, 100) is None
    assert classify_dip(100, None) is None
    assert suggest_topup(None, 5000) is None
    assert suggest_topup("DEEP", 0) is None


def test_suggest_topup_returns_a_range_not_a_number():
    """Never present a single "correct" top-up amount."""
    assert suggest_topup("MODERATE", 5000) == (2500.0, 5000.0)
    assert suggest_topup("DEEP", 5000) == (5000.0, 10_000.0)


def _frame(closes: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes})


def test_three_month_high_uses_last_quarter_only():
    # An old spike outside the window must not count as the 3-month high.
    closes = [500.0] + [100.0] * THREE_MONTH_TRADING_DAYS
    assert three_month_high(_frame(closes)) == 100.0


def test_three_month_high_ignores_nan_closes():
    """yfinance returns NaN closes on thin names; max() over them poisons the
    dip maths (see the same guard in fetch/prices.py)."""
    assert three_month_high(_frame([100.0, np.nan, 120.0, np.nan])) == 120.0
    assert three_month_high(_frame([np.nan, np.nan])) is None


def test_three_month_high_short_history_and_empty():
    assert three_month_high(_frame([100.0, 110.0])) == 110.0
    assert three_month_high(_frame([])) is None
    assert three_month_high(pd.DataFrame({"Open": [1.0]})) is None
    assert three_month_high(None) is None
    # Non-positive closes are not a valid high.
    assert three_month_high(_frame([0.0, -5.0])) is None


def test_scenario_projections_preserve_rate_order():
    projections = scenario_projections(5000, 20, (10.0, 12.0, 14.0))
    assert [p.annual_rate_pct for p in projections] == [10.0, 12.0, 14.0]
    corpora = [p.projected_corpus for p in projections]
    assert corpora == sorted(corpora)


def test_elapsed_plan_years_counts_completed_anniversaries():
    from datetime import datetime

    from stockbot.sip import UTC, current_instalment, elapsed_plan_years

    started = datetime(2024, 3, 15, tzinfo=UTC)
    assert elapsed_plan_years(started, datetime(2025, 3, 14, tzinfo=UTC)) == 0
    assert elapsed_plan_years(started, datetime(2025, 3, 15, tzinfo=UTC)) == 1
    assert elapsed_plan_years(started, datetime(2026, 3, 16, tzinfo=UTC)) == 2
    # A naive started_at (as SQLite can hand back) must not raise — the naive
    # datetime here is the point of the test, hence the noqa.
    naive_start = datetime(2024, 3, 15)  # noqa: DTZ001 - exercises the naive path
    assert elapsed_plan_years(naive_start, datetime(2026, 6, 1, tzinfo=UTC)) == 2
    # Clock skew must not produce a negative exponent.
    assert elapsed_plan_years(started, datetime(2023, 1, 1, tzinfo=UTC)) == 0

    # The instalment actually due, not the original amount.
    assert current_instalment(5000, 10, started, datetime(2026, 3, 16, tzinfo=UTC)) == 6050.0
    assert current_instalment(5000, 0, started, datetime(2026, 3, 16, tzinfo=UTC)) == 5000.0


def test_three_month_high_reads_whichever_series_it_is_given():
    """The caller must pass the same series the compared price came from —
    an adjusted high against an unadjusted price hides real dips after a split."""
    from stockbot.sip import classify_dip, three_month_high

    unadjusted = pd.DataFrame({"Close": [400.0, 420.0, 360.0]})
    adjusted = pd.DataFrame({"Close": [200.0, 210.0, 180.0]})  # post 2:1 split
    assert three_month_high(unadjusted) == 420.0
    assert three_month_high(adjusted) == 210.0
    # Quoted (unadjusted) price vs unadjusted high finds the dip.
    assert classify_dip(360.0, three_month_high(unadjusted)) == "DEEP"
    # Mixing series hides it entirely.
    assert classify_dip(360.0, three_month_high(adjusted)) is None
