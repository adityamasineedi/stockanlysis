"""Risk policy, financial plan and holdings persistence."""

from __future__ import annotations

import pytest


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Point storage at a throwaway DB — tables are created on connect."""
    from stockbot import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "portfolio.db")
    return storage


def test_policy_round_trip_and_defaults(store):
    policy = store.save_risk_policy(42, 500_000)
    assert policy.total_capital_inr == 500_000.0
    assert policy.max_position_pct == 10.0  # bot's proposal
    assert policy.max_sector_pct == 25.0
    assert policy.emergency_fund_months is None

    assert store.get_risk_policy(42) == policy
    assert store.get_risk_policy(999) is None


def test_policy_updates_in_place(store):
    store.save_risk_policy(42, 500_000)
    store.save_risk_policy(42, 750_000, max_position_pct=8.0, emergency_fund_months=6)

    policy = store.get_risk_policy(42)
    assert policy.total_capital_inr == 750_000.0
    assert policy.max_position_pct == 8.0
    assert policy.emergency_fund_months == 6.0


def test_plan_round_trip_and_optional_fields(store):
    plan = store.save_financial_plan(
        42,
        current_age=33,
        target_age=40,
        monthly_investment_inr=200_000,
        desired_monthly_spend_inr=100_000,
    )
    assert plan.years_to_target == 7
    assert plan.desired_monthly_spend_inr == 100_000.0
    # Undeclared stays undeclared rather than defaulting to zero, which would
    # read as "earns nothing after retiring".
    assert plan.monthly_income_inr is None
    assert plan.post_retirement_income_inr is None
    assert plan.monthly_surplus_inr is None

    assert store.get_financial_plan(42) == plan
    assert store.get_financial_plan(999) is None


def test_plan_updates_in_place(store):
    store.save_financial_plan(
        42,
        current_age=33,
        target_age=40,
        monthly_investment_inr=200_000,
        desired_monthly_spend_inr=100_000,
    )
    store.save_financial_plan(
        42,
        current_age=34,
        target_age=45,
        monthly_investment_inr=250_000,
        desired_monthly_spend_inr=120_000,
        monthly_income_inr=400_000,
        monthly_expenses_inr=150_000,
        post_retirement_income_inr=600_000,
    )

    plan = store.get_financial_plan(42)
    assert plan.years_to_target == 11
    assert plan.monthly_investment_inr == 250_000.0
    assert plan.post_retirement_income_inr == 600_000.0
    # Surplus is what is left over, not what is invested — the gap between the
    # two is the headroom the savings lever has.
    assert plan.monthly_surplus_inr == 250_000.0


def test_reaching_the_target_age_leaves_no_negative_horizon(store):
    """Age past target must not produce a negative number of years, which would
    silently invert every projection."""
    plan = store.save_financial_plan(
        42,
        current_age=45,
        target_age=40,
        monthly_investment_inr=200_000,
        desired_monthly_spend_inr=100_000,
    )
    assert plan.years_to_target == 0


def test_plan_and_policy_are_separate_records(store):
    """Two tables on purpose: limits in one, the goal in the other. Declaring
    one must not imply the other."""
    store.save_risk_policy(42, 5_000_000)
    assert store.get_financial_plan(42) is None

    store.save_financial_plan(
        7,
        current_age=33,
        target_age=40,
        monthly_investment_inr=200_000,
        desired_monthly_spend_inr=100_000,
    )
    assert store.get_risk_policy(7) is None
    assert store.get_financial_plan(42) is None  # per chat


def test_holding_round_trip_and_cost_basis(store):
    holding = store.save_holding(42, "bel", 25, 412.50)
    assert holding.ticker == "BEL"  # normalized
    assert holding.cost_basis_inr == 10_312.50

    assert store.get_holding(42, "BEL") == holding
    assert store.get_holding(42, "MISSING") is None


def test_updating_a_holding_preserves_opened_at(store):
    """The holding period stays measurable across top-ups; only quantity,
    cost and updated_at move."""
    first = store.save_holding(42, "BEL", 25, 412.50)
    second = store.save_holding(42, "BEL", 40, 405.00)

    assert second.opened_at == first.opened_at
    assert second.quantity == 40.0
    assert second.avg_cost == 405.00
    assert len(store.list_holdings(42)) == 1  # replaced, not stacked


def test_list_and_delete_holdings(store):
    store.save_holding(42, "CRISIL", 5, 1600.0)
    store.save_holding(42, "BEL", 25, 412.50)

    assert [h.ticker for h in store.list_holdings(42)] == ["BEL", "CRISIL"]  # sorted
    assert store.delete_holding(42, "BEL") is True
    assert [h.ticker for h in store.list_holdings(42)] == ["CRISIL"]
    # Nothing to remove is reported, not silently swallowed.
    assert store.delete_holding(42, "BEL") is False


def test_holdings_and_policy_are_isolated_per_chat(store):
    store.save_risk_policy(1, 500_000)
    store.save_holding(1, "BEL", 25, 412.50)
    store.save_holding(2, "CRISIL", 5, 1600.0)

    assert store.get_risk_policy(2) is None
    assert [h.ticker for h in store.list_holdings(1)] == ["BEL"]
    assert [h.ticker for h in store.list_holdings(2)] == ["CRISIL"]


def test_seed_holding_reproduces_the_sip_ledger(store):
    """The ledger stores each contribution at its own price, so it already
    knows the units accumulated and what they cost — exactly a position."""
    store.save_sip_plan(42, "BEL", 5000)
    store.record_sip_contribution(42, "BEL", 5000, price_at_contribution=400.0)
    store.record_sip_contribution(42, "BEL", 5000, price_at_contribution=380.0)

    summary = store.summarize_sip_contributions(42, "BEL")
    holding = store.seed_holding_from_sip(42, "BEL")

    assert holding.quantity == pytest.approx(summary.units_estimate)
    assert holding.cost_basis_inr == pytest.approx(summary.total_invested, abs=1.0)
    assert holding.avg_cost == pytest.approx(10_000 / summary.units_estimate, abs=0.01)


def test_seed_refuses_when_the_ledger_cannot_say(store):
    """A contribution logged without a price makes units unknowable; seeding
    anyway would understate the quantity and overstate the average cost."""
    store.save_sip_plan(42, "BEL", 5000)
    store.record_sip_contribution(42, "BEL", 5000, price_at_contribution=400.0)
    store.record_sip_contribution(42, "BEL", 5000)  # price unknown

    assert store.seed_holding_from_sip(42, "BEL") is None
    assert store.get_holding(42, "BEL") is None
    # An empty ledger seeds nothing rather than a zero-quantity position.
    assert store.seed_holding_from_sip(42, "NOTHING") is None
