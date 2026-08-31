"""Risk policy and holdings persistence."""

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
