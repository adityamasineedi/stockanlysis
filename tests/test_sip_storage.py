"""SIP plan and contribution persistence."""

from __future__ import annotations

import pytest


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Point storage at a throwaway DB — tables are created on connect."""
    from stockbot import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "sip.db")
    return storage


def test_save_and_read_back_plan(store):
    saved = store.save_sip_plan(42, "heromotoco", 5000, risk_profile="aggressive", step_up_pct=10)
    assert saved.ticker == "HEROMOTOCO"  # normalized
    assert saved.active is True
    assert saved.horizon_years == 20  # spec default

    loaded = store.get_sip_plan(42)
    assert loaded == saved
    assert store.get_sip_plan(999) is None


def test_re_running_sip_replans_instead_of_stacking(store):
    store.save_sip_plan(42, "BEL", 5000)
    store.save_sip_plan(42, "CRISIL", 8000, step_up_pct=5)

    plan = store.get_sip_plan(42)
    assert plan.ticker == "CRISIL"
    assert plan.monthly_amount == 8000
    assert plan.step_up_pct == 5
    assert len(store.list_active_sip_plans()) == 1


def test_pause_and_resume_keeps_the_plan(store):
    store.save_sip_plan(42, "BEL", 5000)

    assert store.set_sip_plan_active(42, False) is True
    assert store.get_sip_plan(42).active is False
    assert store.list_active_sip_plans() == []

    assert store.set_sip_plan_active(42, True) is True
    assert len(store.list_active_sip_plans()) == 1
    # Unknown chat reports failure rather than silently doing nothing.
    assert store.set_sip_plan_active(999, False) is False


def test_contributions_are_append_only(store):
    store.save_sip_plan(42, "BEL", 5000)
    first = store.record_sip_contribution(42, "BEL", 5000, price_at_contribution=400.0)
    second = store.record_sip_contribution(42, "BEL", 5000, price_at_contribution=380.0)
    assert second != first

    summary = store.summarize_sip_contributions(42)
    assert summary.contributions == 2
    assert summary.total_invested == 10_000.0
    # Units accrue at each contribution's own price — the point of averaging.
    assert summary.units_estimate == pytest.approx(5000 / 400 + 5000 / 380, rel=1e-6)


def test_units_withheld_when_any_contribution_lacks_a_price(store):
    """Summing only the priced rows would understate units and overstate the
    average cost, so report None instead of a wrong number."""
    store.save_sip_plan(42, "BEL", 5000)
    store.record_sip_contribution(42, "BEL", 5000, price_at_contribution=400.0)
    store.record_sip_contribution(42, "BEL", 5000)  # price unknown

    summary = store.summarize_sip_contributions(42)
    assert summary.contributions == 2
    assert summary.total_invested == 10_000.0
    assert summary.units_estimate is None


def test_topup_flag_and_empty_ledger(store):
    store.save_sip_plan(42, "BEL", 5000)
    store.record_sip_contribution(42, "BEL", 2500, price_at_contribution=350.0, was_topup=True)
    assert store.summarize_sip_contributions(42).total_invested == 2500.0

    empty = store.summarize_sip_contributions(777)
    assert empty.contributions == 0
    assert empty.total_invested == 0.0
    assert empty.units_estimate is None


def test_plans_are_isolated_per_chat(store):
    store.save_sip_plan(1, "BEL", 5000)
    store.save_sip_plan(2, "CRISIL", 9000)
    store.record_sip_contribution(1, "BEL", 5000, price_at_contribution=400.0)

    assert store.get_sip_plan(1).ticker == "BEL"
    assert store.get_sip_plan(2).ticker == "CRISIL"
    assert store.summarize_sip_contributions(2).contributions == 0
    assert len(store.list_active_sip_plans()) == 2
