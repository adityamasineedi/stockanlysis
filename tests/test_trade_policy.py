"""trade_policy.py — trade-friendly constitution relaxations."""

from __future__ import annotations

import pytest

from stockbot.trade_policy import (
    business_context_blocks_preflight,
    five_year_allows_buy_zone,
    five_year_blocks_capital_range,
    prescan_required_for_analyze,
    wc_gap_blocks_buy_zone,
)


@pytest.fixture(autouse=True)
def _strict_mode(monkeypatch):
    monkeypatch.setattr("stockbot.trade_policy.settings.trade_friendly_mode", False)
    monkeypatch.setattr("stockbot.trade_policy.settings.require_prescan_for_analyze", True)


def test_prescan_skipped_in_trade_friendly_mode(monkeypatch):
    monkeypatch.setattr("stockbot.trade_policy.settings.trade_friendly_mode", True)
    assert prescan_required_for_analyze() is False


def test_five_year_uncertain_blocked_in_strict_mode():
    test = {
        "answer": "UNCERTAIN",
        "confidence": "HIGH",
        "evidence_for": ["revenue CAGR 14% FY22-25", "ROCE stable ~18%"],
        "evidence_against": [],
    }
    assert five_year_allows_buy_zone(test) is False
    assert five_year_blocks_capital_range({"five_year_business_test": test}) is not None


def test_five_year_uncertain_allowed_with_evidence_in_trade_friendly(monkeypatch):
    monkeypatch.setattr("stockbot.trade_policy.settings.trade_friendly_mode", True)
    test = {
        "answer": "UNCERTAIN",
        "confidence": "HIGH",
        "evidence_for": ["revenue CAGR 14% FY22-25", "ROCE stable ~18%"],
        "evidence_against": ["margin compressed 80bp"],
    }
    assert five_year_allows_buy_zone(test) is True
    assert five_year_blocks_capital_range({"five_year_business_test": test}) is None


def test_five_year_no_always_blocks_even_in_trade_friendly(monkeypatch):
    monkeypatch.setattr("stockbot.trade_policy.settings.trade_friendly_mode", True)
    test = {"answer": "NO", "confidence": "HIGH", "evidence_for": ["x"], "evidence_against": []}
    assert five_year_allows_buy_zone(test) is False


def test_wc_inconclusive_soft_in_trade_friendly(monkeypatch):
    monkeypatch.setattr("stockbot.trade_policy.settings.trade_friendly_mode", True)
    assert wc_gap_blocks_buy_zone("INCONCLUSIVE") is False
    assert wc_gap_blocks_buy_zone("WORKING_CAPITAL_STRESS") is True


def test_business_context_preflight_relaxed_with_three_years(monkeypatch):
    monkeypatch.setattr("stockbot.trade_policy.settings.trade_friendly_mode", True)
    assert business_context_blocks_preflight(financial_years=4) is False
