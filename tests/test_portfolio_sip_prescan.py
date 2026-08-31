"""Tests for portfolio SIP prescan integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stockbot.portfolio_sip import target_split_whole_share_lines
from stockbot.portfolio_sip_prescan import (
    evaluate_prescan_gate,
    rank_symbols_by_prescan,
)
from stockbot.portfolio_sip_schema import PrescanGateConfig, SymbolConfig


def test_evaluate_prescan_gate_blocks_not_suitable():
    gate = PrescanGateConfig(enabled=True)
    row = {
        "verdict": "NOT_SUITABLE_FOR_3Y_RESEARCH",
        "suitable_for_deep_analysis": False,
        "logged_at": datetime.now(UTC).isoformat(),
    }
    result = evaluate_prescan_gate("BAD", row, gate)
    assert result.blocked
    assert "NOT_SUITABLE" in (result.note or "")


def test_evaluate_prescan_gate_allows_proceed():
    gate = PrescanGateConfig(enabled=True)
    row = {
        "verdict": "AUTO_DEEP_ANALYSIS",
        "suitable_for_deep_analysis": True,
        "logged_at": datetime.now(UTC).isoformat(),
    }
    result = evaluate_prescan_gate("GOOD", row, gate)
    assert not result.blocked


def test_evaluate_prescan_gate_pending_when_missing():
    gate = PrescanGateConfig(enabled=True, skip_when_missing=False)
    result = evaluate_prescan_gate("NEW", None, gate)
    assert not result.blocked
    assert result.note == "prescan pending"


def test_target_split_prescan_skip_blocks_symbol():
    symbols = (
        SymbolConfig("GOOD", target_amount_monthly=1000, max_amount_monthly=2000),
        SymbolConfig("BAD", target_amount_monthly=1000, max_amount_monthly=2000),
    )
    prices = {"GOOD": 100.0, "BAD": 100.0}
    gate = PrescanGateConfig(enabled=True)
    pmap = {
        "BAD": {
            "verdict": "NOT_SUITABLE_FOR_3Y_RESEARCH",
            "suitable_for_deep_analysis": False,
            "logged_at": datetime.now(UTC).isoformat(),
        }
    }
    lines = target_split_whole_share_lines(
        symbols,
        2000.0,
        prices,
        month=8,
        prescan_gate=gate,
        prescan_map=pmap,
    )
    by_sym = {line.symbol: line for line in lines}
    assert by_sym["GOOD"].shares >= 1
    assert by_sym["BAD"].prescan_skip
    assert by_sym["BAD"].shares == 0


def test_rank_symbols_by_prescan_orders_higher_quant_first():
    symbols = (
        SymbolConfig("LOW"),
        SymbolConfig("HIGH"),
    )
    pmap = {
        "LOW": {"quant_score": 40.0, "quality_score": 50.0},
        "HIGH": {"quant_score": 85.0, "quality_score": 70.0},
    }
    ranked = rank_symbols_by_prescan(symbols, pmap)
    assert ranked[0].symbol == "HIGH"
    assert ranked[1].symbol == "LOW"


def test_prescan_gate_stale_when_skip_when_missing():
    gate = PrescanGateConfig(enabled=True, require_recent_days=30, skip_when_missing=True)
    old = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    row = {
        "verdict": "AUTO_DEEP_ANALYSIS",
        "suitable_for_deep_analysis": True,
        "logged_at": old,
    }
    result = evaluate_prescan_gate("OLD", row, gate)
    assert result.blocked
    assert "stale" in (result.note or "")
