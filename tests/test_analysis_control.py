"""Tests for cooperative /stop cancellation."""

from __future__ import annotations

from unittest.mock import MagicMock

from stockbot.analysis_control import (
    active_operation,
    begin_operation,
    cancel_requested,
    end_operation,
    request_cancel,
)
from stockbot.pipeline import run_full_analysis


def test_request_cancel_when_idle_returns_none() -> None:
    end_operation()
    assert active_operation() is None
    assert request_cancel() is None


def test_request_cancel_sets_flag_while_operation_active() -> None:
    end_operation()
    begin_operation("analyze BEL")
    try:
        label = request_cancel()
        assert label == "analyze BEL"
        assert cancel_requested() is True
    finally:
        end_operation()
    assert cancel_requested() is False


def test_run_full_analysis_returns_cancelled_before_paid_work(monkeypatch) -> None:
    end_operation()
    begin_operation("analyze TCS")
    request_cancel()
    try:
        fake_ticker = MagicMock()
        fake_ticker.symbol = "TCS"

        def boom(*args, **kwargs):
            raise AssertionError("_run_paid_analysis should not run when cancelled")

        monkeypatch.setattr(
            "stockbot.pipeline.resolve_ticker",
            lambda query, table: fake_ticker,
        )
        monkeypatch.setattr("stockbot.pipeline.load_symbol_table", lambda: None)
        from stockbot.storage import CacheLookup

        monkeypatch.setattr(
            "stockbot.pipeline.storage.lookup_cached",
            lambda symbol, max_age_days=None: CacheLookup(hit=None, miss_reason="test"),
        )
        monkeypatch.setattr("stockbot.pipeline._run_paid_analysis", boom)

        result = run_full_analysis("TCS")
        assert result.status == "cancelled"
    finally:
        end_operation()


def test_batch_prescan_stops_when_cancel_requested(monkeypatch) -> None:
    from stockbot.portfolio_sip_prescan import batch_prescan_symbols

    calls: list[str] = []

    def fake_eligibility(symbol: str, *, config) -> MagicMock:
        calls.append(symbol)
        outcome = MagicMock()
        outcome.ticker = symbol
        outcome.verdict = "AUTO_DEEP_ANALYSIS"
        outcome.suitable_for_deep_analysis = True
        outcome.quant_score = 70.0
        return outcome

    monkeypatch.setattr(
        "stockbot.portfolio_sip_prescan.check_deep_analysis_eligibility",
        fake_eligibility,
    )

    end_operation()
    begin_operation("portfolio prescan")
    request_cancel()
    try:
        items = batch_prescan_symbols(("AAA", "BBB", "CCC"), skip_ai=True, delay_seconds=0)
    finally:
        end_operation()

    assert calls == []
    assert len(items) == 0
