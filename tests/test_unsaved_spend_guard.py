"""Tests for unsaved-ticker spend guard (orphan session cost leak)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stockbot.costs import (
    UNSAVED_SPEND_BLOCK_INR,
    log_call,
    should_block_unsaved_spend,
    unsaved_ticker_spend_inr,
)


def test_unsaved_spend_blocks_after_orphan_calls(monkeypatch, tmp_path):
    from stockbot import costs, storage

    db = tmp_path / "guard.db"
    monkeypatch.setattr(costs, "DB_PATH", db)
    monkeypatch.setattr(storage, "DB_PATH", db)

    # Bill Stage 2 without saving an analysis (~₹40+ at Sonnet rates).
    log_call(
        "claude-sonnet-5",
        input_tokens=200_000,
        output_tokens=80_000,
        stage="stage2",
        ticker="HBLENGINE",
        called_at=datetime.now(UTC) - timedelta(hours=1),
    )
    spend = unsaved_ticker_spend_inr("HBLENGINE")
    assert spend >= UNSAVED_SPEND_BLOCK_INR
    blocked, amount = should_block_unsaved_spend("HBLENGINE")
    assert amount == spend
    assert blocked is True


def test_unsaved_spend_clears_after_saved_analysis(monkeypatch, tmp_path):
    from stockbot import costs, storage

    db = tmp_path / "guard2.db"
    monkeypatch.setattr(costs, "DB_PATH", db)
    monkeypatch.setattr(storage, "DB_PATH", db)

    past = datetime.now(UTC) - timedelta(hours=2)
    log_call(
        "claude-sonnet-5",
        input_tokens=20_000,
        output_tokens=10_000,
        stage="stage2",
        ticker="GESHIP",
        called_at=past,
    )
    storage.save_analysis(
        ticker="GESHIP",
        verdict_json={"verdict": "WATCH"},
        report_md="# ok",
        brief_text="brief",
        stage1_tokens=1,
        stage2_tokens=1,
        cost_inr=12.0,
        validation_passed=True,
    )
    # Calls before the saved analysis must not block a new run.
    assert unsaved_ticker_spend_inr("GESHIP") == 0.0
    blocked, _ = should_block_unsaved_spend("GESHIP")
    assert blocked is False
