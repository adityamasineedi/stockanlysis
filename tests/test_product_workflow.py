"""Tests for product workflow and pick calibration."""

from __future__ import annotations

from stockbot.calibration import TierSpread
from stockbot.pick_calibration import (
    PickCalibrationReport,
    build_pick_calibration,
    build_pick_tune_advice,
)
from stockbot.product_workflow import format_daily_workflow, format_portfolio_workflow


def test_daily_workflow_mentions_pick_and_analyze() -> None:
    text = format_daily_workflow()
    assert "/pick" in text
    assert "/analyze" in text
    assert "Daily tip" in text


def test_portfolio_workflow_mentions_sip_prescan_and_track_pick() -> None:
    text = format_portfolio_workflow()
    assert "/sip prescan" in text
    assert "/track pick" in text
    assert "12–18" in text


def test_pick_calibration_empty_history() -> None:
    report = build_pick_calibration(prices={})
    assert report.total_rows >= 0
    assert build_pick_tune_advice(report).lines


def test_pick_tune_advice_with_mock_spread() -> None:
    report = PickCalibrationReport(
        total_rows=20,
        scored=12,
        spread=TierSpread(
            positive_label="pick",
            negative_label="no_pick",
            positive_median_pct=8.0,
            negative_median_pct=2.0,
            n_positive=6,
            n_negative=6,
        ),
        pick_median=8.0,
        no_pick_median=2.0,
        n_pick=6,
        n_no_pick=6,
        by_quant_band=[("quant 50–54", 6, 7.0), ("quant 65+", 6, 5.0)],
        n_pillar_only=3,
        pillar_only_median=10.0,
    )
    advice = build_pick_tune_advice(report)
    joined = " ".join(advice.lines)
    assert "useful" in joined.lower() or "discriminating" in joined.lower()
