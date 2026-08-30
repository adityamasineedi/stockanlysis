"""Entry-snapshot plumbing that makes screening records measurable forward.

A score with no price and no date attached can never be checked against what
the stock actually did afterwards, so these assert the price/timestamp survive
every hop from StockMetrics into the persisted records.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime

from stockbot.portfolio_screener.models import (
    ComponentScores,
    DataValidationResult,
    HardFilterResult,
    QuantScreenResult,
    StockMetrics,
)
from stockbot.portfolio_screener.portfolio_selector import build_screen_record
from stockbot.portfolio_screener.quant_engine import compute_quant_score
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig


def _quant_result(**kwargs: object) -> QuantScreenResult:
    components = ComponentScores(
        business_quality=70.0,
        financial_strength=70.0,
        growth=70.0,
        cash_flow_quality=70.0,
        capital_efficiency=70.0,
        valuation=70.0,
        balance_sheet=70.0,
        earnings_quality=70.0,
        risk=70.0,
    )
    defaults: dict[str, object] = {
        "ticker": "TCS",
        "base_score": 70.0,
        "red_flag_penalty": 0.0,
        "final_quant_score": 70.0,
        "components": components,
        "red_flags": [],
        "data_validation": DataValidationResult(
            ticker="TCS",
            data_completeness_score=100.0,
            data_quality_score=100.0,
            data_confidence="HIGH",
            missing_metrics={},
            contradictions=[],
            critical_ok=True,
        ),
        "hard_filter": HardFilterResult(ticker="TCS", status="PASS", reasons=[]),
        "sector": "Technology",
        "industry": "IT Services",
    }
    defaults.update(kwargs)
    return QuantScreenResult(**defaults)  # type: ignore[arg-type]


def test_quant_score_carries_scan_price() -> None:
    """compute_quant_score must forward the price the scan actually saw."""
    metrics = StockMetrics(
        ticker="TCS",
        current_price_abs=3421.5,
        data_timestamp=datetime.now(UTC),
    )
    quant = compute_quant_score(metrics, ScreenerRunConfig(skip_ai=True, dry_run=True))
    assert quant.current_price_abs == 3421.5
    assert quant.data_timestamp == metrics.data_timestamp


def test_screen_record_captures_entry_snapshot() -> None:
    """The persisted screening record needs both price and date to be scorable."""
    scanned = datetime.now(UTC)
    quant = _quant_result(current_price_abs=1234.5, data_timestamp=scanned)

    record = build_screen_record(quant, ai=None, final_score=70.0)

    assert record.price_at_scan == 1234.5
    assert record.scanned_at == scanned


def test_entry_snapshot_is_none_safe() -> None:
    """A failed price fetch must not break record building — just no snapshot."""
    record = build_screen_record(_quant_result(), ai=None, final_score=70.0)

    assert record.price_at_scan is None
    assert record.scanned_at is None


def test_data_dir_follows_env_override(monkeypatch, tmp_path) -> None:
    """DATA_DIR must be redirectable to a mounted volume, or every redeploy
    silently wipes the analysis history and prescan log."""
    from stockbot import config

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "persistent"))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.DATA_DIR == tmp_path / "persistent"
        # Everything stateful must sit under the override, not the repo tree.
        assert reloaded.DB_PATH.is_relative_to(tmp_path / "persistent")
        assert reloaded.PORTFOLIO_DIR.is_relative_to(tmp_path / "persistent")
        assert reloaded.SCREENER_CACHE_DIR.is_relative_to(tmp_path / "persistent")
    finally:
        monkeypatch.delenv("STOCKBOT_DATA_DIR", raising=False)
        importlib.reload(config)


def test_data_dir_defaults_to_repo_data_when_unset(monkeypatch) -> None:
    from stockbot import config

    monkeypatch.delenv("STOCKBOT_DATA_DIR", raising=False)
    reloaded = importlib.reload(config)
    assert reloaded.DATA_DIR == reloaded.PROJECT_ROOT / "data"
