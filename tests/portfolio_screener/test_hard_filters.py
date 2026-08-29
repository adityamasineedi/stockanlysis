"""Tests for hard exclusion filters — loss-maker IC and leverage wording."""

from __future__ import annotations

from stockbot.portfolio_screener.hard_filters import apply_hard_filters
from stockbot.portfolio_screener.models import DataValidationResult, StockMetrics


def _validation_ok() -> DataValidationResult:
    return DataValidationResult(
        ticker="X",
        data_completeness_score=100.0,
        data_quality_score=90.0,
        data_confidence="HIGH",
        missing_metrics={},
        contradictions=[],
        critical_ok=True,
    )


def test_loss_maker_skips_interest_coverage_hard_exclude() -> None:
    metrics = StockMetrics(
        ticker="AEQUS",
        sector="Industrials",
        industry="Industrial Machinery",
        years_available=5,
        net_income=-50.0,
        net_income_series=[-40.0, -45.0, -50.0],
        operating_cash_flow=-30.0,
        ocf_series=[-20.0, -25.0, -30.0],
        debt=500.0,
        debt_equity=2.0,
        interest_coverage=0.75,
        roe=-10.0,
        roce=-8.0,
        ocf_to_pat=-0.6,
    )
    result = apply_hard_filters(metrics, _validation_ok())
    assert result.status == "HARD_EXCLUDE"
    assert any("persistent losses" in r for r in result.reasons)
    assert not any("interest coverage" in r for r in result.reasons)


def test_high_debt_uses_de_ratio_not_misleading_multiplier() -> None:
    metrics = StockMetrics(
        ticker="ASHOKLEY",
        sector="Industrials",
        industry="Auto Manufacturers",
        years_available=5,
        net_income=100.0,
        net_income_series=[80.0, 90.0, 100.0],
        operating_cash_flow=50.0,
        ocf_series=[10.0, 20.0, 50.0],
        debt_series=[50.0, 200.0, 350.0],
        debt=350.0,
        debt_equity=4.49,
        interest_coverage=1.2,
        roe=8.0,
        roce=10.0,
        ocf_to_pat=0.5,
    )
    result = apply_hard_filters(metrics, _validation_ok())
    assert result.status == "HARD_EXCLUDE"
    joined = " ".join(result.reasons)
    assert "7.0x" not in joined
    assert "D/E" in joined


def test_crisil_not_classified_as_bank() -> None:
    from stockbot.portfolio_screener.issuer_routing import classify_issuer

    metrics = StockMetrics(
        ticker="CRISIL",
        sector="Financial Services",
        industry="Financial Data & Stock Exchanges",
        years_available=5,
    )
    assert classify_issuer(metrics) == "RATING_ANALYTICS"
