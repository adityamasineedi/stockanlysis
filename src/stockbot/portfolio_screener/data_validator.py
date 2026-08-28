"""Stage 1 — data validation before scoring."""

from __future__ import annotations

from stockbot.portfolio_screener.models import DataValidationResult, StockMetrics
from stockbot.portfolio_screener.score_utils import clamp
from stockbot.portfolio_screener.scoring_config import (
    ConfidenceLevel,
    HardFilterThresholds,
)

# Soft metrics improve completeness but are not critical for DATA_INSUFFICIENT.
_SOFT_METRICS = (
    "market_cap_cr",
    "ebitda",
    "ebit",
    "free_cash_flow",
    "roce",
    "debt",
    "cash",
    "net_debt",
    "interest_coverage",
    "pe",
    "pb",
    "ev_ebitda",
    "dividend_yield_pct",
    "promoter_holding_pct",
    "pledged_promoter_holding_pct",
    "share_dilution_pct",
    "forward_pe",
    "industry",
    "revenue_cagr_3y",
    "eps_cagr_3y",
)


def _metric_present(metrics: StockMetrics, name: str) -> bool:
    value = getattr(metrics, name, None)
    return value is not None


def _detect_contradictions(metrics: StockMetrics) -> list[str]:
    issues: list[str] = []
    if (
        metrics.net_income is not None
        and metrics.operating_cash_flow is not None
        and metrics.net_income > 0
        and metrics.operating_cash_flow < 0
        and abs(metrics.operating_cash_flow) > abs(metrics.net_income) * 2
    ):
        issues.append("PAT positive but OCF deeply negative — earnings quality concern")

    if metrics.equity is not None and metrics.equity < 0:
        issues.append("Negative net worth / equity")

    if (
        metrics.revenue is not None
        and metrics.net_income is not None
        and metrics.revenue > 0
        and metrics.net_income > metrics.revenue
    ):
        issues.append("Net income exceeds revenue — possible data inconsistency")

    if metrics.pe is not None and metrics.pe < 0:
        issues.append("Negative P/E with reported metrics — loss-making or data quirk")

    if (
        metrics.debt is not None
        and metrics.cash is not None
        and metrics.net_debt is not None
        and abs((metrics.debt - metrics.cash) - metrics.net_debt) > 1.0
    ):
        issues.append("net_debt does not reconcile with debt − cash")

    return issues


def validate_stock_data(
    metrics: StockMetrics,
    thresholds: HardFilterThresholds | None = None,
) -> DataValidationResult:
    thresholds = thresholds or HardFilterThresholds()

    critical_missing: dict[str, str] = {}
    for name in thresholds.require_critical_metrics:
        if not _metric_present(metrics, name):
            critical_missing[name] = metrics.missing.get(name, "unavailable")

    # Key trio (ROE / leverage / cash conversion): allow a single gap so a
    # Screener ratios omission doesn't alone force DATA_INSUFFICIENT when
    # P&L+BS still support scoring. ≥2 missing → treat as critical failure.
    key_trio_missing: dict[str, str] = {}
    for name in thresholds.key_trio_metrics:
        if not _metric_present(metrics, name):
            key_trio_missing[name] = metrics.missing.get(name, "unavailable")
    if len(key_trio_missing) >= 2:
        critical_missing.update(key_trio_missing)

    soft_missing = 0
    soft_total = len(_SOFT_METRICS)
    for name in _SOFT_METRICS:
        if not _metric_present(metrics, name):
            soft_missing += 1

    # Completeness counts core critical + key trio + soft metrics
    trio_total = len(thresholds.key_trio_metrics)
    trio_present = trio_total - len(key_trio_missing)
    critical_total = len(thresholds.require_critical_metrics)
    critical_present = critical_total - len(
        {k: v for k, v in critical_missing.items() if k in thresholds.require_critical_metrics}
    )
    denom = critical_total + trio_total + soft_total
    completeness = 0.0
    if denom > 0:
        completeness = (
            (critical_present + trio_present + (soft_total - soft_missing)) / denom
        ) * 100.0

    contradictions = _detect_contradictions(metrics)
    # Quality penalises contradictions and thin history.
    quality = completeness
    quality -= 10.0 * len(contradictions)
    if metrics.years_available < 3:
        quality -= 15.0
    elif metrics.years_available < 5:
        quality -= 5.0
    quality = clamp(quality)

    confidence: ConfidenceLevel
    if completeness >= 80 and not contradictions and metrics.years_available >= 4:
        confidence = "HIGH"
    elif completeness >= 55 and len(contradictions) <= 1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Single key-trio gap stays in missing_metrics for data_concerns, but does
    # not alone set critical_ok=False (unless folded in via ≥2 rule above).
    all_missing = {
        **critical_missing,
        **{k: v for k, v in key_trio_missing.items() if k not in critical_missing},
        **{
            k: v
            for k, v in metrics.missing.items()
            if k not in critical_missing and k not in key_trio_missing
        },
    }

    return DataValidationResult(
        ticker=metrics.ticker,
        data_completeness_score=round(completeness, 2),
        data_quality_score=round(quality, 2),
        data_confidence=confidence,
        missing_metrics=all_missing,
        contradictions=contradictions,
        critical_ok=len(critical_missing) == 0,
    )
