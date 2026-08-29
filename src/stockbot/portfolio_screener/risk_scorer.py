"""Risk score (0–100). Higher = lower risk."""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    clamp,
    linear_score,
    stability_score,
    weighted_mean,
)
from stockbot.portfolio_screener.scoring_config import ValuationRisk

_VALUATION_RISK_SCORE = {
    "LOW": 90.0,
    "MEDIUM": 70.0,
    "HIGH": 40.0,
    "EXTREME": 15.0,
}


def score_risk(
    metrics: StockMetrics,
    *,
    valuation_risk: ValuationRisk = "MEDIUM",
    cyclical_sector: bool = False,
) -> float:
    governance = 100.0
    if metrics.pledged_promoter_holding_pct is not None:
        governance = linear_score(
            metrics.pledged_promoter_holding_pct,
            bad=50.0,
            good=0.0,
            higher_is_better=False,
        ) or 50.0

    size_score = None
    if metrics.market_cap_cr is not None:
        size_score = linear_score(metrics.market_cap_cr, bad=500.0, good=20000.0)

    liquidity_proxy = size_score  # without volume data, size is the proxy

    cyclicality_penalty = 55.0 if cyclical_sector else 80.0

    parts = [
        (linear_score(metrics.debt_equity, bad=2.5, good=0.2, higher_is_better=False), 0.20),
        (stability_score(metrics.net_income_series), 0.20),
        (stability_score(metrics.operating_margin_series), 0.15),
        (_VALUATION_RISK_SCORE.get(valuation_risk, 70.0), 0.15),
        (governance, 0.15),
        (size_score, 0.10),
        (liquidity_proxy, 0.05),
    ]
    score, coverage = weighted_mean(parts)
    blended = score * (0.6 + 0.4 * coverage)
    # Blend in cyclicality as a soft ceiling, not a silent metric rewrite.
    blended = 0.85 * blended + 0.15 * cyclicality_penalty
    return clamp(blended)


def score_balance_sheet(metrics: StockMetrics) -> float:
    """Narrow balance-sheet pillar (separate from broader financial strength)."""
    parts = [
        (linear_score(metrics.debt_equity, bad=2.0, good=0.3, higher_is_better=False), 0.40),
        (linear_score(metrics.current_ratio, bad=0.9, good=2.0), 0.30),
        (
            100.0
            if metrics.net_debt is not None and metrics.net_debt <= 0
            else linear_score(metrics.net_debt_ebitda, bad=3.5, good=0.0, higher_is_better=False),
            0.30,
        ),
    ]
    score, coverage = weighted_mean(parts)
    return clamp(score * (0.6 + 0.4 * coverage))


def score_earnings_quality(metrics: StockMetrics) -> float:
    from stockbot.portfolio_screener.issuer_routing import (
        FINANCIAL_SCORECARD_ISSUERS,
        classify_issuer,
    )

    if classify_issuer(metrics) in FINANCIAL_SCORECARD_ISSUERS:
        # CFO/PAT is meaningless for banks — neutral mid score
        return 65.0

    parts = [
        (linear_score(metrics.ocf_to_pat, bad=0.3, good=1.2), 0.45),
        (linear_score(metrics.fcf_to_pat, bad=0.0, good=1.0), 0.25),
        (stability_score(metrics.eps_series), 0.30),
    ]
    score, coverage = weighted_mean(parts)
    return clamp(score * (0.6 + 0.4 * coverage))
