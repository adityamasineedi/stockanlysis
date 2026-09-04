"""Risk score (0–100). Higher = lower risk."""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    clamp,
    finalize_pillar,
    linear_score,
    stability_score,
    weighted_mean,
)
from stockbot.portfolio_screener.scoring_config import ValuationRisk


def score_risk(
    metrics: StockMetrics,
    *,
    valuation_risk: ValuationRisk = "MEDIUM",
    cyclical_sector: bool = False,
) -> float:
    """Governance / size / liquidity / cyclicality.

    D/E lives in financial_strength, earnings-path stability in earnings_quality,
    and rich P/E in the valuation pillar — do not re-hit them here.
    ``valuation_risk`` is accepted for call-site compatibility but unused.
    """
    _ = valuation_risk

    from stockbot.portfolio_screener.issuer_routing import (
        FINANCIAL_SCORECARD_ISSUERS,
        classify_issuer,
    )

    issuer = classify_issuer(metrics)

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

    liquidity_score = None
    if metrics.adv_inr_cr is not None:
        liquidity_score = linear_score(metrics.adv_inr_cr, bad=0.5, good=50.0)
    elif size_score is not None:
        liquidity_score = size_score  # fallback when ADV missing

    cyclicality_penalty = 55.0 if cyclical_sector else 80.0
    if issuer in FINANCIAL_SCORECARD_ISSUERS:
        # Banks: skip industrial margin-path noise; lean on size/liquidity/gov.
        parts = [
            (governance, 0.35),
            (size_score, 0.35),
            (liquidity_score, 0.30),
        ]
    else:
        parts = [
            (stability_score(metrics.operating_margin_series), 0.25),
            (governance, 0.30),
            (size_score, 0.25),
            (liquidity_score, 0.20),
        ]
    score, coverage = weighted_mean(parts)
    blended = finalize_pillar(score, coverage)
    blended = 0.85 * blended + 0.15 * cyclicality_penalty
    return clamp(blended)


def score_balance_sheet(metrics: StockMetrics) -> float:
    """Liquidity / net-debt slice — D/E lives in financial_strength only."""
    from stockbot.portfolio_screener.issuer_routing import (
        FINANCIAL_SCORECARD_ISSUERS,
        classify_issuer,
    )

    issuer = classify_issuer(metrics)
    if issuer in FINANCIAL_SCORECARD_ISSUERS:
        return 65.0

    nd_bad = 5.5 if issuer == "UTILITY" else 3.5
    parts = [
        (linear_score(metrics.current_ratio, bad=0.9, good=2.0), 0.50),
        (
            100.0
            if metrics.net_debt is not None and metrics.net_debt <= 0
            else linear_score(
                metrics.net_debt_ebitda,
                bad=nd_bad,
                good=0.0,
                higher_is_better=False,
            ),
            0.50,
        ),
    ]
    score, coverage = weighted_mean(parts)
    return finalize_pillar(score, coverage)


def score_earnings_quality(metrics: StockMetrics) -> float:
    from stockbot.portfolio_screener.issuer_routing import (
        FINANCIAL_SCORECARD_ISSUERS,
        classify_issuer,
    )

    if classify_issuer(metrics) in FINANCIAL_SCORECARD_ISSUERS:
        # CFO/PAT is meaningless for banks — neutral mid score
        return 65.0

    # Cash conversion (OCF/FCF) lives in cash_flow_quality only. Earnings
    # quality here is path stability of reported profits / EPS.
    parts = [
        (stability_score(metrics.eps_series), 0.60),
        (stability_score(metrics.net_income_series), 0.40),
    ]
    score, coverage = weighted_mean(parts)
    return finalize_pillar(score, coverage)
