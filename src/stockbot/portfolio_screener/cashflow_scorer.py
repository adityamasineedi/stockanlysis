"""Cash-flow quality score (0–100)."""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    finalize_pillar,
    linear_score,
    series_present,
    stability_score,
    weighted_mean,
)

_WC_TIMING_ISSUERS = frozenset({"DEFENCE_EPC_PROJECT", "EPC_PROJECT_BUSINESS", "UTILITY"})


def score_cash_flow(metrics: StockMetrics) -> float:
    from stockbot.portfolio_screener.issuer_routing import (
        FINANCIAL_SCORECARD_ISSUERS,
        classify_issuer,
        is_loss_making,
    )

    issuer = classify_issuer(metrics)

    # Banks / NBFCs: deposit/loan CFO is not industrial OCF/PAT.
    if issuer in FINANCIAL_SCORECARD_ISSUERS:
        return 65.0

    ocf_consistency: float | None = None
    ocf = series_present(metrics.ocf_series)
    if ocf:
        positive_ratio = sum(1 for v in ocf if v > 0) / len(ocf)
        ocf_consistency = positive_ratio * 100.0

    fcf_consistency: float | None = None
    fcf = series_present(metrics.fcf_series)
    if fcf:
        positive_ratio = sum(1 for v in fcf if v > 0) / len(fcf)
        fcf_consistency = positive_ratio * 100.0

    # Loss-makers: OCF/PAT of two negatives is meaningless — path only.
    if issuer == "LOSS_MAKING_GROWTH" or is_loss_making(metrics):
        parts = [
            (ocf_consistency, 0.60),
            (stability_score(metrics.ocf_series, prefer_positive=True), 0.40),
        ]
        score, coverage = weighted_mean(parts)
        return finalize_pillar(score, coverage)

    # Defence / EPC / utility: WC timing — soften OCF/PAT anchors, lean on consistency.
    if issuer in _WC_TIMING_ISSUERS:
        parts = [
            (linear_score(metrics.ocf_to_pat, bad=-0.5, good=1.0), 0.20),
            (linear_score(metrics.fcf_to_pat, bad=-0.3, good=0.8), 0.15),
            (ocf_consistency, 0.30),
            (fcf_consistency, 0.15),
            (stability_score(metrics.ocf_series, prefer_positive=True), 0.20),
        ]
        score, coverage = weighted_mean(parts)
        return finalize_pillar(score, coverage)

    parts = [
        (linear_score(metrics.ocf_to_pat, bad=0.3, good=1.2), 0.30),
        (linear_score(metrics.fcf_to_pat, bad=0.0, good=1.0), 0.20),
        (linear_score(metrics.fcf_margin, bad=-0.05, good=0.15), 0.15),
        (ocf_consistency, 0.15),
        (fcf_consistency, 0.10),
        (stability_score(metrics.ocf_series, prefer_positive=True), 0.10),
    ]
    score, coverage = weighted_mean(parts)
    return finalize_pillar(score, coverage)
