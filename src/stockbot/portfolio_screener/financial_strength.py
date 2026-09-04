"""Financial strength score (0–100)."""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    finalize_pillar,
    linear_score,
    series_present,
    weighted_mean,
)


def score_financial_strength(metrics: StockMetrics) -> float:
    from stockbot.portfolio_screener.issuer_routing import (
        FINANCIAL_SCORECARD_ISSUERS,
        classify_issuer,
    )

    issuer = classify_issuer(metrics)
    if issuer in FINANCIAL_SCORECARD_ISSUERS:
        # Avoid punishing banks for deposit/borrowings leverage.
        parts = [
            (linear_score(metrics.roe, bad=5.0, good=18.0), 0.45),
            (linear_score(metrics.roce, bad=5.0, good=15.0), 0.25),
            (
                80.0
                if metrics.net_income is not None and metrics.net_income > 0
                else 30.0,
                0.30,
            ),
        ]
        score, coverage = weighted_mean(parts)
        return finalize_pillar(score, coverage)

    # Utilities: regulated leverage — softer industrial curves.
    if issuer == "UTILITY":
        de_bad, de_good = 4.0, 0.8
        nd_bad = 6.0
        ic_bad, ic_good = 1.0, 6.0
    else:
        de_bad, de_good = 2.5, 0.2
        nd_bad = 4.0
        ic_bad, ic_good = 1.5, 10.0

    debt_trend_score: float | None = None
    debt = series_present(metrics.debt_series)
    if len(debt) >= 2 and debt[0] > 0:
        ratio = debt[-1] / debt[0]
        debt_trend_score = linear_score(ratio, bad=2.5, good=0.7, higher_is_better=False)
    elif metrics.debt is not None and metrics.debt == 0:
        debt_trend_score = 100.0

    cash_score: float | None = None
    if metrics.cash is not None and metrics.debt is not None:
        if metrics.debt <= 0:
            cash_score = 100.0
        else:
            cash_score = linear_score(
                metrics.cash / max(metrics.debt, 1e-9),
                bad=0.1,
                good=1.0,
            )

    # FCF polarity lives in cash_flow_quality — do not re-score it here.
    # D/E is owned here (not again in BS / risk).
    parts = [
        (linear_score(metrics.debt_equity, bad=de_bad, good=de_good, higher_is_better=False), 0.30),
        (
            linear_score(metrics.net_debt_ebitda, bad=nd_bad, good=0.0, higher_is_better=False),
            0.25,
        ),
        (linear_score(metrics.interest_coverage, bad=ic_bad, good=ic_good), 0.25),
        (linear_score(metrics.current_ratio, bad=0.8, good=2.0), 0.10),
        (debt_trend_score, 0.05),
        (cash_score, 0.05),
    ]
    score, coverage = weighted_mean(parts)
    return finalize_pillar(score, coverage)
