"""Stage 2 — configurable hard exclusion filter."""

from __future__ import annotations

from stockbot.portfolio_screener.models import (
    DataValidationResult,
    HardFilterResult,
    StockMetrics,
)
from stockbot.portfolio_screener.score_utils import series_present
from stockbot.portfolio_screener.scoring_config import (
    HardFilterThresholds,
    ValuationRisk,
)


def classify_valuation_risk(
    metrics: StockMetrics,
    *,
    sector_pe_expensive: float | None = None,
) -> ValuationRisk:
    """Valuation risk label — never a hard exclusion by itself."""
    pe = metrics.pe
    if pe is None or pe <= 0:
        return "MEDIUM"
    expensive = sector_pe_expensive if sector_pe_expensive is not None else 40.0
    if pe >= expensive * 2.0:
        return "EXTREME"
    if pe >= expensive:
        return "HIGH"
    if pe >= expensive * 0.7:
        return "MEDIUM"
    return "LOW"


def apply_hard_filters(
    metrics: StockMetrics,
    validation: DataValidationResult,
    thresholds: HardFilterThresholds | None = None,
    *,
    sector_pe_expensive: float | None = None,
    human_override: bool = False,
) -> HardFilterResult:
    thresholds = thresholds or HardFilterThresholds()
    reasons: list[str] = []
    valuation_risk = classify_valuation_risk(
        metrics, sector_pe_expensive=sector_pe_expensive
    )

    if not validation.critical_ok:
        return HardFilterResult(
            ticker=metrics.ticker,
            status="DATA_INSUFFICIENT",
            reasons=[
                f"critical metric missing: {name} ({reason})"
                for name, reason in validation.missing_metrics.items()
                if name in thresholds.require_critical_metrics
            ]
            or ["critical data incomplete"],
            valuation_risk=valuation_risk,
            human_override=human_override,
        )

    # Persistent negative OCF
    ocf = series_present(metrics.ocf_series)
    if len(ocf) >= thresholds.persistent_negative_ocf_years:
        tail = ocf[-thresholds.persistent_negative_ocf_years :]
        if all(v < 0 for v in tail):
            reasons.append(
                f"persistent negative OCF for {thresholds.persistent_negative_ocf_years}+ years"
            )

    # Persistent losses
    pat = series_present(metrics.net_income_series)
    if len(pat) >= thresholds.persistent_loss_years:
        tail = pat[-thresholds.persistent_loss_years :]
        if all(v < 0 for v in tail):
            reasons.append(
                f"persistent losses for {thresholds.persistent_loss_years}+ years"
            )

    # Interest coverage
    if (
        metrics.interest_coverage is not None
        and metrics.debt is not None
        and metrics.debt > 0
        and metrics.interest_coverage < thresholds.min_interest_coverage
    ):
        reasons.append(
            f"weak interest coverage {metrics.interest_coverage:.2f} "
            f"< {thresholds.min_interest_coverage}"
        )

    # Debt surge + high leverage
    debt_series = series_present(metrics.debt_series)
    if len(debt_series) >= 3 and debt_series[0] > 0:
        growth = debt_series[-1] / debt_series[0]
        if growth >= 3.0 and (
            metrics.debt_equity is not None
            and metrics.debt_equity > thresholds.max_debt_equity
        ):
            reasons.append(
                f"severe debt increase ({growth:.1f}x) with D/E "
                f"{metrics.debt_equity:.2f}"
            )

    if (
        metrics.debt_equity is not None
        and metrics.debt_equity > thresholds.max_debt_equity * 1.5
    ):
        reasons.append(f"extreme D/E {metrics.debt_equity:.2f}")

    if (
        metrics.net_debt_ebitda is not None
        and metrics.net_debt_ebitda > thresholds.max_net_debt_ebitda
    ):
        reasons.append(
            f"net debt/EBITDA {metrics.net_debt_ebitda:.2f} > "
            f"{thresholds.max_net_debt_ebitda}"
        )

    if metrics.equity is not None and metrics.equity < 0:
        reasons.append("negative net worth")

    # Governance
    if (
        metrics.promoter_pledge_pct is not None
        and metrics.promoter_pledge_pct >= thresholds.max_promoter_pledge_pct
    ):
        reasons.append(
            f"promoter pledge {metrics.promoter_pledge_pct:.1f}% of promoter holding"
        )

    # Earnings quality: large persistent OCF vs PAT gap (PAT positive, OCF weak)
    if len(ocf) >= 3 and len(pat) >= 3:
        recent_pat = pat[-3:]
        recent_ocf = ocf[-3:]
        if all(p > 0 for p in recent_pat) and all(
            o < p * 0.3 for o, p in zip(recent_ocf, recent_pat, strict=True)
        ):
            reasons.append("persistent OCF << PAT — severe earnings quality gap")

    if reasons and not (human_override and thresholds.allow_human_override):
        return HardFilterResult(
            ticker=metrics.ticker,
            status="HARD_EXCLUDE",
            reasons=reasons,
            valuation_risk=valuation_risk,
            human_override=False,
        )

    if reasons and human_override and thresholds.allow_human_override:
        return HardFilterResult(
            ticker=metrics.ticker,
            status="PASS",
            reasons=[f"HUMAN_OVERRIDE: {r}" for r in reasons],
            valuation_risk=valuation_risk,
            human_override=True,
        )

    return HardFilterResult(
        ticker=metrics.ticker,
        status="PASS",
        reasons=[],
        valuation_risk=valuation_risk,
        human_override=False,
    )
