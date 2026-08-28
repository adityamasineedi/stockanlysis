"""Stage 2 — configurable hard exclusion filter."""

from __future__ import annotations

from stockbot.portfolio_screener.issuer_routing import (
    FINANCIAL_SCORECARD_ISSUERS,
    classify_issuer,
    fundamentals_fetch_failed,
    is_loss_making,
)
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
    issuer = classify_issuer(metrics)
    is_financial = issuer in FINANCIAL_SCORECARD_ISSUERS
    loss_making = issuer == "LOSS_MAKING_GROWTH" or is_loss_making(metrics)

    if fundamentals_fetch_failed(metrics):
        return HardFilterResult(
            ticker=metrics.ticker,
            status="DATA_UNAVAILABLE",
            reasons=["fundamentals fetch failed — no investability conclusion"],
            valuation_risk=valuation_risk,
            human_override=human_override,
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

    # Persistent negative OCF — skip for banks (deposit/loan flows dominate CFO)
    if not is_financial:
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

    # Leverage / interest coverage — skip for financials and loss-makers (IC is
    # not a meaningful solvency test when EBIT/PAT is negative).
    if not is_financial and not loss_making:
        if (
            metrics.interest_coverage is not None
            and metrics.debt is not None
            and metrics.debt > 0
            and metrics.interest_coverage < thresholds.min_interest_coverage
        ):
            # Utilities: hard-exclude only if coverage is truly broken (< 1.0)
            floor = 1.0 if issuer == "UTILITY" else thresholds.min_interest_coverage
            if metrics.interest_coverage < floor:
                reasons.append(
                    f"weak interest coverage {metrics.interest_coverage:.2f} < {floor}"
                )

        debt_series = series_present(metrics.debt_series)
        max_de = (
            thresholds.max_debt_equity * 1.2
            if issuer == "UTILITY"
            else thresholds.max_debt_equity
        )
        if metrics.debt_equity is not None and metrics.debt_equity > max_de:
            if len(debt_series) >= 3 and debt_series[0] > 0:
                d0, d_last = debt_series[0], debt_series[-1]
                if d0 >= d_last * 0.25:
                    reasons.append(
                        f"very high D/E {metrics.debt_equity:.2f} with rising absolute "
                        f"debt ({d0:.0f}→{d_last:.0f} Cr over {len(debt_series)}y)"
                    )
                else:
                    reasons.append(
                        f"very high consolidated D/E {metrics.debt_equity:.2f} sustained — "
                        "incompatible with 3-year profitable-compounder gate"
                    )
            else:
                reasons.append(
                    f"very high consolidated D/E {metrics.debt_equity:.2f} — "
                    "incompatible with 3-year profitable-compounder gate"
                )

        extreme_de = thresholds.max_debt_equity * (1.8 if issuer == "UTILITY" else 1.5)
        if (
            metrics.debt_equity is not None
            and metrics.debt_equity > extreme_de
            and not any("very high" in r or "D/E" in r for r in reasons)
        ):
            reasons.append(
                f"extreme D/E {metrics.debt_equity:.2f} — requires dedicated "
                "leveraged-cycle analysis instead of auto 3y compounding"
            )

        max_nd = thresholds.max_net_debt_ebitda * (1.3 if issuer == "UTILITY" else 1.0)
        if metrics.net_debt_ebitda is not None and metrics.net_debt_ebitda > max_nd:
            reasons.append(
                f"net debt/EBITDA {metrics.net_debt_ebitda:.2f} > {max_nd:.1f}"
            )

    if metrics.equity is not None and metrics.equity < 0:
        reasons.append("negative net worth")

    if (
        metrics.pledged_promoter_holding_pct is not None
        and metrics.pledged_promoter_holding_pct >= thresholds.max_promoter_pledge_pct
    ):
        reasons.append(
            f"pledged_promoter_holding_pct {metrics.pledged_promoter_holding_pct:.1f}% "
            f"of promoter holding (>= {thresholds.max_promoter_pledge_pct:.0f}%)"
        )

    # Persistent OCF<<PAT — skip financials and WC-heavy (handled as WATCH in routing)
    if not is_financial and issuer not in {"DEFENCE_EPC_PROJECT", "UTILITY"}:
        ocf = series_present(metrics.ocf_series)
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
        human_override=human_override,
    )
