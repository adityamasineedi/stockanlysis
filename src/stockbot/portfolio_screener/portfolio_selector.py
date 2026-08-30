"""Final candidate selection with diversification + correlation constraints."""

from __future__ import annotations

from stockbot.portfolio_screener.correlation import CorrelationInfo
from stockbot.portfolio_screener.models import (
    AIRankResult,
    QuantScreenResult,
    StockScreenRecord,
)
from stockbot.portfolio_screener.scoring_config import (
    CandidateBand,
    PortfolioConstraints,
    ScreenerRunConfig,
)


def candidate_band(score: float, constraints: PortfolioConstraints) -> CandidateBand:
    if score >= constraints.strong_candidate_min:
        return "STRONG_CANDIDATE"
    if score >= constraints.candidate_min:
        return "CANDIDATE"
    if score >= constraints.watchlist_min:
        return "WATCHLIST"
    return "REMOVE"


def combine_scores(
    quant: float,
    ai: float | None,
    config: ScreenerRunConfig,
) -> float:
    config.blend.validate()
    if ai is None:
        return quant
    return quant * config.blend.quant_weight + ai * config.blend.ai_weight


def build_screen_record(
    quant: QuantScreenResult,
    *,
    ai: AIRankResult | None = None,
    final_score: float | None = None,
    corr: CorrelationInfo | None = None,
) -> StockScreenRecord:
    rec = StockScreenRecord(
        ticker=quant.ticker,
        sector=quant.sector,
        industry=quant.industry,
        hard_filter_status=quant.hard_filter.status,
        hard_filter_reason=list(quant.hard_filter.reasons),
        quant_score=quant.final_quant_score,
        base_score=quant.base_score,
        red_flag_penalty=quant.red_flag_penalty,
        ai_score=ai.ai_score if ai else None,
        final_score=final_score,
        quality_score=quant.components.business_quality,
        growth_score=quant.components.growth,
        valuation_score=quant.components.valuation,
        financial_strength_score=quant.components.financial_strength,
        risk_score=quant.components.risk,
        cash_flow_score=quant.components.cash_flow_quality,
        capital_efficiency_score=quant.components.capital_efficiency,
        balance_sheet_score=quant.components.balance_sheet,
        earnings_quality_score=quant.components.earnings_quality,
        valuation_risk=quant.components.valuation_risk,
        growth_trend=quant.components.growth_trend,
        data_confidence=quant.data_validation.data_confidence,
        data_completeness=quant.data_validation.data_completeness_score,
        data_quality=quant.data_validation.data_quality_score,
        red_flags=[f"{f.severity}:{f.code}" for f in quant.red_flags],
        key_risks=list(quant.hard_filter.reasons),
        sent_to_ai=ai is not None,
        ai_detail=ai,
        price_at_scan=quant.current_price_abs,
        scanned_at=quant.data_timestamp,
    )
    if corr is not None:
        rec.correlation_risk = corr.correlation_risk
        rec.correlation_cluster = corr.correlation_cluster
    if ai is not None and ai.key_risk:
        rec.key_risks.append(ai.key_risk)
    return rec


def select_portfolio(
    quant_results: list[QuantScreenResult],
    ai_results: list[AIRankResult],
    corr_map: dict[str, CorrelationInfo],
    config: ScreenerRunConfig,
) -> tuple[list[StockScreenRecord], list[StockScreenRecord], str]:
    """Returns (selected, rejected, status).

    Does not force exactly min or max — confidence-based with diversification.
    Never fills artificially below quality threshold.
    """
    constraints = config.constraints
    ai_by_ticker = {a.ticker: a for a in ai_results}

    records: list[StockScreenRecord] = []
    for q in quant_results:
        ai = ai_by_ticker.get(q.ticker)
        if q.hard_filter.status == "HARD_EXCLUDE":
            rec = build_screen_record(q, ai=None, final_score=q.final_quant_score, corr=corr_map.get(q.ticker))
            rec.selection_status = "HARD_EXCLUDED"
            rec.rejection_reason = "; ".join(q.hard_filter.reasons) or "hard exclude"
            records.append(rec)
            continue
        if q.hard_filter.status == "DATA_INSUFFICIENT":
            rec = build_screen_record(q, ai=None, final_score=None, corr=corr_map.get(q.ticker))
            rec.selection_status = "HARD_EXCLUDED"
            rec.rejection_reason = "DATA_INSUFFICIENT: " + (
                "; ".join(q.hard_filter.reasons) or "critical data missing"
            )
            records.append(rec)
            continue

        final = combine_scores(
            q.final_quant_score,
            ai.ai_score if ai else None,
            config,
        )
        rec = build_screen_record(
            q,
            ai=ai,
            final_score=round(final, 2),
            corr=corr_map.get(q.ticker),
        )
        if ai is not None and not ai.keep_for_deep_analysis:
            rec.selection_status = "AI_REJECTED"
            rec.rejection_reason = ai.key_risk or "AI recommended skip"
            records.append(rec)
            continue

        band = candidate_band(final, constraints)
        rec.candidate_band = band
        if band == "REMOVE":
            rec.selection_status = "BELOW_THRESHOLD"
            rec.rejection_reason = f"final_score {final:.1f} < {constraints.min_final_score}"
        else:
            rec.selection_status = "SELECTED"  # provisional — diversification may drop
            rec.selection_reason = band
        records.append(rec)

    # Rank provisional candidates
    provisional = [
        r
        for r in records
        if r.selection_status == "SELECTED" and r.final_score is not None
    ]
    provisional.sort(key=lambda r: r.final_score or 0.0, reverse=True)

    selected: list[StockScreenRecord] = []
    selected_tickers: set[str] = set()
    sector_counts: dict[str, int] = {}
    industry_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}

    def _can_add(rec: StockScreenRecord, *, allow_diversification_relax: bool) -> tuple[bool, str]:
        if len(selected) >= constraints.max_stocks:
            return False, "max stocks reached"
        sector = rec.sector or "Unknown"
        industry = rec.industry or "Unknown"
        max_sector = int(constraints.max_stocks * constraints.max_sector_weight)
        max_industry = int(constraints.max_stocks * constraints.max_industry_weight)
        max_sector = max(1, max_sector)
        max_industry = max(1, max_industry)

        if sector_counts.get(sector, 0) >= max_sector and not allow_diversification_relax:
            return False, f"sector cap ({sector})"
        if industry_counts.get(industry, 0) >= max_industry and not allow_diversification_relax:
            return False, f"industry cap ({industry})"

        cluster = rec.correlation_cluster
        if cluster and cluster_counts.get(cluster, 0) >= constraints.max_per_correlation_cluster:
            return False, f"correlation cluster {cluster}"
        return True, ""

    def _commit(rec: StockScreenRecord, reason: str) -> None:
        rec.selection_status = "SELECTED"
        rec.selection_reason = reason
        rec.rejection_reason = ""
        selected.append(rec)
        selected_tickers.add(rec.ticker)
        sector_counts[rec.sector or "Unknown"] = sector_counts.get(rec.sector or "Unknown", 0) + 1
        industry_counts[rec.industry or "Unknown"] = (
            industry_counts.get(rec.industry or "Unknown", 0) + 1
        )
        if rec.correlation_cluster:
            cluster_counts[rec.correlation_cluster] = (
                cluster_counts.get(rec.correlation_cluster, 0) + 1
            )

    # First pass: prefer STRONG then CANDIDATE then WATCHLIST, respecting caps
    for band_name in ("STRONG_CANDIDATE", "CANDIDATE", "WATCHLIST"):
        for rec in provisional:
            if rec.candidate_band != band_name:
                continue
            if rec.ticker in selected_tickers or rec.selection_status != "SELECTED":
                continue
            ok, why = _can_add(rec, allow_diversification_relax=False)
            if not ok:
                rec.selection_status = "DIVERSIFICATION_DROPPED"
                rec.rejection_reason = why
                continue
            _commit(rec, band_name)

    # Diversification substitution: pull next-best dropped names that fit caps
    if len(selected) < constraints.max_stocks:
        dropped = [
            r
            for r in provisional
            if r.selection_status == "DIVERSIFICATION_DROPPED"
            and r.ticker not in selected_tickers
        ]
        dropped.sort(key=lambda r: r.final_score or 0.0, reverse=True)
        for rec in dropped:
            if len(selected) >= constraints.max_stocks:
                break
            ok, _ = _can_add(rec, allow_diversification_relax=False)
            if not ok:
                continue
            _commit(rec, "DIVERSIFICATION_SUBSTITUTION")

    # Assign ranks
    selected.sort(key=lambda r: r.final_score or 0.0, reverse=True)
    for i, rec in enumerate(selected, start=1):
        rec.ranking = i
        rec.selection_status = "SELECTED"

    rejected: list[StockScreenRecord] = []
    for rec in records:
        if rec.ticker in selected_tickers:
            continue
        if rec.selection_status == "SELECTED":
            rec.selection_status = "DIVERSIFICATION_DROPPED"
            rec.rejection_reason = rec.rejection_reason or "not selected after constraints"
        if (
            not rec.sent_to_ai
            and rec.hard_filter_status == "PASS"
            and rec.ticker not in ai_by_ticker
            and rec.selection_status
            not in (
                "HARD_EXCLUDED",
                "BELOW_THRESHOLD",
                "AI_REJECTED",
                "DIVERSIFICATION_DROPPED",
            )
        ):
            rec.selection_status = "NOT_SENT_TO_AI"
        rejected.append(rec)

    if len(selected) < constraints.min_stocks:
        status = "INSUFFICIENT_HIGH_QUALITY_CANDIDATES"
    else:
        status = "READY_FOR_DEEP_ANALYSIS"

    return selected, rejected, status
