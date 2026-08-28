"""Compact brief enrichment — metadata, prescan quant summary, curated news.

Uses data already fetched in assemble_brief() plus one cheap yfinance info
pull. Does not re-fetch Screener fundamentals.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from stockbot.models import (
    Brief,
    BriefMetadata,
    NewsItems,
    NewsSummaryItem,
    PrescanSummary,
    PriceData,
    RedFlag,
    StreetConsensus,
    Technicals,
    TickerInfo,
)
from stockbot.portfolio_screener.issuer_routing import decide_eligibility_route
from stockbot.portfolio_screener.metrics import extract_metrics, fetch_market_metadata
from stockbot.portfolio_screener.portfolio_selector import candidate_band
from stockbot.portfolio_screener.quant_engine import compute_quant_score
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig

Stage2Mode = Literal["LITE", "FULL"]
_LITE_ISSUER_CLASSES = frozenset({"NON_FINANCIAL"})

NEWS_SUMMARY_LIMIT = 15
NEWS_LOOKBACK_DAYS = 365

_ORDER_BOOK_KEYWORDS = (
    "order book",
    "order inflow",
    "order pipeline",
    "contract",
    "mou",
    "deal size",
    "backlog",
    "order win",
)
_MANAGEMENT_KEYWORDS = (
    "ceo",
    "md resign",
    "cfo resign",
    "board",
    "management",
    "director",
)
_GUIDANCE_KEYWORDS = ("guidance", "profit warning", "outlook", "forecast")
_MNA_KEYWORDS = ("acquisition", "merger", "strategic partnership", "takeover")
_BROKER_KEYWORDS = ("broker", "target price", "upgrade", "downgrade", "initiates")


def _headline_source(headline: str) -> str:
    for sep in (" - ", " | ", " — "):
        if sep in headline:
            tail = headline.rsplit(sep, 1)[-1].strip()
            if 2 <= len(tail) <= 48:
                return tail
    return "unknown"


def _classify_news_type(headline: str) -> str:
    lowered = headline.lower()
    if any(k in lowered for k in _ORDER_BOOK_KEYWORDS):
        return "order_book"
    if any(k in lowered for k in _MNA_KEYWORDS):
        return "corporate_action"
    if any(k in lowered for k in _GUIDANCE_KEYWORDS):
        return "guidance"
    if any(k in lowered for k in _MANAGEMENT_KEYWORDS):
        return "management"
    if any(k in lowered for k in _BROKER_KEYWORDS):
        return "broker_view"
    return "general"


def _news_note(headline: str, news_type: str) -> str:
    if news_type == "order_book":
        return "Check order-book / contract context against filings — headline only."
    if news_type == "broker_view":
        return "External view — not verified; do not treat targets as facts."
    if news_type == "management":
        return "Governance / leadership event — verify in exchange filings."
    if news_type == "guidance":
        return "Management or media guidance — cross-check with results."
    if news_type == "corporate_action":
        return "Corporate action headline — verify structure and funding."
    return "Supplement fundamentals only — headline summary, not verified fact."


def _news_rank(item: RedFlag, *, today: date) -> float:
    age_days = max(0, (today - item.published_date).days)
    recency = max(0.0, 1.0 - age_days / NEWS_LOOKBACK_DAYS)
    news_type = _classify_news_type(item.headline)
    type_boost = {
        "order_book": 0.35,
        "management": 0.25,
        "guidance": 0.20,
        "corporate_action": 0.20,
        "broker_view": 0.10,
        "general": 0.0,
    }.get(news_type, 0.0)
    return recency + type_boost


def build_news_summary(news: NewsItems | None) -> tuple[NewsSummaryItem, ...]:
    """Rank and trim general news into compact bullets for LLM stages."""
    if news is None or not news.general:
        return ()

    today = datetime.now(UTC).date()
    cutoff = today - timedelta(days=NEWS_LOOKBACK_DAYS)
    candidates = [item for item in news.general if item.published_date >= cutoff]
    ranked = sorted(candidates, key=lambda item: _news_rank(item, today=today), reverse=True)

    summary: list[NewsSummaryItem] = []
    seen: set[str] = set()
    for item in ranked:
        key = item.headline.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        news_type = _classify_news_type(item.headline)
        summary.append(
            NewsSummaryItem(
                date=item.published_date.isoformat(),
                source=_headline_source(item.headline),
                news_type=news_type,
                headline=item.headline,
                note=_news_note(item.headline, news_type),
            )
        )
        if len(summary) >= NEWS_SUMMARY_LIMIT:
            break
    return tuple(summary)


def build_brief_metadata(
    ticker: TickerInfo,
    price: PriceData,
    technicals: Technicals,
) -> BriefMetadata:
    meta = fetch_market_metadata(ticker.symbol)
    return BriefMetadata(
        ticker=ticker.symbol,
        company_name=ticker.company_name,
        sector=str(meta["sector"]) if meta.get("sector") else None,
        industry=str(meta["industry"]) if meta.get("industry") else None,
        market_cap_cr=float(meta["market_cap_cr"])
        if meta.get("market_cap_cr") is not None
        else None,
        ttm_pe=float(meta["trailing_pe"]) if meta.get("trailing_pe") is not None else None,
        ttm_pb=float(meta["pb"]) if meta.get("pb") is not None else None,
        price=round(price.current_price_abs, 2),
        price_date=price.price_date.isoformat(),
        range_52w_low=round(price.week52_low_abs, 2),
        range_52w_high=round(price.week52_high_abs, 2),
        rsi_14=round(technicals.rsi14, 2) if technicals.rsi14 is not None else None,
    )


def _street_consensus_tension(price_vs_target_pct: float | None) -> str:
    if price_vs_target_pct is None:
        return "MISSING"
    if abs(price_vs_target_pct) >= 15.0:
        return "HIGH"
    if abs(price_vs_target_pct) >= 8.0:
        return "MEDIUM"
    return "NONE"


def build_street_consensus(ticker: TickerInfo, price: PriceData) -> StreetConsensus:
    """yfinance analyst aggregate for tension diagnostics only."""
    meta = fetch_market_metadata(ticker.symbol)
    target_mean = meta.get("target_mean_price")
    target_low = meta.get("target_low_price")
    target_high = meta.get("target_high_price")
    analyst_count = meta.get("analyst_count")
    recommendation = meta.get("recommendation_key")

    target_mean_f = float(target_mean) if target_mean is not None else None
    price_vs_target: float | None = None
    if target_mean_f is not None and target_mean_f > 0:
        price_vs_target = round(
            (price.current_price_abs - target_mean_f) / target_mean_f * 100.0,
            2,
        )

    if analyst_count is None and target_mean_f is None:
        note = "No yfinance analyst consensus available — do not invent street targets."
    else:
        note = (
            "yfinance aggregate analyst data — unverified third-party estimates; "
            "use only as a tension check vs your own fair-value work."
        )

    return StreetConsensus(
        source="yfinance",
        analyst_count=int(analyst_count) if analyst_count is not None else None,
        recommendation_key=str(recommendation) if recommendation else None,
        target_mean_price=target_mean_f,
        target_low_price=float(target_low) if target_low is not None else None,
        target_high_price=float(target_high) if target_high is not None else None,
        price_vs_target_pct=price_vs_target,
        tension=_street_consensus_tension(price_vs_target),
        note=note,
    )


def build_prescan_summary(brief: Brief) -> PrescanSummary | None:
    """Quant-only prescan summary from brief fetch results (no eligibility AI)."""
    market_meta = fetch_market_metadata(brief.ticker.symbol)
    metrics = extract_metrics(
        brief.ticker,
        financials=brief.financials,
        price=brief.price,
        shareholding=brief.shareholding,
        market_meta=market_meta,
    )
    config = ScreenerRunConfig(skip_ai=True)
    quant = compute_quant_score(metrics, config)
    routing = decide_eligibility_route(metrics, quant)
    band = candidate_band(quant.final_quant_score, config.constraints)
    cash = routing.cash_conversion
    major_flags = tuple(
        f.code for f in quant.red_flags if f.severity in {"severe", "major", "moderate"}
    )[:8]

    return PrescanSummary(
        quant_score=round(quant.final_quant_score, 2),
        quality_score=round(quant.components.business_quality, 1),
        growth_score=round(quant.components.growth, 1),
        strength_score=round(quant.components.financial_strength, 1),
        band=band,
        issuer_class=routing.issuer_class,
        route=routing.route,
        eligibility_verdict=routing.eligibility,
        cash_conversion_status=cash.status,
        ocf_pat_current=cash.ocf_pat_current,
        ocf_pat_3y=cash.ocf_pat_3y,
        data_confidence=quant.data_validation.data_confidence,
        major_flags=major_flags,
    )


def enrich_brief(brief: Brief) -> Brief:
    metadata = build_brief_metadata(brief.ticker, brief.price, brief.technicals)
    prescan_summary = build_prescan_summary(brief)
    news_summary = build_news_summary(brief.news)
    street_consensus = build_street_consensus(brief.ticker, brief.price)
    return Brief(
        ticker=brief.ticker,
        price=brief.price,
        technicals=brief.technicals,
        financials=brief.financials,
        shareholding=brief.shareholding,
        news=brief.news,
        annual_report=brief.annual_report,
        missing=brief.missing,
        token_count=brief.token_count,
        confidence_ceiling=brief.confidence_ceiling,
        generated_at=brief.generated_at,
        metadata=metadata,
        prescan_summary=prescan_summary,
        news_summary=news_summary,
        street_consensus=street_consensus,
    )


def format_metadata_json(
    metadata: BriefMetadata | None,
    street_consensus: StreetConsensus | None = None,
) -> str:
    if metadata is None:
        return "MISSING: metadata block not built"
    payload = {
        "ticker": metadata.ticker,
        "company_name": metadata.company_name,
        "sector": metadata.sector,
        "industry": metadata.industry,
        "market_cap_cr": metadata.market_cap_cr,
        "ttm_pe": metadata.ttm_pe,
        "ttm_pb": metadata.ttm_pb,
        "price_stats": {
            "price": metadata.price,
            "price_date": metadata.price_date,
            "range_52w": [metadata.range_52w_low, metadata.range_52w_high],
            "rsi_14": metadata.rsi_14,
        },
        "street_consensus": format_street_consensus_payload(street_consensus),
    }
    return json.dumps(payload, indent=2)


def format_street_consensus_payload(
    consensus: StreetConsensus | None,
) -> dict[str, object]:
    if consensus is None:
        return {"tension": "MISSING", "note": "Street consensus block not built"}
    return {
        "source": consensus.source,
        "analyst_count": consensus.analyst_count,
        "recommendation_key": consensus.recommendation_key,
        "target_mean_price": consensus.target_mean_price,
        "target_low_price": consensus.target_low_price,
        "target_high_price": consensus.target_high_price,
        "price_vs_target_pct": consensus.price_vs_target_pct,
        "tension": consensus.tension,
        "note": consensus.note,
    }


def format_prescan_summary_json(summary: PrescanSummary | None) -> str:
    if summary is None:
        return "MISSING: prescan summary not built (financials unavailable)"
    payload = {
        "quant_score": summary.quant_score,
        "quality_score": summary.quality_score,
        "growth_score": summary.growth_score,
        "strength_score": summary.strength_score,
        "band": summary.band,
        "issuer_class": summary.issuer_class,
        "route": summary.route,
        "eligibility_verdict": summary.eligibility_verdict,
        "cash_conversion_status": summary.cash_conversion_status,
        "ocf_pat_current": summary.ocf_pat_current,
        "ocf_pat_3y": summary.ocf_pat_3y,
        "data_confidence": summary.data_confidence,
        "major_flags": list(summary.major_flags),
    }
    return json.dumps(payload, indent=2)


def format_news_summary_json(items: tuple[NewsSummaryItem, ...]) -> str:
    if not items:
        return "[]"
    payload = [
        {
            "date": item.date,
            "source": item.source,
            "type": item.news_type,
            "headline": item.headline,
            "note": item.note,
        }
        for item in items
    ]
    return json.dumps(payload, indent=2)


def stage2_mode_from_prescan(summary: PrescanSummary | None) -> tuple[Stage2Mode, tuple[str, ...]]:
    """Mirror analysis_routing LITE/FULL rules from a prescan summary."""
    if summary is None:
        return "FULL", ("prescan_summary missing",)

    reasons: list[str] = []
    if summary.eligibility_verdict and summary.eligibility_verdict != "AUTO_DEEP_ANALYSIS":
        reasons.append(f"eligibility={summary.eligibility_verdict}")
    if summary.issuer_class and summary.issuer_class not in _LITE_ISSUER_CLASSES:
        reasons.append(f"issuer_class={summary.issuer_class}")
    if summary.data_confidence and summary.data_confidence != "HIGH":
        reasons.append(f"data_confidence={summary.data_confidence}")
    if summary.major_flags:
        reasons.append(f"quant_red_flags={len(summary.major_flags)}")

    if not reasons:
        return "LITE", ("clean AUTO_DEEP prescan — lite Stage 2 eligible",)
    return "FULL", tuple(reasons)
