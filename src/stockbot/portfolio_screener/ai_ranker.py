"""AI contextual ranking layer — cheap model, structured inputs only.

Does NOT perform full stock analysis. Cannot override HARD_EXCLUDE.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from anthropic import Anthropic

from stockbot.config import PROMPTS_DIR, settings
from stockbot.llm.client import call_anthropic_and_log
from stockbot.portfolio_screener.cost_tracker import ScreenerCostTracker
from stockbot.portfolio_screener.models import AIRankResult, QuantScreenResult
from stockbot.portfolio_screener.scoring_config import (
    ConfidenceLevel,
    ScreenerRunConfig,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = PROMPTS_DIR / "portfolio-screener-ranker-v1.md"

_DEFAULT_SYSTEM = """You are a portfolio pre-screener ranking agent for long-term equity investing.
Your job is ONLY to review quantitative screening results, flag inconsistencies /
contextual risks, and rank which stocks deserve expensive deep analysis.

Rules:
- Do NOT invent missing financial data.
- Do NOT produce BUY / WATCH / SKIP / fair value / target price.
- Do NOT promote any stock with hard_filter_status HARD_EXCLUDE or DATA_INSUFFICIENT.
- Prefer business quality, financial strength, earnings/cash-flow quality, and risk balance.
- Return ONLY a JSON array (no markdown fences) of objects with keys:
  ticker, rank, ai_score (0-100), confidence (HIGH|MEDIUM|LOW),
  keep_for_deep_analysis (bool), key_reason, key_risk, data_concerns (array of strings).
"""


def load_ranker_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return _DEFAULT_SYSTEM


def _payload_for_ai(results: list[QuantScreenResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "ticker": r.ticker,
                "sector": r.sector,
                "industry": r.industry,
                "hard_filter_status": r.hard_filter.status,
                "hard_filter_reasons": r.hard_filter.reasons,
                "quant_score": r.final_quant_score,
                "base_score": r.base_score,
                "red_flag_penalty": r.red_flag_penalty,
                "red_flags": [f"{f.severity}:{f.code}:{f.message}" for f in r.red_flags],
                "components": {
                    "business_quality": r.components.business_quality,
                    "financial_strength": r.components.financial_strength,
                    "growth": r.components.growth,
                    "growth_trend": r.components.growth_trend,
                    "cash_flow_quality": r.components.cash_flow_quality,
                    "capital_efficiency": r.components.capital_efficiency,
                    "valuation": r.components.valuation,
                    "valuation_risk": r.components.valuation_risk,
                    "valuation_confidence": r.components.valuation_confidence,
                    "balance_sheet": r.components.balance_sheet,
                    "earnings_quality": r.components.earnings_quality,
                    "risk": r.components.risk,
                    "moat_confidence": r.components.moat_confidence,
                },
                "data_confidence": r.data_validation.data_confidence,
                "data_completeness": r.data_validation.data_completeness_score,
                "missing_critical": [
                    k
                    for k in r.data_validation.missing_metrics
                    if k
                    in (
                        "current_price_abs",
                        "revenue",
                        "net_income",
                        "eps",
                        "operating_cash_flow",
                        "roe",
                        "sector",
                    )
                ],
                "contradictions": r.data_validation.contradictions,
            }
        )
    return rows


def _parse_ai_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # Find array bounds if prose leaked
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, list):
        raise TypeError("AI ranker response is not a JSON array")
    return data


def _coerce_confidence(raw: Any) -> ConfidenceLevel:
    s = str(raw or "LOW").upper()
    if s in ("HIGH", "MEDIUM", "LOW"):
        return s  # type: ignore[return-value]
    return "LOW"


def _fallback_ranks(results: list[QuantScreenResult]) -> list[AIRankResult]:
    """Deterministic fallback when AI is skipped or fails."""
    ordered = sorted(results, key=lambda r: r.final_quant_score, reverse=True)
    out: list[AIRankResult] = []
    for i, r in enumerate(ordered, start=1):
        blocked = r.hard_filter.status in ("HARD_EXCLUDE", "DATA_INSUFFICIENT")
        out.append(
            AIRankResult(
                ticker=r.ticker,
                rank=i,
                ai_score=r.final_quant_score,
                confidence="MEDIUM",
                keep_for_deep_analysis=not blocked and r.final_quant_score >= 60,
                key_reason="Deterministic fallback (AI skipped or unavailable)",
                key_risk="; ".join(r.hard_filter.reasons) or "See quant red flags",
                data_concerns=list(r.data_validation.contradictions),
            )
        )
    return out


def rank_with_ai(
    quant_results: list[QuantScreenResult],
    config: ScreenerRunConfig,
    *,
    cost_tracker: ScreenerCostTracker | None = None,
    client: Anthropic | None = None,
) -> list[AIRankResult]:
    eligible = [
        r
        for r in quant_results
        if r.hard_filter.status == "PASS"
    ]
    # Still include hard-excluded in payload? Spec says AI must not promote them.
    # Send only PASS names for ranking, but keep excluded out of keep list.
    shortlist = sorted(eligible, key=lambda r: r.final_quant_score, reverse=True)
    shortlist = shortlist[: config.constraints.ai_shortlist_size]

    if config.skip_ai or config.dry_run or not shortlist:
        return _fallback_ranks(shortlist if shortlist else eligible)

    if not settings.anthropic_api_key:
        logger.warning("No ANTHROPIC_API_KEY — using deterministic AI fallback")
        return _fallback_ranks(shortlist)

    system = load_ranker_prompt()
    user_msg = (
        "Rank these quantitatively screened stocks for deep-analysis priority.\n"
        "Return JSON array only.\n\n"
        + json.dumps(_payload_for_ai(shortlist), indent=2)
    )

    anthropic_client = client or Anthropic(api_key=settings.anthropic_api_key)
    try:
        response, cost_inr = call_anthropic_and_log(
            anthropic_client,
            stage="portfolio_screener_rank",
            ticker="PORTFOLIO",
            model=config.ai_model,
            max_tokens=4096,
            system=[{"type": "text", "text": system}],
            messages=[{"role": "user", "content": user_msg}],
            stream=False,
        )
        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        raw = "\n".join(text_parts)
        usage = response.usage
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        if cost_tracker is not None:
            cost_tracker.record_ai_call(
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_inr=cost_inr,
            )
        parsed = _parse_ai_json(raw)
    except Exception:
        logger.exception("AI ranker failed — falling back to quant order")
        return _fallback_ranks(shortlist)

    hard_blocked = {
        r.ticker
        for r in quant_results
        if r.hard_filter.status in ("HARD_EXCLUDE", "DATA_INSUFFICIENT")
    }
    quant_by_ticker = {r.ticker: r for r in shortlist}

    results: list[AIRankResult] = []
    seen: set[str] = set()
    for item in parsed:
        ticker = str(item.get("ticker", "")).upper().strip()
        if not ticker or ticker in seen:
            continue
        if ticker not in quant_by_ticker:
            continue
        seen.add(ticker)
        keep = bool(item.get("keep_for_deep_analysis", True))
        if ticker in hard_blocked:
            keep = False
        try:
            ai_score = float(item.get("ai_score", quant_by_ticker[ticker].final_quant_score))
        except (TypeError, ValueError):
            ai_score = quant_by_ticker[ticker].final_quant_score
        try:
            rank = int(item.get("rank", len(results) + 1))
        except (TypeError, ValueError):
            rank = len(results) + 1
        concerns = item.get("data_concerns") or []
        if not isinstance(concerns, list):
            concerns = [str(concerns)]
        results.append(
            AIRankResult(
                ticker=ticker,
                rank=rank,
                ai_score=max(0.0, min(100.0, ai_score)),
                confidence=_coerce_confidence(item.get("confidence")),
                keep_for_deep_analysis=keep,
                key_reason=str(item.get("key_reason") or ""),
                key_risk=str(item.get("key_risk") or ""),
                data_concerns=[str(c) for c in concerns],
            )
        )

    # Ensure every shortlisted ticker appears
    for r in shortlist:
        if r.ticker not in seen:
            results.append(
                AIRankResult(
                    ticker=r.ticker,
                    rank=len(results) + 1,
                    ai_score=r.final_quant_score,
                    confidence="LOW",
                    keep_for_deep_analysis=True,
                    key_reason="Missing from AI response — retained via quant score",
                    key_risk="",
                    data_concerns=["AI omitted ticker"],
                )
            )

    results.sort(key=lambda x: x.rank)
    return results


__all__ = ["load_ranker_prompt", "rank_with_ai"]
