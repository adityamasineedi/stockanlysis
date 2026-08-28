"""AI contextual ranking layer — cheap model, structured inputs only.

Default provider resolution (cheapest solid option with an available key):
  1. OpenAI gpt-4o-mini   (~$0.15 / $0.60 per MTok)
  2. DeepSeek V4 Flash    (~$0.22 / $0.66 off-peak)
  3. Anthropic Haiku 4.5  (~$1 / $5)

Does NOT perform full stock analysis. Cannot override HARD_EXCLUDE.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from anthropic import Anthropic

from stockbot.config import PROMPTS_DIR, settings
from stockbot.costs import log_call
from stockbot.llm.client import call_anthropic_and_log
from stockbot.portfolio_screener.cost_tracker import ScreenerCostTracker
from stockbot.portfolio_screener.models import AIRankResult, QuantScreenResult
from stockbot.portfolio_screener.scoring_config import (
    AI_RANKER_MODELS,
    ConfidenceLevel,
    ScreenerRunConfig,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = PROMPTS_DIR / "portfolio-screener-ranker-v1.md"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
OPENAI_API_BASE = "https://api.openai.com/v1"
MAX_RANKER_TOKENS = 4096

AiProvider = Literal["openai", "deepseek", "anthropic"]

_DEFAULT_SYSTEM = """You are a portfolio pre-screener ranking agent for long-term equity investing.
Your job is ONLY to review quantitative screening results, flag inconsistencies /
contextual risks, and rank which stocks deserve expensive deep analysis.

Rules:
- Do NOT invent missing financial data.
- Do NOT produce BUY / WATCH / SKIP / fair value / target price.
- Do NOT promote any stock with hard_filter_status HARD_EXCLUDE or DATA_INSUFFICIENT.
- Prefer business quality, financial strength, earnings/cash-flow quality, and risk balance.
- Return ONLY a JSON object: {"stocks": [ ... ]} where each element has keys:
  ticker, rank, ai_score (0-100), confidence (HIGH|MEDIUM|LOW),
  keep_for_deep_analysis (bool), key_reason, key_risk, data_concerns (array of strings).
"""


@dataclass(frozen=True)
class RankerCallResult:
    text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_inr: float
    provider: str
    model: str


def load_ranker_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return _DEFAULT_SYSTEM


def resolve_ai_ranker(config: ScreenerRunConfig) -> tuple[AiProvider, str] | None:
    """Pick provider+model. Returns None if no usable API key is configured."""
    requested = (config.ai_provider or "auto").lower().strip()
    explicit_model = config.ai_model

    def _model_for(provider: AiProvider) -> str:
        return explicit_model or AI_RANKER_MODELS[provider]

    if requested == "openai":
        if not settings.openai_api_key:
            logger.warning("ai_provider=openai but OPENAI_API_KEY is unset")
            return None
        return "openai", _model_for("openai")
    if requested == "deepseek":
        if not settings.deepseek_api_key:
            logger.warning("ai_provider=deepseek but DEEPSEEK_API_KEY is unset")
            return None
        return "deepseek", _model_for("deepseek")
    if requested == "anthropic":
        if not settings.anthropic_api_key:
            logger.warning("ai_provider=anthropic but ANTHROPIC_API_KEY is unset")
            return None
        return "anthropic", _model_for("anthropic")

    # auto: cheapest solid option with an available key
    if settings.openai_api_key:
        return "openai", _model_for("openai")
    if settings.deepseek_api_key:
        return "deepseek", _model_for("deepseek")
    if settings.anthropic_api_key:
        return "anthropic", _model_for("anthropic")
    return None


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
    # Prefer object/array bounds if prose leaked
    obj_start = text.find("{")
    arr_start = text.find("[")
    if obj_start >= 0 and (arr_start < 0 or obj_start < arr_start):
        end = text.rfind("}")
        if end > obj_start:
            text = text[obj_start : end + 1]
    elif arr_start >= 0:
        end = text.rfind("]")
        if end > arr_start:
            text = text[arr_start : end + 1]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("stocks", "rankings", "results", "candidates"):
            inner = data.get(key)
            if isinstance(inner, list):
                return inner
        raise TypeError("AI ranker JSON object missing stocks/rankings array")
    raise TypeError("AI ranker response is not a JSON array or object")


def _coerce_confidence(raw: Any) -> ConfidenceLevel:
    s = str(raw or "LOW").upper()
    if s in ("HIGH", "MEDIUM", "LOW"):
        return s  # type: ignore[return-value]
    return "LOW"


def _fallback_ranks(results: list[QuantScreenResult]) -> list[AIRankResult]:
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


def _user_message(shortlist: list[QuantScreenResult]) -> str:
    return (
        "Rank these quantitatively screened stocks for deep-analysis priority.\n"
        'Return ONLY JSON: {"stocks":[...]} with one object per ticker.\n\n'
        + json.dumps(_payload_for_ai(shortlist), indent=2)
    )


def _call_openai_compatible(
    *,
    provider: Literal["openai", "deepseek"],
    api_base: str,
    api_key: str,
    model: str,
    system: str,
    user_msg: str,
) -> RankerCallResult:
    response = httpx.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_RANKER_TOKENS,
            "temperature": 0,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    # DeepSeek reports cache hits separately; OpenAI may use prompt_tokens_details.
    cached_tokens = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    if cached_tokens == 0:
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = int(details.get("cached_tokens", 0) or 0)
    # For DeepSeek pricing, input_tokens should be cache-MISS remainder.
    miss_tokens = input_tokens
    if provider == "deepseek" and cached_tokens:
        miss_tokens = max(0, input_tokens - cached_tokens)

    cost_inr = log_call(
        model=model,
        input_tokens=miss_tokens if provider == "deepseek" else max(0, input_tokens - cached_tokens),
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        provider=provider,
        called_at=datetime.now(UTC),
    )
    return RankerCallResult(
        text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cost_inr=cost_inr,
        provider=provider,
        model=model,
    )


def _call_anthropic_ranker(
    *,
    model: str,
    system: str,
    user_msg: str,
    client: Anthropic | None = None,
) -> RankerCallResult:
    anthropic_client = client or Anthropic(api_key=settings.anthropic_api_key)
    response, cost_inr = call_anthropic_and_log(
        anthropic_client,
        stage="portfolio_screener_rank",
        ticker="PORTFOLIO",
        model=model,
        max_tokens=MAX_RANKER_TOKENS,
        system=[{"type": "text", "text": system}],
        messages=[{"role": "user", "content": user_msg}],
        stream=False,
    )
    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    usage = response.usage
    return RankerCallResult(
        text="\n".join(text_parts),
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cached_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cost_inr=cost_inr,
        provider="anthropic",
        model=model,
    )


def _materialize_ranks(
    parsed: list[dict[str, Any]],
    *,
    shortlist: list[QuantScreenResult],
    quant_results: list[QuantScreenResult],
) -> list[AIRankResult]:
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
        if not ticker or ticker in seen or ticker not in quant_by_ticker:
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


def rank_with_ai(
    quant_results: list[QuantScreenResult],
    config: ScreenerRunConfig,
    *,
    cost_tracker: ScreenerCostTracker | None = None,
    client: Anthropic | None = None,
) -> list[AIRankResult]:
    eligible = [r for r in quant_results if r.hard_filter.status == "PASS"]
    shortlist = sorted(eligible, key=lambda r: r.final_quant_score, reverse=True)
    shortlist = shortlist[: config.constraints.ai_shortlist_size]

    if config.skip_ai or config.dry_run or not shortlist:
        return _fallback_ranks(shortlist if shortlist else eligible)

    resolved = resolve_ai_ranker(config)
    if resolved is None:
        logger.warning("No AI provider key available for pre-screener — quant fallback")
        return _fallback_ranks(shortlist)

    provider, model = resolved
    system = load_ranker_prompt()
    # Prefer object wrapper for OpenAI/DeepSeek json_object mode.
    if provider in ("openai", "deepseek") and '"stocks"' not in system:
        system = system + '\n\nWrap the array as {"stocks": [...]}.'
    user_msg = _user_message(shortlist)

    logger.info("pre-screener AI ranker provider=%s model=%s n=%d", provider, model, len(shortlist))

    try:
        if provider == "openai":
            call = _call_openai_compatible(
                provider="openai",
                api_base=OPENAI_API_BASE,
                api_key=settings.openai_api_key,
                model=model,
                system=system,
                user_msg=user_msg,
            )
        elif provider == "deepseek":
            call = _call_openai_compatible(
                provider="deepseek",
                api_base=DEEPSEEK_API_BASE,
                api_key=settings.deepseek_api_key,
                model=model,
                system=system,
                user_msg=user_msg,
            )
        else:
            call = _call_anthropic_ranker(
                model=model, system=system, user_msg=user_msg, client=client
            )

        if cost_tracker is not None:
            cost_tracker.record_ai_call(
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                cost_inr=call.cost_inr,
            )
        parsed = _parse_ai_json(call.text)
    except Exception:
        logger.exception(
            "AI ranker failed (provider=%s model=%s) — falling back to quant order",
            provider,
            model,
        )
        return _fallback_ranks(shortlist)

    return _materialize_ranks(parsed, shortlist=shortlist, quant_results=quant_results)


__all__ = ["load_ranker_prompt", "rank_with_ai", "resolve_ai_ranker"]
