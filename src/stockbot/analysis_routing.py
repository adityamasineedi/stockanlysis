"""Stage 2 routing — lite vs full analysis path.

Uses deterministic prescan metrics (no eligibility AI call) plus Stage 1
extraction red flags to decide whether the expensive 16-section Sonnet
report is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from stockbot.config import settings
from stockbot.llm.extract import ExtractionResult
from stockbot.models import TickerInfo
from stockbot.portfolio_screener.data_loader import fetch_universe_metrics
from stockbot.portfolio_screener.issuer_routing import decide_eligibility_route
from stockbot.portfolio_screener.quant_engine import compute_quant_score
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig

logger = logging.getLogger(__name__)

Stage2Mode = Literal["LITE", "FULL"]

_LITE_ISSUER_CLASSES = frozenset({"NON_FINANCIAL"})


@dataclass(frozen=True)
class AnalysisRouting:
    stage2_mode: Stage2Mode
    eligibility_verdict: str | None
    issuer_class: str | None
    data_confidence: str | None
    quant_red_flags_count: int
    reasons: tuple[str, ...]


def _quant_prescan_routing(ticker: TickerInfo) -> AnalysisRouting:
    config = ScreenerRunConfig(skip_ai=True)
    metrics_list = fetch_universe_metrics([ticker])
    metrics = metrics_list[0]
    quant = compute_quant_score(metrics, config)
    route = decide_eligibility_route(metrics, quant)
    flag_count = len(quant.red_flags)
    severe_or_major = sum(1 for f in quant.red_flags if f.severity in {"severe", "major"})

    reasons: list[str] = []
    mode: Stage2Mode = "FULL"

    if route.eligibility != "AUTO_DEEP_ANALYSIS":
        reasons.append(f"eligibility={route.eligibility}")
    if route.issuer_class not in _LITE_ISSUER_CLASSES:
        reasons.append(f"issuer_class={route.issuer_class}")
    if quant.data_validation.data_confidence != "HIGH":
        reasons.append(f"data_confidence={quant.data_validation.data_confidence}")
    if flag_count > 0:
        reasons.append(f"quant_red_flags={flag_count}")
    if severe_or_major > 0:
        reasons.append(f"severe_or_major_flags={severe_or_major}")

    if not reasons:
        mode = "LITE"
        reasons.append("clean AUTO_DEEP prescan — lite Stage 2 eligible")

    logger.info(
        "Stage 2 routing for %s: mode=%s (%s)",
        ticker.symbol,
        mode,
        "; ".join(reasons),
    )
    return AnalysisRouting(
        stage2_mode=mode,
        eligibility_verdict=route.eligibility,
        issuer_class=route.issuer_class,
        data_confidence=quant.data_validation.data_confidence,
        quant_red_flags_count=flag_count,
        reasons=tuple(reasons),
    )


def resolve_stage2_mode(
    ticker: TickerInfo,
    extraction: ExtractionResult,
    *,
    prescan: AnalysisRouting | None = None,
) -> Stage2Mode:
    """Final mode after Stage 1 — extraction news flags always force FULL."""
    if settings.force_stage2_full:
        logger.info(
            "%s: FORCE_STAGE2_FULL enabled — using FULL Stage 2 regardless of routing",
            ticker.symbol,
        )
        return "FULL"
    prescan = prescan or _quant_prescan_routing(ticker)
    if extraction.red_flags_found:
        logger.info(
            "%s: upgrading Stage 2 to FULL (%d extraction red flag(s))",
            ticker.symbol,
            len(extraction.red_flags_found),
        )
        return "FULL"
    if prescan.stage2_mode == "LITE":
        return "LITE"
    return "FULL"


def compute_stage2_routing(ticker: TickerInfo) -> AnalysisRouting:
    """Quant-only prescan routing (no LLM). Call before Stage 1."""
    return _quant_prescan_routing(ticker)
