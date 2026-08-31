"""Prescan integration for portfolio SIP — gate, ranking, batch scan."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from stockbot.portfolio_screener.eligibility import check_deep_analysis_eligibility
from stockbot.portfolio_screener.outcome_log import load_prescan_outcomes
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig
from stockbot.portfolio_sip_schema import PrescanGateConfig, SymbolConfig

logger = logging.getLogger(__name__)

PROCEED_VERDICTS = frozenset({"AUTO_DEEP_ANALYSIS", "SECTOR_SPECIFIC_REVIEW"})
MONITOR_ONLY_VERDICTS = frozenset({"HOLDING_MONITOR_ONLY"})


@dataclass(frozen=True)
class PrescanGateResult:
    blocked: bool
    note: str | None = None
    verdict: str | None = None
    suitable: bool | None = None


@dataclass(frozen=True)
class PrescanBatchItem:
    symbol: str
    verdict: str | None
    suitable: bool
    quant_score: float | None
    error: str | None = None


def prescan_outcome_map() -> dict[str, dict[str, Any]]:
    """Latest prescan row per ticker, keyed by uppercase symbol."""
    return {
        str(row.get("ticker") or "").upper(): row
        for row in load_prescan_outcomes()
        if row.get("ticker")
    }


def _parse_logged_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def evaluate_prescan_gate(
    symbol: str,
    row: dict[str, Any] | None,
    gate: PrescanGateConfig,
    *,
    prescan_exempt: bool = False,
) -> PrescanGateResult:
    """Return whether allocation should be blocked for this symbol."""
    if not gate.enabled:
        return PrescanGateResult(blocked=False)

    if prescan_exempt:
        return PrescanGateResult(blocked=False, note="prescan exempt (ETF/non-equity)")

    if row is None:
        if gate.skip_when_missing:
            return PrescanGateResult(
                blocked=True,
                note="prescan missing",
                verdict=None,
                suitable=None,
            )
        return PrescanGateResult(blocked=False, note="prescan pending")

    verdict = str(row.get("verdict") or "")
    suitable = bool(row.get("suitable_for_deep_analysis"))
    if gate.require_recent_days > 0:
        logged = _parse_logged_at(str(row.get("logged_at") or ""))
        if logged is None:
            if gate.skip_when_missing:
                return PrescanGateResult(
                    blocked=True,
                    note="prescan stale",
                    verdict=verdict or None,
                    suitable=suitable,
                )
        else:
            age_days = (datetime.now(UTC) - logged).days
            if age_days > gate.require_recent_days:
                if gate.skip_when_missing:
                    return PrescanGateResult(
                        blocked=True,
                        note=f"prescan stale ({age_days}d)",
                        verdict=verdict or None,
                        suitable=suitable,
                    )
                return PrescanGateResult(
                    blocked=False,
                    note=f"prescan stale ({age_days}d)",
                    verdict=verdict or None,
                    suitable=suitable,
                )

    if not suitable and verdict not in PROCEED_VERDICTS:
        if gate.allow_holding_monitor and verdict in MONITOR_ONLY_VERDICTS:
            return PrescanGateResult(
                blocked=False,
                note="prescan monitor-only (SIP ok)",
                verdict=verdict or None,
                suitable=False,
            )
        return PrescanGateResult(
            blocked=True,
            note=f"prescan {verdict}",
            verdict=verdict or None,
            suitable=False,
        )

    return PrescanGateResult(
        blocked=False,
        verdict=verdict or None,
        suitable=suitable,
    )


def rank_symbols_by_prescan(
    symbols: tuple[SymbolConfig, ...],
    prescan_map: dict[str, dict[str, Any]],
) -> tuple[SymbolConfig, ...]:
    """Higher quant_score first; unknown symbols last."""

    def sort_key(sym: SymbolConfig) -> tuple[float, float, str]:
        row = prescan_map.get(sym.symbol)
        if row is None:
            return (-1.0, -1.0, sym.symbol)
        quant = row.get("quant_score")
        quality = row.get("quality_score")
        q = float(quant) if isinstance(quant, (int, float)) else -1.0
        qual = float(quality) if isinstance(quality, (int, float)) else -1.0
        return (-q, -qual, sym.symbol)

    return tuple(sorted(symbols, key=sort_key))


def batch_prescan_symbols(
    symbols: tuple[str, ...],
    *,
    skip_ai: bool = True,
    delay_seconds: float = 2.0,
) -> list[PrescanBatchItem]:
    """Run prescan for each symbol (writes prescan_outcomes.jsonl)."""
    config = ScreenerRunConfig(ai_provider="auto", skip_ai=skip_ai)
    unique = tuple(dict.fromkeys(s.upper() for s in symbols))
    results: list[PrescanBatchItem] = []

    for index, symbol in enumerate(unique):
        if index > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        try:
            outcome = check_deep_analysis_eligibility(symbol, config=config)
            results.append(
                PrescanBatchItem(
                    symbol=str(outcome.ticker or symbol).upper(),
                    verdict=str(outcome.verdict),
                    suitable=outcome.suitable_for_deep_analysis,
                    quant_score=(
                        round(float(outcome.quant_score), 2)
                        if outcome.quant_score is not None
                        else None
                    ),
                )
            )
        except Exception as exc:
            logger.exception("portfolio prescan failed for %s", symbol)
            results.append(
                PrescanBatchItem(
                    symbol=symbol,
                    verdict=None,
                    suitable=False,
                    quant_score=None,
                    error=str(exc)[:120],
                )
            )
    return results


def format_prescan_batch_summary_html(items: list[PrescanBatchItem]) -> str:
    from html import escape

    proceed = sum(1 for item in items if item.suitable and not item.error)
    blocked = sum(1 for item in items if not item.suitable and not item.error)
    errors = sum(1 for item in items if item.error)
    lines = [
        f"<b>Portfolio prescan complete</b> — {len(items)} symbol(s)",
        f"Proceed: {proceed} · Blocked: {blocked} · Errors: {errors}",
        "",
        "<pre>Symbol     Verdict              Q</pre>",
    ]
    for item in items:
        verdict = item.verdict or item.error or "—"
        q = f"{item.quant_score:.0f}" if item.quant_score is not None else "—"
        mark = "✅" if item.suitable and not item.error else "⏸"
        lines.append(
            f"{mark} <code>{escape(item.symbol):8}</code> "
            f"{escape(str(verdict)[:22]):22} {q:>3}"
        )
    lines.append("")
    lines.append("<i>Run <code>/sip plan</code> to see gated allocation.</i>")
    return "\n".join(lines)


def portfolio_reminder_chat_ids() -> frozenset[int]:
    """Chats to receive monthly portfolio SIP plan reminders."""
    from stockbot.config import parse_telegram_allowed_chat_ids
    from stockbot.storage import list_sip_contribution_chat_ids

    allowed = parse_telegram_allowed_chat_ids()
    if allowed:
        return allowed
    return list_sip_contribution_chat_ids()


def all_portfolio_symbols(config) -> tuple[str, ...]:
    from stockbot.portfolio_sip_schema import symbol_names

    symbols: list[str] = []
    for bucket in config.portfolios:
        if bucket.enabled:
            symbols.extend(symbol_names(bucket))
    return tuple(dict.fromkeys(symbols))
