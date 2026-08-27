"""Audit logging for every selected or rejected stock."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from stockbot.config import LOGS_DIR
from stockbot.portfolio_screener.models import ScreeningResult, StockScreenRecord

logger = logging.getLogger(__name__)


def format_human_table(records: list[StockScreenRecord]) -> str:
    header = (
        "Rank | Stock | Sector | Quant | AI | Final | Quality | Growth | "
        "Value | Risk | Decision"
    )
    lines = [header, "-" * len(header)]
    for s in records:
        lines.append(
            f"{s.ranking or '-':>4} | {s.ticker:<10} | {(s.sector or '-')[:14]:<14} | "
            f"{_fmt(s.quant_score):>5} | {_fmt(s.ai_score):>5} | {_fmt(s.final_score):>5} | "
            f"{_fmt(s.quality_score):>7} | {_fmt(s.growth_score):>6} | "
            f"{_fmt(s.valuation_score):>5} | {_fmt(s.risk_score):>4} | "
            f"{s.selection_status}"
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def write_audit_artifact(result: ScreeningResult, path: Path | None = None) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = LOGS_DIR / f"portfolio_screen_{stamp}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    logger.info("Wrote screening audit artifact to %s", path)
    return path


def log_stock_decision(record: StockScreenRecord) -> None:
    logger.info(
        "screen_decision ticker=%s status=%s hard=%s quant=%s ai=%s final=%s "
        "reason=%s reject=%s",
        record.ticker,
        record.selection_status,
        record.hard_filter_status,
        record.quant_score,
        record.ai_score,
        record.final_score,
        record.selection_reason or "-",
        record.rejection_reason or "-",
    )
