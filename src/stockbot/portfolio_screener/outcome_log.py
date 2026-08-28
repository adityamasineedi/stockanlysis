"""Append-only pre-scan outcome log for fetch/hard-filter monitoring.

Tracks how often DATA_INSUFFICIENT vs weak-fundamentals rejects occur after
ratio compute fallbacks (e.g. BBOX-style missing Screener ROE %).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stockbot.config import PORTFOLIO_DIR

logger = logging.getLogger(__name__)

OUTCOMES_PATH = PORTFOLIO_DIR / "prescan_outcomes.jsonl"

_USER_LABELS = {
    "roe": "ROE",
    "roce": "ROCE",
    "debt_equity": "D/E",
    "ocf_to_pat": "OCF/PAT",
    "interest_coverage": "Interest coverage",
    "net_debt_ebitda": "Net debt/EBITDA",
    "current_ratio": "Current ratio",
}

# Only surface high-stakes derived ratios to users (not every margin/turnover).
_TELEGRAM_FIELDS = frozenset(_USER_LABELS)

_SOURCE_HINT = {
    "computed": "derived from statements — cross-check Screener.in",
    "yfinance": "from Yahoo Finance — cross-check Screener.in",
}

# Always statement-derived (never a Screener "ratios" row) — softer wording.
_ALWAYS_DERIVED = frozenset({"ocf_to_pat", "net_debt_ebitda", "interest_coverage", "debt_equity"})
_ALWAYS_DERIVED_HINT = "derived from P&L/BS/CF (not a Screener ratio row)"


def format_computed_metric_warnings(
    metric_sources: dict[str, str],
    metrics_values: dict[str, float | None] | None = None,
) -> list[str]:
    """Human-readable lines for Telegram when a metric was not Screener-fetched."""
    lines: list[str] = []
    values = metrics_values or {}
    for field, source in sorted(metric_sources.items()):
        if source not in ("computed", "yfinance"):
            continue
        if field not in _TELEGRAM_FIELDS:
            continue
        label = _USER_LABELS[field]
        if field in _ALWAYS_DERIVED and source == "computed":
            hint = _ALWAYS_DERIVED_HINT
        else:
            hint = _SOURCE_HINT[source]
        raw = values.get(field)
        if isinstance(raw, (int, float)):
            if field in ("roe", "roce"):
                shown = f"{raw:.1f}%"
            else:
                shown = f"{raw:.2f}"
            lines.append(f"{label} {shown} — {hint}")
        else:
            lines.append(f"{label} — {hint}")
    return lines


def log_prescan_outcome(record: dict[str, Any]) -> None:
    """Structured log + JSONL append for later aggregation."""
    payload = {
        **record,
        "logged_at": datetime.now(UTC).isoformat(),
    }
    logger.info(
        "prescan_outcome ticker=%s verdict=%s hard=%s quant=%s band=%s "
        "reject_class=%s computed=%s missing_key=%s",
        payload.get("ticker"),
        payload.get("verdict"),
        payload.get("hard_filter_status"),
        payload.get("quant_score"),
        payload.get("candidate_band"),
        payload.get("reject_class"),
        payload.get("computed_metrics"),
        payload.get("missing_key_trio"),
    )
    try:
        PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
        with OUTCOMES_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("Failed to append pre-scan outcome to %s", OUTCOMES_PATH)


def classify_reject(
    *,
    hard_status: str | None,
    verdict: str,
    quant_score: float | None,
) -> str:
    """Bucket for monitoring: data_gap vs weak_fundamentals vs other."""
    if hard_status == "DATA_UNAVAILABLE" or verdict in {
        "DATA_UNAVAILABLE",
        "DATA_UNAVAILABLE_RETRY",
    }:
        return "data_unavailable"
    if verdict in {"MODEL_NOT_APPLICABLE", "SECTOR_SPECIFIC_REVIEW"}:
        return "sector_specific_review"
    if verdict in {"REVIEW_EXCEPTION"}:
        return "review_exception"
    if verdict == "HOLDING_MONITOR_ONLY":
        return "holding_monitor"
    if hard_status == "DATA_INSUFFICIENT":
        return "data_gap"
    if hard_status == "HARD_EXCLUDE":
        return "hard_exclude"
    if verdict in {"NOT_SUITABLE", "NOT_SUITABLE_FOR_3Y_RESEARCH"} and (
        quant_score is not None and quant_score < 60
    ):
        return "weak_fundamentals"
    if verdict in {"MARGINAL", "SECTOR_SPECIFIC_REVIEW"}:
        return "sector_specific_review"
    if verdict in {"SUITABLE_FOR_DEEP_ANALYSIS", "AUTO_DEEP_ANALYSIS"}:
        return "auto_deep"
    return "other"


def summarize_outcomes(path: Path | None = None, *, limit: int = 500) -> dict[str, Any]:
    """Read recent JSONL rows and count reject classes (ops helper)."""
    target = path or OUTCOMES_PATH
    counts: dict[str, int] = {}
    n = 0
    if not target.exists():
        return {"rows": 0, "by_reject_class": counts, "path": str(target)}
    lines = target.read_text(encoding="utf-8").splitlines()[-limit:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        key = str(row.get("reject_class") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {"rows": n, "by_reject_class": counts, "path": str(target)}
