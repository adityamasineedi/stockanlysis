"""Soft pick policy — rank prescan history without over-filtering.

Hard rejects only true deal-breakers (HARD_EXCLUDE, CRITICAL cash, data gaps).
Soft score and pillar strength gate inclusion; prescan MONITOR is not a sell signal.
Final buy decision still requires /analyze verdict + buy range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from stockbot.config import settings

PickTier = Literal["analyze_now", "analyze_if_interested"]

_HARD_REJECT_STATUSES = frozenset({"HARD_EXCLUDE", "DATA_UNAVAILABLE", "DATA_INSUFFICIENT"})
_HARD_REJECT_VERDICTS = frozenset(
    {
        "NOT_SUITABLE_FOR_3Y_RESEARCH",
        "NOT_SUITABLE",
        "DATA_UNAVAILABLE",
        "DATA_UNAVAILABLE_RETRY",
        "NOT_FOUND",
        "AMBIGUOUS",
    }
)
_ANALYZE_NOW_VERDICTS = frozenset({"AUTO_DEEP_ANALYSIS", "SECTOR_SPECIFIC_REVIEW"})
_PILLAR_KEYS = ("quality_score", "growth_score", "strength_score")


def pick_min_quant_score() -> float:
    return settings.pick_min_quant_score


def pick_min_pillar_score() -> float:
    return settings.pick_min_pillar_score


def _float_score(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def pick_skip_reason(row: dict[str, Any]) -> str | None:
    """Return a short reason when a row fails the soft pick policy."""
    hard = str(row.get("hard_filter_status") or "")
    if hard in _HARD_REJECT_STATUSES:
        return f"hard filter: {hard}"

    verdict = str(row.get("verdict") or "")
    if verdict in _HARD_REJECT_VERDICTS:
        return f"verdict: {verdict}"

    cash = str(row.get("cash_conversion_status") or "")
    if cash == "CRITICAL":
        return "cash conversion CRITICAL"

    quant = _float_score(row, "quant_score")
    min_quant = pick_min_quant_score()
    min_pillar = pick_min_pillar_score()
    pillar_hit = any(
        (score := _float_score(row, key)) is not None and score >= min_pillar
        for key in _PILLAR_KEYS
    )
    if row.get("quality_override"):
        return None
    if quant is not None and quant >= min_quant:
        return None
    if pillar_hit:
        return None
    if quant is None:
        return "quant score missing"
    return f"overall {quant:.1f} < {min_quant:.0f} and no Q/G/S pillar ≥ {min_pillar:.0f}"


def is_pick_eligible(row: dict[str, Any]) -> bool:
    return pick_skip_reason(row) is None


def pick_tier(row: dict[str, Any]) -> PickTier:
    verdict = str(row.get("verdict") or "")
    if verdict in _ANALYZE_NOW_VERDICTS:
        return "analyze_now"
    return "analyze_if_interested"


@dataclass(frozen=True)
class PickPolicySummary:
    min_quant: float
    min_pillar: float
    eligible_count: int
    skipped_count: int


def query_pick_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter prescan rows using the soft pick policy, best first."""
    eligible = [row for row in rows if is_pick_eligible(row)]
    eligible.sort(
        key=lambda r: (
            0 if pick_tier(r) == "analyze_now" else 1,
            -(_float_score(r, "quant_score") or -1.0),
            -(_float_score(r, "quality_score") or -1.0),
            str(r.get("ticker") or ""),
        ),
    )
    return eligible


def summarize_pick_policy(rows: list[dict[str, Any]]) -> PickPolicySummary:
    eligible = sum(1 for row in rows if is_pick_eligible(row))
    return PickPolicySummary(
        min_quant=pick_min_quant_score(),
        min_pillar=pick_min_pillar_score(),
        eligible_count=eligible,
        skipped_count=max(0, len(rows) - eligible),
    )
