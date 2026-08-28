"""Evidence-based expected return ranges (CAGR), not yearly return ladders.

Python computes bear/base/bull CAGR ranges from fair-value scenarios and
current price. The model supplies assumptions and confidence only — never
trusted for CAGR arithmetic (same prevention pattern as compute_valuation).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from stockbot.llm.verdict import ExpectedReturnInputs, ValuationComputed, VerdictJSON

DEFAULT_HORIZON_YEARS = 3
DEFAULT_NOTE = (
    "Probabilistic scenario ranges over the horizon — not guaranteed yearly "
    "returns; actual path will be volatile."
)

ExpectedReturnDisplayMode = Literal["EDUCATIONAL_ONLY", "SCENARIO_RANGES"]

_YEARLY_LADDER_RE = re.compile(
    r"\b(?:year\s*[12345]|y\s*[12345])\s*[=:]\s*\d+(?:\.\d+)?\s*%",
    re.IGNORECASE,
)


class ExpectedReturnComputed(BaseModel):
    horizon_years: int
    bear_cagr_range_pct: tuple[float, float]
    base_cagr_range_pct: tuple[float, float]
    bull_cagr_range_pct: tuple[float, float]
    assumptions: list[str]
    confidence: str
    note: str
    display_mode: ExpectedReturnDisplayMode
    computed_from: str = "fair_value_scenarios"


def parse_horizon_years(holding_period: str, override: int | None = None) -> int:
    if override is not None:
        return max(2, min(5, override))
    nums = [int(n) for n in re.findall(r"\d+", holding_period or "")]
    if nums:
        return max(2, min(5, nums[0]))
    return DEFAULT_HORIZON_YEARS


def _cagr_pct(current_price: float, future_price: float, years: int) -> float:
    if current_price <= 0 or future_price <= 0 or years <= 0:
        return 0.0
    return ((future_price / current_price) ** (1.0 / years) - 1.0) * 100.0


def _cagr_range_pct(
    current_price: float,
    fair_low: float,
    fair_high: float,
    years: int,
) -> tuple[float, float]:
    low = _cagr_pct(current_price, fair_low, years)
    high = _cagr_pct(current_price, fair_high, years)
    return (round(min(low, high), 1), round(max(low, high), 1))


def resolve_display_mode(verdict: VerdictJSON) -> ExpectedReturnDisplayMode:
    """Mazdock-style unresolved thesis → educational ranges only."""
    if verdict.buy_range_allowed is True:
        five_y = verdict.five_year_business_test
        wc = (verdict.wc_gap_classification or "").strip().upper()
        wc_ok = wc in ("", "TEMPORARY_BILLING_CYCLE")
        if five_y and five_y.answer == "YES" and wc_ok and not verdict.anti_chase_flag:
            return "SCENARIO_RANGES"
    return "EDUCATIONAL_ONLY"


def resolve_display_mode_from_dict(verdict_json: dict) -> ExpectedReturnDisplayMode:
    if verdict_json.get("buy_range_allowed") is not True:
        return "EDUCATIONAL_ONLY"
    five_y = verdict_json.get("five_year_business_test") or {}
    if not isinstance(five_y, dict) or five_y.get("answer") != "YES":
        return "EDUCATIONAL_ONLY"
    wc = str(verdict_json.get("wc_gap_classification") or "").strip().upper()
    if wc and wc != "TEMPORARY_BILLING_CYCLE":
        return "EDUCATIONAL_ONLY"
    if verdict_json.get("anti_chase_flag"):
        return "EDUCATIONAL_ONLY"
    return "SCENARIO_RANGES"


def _default_assumptions(verdict: VerdictJSON) -> list[str]:
    parts = [
        "CAGR ranges derived from bear/base/bull fair-value scenarios vs current price",
        f"Horizon aligned to holding period ({verdict.holding_period})",
    ]
    if verdict.missing_data_impact and verdict.missing_data_impact.lower() not in (
        "none",
        "no meaningful impact",
    ):
        parts.append(f"Data gaps may widen actual outcomes: {verdict.missing_data_impact[:120]}")
    return parts


def compute_expected_return(
    verdict: VerdictJSON,
    valuation: ValuationComputed,
    inputs: ExpectedReturnInputs | None = None,
) -> ExpectedReturnComputed:
    inputs = inputs or ExpectedReturnInputs()
    years = parse_horizon_years(verdict.holding_period, inputs.horizon_years)
    price = verdict.current_price_abs
    assumptions = inputs.assumptions or _default_assumptions(verdict)
    note = (inputs.note or DEFAULT_NOTE).strip()
    return ExpectedReturnComputed(
        horizon_years=years,
        bear_cagr_range_pct=_cagr_range_pct(
            price, *valuation.fair_value_bear_abs, years
        ),
        base_cagr_range_pct=_cagr_range_pct(
            price, *valuation.fair_value_base_abs, years
        ),
        bull_cagr_range_pct=_cagr_range_pct(
            price, *valuation.fair_value_bull_abs, years
        ),
        assumptions=assumptions,
        confidence=inputs.confidence,
        note=note,
        display_mode=resolve_display_mode(verdict),
    )


_VERDICT_KEYS = frozenset(VerdictJSON.model_fields.keys())


def merge_expected_return_into_verdict_json(verdict_json: dict) -> dict:
    """Compute or refresh expected_return on a stored verdict dict."""
    from stockbot.llm.verdict import ValuationInputs, compute_valuation

    raw_inputs = verdict_json.get("valuation_inputs")
    if not isinstance(raw_inputs, dict):
        return verdict_json
    if verdict_json.get("current_price_abs") is None:
        return verdict_json

    try:
        verdict = VerdictJSON.model_validate(
            {k: v for k, v in verdict_json.items() if k in _VERDICT_KEYS}
        )
        valuation = compute_valuation(ValuationInputs.model_validate(raw_inputs))
    except Exception:
        return verdict_json

    computed = compute_expected_return(verdict, valuation, verdict.expected_return)
    updated = dict(verdict_json)
    updated["expected_return"] = computed.model_dump(mode="json")
    return updated


def format_cagr_range(pair: tuple[float, float] | list[float]) -> str:
    low, high = float(pair[0]), float(pair[1])
    return f"{low:.1f}%–{high:.1f}%"


def format_expected_return_telegram(expected_return: dict) -> list[str]:
    """HTML lines for the Telegram verdict card."""
    if not isinstance(expected_return, dict):
        return []
    horizon = expected_return.get("horizon_years", DEFAULT_HORIZON_YEARS)
    bear = expected_return.get("bear_cagr_range_pct")
    base = expected_return.get("base_cagr_range_pct")
    bull = expected_return.get("bull_cagr_range_pct")
    if not all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in (bear, base, bull)):
        return []

    lines = [
        f"Expected {horizon}y CAGR (scenarios, not guaranteed):",
        (
            f"Bear {format_cagr_range(bear)} · "
            f"Base {format_cagr_range(base)} · "
            f"Bull {format_cagr_range(bull)}"
        ),
    ]
    mode = expected_return.get("display_mode")
    if mode == "EDUCATIONAL_ONLY":
        lines.append("Educational scenario ranges only — no buy zone issued")
    note = expected_return.get("note")
    if note and str(note).strip():
        lines.append(str(note).strip())
    return lines


def report_contains_yearly_return_ladder(report_text: str) -> bool:
    return bool(_YEARLY_LADDER_RE.search(report_text))
