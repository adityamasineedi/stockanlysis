"""Deterministic Quality-First constitution gates applied after Stage 2.

Prevention over detection: compute anti-chase and related overrides in
Python rather than trusting the model to set flags consistently.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from stockbot.models import Brief

if TYPE_CHECKING:
    from stockbot.llm.verdict import ValuationComputed, VerdictJSON

# Price at/above base-case fair-value top, or rich trailing P/E with
# non-HIGH earnings quality → pause new capital (constitution anti-chase).
ANTI_CHASE_PE_THRESHOLD = 35.0
_PRICE_ABOVE_BASE_FV_TOLERANCE = 0.005  # 0.5% — float wiggle
_VALUATION_TENSION_BEAR_MULTIPLIER = 2.0


def _trailing_pe(verdict: VerdictJSON, brief: Brief) -> float | None:
    if brief.financials is None:
        return None
    pnl = brief.financials.pnl
    if "EPS in Rs" not in pnl.index:
        return None
    row = pnl.loc["EPS in Rs"]
    value = row["TTM"] if "TTM" in pnl.columns else row.iloc[-1]
    try:
        eps = float(value)
    except (TypeError, ValueError):
        return None
    if eps <= 0:
        return None
    return verdict.current_price_abs / eps


def should_anti_chase(
    verdict: VerdictJSON,
    valuation: ValuationComputed,
    brief: Brief,
) -> tuple[bool, str]:
    """Return (True, reason) when anti_chase_flag must be set."""
    base_high = valuation.fair_value_base_abs[1]
    price = verdict.current_price_abs
    if price >= base_high * (1.0 - _PRICE_ABOVE_BASE_FV_TOLERANCE):
        return (
            True,
            f"price {price:.2f} at/above base fair-value top {base_high:.2f}",
        )

    trailing_pe = _trailing_pe(verdict, brief)
    if (
        trailing_pe is not None
        and trailing_pe >= ANTI_CHASE_PE_THRESHOLD
        and (verdict.earnings_quality or "").upper() != "HIGH"
    ):
        return (
            True,
            (
                f"trailing P/E {trailing_pe:.1f}x >= {ANTI_CHASE_PE_THRESHOLD:.0f}x "
                f"with earnings_quality={verdict.earnings_quality!r}"
            ),
        )

    return False, ""


def should_anti_chase_from_dict(verdict_json: dict) -> tuple[bool, str]:
    """Anti-chase from stored verdict_json (Telegram card / cache — no Brief needed)."""
    price = verdict_json.get("current_price_abs")
    if price is None:
        return False, ""

    base_high: float | None = None
    fv = verdict_json.get("fair_value_base_abs")
    if isinstance(fv, (list, tuple)) and len(fv) >= 2 and fv[1] is not None:
        base_high = float(fv[1])
    else:
        raw_inputs = verdict_json.get("valuation_inputs")
        if isinstance(raw_inputs, dict):
            from stockbot.llm.verdict import ValuationInputs, compute_valuation

            valuation = compute_valuation(ValuationInputs.model_validate(raw_inputs))
            base_high = valuation.fair_value_base_abs[1]

    if base_high is not None and float(price) >= base_high * (
        1.0 - _PRICE_ABOVE_BASE_FV_TOLERANCE
    ):
        return (
            True,
            f"price {float(price):.2f} at/above base fair-value top {base_high:.2f}",
        )

    earnings_quality = (verdict_json.get("earnings_quality") or "").upper()
    raw_inputs = verdict_json.get("valuation_inputs")
    if isinstance(raw_inputs, dict) and earnings_quality != "HIGH":
        eps_base = raw_inputs.get("eps_base")
        try:
            eps = float(eps_base) if eps_base is not None else 0.0
        except (TypeError, ValueError):
            eps = 0.0
        if eps > 0:
            pe = float(price) / eps
            if pe >= ANTI_CHASE_PE_THRESHOLD:
                return (
                    True,
                    (
                        f"P/E vs base EPS {pe:.1f}x >= {ANTI_CHASE_PE_THRESHOLD:.0f}x "
                        f"with earnings_quality={earnings_quality!r}"
                    ),
                )

    return False, ""


def compute_valuation_tension(
    price: float,
    valuation: ValuationComputed,
) -> str:
    """Secondary tension flag — cross-check only, not a primary input."""
    bear_high = valuation.fair_value_bear_abs[1]
    base_low, base_high = valuation.fair_value_base_abs
    base_mid = (base_low + base_high) / 2.0
    if price >= base_high * (1.0 - _PRICE_ABOVE_BASE_FV_TOLERANCE):
        return "HIGH"
    if bear_high > 0 and price >= bear_high * _VALUATION_TENSION_BEAR_MULTIPLIER:
        return "HIGH"
    if price > base_mid:
        return "MEDIUM"
    return "NONE"


def compute_valuation_tension_from_dict(verdict_json: dict) -> str:
    price = verdict_json.get("current_price_abs")
    if price is None:
        return "NONE"
    bear_high: float | None = None
    base_low: float | None = None
    base_high: float | None = None
    bear_fv = verdict_json.get("fair_value_bear_abs")
    base_fv = verdict_json.get("fair_value_base_abs")
    if isinstance(bear_fv, (list, tuple)) and len(bear_fv) >= 2:
        bear_high = float(bear_fv[1])
    if isinstance(base_fv, (list, tuple)) and len(base_fv) >= 2:
        base_low = float(base_fv[0])
        base_high = float(base_fv[1])
    if base_high is None or bear_high is None:
        raw_inputs = verdict_json.get("valuation_inputs")
        if isinstance(raw_inputs, dict):
            from stockbot.llm.verdict import ValuationInputs, compute_valuation

            valuation = compute_valuation(ValuationInputs.model_validate(raw_inputs))
            bear_high = valuation.fair_value_bear_abs[1]
            base_low, base_high = valuation.fair_value_base_abs
    if base_high is None or base_low is None or bear_high is None:
        return "NONE"
    base_mid = (base_low + base_high) / 2.0
    price_f = float(price)
    if price_f >= base_high * (1.0 - _PRICE_ABOVE_BASE_FV_TOLERANCE):
        return "HIGH"
    if bear_high > 0 and price_f >= bear_high * _VALUATION_TENSION_BEAR_MULTIPLIER:
        return "HIGH"
    if price_f > base_mid:
        return "MEDIUM"
    return "NONE"


def refresh_constitution_fields(verdict_json: dict) -> dict:
    """Re-apply deterministic constitution fields (cache hits / display)."""
    updated = dict(verdict_json)
    anti_chase, _ = should_anti_chase_from_dict(updated)
    updated["anti_chase_flag"] = anti_chase
    updated["external_valuation_tension"] = compute_valuation_tension_from_dict(updated)
    if anti_chase:
        updated["buy_range_allowed"] = False
        updated["add_range_allowed"] = False
        updated["buy_zone_abs"] = None
    from stockbot.expected_return import merge_expected_return_into_verdict_json

    return merge_expected_return_into_verdict_json(updated)


def sync_live_price_into_verdict(
    verdict_json: dict,
    *,
    live_price_abs: float,
    live_price_date: date,
) -> dict:
    """Overlay a live market price onto a cached verdict — no LLM re-run.

    Preserves ``analysis_price_abs`` / ``analysis_price_date`` (the price
    baked in at the original paid run) while updating ``current_price_abs``
    and ``price_date`` for display and constitution gates.
    """
    updated = dict(verdict_json)
    if updated.get("analysis_price_abs") is None:
        updated["analysis_price_abs"] = updated.get("current_price_abs")
    if updated.get("analysis_price_date") is None:
        updated["analysis_price_date"] = updated.get("price_date")
    updated["current_price_abs"] = round(float(live_price_abs), 2)
    updated["price_date"] = live_price_date.isoformat()
    updated["price_synced_at"] = datetime.now(UTC).isoformat()
    return refresh_constitution_fields(updated)


def apply_constitution_overrides(
    verdict: VerdictJSON,
    valuation: ValuationComputed,
    brief: Brief,
) -> VerdictJSON:
    """Apply non-negotiable constitution overrides before storage/display."""
    anti_chase, _reason = should_anti_chase(verdict, valuation, brief)
    tension = compute_valuation_tension(verdict.current_price_abs, valuation)
    updates: dict[str, object] = {"external_valuation_tension": tension}
    if anti_chase:
        updates["anti_chase_flag"] = True
        if verdict.buy_range_allowed or verdict.buy_zone_abs is not None:
            updates["buy_range_allowed"] = False
            updates["add_range_allowed"] = False
            updates["buy_zone_abs"] = None
    elif verdict.anti_chase_flag is None:
        updates["anti_chase_flag"] = False
    return verdict.model_copy(update=updates)
