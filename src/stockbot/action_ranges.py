"""Deterministic buy/add action ranges for Telegram and display.

Buy zone levels come from the model's ``buy_zone_abs`` (validated in
``validate.py`` against ``RISK_DISCOUNT_BANDS`` vs base fair value).

Add-more (on-dip) zones are computed in Python — v2 master prompt:
"Add More Zone (around bear fair value)" plus constitution: add only on
valuation-supported dips, never on price alone.

When an initial buy range exists, the add-more band is the bear-case fair
value slice *below* the first-tranche floor (bear low → min(bear high,
buy low)). Without a buy zone, the full bear fair-value band is used.
"""

from __future__ import annotations

_THESIS_BLOCKS_ADD = frozenset({"THESIS_AT_RISK", "THESIS_BROKEN"})


def _float_pair(pair: object) -> tuple[float, float] | None:
    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return None
    try:
        low, high = float(pair[0]), float(pair[1])
    except (TypeError, ValueError):
        return None
    if low <= 0 or high <= 0:
        return None
    return (low, high) if low <= high else (high, low)


def compute_add_more_zone_abs(
    *,
    fair_value_bear_abs: tuple[float, float],
    buy_zone_abs: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """On-dip add band anchored to bear fair value (Python-computed)."""
    bear_low, bear_high = fair_value_bear_abs
    if buy_zone_abs is not None:
        buy_low, _buy_high = buy_zone_abs
        add_high = min(bear_high, buy_low)
        add_low = bear_low
    else:
        add_low, add_high = bear_low, bear_high

    if add_low <= 0 or add_high <= 0 or add_low >= add_high:
        return None
    return (round(add_low, 2), round(add_high, 2))


def capital_range_blocked_reason(verdict_json: dict) -> str | None:
    """Gates that block *any* new capital — the buy range and add-more alike.

    Split out of ``add_more_range_blocked_reason`` so the buy line can name
    the same reasons. The buy line used to hand-roll only the anti-chase and
    WC checks, so a five-year test of NO/UNCERTAIN — a primary buy-zone
    blocker per the v3 prompt — printed a bare "not issued" with no reason,
    while the add-more line on the very same card said "(five-year: …)".
    """
    from stockbot.constitution_gates import (
        should_anti_chase_from_dict,
        wc_gap_blocks_buy_zone,
    )

    if bool(verdict_json.get("anti_chase_flag")) or should_anti_chase_from_dict(verdict_json)[0]:
        return "anti-chase: pause new capital"

    wc_gap = verdict_json.get("wc_gap_classification")
    wc_norm = str(wc_gap).strip().upper() if wc_gap else None
    if wc_gap_blocks_buy_zone(wc_gap) and wc_norm:
        return f"WC: {wc_norm}"

    five_year = verdict_json.get("five_year_business_test") or {}
    answer = str(five_year.get("answer") or "").strip().upper() if isinstance(five_year, dict) else ""
    if answer and answer != "YES":
        return f"five-year test: {answer}"

    return None


def add_more_range_blocked_reason(verdict_json: dict) -> str | None:
    """Return a short reason when constitution gates block add-more display."""
    shared = capital_range_blocked_reason(verdict_json)
    if shared is not None:
        return shared

    thesis = str(verdict_json.get("thesis_status") or "").strip().upper()
    if thesis in _THESIS_BLOCKS_ADD:
        return f"thesis: {thesis}"

    if verdict_json.get("add_range_allowed") is not True:
        return "add range not allowed"

    return None


def buy_zone_price_ceiling(verdict_json: dict) -> tuple[float, str] | None:
    """Highest price at which a buy zone could be issued, and the risk level.

    A buy zone is fair value minus a risk-scaled margin of safety, so a stock
    trading above this ceiling cannot have one however good the business is.
    The card showed the gate that blocked the range but never this bar, so
    "not issued" on a fairly-valued stock looked like a malfunction rather
    than a price judgement.

    This is necessary, not sufficient — the constitution gates still apply.
    """
    from stockbot.validate import RISK_DISCOUNT_BANDS

    risk = str(verdict_json.get("risk") or "").strip().upper()
    band = RISK_DISCOUNT_BANDS.get(risk)
    if band is None:
        return None

    base = _resolve_base_fv_floats(verdict_json)
    if base is None:
        return None

    fv_mid = (base[0] + base[1]) / 2
    if fv_mid <= 0:
        return None
    # Shallowest discount in the band gives the highest qualifying price.
    return round(fv_mid * (1 - band[0]), 2), risk


def _resolve_base_fv_floats(verdict_json: dict) -> tuple[float, float] | None:
    base = _float_pair(verdict_json.get("fair_value_abs"))
    if base is not None:
        return base

    raw_inputs = verdict_json.get("valuation_inputs")
    if isinstance(raw_inputs, dict):
        try:
            from stockbot.llm.verdict import ValuationInputs, compute_valuation

            valuation = compute_valuation(ValuationInputs.model_validate(raw_inputs))
            return valuation.fair_value_base_abs
        except Exception:  # noqa: BLE001 - best-effort derivation, fall back to no ceiling
            return None
    return None


def resolve_add_more_zone_abs(verdict_json: dict) -> tuple[float, float] | None:
    """Compute add-more zone from verdict JSON when gates pass."""
    if add_more_range_blocked_reason(verdict_json) is not None:
        return None

    bear = _resolve_bear_fv_floats(verdict_json)
    if bear is None:
        return None

    buy = _float_pair(verdict_json.get("buy_zone_abs"))
    return compute_add_more_zone_abs(
        fair_value_bear_abs=bear,
        buy_zone_abs=buy,
    )


def _resolve_bear_fv_floats(verdict_json: dict) -> tuple[float, float] | None:
    bear = _float_pair(verdict_json.get("fair_value_bear_abs"))
    if bear is not None:
        return bear

    raw_inputs = verdict_json.get("valuation_inputs")
    if isinstance(raw_inputs, dict):
        try:
            from stockbot.llm.verdict import ValuationInputs, compute_valuation

            valuation = compute_valuation(ValuationInputs.model_validate(raw_inputs))
            return valuation.fair_value_bear_abs
        except Exception:  # noqa: BLE001 - best-effort derivation, fall back to no bear range
            return None
    return None
