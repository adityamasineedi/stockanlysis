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


def add_more_range_blocked_reason(verdict_json: dict) -> str | None:
    """Return a short reason when constitution gates block add-more display."""
    from stockbot.constitution_gates import should_anti_chase_from_dict, wc_gap_blocks_buy_zone

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

    thesis = str(verdict_json.get("thesis_status") or "").strip().upper()
    if thesis in _THESIS_BLOCKS_ADD:
        return f"thesis: {thesis}"

    if verdict_json.get("add_range_allowed") is not True:
        return "add range not allowed"

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
        except Exception:
            return None
    return None
