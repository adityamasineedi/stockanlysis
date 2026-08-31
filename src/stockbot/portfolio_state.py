"""Position sizing against the holder's own capital — pure, no I/O.

The analysis pipeline answers "is this a good business at this price". Nothing
answered "how much of *my* money belongs in it", because nothing knew what the
holder owns or has. That gap is visible in the constitution itself, which
declares ``"maximum_intended_position_pct": null`` and notes that limits "come
from user risk policy, not invented by the bot" — the bot correctly refuses to
make the number up, and until now had nowhere to read it from.

Every function here returns None rather than a number it cannot justify. A
percentage computed against an unknown denominator is worse than no
percentage: it looks authoritative and is arbitrary. This mirrors
``summarize_sip_contributions``, which withholds ``units_estimate`` when any
contribution lacks a price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The bot proposes; the user's policy decides. 10% is a common single-name cap
# for a concentrated-but-not-reckless equity book, and is overridable.
DEFAULT_MAX_POSITION_PCT = 10.0

# Intended position is built in four ~25% tranches (constitution principle 3).
TRANCHE_COUNT = 4


def _finite_positive(value: object) -> float | None:
    """A usable money/quantity figure, or None.

    float() parses "nan" and "inf", and `nan <= 0` is False, so a bare
    comparison lets non-finite values through — the same hole that once let a
    NaN SIP plan save and render "₹nan" on every message.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


@dataclass(frozen=True)
class PositionSizing:
    """What one holding is worth, and what room is left in it."""

    value_inr: float
    pct_of_capital: float
    headroom_inr: float
    max_position_pct: float

    @property
    def over_cap(self) -> bool:
        return self.pct_of_capital > self.max_position_pct


def position_value(quantity: object, price: object) -> float | None:
    """Market value of a holding. None when either input is unusable."""
    qty = _finite_positive(quantity)
    unit = _finite_positive(price)
    if qty is None or unit is None:
        return None
    return round(qty * unit, 2)


def position_pct(value_inr: object, total_capital_inr: object) -> float | None:
    """Share of total capital, or None without a usable denominator."""
    value = _finite_positive(value_inr)
    capital = _finite_positive(total_capital_inr)
    if value is None or capital is None:
        return None
    return round(value / capital * 100.0, 2)


def headroom_inr(
    value_inr: object,
    total_capital_inr: object,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
) -> float | None:
    """Rupees that could still be added before the per-position cap binds.

    Clamped at zero: a position already over its cap has no room, and a
    negative "headroom" would read as though selling were required, which is a
    different judgement this function is not making.
    """
    capital = _finite_positive(total_capital_inr)
    cap_pct = _finite_positive(max_position_pct)
    if capital is None or cap_pct is None:
        return None
    value = _finite_positive(value_inr) or 0.0
    return round(max(capital * cap_pct / 100.0 - value, 0.0), 2)


def size_position(
    quantity: object,
    price: object,
    total_capital_inr: object,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
) -> PositionSizing | None:
    """The three numbers together, or None if any input is unusable."""
    value = position_value(quantity, price)
    pct = position_pct(value, total_capital_inr)
    room = headroom_inr(value, total_capital_inr, max_position_pct)
    if value is None or pct is None or room is None:
        return None
    return PositionSizing(
        value_inr=value,
        pct_of_capital=pct,
        headroom_inr=room,
        max_position_pct=float(max_position_pct),
    )


def intended_position_inr(
    total_capital_inr: object, max_position_pct: float = DEFAULT_MAX_POSITION_PCT
) -> float | None:
    """Full size a position is allowed to reach under the policy."""
    capital = _finite_positive(total_capital_inr)
    cap_pct = _finite_positive(max_position_pct)
    if capital is None or cap_pct is None:
        return None
    return round(capital * cap_pct / 100.0, 2)


def tranche_amounts(intended_inr: object, count: int = TRANCHE_COUNT) -> list[float] | None:
    """The intended position split into equal tranches, in rupees.

    ``position_building_plan`` has no Python renderer — it exists only as prose
    the prompt asks the model to print, so its tranches are percentages of an
    unstated total. With a policy in hand they become spendable amounts.

    The last tranche absorbs the rounding remainder so the parts sum exactly to
    the whole; four slices of a number not divisible by four otherwise lose
    paise, and a plan whose parts don't add up invites doubt about the rest.
    """
    total = _finite_positive(intended_inr)
    if total is None or count < 1:
        return None
    slice_inr = round(total / count, 2)
    tranches = [slice_inr] * (count - 1)
    tranches.append(round(total - slice_inr * (count - 1), 2))
    return tranches


def concentration_breaches(
    positions: dict[str, float], total_capital_inr: object,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
) -> list[tuple[str, float]]:
    """(ticker, pct) for every holding over the cap, worst first.

    Positions whose percentage cannot be computed are omitted rather than
    reported as compliant — absence of evidence is not a passing check.
    """
    breaches: list[tuple[str, float]] = []
    cap_pct = _finite_positive(max_position_pct)
    if cap_pct is None:
        return breaches
    for ticker, value in positions.items():
        pct = position_pct(value, total_capital_inr)
        if pct is not None and pct > cap_pct:
            breaches.append((ticker, pct))
    breaches.sort(key=lambda item: (-item[1], item[0]))
    return breaches
