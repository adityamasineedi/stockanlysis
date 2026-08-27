"""render.py — placeholder-token substitution for v3 master-prompt reports.

v3 requires Stage 2 to write `{{token}}` placeholders instead of literal
numbers in prose (see prompts/master-stock-analysis-prompt-v3.md, "⬥
PLACEHOLDER TOKENS"), so Python substitutes the canonical values at render
time. This makes number drift between prose and the structured verdict
impossible rather than merely detectable after the fact — the same
prevention-over-detection principle behind moving fair-value arithmetic
out of the model (see llm/verdict.py's compute_valuation).

Signature note: the requested signature was
    render_report(report_md, technicals, verdict, shareholding) -> str
but v3's own token list includes {{week52_high}} / {{week52_low}}, which
only exist on PriceData — there is no way to satisfy v3's declared token
contract without it. Extended the signature to accept `price` too, and
`verdict` is the typed VerdictJSON (plus a separately-computed
ValuationComputed for the fair-value tokens) rather than a raw dict, so
this can't drift from extract_verdict_json's actual return type.
"""

from __future__ import annotations

import re

from stockbot.llm.verdict import ValuationComputed, VerdictJSON
from stockbot.models import PriceData, Shareholding, Technicals

# The master prompt's own examples show tokens wrapped in backticks
# (`` `{{current_price}}` ``) as markdown-authoring guidance for the model —
# but that's about how the model marks a token in ITS prose, not part of the
# substituted value. The original regex only matched the `{{...}}` interior,
# leaving the backticks in place around the rendered number ("`₹410.65`",
# "`2026-08-26`") — literal backticks surviving into the delivered report.
# Consume an optional backtick on either side so the whole thing — backticks
# included — is replaced by the plain formatted value.
_TOKEN_RE = re.compile(r"`?\{\{(\w+)\}\}`?")
_ANY_BRACE_RE = re.compile(r"\{\{.*?\}\}")

# Tokens formatted as ₹ money to 2dp, per Fix 1's example ("₹410.65, ...,
# 408.00"). Tokens not in this set (rsi14, the _pct tokens) get their own
# formatting below.
_MONEY_TOKENS = {
    "current_price",
    "week52_high",
    "week52_low",
    "sma50",
    "sma200",
    "support",
    "resistance",
    "fair_value_bear",
    "fair_value_base",
    "fair_value_bull",
    "fair_value_base_low",
    "fair_value_base_high",
    "buy_zone_low",
    "buy_zone_high",
}
_PERCENT_TOKENS = {"upside_pct", "downside_pct", "promoter_pct", "pledge_pct"}


class PlaceholderError(Exception):
    pass


def _nearest_below(levels: list[float], price: float) -> float | None:
    candidates = [lv for lv in levels if lv <= price]
    if candidates:
        return max(candidates)
    return min(levels) if levels else None


def _nearest_above(levels: list[float], price: float) -> float | None:
    candidates = [lv for lv in levels if lv >= price]
    if candidates:
        return min(candidates)
    return max(levels) if levels else None


def _midpoint(range_: tuple[float, float]) -> float:
    return (range_[0] + range_[1]) / 2


def _format(name: str, value: float) -> str:
    if name == "price_date":
        return str(value)
    if name == "rsi14":
        return f"{value:.1f}"
    if name in _PERCENT_TOKENS:
        return f"{value:.1f}%"
    if name in _MONEY_TOKENS:
        return f"₹{value:.2f}"
    # Should be unreachable — every real token above is classified. A new
    # token added to _build_tokens without a formatting rule here is a bug,
    # not a runtime data problem, so this fails loudly rather than guessing.
    raise PlaceholderError(f"No formatting rule for token {{{{{name}}}}}")


def _build_tokens(
    price: PriceData,
    technicals: Technicals,
    verdict: VerdictJSON,
    valuation: ValuationComputed,
    shareholding: Shareholding | None,
) -> dict[str, float | str | None]:
    current_price = verdict.current_price_abs
    fair_value_base_mid = _midpoint(valuation.fair_value_base_abs)
    fair_value_bear_mid = _midpoint(valuation.fair_value_bear_abs)

    return {
        "current_price": current_price,
        "price_date": verdict.price_date.isoformat(),
        "week52_high": price.week52_high_abs,
        "week52_low": price.week52_low_abs,
        "sma50": technicals.sma50,
        "sma200": technicals.sma200,
        "rsi14": technicals.rsi14,
        "support": _nearest_below(technicals.support_abs, current_price),
        "resistance": _nearest_above(technicals.resistance_abs, current_price),
        "fair_value_bear": fair_value_bear_mid,
        "fair_value_base": fair_value_base_mid,
        "fair_value_bull": _midpoint(valuation.fair_value_bull_abs),
        # The headline "Fair Value" figure must be the BASE case's own
        # range (low, high) — not a bear-low-to-bull-high span, which mixes
        # two different scenarios into one number and can be enormous (a
        # 141% span isn't a "fair value", it's the whole scenario fan).
        "fair_value_base_low": valuation.fair_value_base_abs[0],
        "fair_value_base_high": valuation.fair_value_base_abs[1],
        "buy_zone_low": verdict.buy_zone_abs[0],
        "buy_zone_high": verdict.buy_zone_abs[1],
        # Both computed the same direction — (target - current) / current —
        # so the sign convention is consistent instead of one token always
        # being forced positive: positive means the target sits above the
        # current price, negative means the current price already exceeds
        # it. Previously downside_pct flipped the subtraction order to force
        # a positive number, which is exactly why one came out signed and
        # the other didn't when price was already above the base case.
        "upside_pct": (fair_value_base_mid - current_price) / current_price * 100,
        "downside_pct": (fair_value_bear_mid - current_price) / current_price * 100,
        "promoter_pct": shareholding.promoter_pct if shareholding else None,
        "pledge_pct": shareholding.pledge_pct_of_promoter_holding if shareholding else None,
    }


def render_report(
    report_md: str,
    price: PriceData,
    technicals: Technicals,
    verdict: VerdictJSON,
    valuation: ValuationComputed,
    shareholding: Shareholding | None,
) -> str:
    tokens = _build_tokens(price, technicals, verdict, valuation, shareholding)

    def _substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in tokens:
            raise PlaceholderError(f"Unknown placeholder token: {{{{{name}}}}} — not in the token set")
        value = tokens[name]
        if value is None:
            # e.g. {{pledge_pct}} when pledge is unconfirmed — must fail
            # loudly, not render "None" or silently drop to a guessed 0.
            raise PlaceholderError(
                f"Placeholder {{{{{name}}}}} has no confirmed value (source data is None) "
                f"and must not be substituted"
            )
        return _format(name, value)

    rendered = _TOKEN_RE.sub(_substitute, report_md)

    leftover = _ANY_BRACE_RE.findall(rendered)
    if leftover:
        raise PlaceholderError(f"Unsubstituted placeholder(s) remain after rendering: {leftover}")

    return rendered
