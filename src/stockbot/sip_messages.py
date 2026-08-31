"""Telegram message bodies for SIP plans.

Kept out of ``bot.py`` so the wording and the scenario-sourcing rules can be
tested without constructing Telegram objects. All maths lives in ``sip.py``;
this module only decides what to say and where the numbers come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from stockbot.sip import (
    DEFAULT_SCENARIO_RATES_PCT,
    DipSeverity,
    classify_dip,
    dip_pct_from_high,
    next_step_up_amount,
    scenario_projections,
    suggest_topup,
)
from stockbot.storage import SipLedgerSummary, SipPlan

# Shown whenever a top-up is suggested. The spec requires it every time, not
# once at setup — a dip message read in isolation must carry its own warning.
TOPUP_RISK_NOTE = (
    "Only add extra if your emergency fund is intact and you can stay invested "
    "5+ years. A falling price is not by itself proof the business is fine."
)

_SCENARIO_LABELS = ("Conservative", "Base", "Optimistic")


@dataclass(frozen=True)
class ScenarioRates:
    """The three CAGRs to project with, and where they came from."""

    rates_pct: tuple[float, ...]
    source: str
    caveat: str


def _age_note(computed_at: datetime | None) -> str:
    """How stale the stored scenarios are — a SIP outlives its analysis."""
    if computed_at is None:
        return ""
    days = (datetime.now(UTC) - computed_at).days
    if days < 45:
        return ""
    if days < 365:
        return f" They were computed about {days // 30} month(s) ago"
    return f" They were computed about {days // 365} year(s) ago"


def resolve_scenario_rates(
    verdict_json: dict | None, *, computed_at: datetime | None = None
) -> ScenarioRates:
    """Prefer the stock's own valuation scenarios over generic bands.

    Generic "large-cap 10-12%" figures describe *funds*; this bot invests in a
    single stock, where dispersion is far wider. When an ``/analyze`` exists we
    already have bear/base/bull CAGRs derived from that company's own
    valuation, so use those and say so.
    """
    expected = (verdict_json or {}).get("expected_return")
    if isinstance(expected, dict):
        try:
            rates = tuple(
                round((float(expected[key][0]) + float(expected[key][1])) / 2, 1)
                for key in ("bear_cagr_range_pct", "base_cagr_range_pct", "bull_cagr_range_pct")
            )
        except (KeyError, TypeError, ValueError, IndexError):
            rates = ()
        if rates:
            horizon = expected.get("horizon_years")
            return ScenarioRates(
                rates_pct=rates,
                source=f"this stock's own {horizon or '3'}-year valuation scenarios",
                caveat=(
                    "Those scenarios were built for a "
                    f"{horizon or 3}-year view; stretching them across the full "
                    "horizon assumes the same rate keeps compounding, which no "
                    "single stock is guaranteed to do."
                    f"{_age_note(computed_at)}"
                    f"{' — run /analyze again to refresh them.' if _age_note(computed_at) else ''}"
                ),
            )

    return ScenarioRates(
        rates_pct=DEFAULT_SCENARIO_RATES_PCT,
        source="generic assumptions you can change",
        caveat=(
            "These are broad market/fund averages, not a forecast for this "
            "stock. Run /analyze on it for scenarios built from its own "
            "valuation."
        ),
    )


def _money(value: float) -> str:
    return f"₹{value:,.0f}"


def format_projection_block(
    plan: SipPlan, rates: ScenarioRates, *, elapsed_years: int = 0
) -> str:
    """The three-scenario growth table plus its assumptions."""
    projections = scenario_projections(
        plan.monthly_amount,
        plan.horizon_years,
        rates.rates_pct,
        step_up_pct=plan.step_up_pct,
    )
    lines = [
        f"<b>What it could grow into ({plan.horizon_years} years)</b>",
        f"<pre>{'Scenario':<14}{'Rate':>6}{'Corpus':>13}",
    ]
    for label, projection in zip(_SCENARIO_LABELS, projections, strict=False):
        lines.append(
            f"{label:<14}{projection.annual_rate_pct:>5.1f}%{_money(projection.projected_corpus):>13}"
        )
    lines.append("</pre>")
    invested = projections[0].total_invested if projections else 0.0
    lines.append(f"You would have put in {_money(invested)} over that period.")
    lines.append(f"<i>Rates from {rates.source}. {rates.caveat}</i>")
    if plan.step_up_pct:
        due = next_step_up_amount(plan.monthly_amount, plan.step_up_pct, elapsed_years + 1)
        lines.append(
            f"<i>Step-up {plan.step_up_pct:.0f}%/yr — next anniversary the "
            f"instalment becomes {_money(due)}.</i>"
        )
    return "\n".join(lines)


def format_dip_block(
    plan: SipPlan, current_price: float, high_3m: float | None
) -> tuple[str, DipSeverity | None]:
    """Dip status and, when one is live, the suggested top-up range."""
    dip = classify_dip(current_price, high_3m)
    drop = dip_pct_from_high(current_price, high_3m)

    if high_3m is None:
        return ("<i>Not enough recent price history to check for a dip.</i>", None)
    if dip is None:
        near = f"{drop:.1f}% below" if drop else "at or above"
        return (f"No dip right now — price is {near} its 3-month high.", None)

    topup = suggest_topup(dip, plan.monthly_amount)
    assert topup is not None  # dip is not None, amount validated at setup
    heading = "📉 Deep dip" if dip == "DEEP" else "📉 Moderate dip"
    return (
        "\n".join(
            [
                (
                    f"<b>{heading}</b> — {drop:.1f}% below the 3-month high "
                    f"({_money(high_3m)} → {_money(current_price)})."
                ),
                f"Optional one-time top-up: {_money(topup[0])}–{_money(topup[1])}.",
                f"<i>{TOPUP_RISK_NOTE}</i>",
            ]
        ),
        dip,
    )


def format_plan_summary(plan: SipPlan) -> str:
    step_up = f" · step-up {plan.step_up_pct:.0f}%/yr" if plan.step_up_pct else ""
    state = "" if plan.active else " · <b>PAUSED</b>"
    return (
        f"<b>SIP — {plan.ticker}</b>\n"
        f"{_money(plan.monthly_amount)}/month · {plan.horizon_years}y · "
        f"{plan.risk_profile}{step_up}{state}"
    )


def format_status(
    plan: SipPlan,
    ledger: SipLedgerSummary,
    rates: ScenarioRates,
    *,
    current_price: float | None = None,
) -> str:
    lines = [format_plan_summary(plan), ""]
    if ledger.contributions == 0:
        lines.append("No contributions logged yet.")
    else:
        lines.append(
            f"Invested so far: {_money(ledger.total_invested)} "
            f"across {ledger.contributions} contribution(s)."
        )
        if ledger.units_estimate is not None and current_price:
            value = ledger.units_estimate * current_price
            # Units are stored rounded, so a genuinely flat position can land a
            # hair either side of zero — don't print "−₹0" for break-even.
            delta = round(value - ledger.total_invested)
            change = "flat" if delta == 0 else f"{'+' if delta > 0 else '−'}{_money(abs(delta))}"
            lines.append(
                f"Current value: {_money(value)} "
                f"({change} on {ledger.units_estimate:.2f} shares)."
            )
        elif ledger.units_estimate is None:
            lines.append(
                "<i>Current value unavailable — some contributions were logged "
                "without a price.</i>"
            )
    lines.extend(["", format_projection_block(plan, rates)])
    return "\n".join(lines)


def format_monthly_reminder(
    plan: SipPlan,
    ledger: SipLedgerSummary,
    rates: ScenarioRates,
    *,
    current_price: float | None,
    high_3m: float | None,
) -> str:
    """The scheduled nudge: confirm the amount, show totals, flag any dip."""
    due = _money(plan.monthly_amount)
    lines = [
        f"🗓 <b>SIP due — {plan.ticker}</b>",
        f"This month's instalment: {due}.",
        "",
    ]
    if ledger.contributions:
        lines.append(f"Invested so far: {_money(ledger.total_invested)}.")
    if current_price is not None:
        dip_text, _ = format_dip_block(plan, current_price, high_3m)
        lines.extend(["", dip_text])
    else:
        lines.append("<i>Live price unavailable this run — dip check skipped.</i>")
    lines.extend(["", format_projection_block(plan, rates)])
    lines.append("")
    lines.append("Log it with <code>/sip paid</code> once you've invested.")
    return "\n".join(lines)
