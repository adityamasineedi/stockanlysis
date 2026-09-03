"""Assemble and render the /plan verdict.

Kept apart from ``retirement`` so the maths stays testable without a stored
plan, and apart from ``bot`` so the wording is testable without Telegram.

The ordering of the output is the point. A feasibility number on its own
invites "so how do I get a better number?", and the honest answer is usually
not "pick better stocks" — so the lever ranking sits directly beneath the
verdict, before any table.
"""

from __future__ import annotations

from dataclasses import dataclass

from stockbot.retirement import (
    NO_OUTLIER_RETURN_PCT,
    RETURN_SCENARIOS_PCT,
    WITHDRAWAL_SCENARIOS_PCT,
    CorpusScenario,
    Feasibility,
    FundingGap,
    LeverEffect,
    compare_levers,
    corpus_scenarios,
    dominant_lever,
    funding_gap,
    max_survivable_permanent_loss_pct,
    required_monthly_investment,
    survives_without_outliers,
)
from stockbot.storage import FinancialPlan

# The assumptions the headline is computed at. Every one is shown alongside the
# number it produced, and each is swept in the grid below it — a single
# confident figure is the thing this module exists to avoid.
BASE_INFLATION_PCT = 6.0
BASE_WITHDRAWAL_PCT = 3.0
BASE_RETURN_PCT = 10.0

VERDICT_EMOJI: dict[Feasibility, str] = {
    "FEASIBLE": "🟢",
    "FEASIBLE_WITH_CONDITIONS": "🟡",
    "HIGHLY_STRETCHED": "🟠",
    "NOT_FEASIBLE": "🔴",
}

VERDICT_WORDS: dict[Feasibility, str] = {
    "FEASIBLE": "ON TRACK",
    "FEASIBLE_WITH_CONDITIONS": "CLOSE, WITH CONDITIONS",
    "HIGHLY_STRETCHED": "HIGHLY STRETCHED",
    "NOT_FEASIBLE": "NOT FEASIBLE AS PLANNED",
}

LEVER_WORDS: dict[str, str] = {
    "savings": "Save 30% more each month",
    "return": "Earn 3 more points of return",
    "time": "Work 2 more years",
}


@dataclass(frozen=True)
class PlanAssessment:
    plan: FinancialPlan
    starting_capital_inr: float
    target_corpus_inr: float
    gap: FundingGap
    coverage_by_return: list[tuple[float, float]]
    levers: list[LeverEffect]
    required_monthly_inr: float | None
    survives_plain: bool | None
    plain_coverage: float | None
    max_permanent_loss_pct: float | None
    corpus_grid: list[CorpusScenario]

    @property
    def feasibility(self) -> Feasibility:
        return self.gap.feasibility

    @property
    def biggest_lever(self) -> LeverEffect | None:
        return dominant_lever(self.levers)

    @property
    def picking_is_not_the_lever(self) -> bool:
        """True when something other than returns moves the outcome most.

        The finding that justifies running this gate before stock selection.
        """
        biggest = self.biggest_lever
        return biggest is not None and biggest.name != "return"


def build_plan_assessment(
    plan: FinancialPlan,
    starting_capital_inr: float,
    *,
    inflation_pct: float = BASE_INFLATION_PCT,
    withdrawal_pct: float = BASE_WITHDRAWAL_PCT,
    return_pct: float = BASE_RETURN_PCT,
) -> PlanAssessment | None:
    """Everything /plan needs, or None when the horizon has already passed."""
    years = plan.years_to_target
    if years <= 0:
        return None

    grid = corpus_scenarios(
        plan.desired_monthly_spend_inr,
        years,
        annual_income_at_target=plan.post_retirement_income_inr or 0.0,
    )
    base = next(
        (
            s
            for s in grid
            if s.inflation_pct == inflation_pct and s.withdrawal_pct == withdrawal_pct
        ),
        None,
    )
    if base is None or base.required_corpus_inr <= 0:
        # Either the spend is unusable, or post-retirement income already covers
        # it. Both need words rather than a corpus, and the caller says so.
        return None

    target = base.required_corpus_inr
    gap = funding_gap(
        target, starting_capital_inr, plan.monthly_investment_inr, years, return_pct
    )
    if gap is None:
        return None

    coverage: list[tuple[float, float]] = []
    for rate in RETURN_SCENARIOS_PCT:
        scenario = funding_gap(
            target, starting_capital_inr, plan.monthly_investment_inr, years, rate
        )
        if scenario is not None:
            coverage.append((rate, scenario.coverage_ratio))

    plain = funding_gap(
        target,
        starting_capital_inr,
        plan.monthly_investment_inr,
        years,
        NO_OUTLIER_RETURN_PCT,
    )

    return PlanAssessment(
        plan=plan,
        starting_capital_inr=starting_capital_inr,
        target_corpus_inr=target,
        gap=gap,
        coverage_by_return=coverage,
        levers=compare_levers(
            starting_capital_inr, plan.monthly_investment_inr, years, return_pct
        ),
        required_monthly_inr=required_monthly_investment(
            target, starting_capital_inr, years, return_pct
        ),
        survives_plain=survives_without_outliers(
            target, starting_capital_inr, plan.monthly_investment_inr, years
        ),
        plain_coverage=plain.coverage_ratio if plain else None,
        max_permanent_loss_pct=max_survivable_permanent_loss_pct(
            gap.projected_corpus_inr, target
        ),
        corpus_grid=grid,
    )


def money(value: float) -> str:
    """Indian scale — crore and lakh, because ₹60,100,000 is unreadable."""
    if value >= 10_000_000:
        return f"₹{value / 10_000_000:,.2f}cr"
    if value >= 100_000:
        return f"₹{value / 100_000:,.2f}L"
    return f"₹{value:,.0f}"


def format_plan_report(assessment: PlanAssessment) -> str:
    plan = assessment.plan
    years = plan.years_to_target
    verdict = assessment.feasibility

    lines = [
        f"🎯 <b>Retire at {plan.target_age:g} — {years} year(s) away</b>",
        "",
        (
            f"{VERDICT_EMOJI[verdict]} <b>{VERDICT_WORDS[verdict]}</b> — on track for "
            f"~{assessment.gap.coverage_ratio * 100:.0f}% of target."
        ),
        "",
        (
            f"Target {money(assessment.target_corpus_inr)} · "
            f"projected {money(assessment.gap.projected_corpus_inr)}"
            + (
                f" · short {money(assessment.gap.gap_inr)}"
                if assessment.gap.gap_inr > 0
                else " · covered"
            )
        ),
        (
            f"<i>at {BASE_INFLATION_PCT:g}% inflation, {BASE_WITHDRAWAL_PCT:g}% withdrawal, "
            f"{BASE_RETURN_PCT:g}% return</i>"
        ),
    ]

    if assessment.levers:
        lines += ["", "<b>What actually closes the gap</b>"]
        for lever in reversed(assessment.levers):  # biggest first
            lines.append(
                f"{LEVER_WORDS.get(lever.name, lever.description)} — "
                f"+{money(lever.gain_inr)} ({lever.gain_pct:+.0f}%)"
            )

        if assessment.picking_is_not_the_lever:
            biggest = assessment.biggest_lever
            lines += [
                "",
                (
                    "<b>Better stock picking is not your biggest lever</b> — "
                    f"{LEVER_WORDS.get(biggest.name, biggest.description).lower()} "
                    "is worth more. Choose stocks carefully because losses are "
                    "permanent, not because picking is what gets you there."
                ),
            ]

    if assessment.required_monthly_inr is not None and assessment.gap.gap_inr > 0:
        lines += [
            "",
            (
                "To close it by investing alone: "
                f"<b>{money(assessment.required_monthly_inr)}/month</b>, "
                f"up from {money(plan.monthly_investment_inr)}."
            ),
        ]
        surplus = plan.monthly_surplus_inr
        if surplus is not None and assessment.required_monthly_inr > surplus:
            lines.append(
                f"<i>That is more than your {money(surplus)} monthly surplus — "
                "the target age or the spend has to move.</i>"
            )

    if assessment.survives_plain is False:
        lines += [
            "",
            (
                "⚠️ <b>This plan does not work without an exceptional winner.</b> "
                f"At a broad-market {NO_OUTLIER_RETURN_PCT:g}% it reaches "
                f"{(assessment.plain_coverage or 0) * 100:.0f}% of target. Plans "
                "that need a multi-bagger usually do not get one."
            ),
        ]
    elif assessment.survives_plain is True:
        lines += [
            "",
            (
                f"✅ Works even at a broad-market {NO_OUTLIER_RETURN_PCT:g}% — it "
                "does not depend on finding an exceptional winner."
            ),
        ]

    if assessment.max_permanent_loss_pct is not None:
        loss = assessment.max_permanent_loss_pct
        lines += [
            "",
            f"Permanent capital loss the plan absorbs: <b>{loss:.0f}%</b>"
            + (
                ". No margin — one thesis failure moves the retirement date."
                if loss <= 0
                else "."
            ),
        ]

    lines += ["", "<b>If the assumptions move</b>", _corpus_table(assessment.corpus_grid)]
    lines += [
        "",
        "<b>If returns disappoint</b>",
        " · ".join(
            f"{rate:g}% → {ratio * 100:.0f}%" for rate, ratio in assessment.coverage_by_return
        ),
    ]
    return "\n".join(lines)


def _corpus_table(grid: list[CorpusScenario]) -> str:
    """Target corpus by inflation × withdrawal rate.

    Two withdrawal columns only — four is unreadable on a phone, and 3% vs 4%
    already spans the disagreement that matters.
    """
    shown = [w for w in (3.0, 4.0) if w in WITHDRAWAL_SCENARIOS_PCT]
    label_width, cell_width = 6, 9

    def row(label: str, cells: list[str]) -> str:
        # Fixed widths in a <code> block, so the columns line up in Telegram's
        # monospace font rather than drifting with the value lengths.
        padded = "  ".join(f"{cell:>{cell_width}}" for cell in cells)
        return f"<code>{label:<{label_width}}{padded}</code>"

    rows = [row("infl", [f"{w:g}% wd" for w in shown])]

    inflations: list[float] = []
    for scenario in grid:
        if scenario.inflation_pct not in inflations:
            inflations.append(scenario.inflation_pct)

    for inflation in inflations:
        cells = []
        for withdrawal in shown:
            match = next(
                (
                    s
                    for s in grid
                    if s.inflation_pct == inflation and s.withdrawal_pct == withdrawal
                ),
                None,
            )
            cells.append(money(match.required_corpus_inr) if match else "—")
        rows.append(row(f"{inflation:g}%", cells))
    return "\n".join(rows)
