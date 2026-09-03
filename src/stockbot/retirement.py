"""Can you actually retire at the target age? — pure maths, no I/O.

The gate that runs *before* stock selection. The point is not to find better
stocks; it is to establish whether stock selection is even the lever that
matters, and to refuse to paper over a savings gap with more risk.

Nothing here assumes a return, an inflation rate, a withdrawal rate or a
corpus. Every one of those is either supplied or swept across a range, and the
range is reported rather than collapsed to a single confident number. A plan
built on one assumed CAGR is a forecast wearing a spreadsheet.

Two results carry more weight than the rest:

- ``dominant_lever`` — over a short horizon, savings usually beats returns, and
  saying so prevents "reach the target by buying riskier stocks".
- ``survives_without_outliers`` — whether the plan still works if none of the
  picks turns into a multi-bagger. If it doesn't, the plan is a lottery ticket
  and should be labelled one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from stockbot.sip import project_corpus

# Swept, never assumed. Indian CPI has spent long stretches in each of these.
INFLATION_SCENARIOS_PCT: tuple[float, ...] = (4.0, 6.0, 8.0)

# The classic 4% rule assumes a 30-year US retirement. A retirement starting at
# 40 may run 50+ years, so lower rates are shown alongside and none is
# presented as safe.
WITHDRAWAL_SCENARIOS_PCT: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0)

# Return scenarios for the accumulation phase. Equity returns are not a rate,
# they are a distribution; these are three draws from it, not a forecast.
RETURN_SCENARIOS_PCT: tuple[float, ...] = (8.0, 10.0, 12.0)

# The return used for "does this work without needing exceptional stocks?".
# Deliberately the low end: if the plan only closes at 15%, it depends on
# outliers.
NO_OUTLIER_RETURN_PCT = 8.0

Feasibility = Literal["FEASIBLE", "FEASIBLE_WITH_CONDITIONS", "HIGHLY_STRETCHED", "NOT_FEASIBLE"]

# How close the projection must come to the target to earn each verdict.
_FEASIBLE_RATIO = 1.0
_CONDITIONS_RATIO = 0.85
_STRETCHED_RATIO = 0.6


def _finite_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _finite_non_negative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def inflate(today_amount: object, inflation_pct: float, years: int) -> float | None:
    """Today's rupees expressed in target-year rupees."""
    amount = _finite_positive(today_amount)
    rate = _finite_non_negative(inflation_pct)
    if amount is None or rate is None or years < 0:
        return None
    return round(amount * (1.0 + rate / 100.0) ** years, 2)


def required_corpus(annual_spend: object, withdrawal_pct: float) -> float | None:
    """Corpus that funds a given annual spend at a given withdrawal rate.

    No withdrawal rate is safe by construction — this is arithmetic, not a
    guarantee, which is why callers sweep several.
    """
    spend = _finite_positive(annual_spend)
    rate = _finite_positive(withdrawal_pct)
    if spend is None or rate is None:
        return None
    return round(spend / (rate / 100.0), 2)


@dataclass(frozen=True)
class CorpusScenario:
    inflation_pct: float
    withdrawal_pct: float
    annual_spend_at_target: float
    required_corpus_inr: float


def corpus_scenarios(
    monthly_spend_today: object,
    years_to_target: int,
    *,
    annual_income_at_target: object = 0.0,
    inflations: tuple[float, ...] = INFLATION_SCENARIOS_PCT,
    withdrawals: tuple[float, ...] = WITHDRAWAL_SCENARIOS_PCT,
) -> list[CorpusScenario]:
    """The corpus grid: every inflation × withdrawal combination.

    ``annual_income_at_target`` is income expected to continue after the target
    age. Only the shortfall needs funding from the portfolio, which is often the
    single biggest lever in the whole plan — but it is subtracted only where
    supplied, and callers should also run the grid with it at zero to see what
    happens if it stops.
    """
    spend = _finite_positive(monthly_spend_today)
    if spend is None or years_to_target < 0:
        return []
    income = _finite_non_negative(annual_income_at_target) or 0.0

    out: list[CorpusScenario] = []
    for inflation in inflations:
        future_monthly = inflate(spend, inflation, years_to_target)
        if future_monthly is None:
            continue
        annual = round(future_monthly * 12, 2)
        portfolio_funded = max(annual - income, 0.0)
        for withdrawal in withdrawals:
            corpus = required_corpus(portfolio_funded, withdrawal)
            if corpus is None:
                # Income covers the whole spend: no corpus needed for income,
                # though the plan still depends on that income continuing.
                corpus = 0.0
            out.append(
                CorpusScenario(
                    inflation_pct=inflation,
                    withdrawal_pct=withdrawal,
                    annual_spend_at_target=annual,
                    required_corpus_inr=corpus,
                )
            )
    return out


def project_wealth(
    starting_capital: object,
    monthly_investment: object,
    years: int,
    annual_return_pct: float,
    *,
    step_up_pct: float = 0.0,
) -> float | None:
    """Starting capital compounded, plus the monthly contributions.

    The contribution half reuses ``sip.project_corpus``, which is already
    verified against the closed-form annuity, rather than re-deriving
    compounding here.
    """
    start = _finite_non_negative(starting_capital)
    monthly = _finite_non_negative(monthly_investment)
    if start is None or monthly is None or years <= 0:
        return None
    if annual_return_pct <= -100.0:
        return None

    grown = start * (1.0 + annual_return_pct / 100.0) ** years
    contributed = (
        project_corpus(monthly, years, annual_return_pct, step_up_pct=step_up_pct).projected_corpus
        if monthly > 0
        else 0.0
    )
    return round(grown + contributed, 2)


@dataclass(frozen=True)
class FundingGap:
    required_corpus_inr: float
    projected_corpus_inr: float
    annual_return_pct: float

    @property
    def gap_inr(self) -> float:
        return round(max(self.required_corpus_inr - self.projected_corpus_inr, 0.0), 2)

    @property
    def coverage_ratio(self) -> float:
        if self.required_corpus_inr <= 0:
            return 1.0
        return round(self.projected_corpus_inr / self.required_corpus_inr, 4)

    @property
    def feasibility(self) -> Feasibility:
        ratio = self.coverage_ratio
        if ratio >= _FEASIBLE_RATIO:
            return "FEASIBLE"
        if ratio >= _CONDITIONS_RATIO:
            return "FEASIBLE_WITH_CONDITIONS"
        if ratio >= _STRETCHED_RATIO:
            return "HIGHLY_STRETCHED"
        return "NOT_FEASIBLE"


def funding_gap(
    required_corpus_inr: object,
    starting_capital: object,
    monthly_investment: object,
    years: int,
    annual_return_pct: float,
    *,
    step_up_pct: float = 0.0,
) -> FundingGap | None:
    required = _finite_positive(required_corpus_inr)
    projected = project_wealth(
        starting_capital, monthly_investment, years, annual_return_pct, step_up_pct=step_up_pct
    )
    if required is None or projected is None:
        return None
    return FundingGap(
        required_corpus_inr=required,
        projected_corpus_inr=projected,
        annual_return_pct=annual_return_pct,
    )


def required_monthly_investment(
    required_corpus_inr: object,
    starting_capital: object,
    years: int,
    annual_return_pct: float,
) -> float | None:
    """Monthly contribution that closes the gap at a given return.

    Solved by scaling: the annuity is linear in the instalment, so one
    projection of ₹1/month gives the factor.
    """
    required = _finite_positive(required_corpus_inr)
    start = _finite_non_negative(starting_capital)
    if required is None or start is None or years <= 0 or annual_return_pct <= -100.0:
        return None

    grown = start * (1.0 + annual_return_pct / 100.0) ** years
    shortfall = required - grown
    if shortfall <= 0:
        return 0.0
    per_rupee = project_corpus(1.0, years, annual_return_pct).projected_corpus
    if per_rupee <= 0:
        return None
    return round(shortfall / per_rupee, 2)


@dataclass(frozen=True)
class LeverEffect:
    """What one controllable change is worth, in rupees of final corpus."""

    name: str
    description: str
    corpus_inr: float
    gain_inr: float

    @property
    def gain_pct(self) -> float:
        base = self.corpus_inr - self.gain_inr
        if base <= 0:
            return 0.0
        return round(self.gain_inr / base * 100.0, 1)


def compare_levers(
    starting_capital: object,
    monthly_investment: object,
    years: int,
    annual_return_pct: float,
    *,
    savings_uplift_pct: float = 30.0,
    return_uplift_pts: float = 3.0,
    extra_years: int = 2,
) -> list[LeverEffect]:
    """Which controllable variable moves the outcome most — Step 5's question.

    Answering this with arithmetic is the whole point of the gate. Over a short
    horizon compounding has little time to work, so a large improvement in
    stock selection (+3 points of return, a top-decile outcome) is usually
    worth less than a moderate rise in savings. If that is true for this user,
    the plan should say so instead of sending them looking for better stocks.

    Returned worst-first so the biggest lever reads last, next to the verdict.
    """
    base = project_wealth(starting_capital, monthly_investment, years, annual_return_pct)
    monthly = _finite_non_negative(monthly_investment)
    if base is None or monthly is None:
        return []

    candidates: list[LeverEffect] = []

    higher_savings = project_wealth(
        starting_capital, monthly * (1 + savings_uplift_pct / 100.0), years, annual_return_pct
    )
    if higher_savings is not None:
        candidates.append(
            LeverEffect(
                "savings",
                f"save {savings_uplift_pct:.0f}% more each month",
                higher_savings,
                round(higher_savings - base, 2),
            )
        )

    higher_return = project_wealth(
        starting_capital, monthly, years, annual_return_pct + return_uplift_pts
    )
    if higher_return is not None:
        candidates.append(
            LeverEffect(
                "return",
                f"earn {return_uplift_pts:.0f} more points of annual return",
                higher_return,
                round(higher_return - base, 2),
            )
        )

    longer = project_wealth(starting_capital, monthly, years + extra_years, annual_return_pct)
    if longer is not None:
        candidates.append(
            LeverEffect(
                "time",
                f"work {extra_years} more year(s)",
                longer,
                round(longer - base, 2),
            )
        )

    candidates.sort(key=lambda lever: lever.gain_inr)
    return candidates


def dominant_lever(levers: list[LeverEffect]) -> LeverEffect | None:
    """The single change worth most. None when there is nothing to compare."""
    return levers[-1] if levers else None


def survives_without_outliers(
    required_corpus_inr: object,
    starting_capital: object,
    monthly_investment: object,
    years: int,
    *,
    conservative_return_pct: float = NO_OUTLIER_RETURN_PCT,
) -> bool | None:
    """Step 14: does the plan still work if no pick becomes a 5x or 10x?

    Modelled as a broad-market-like return with no stock-picking edge. If the
    answer is no, the retirement depends on finding exceptional stocks, and the
    plan should be presented as depending on them rather than quietly assuming
    they arrive.
    """
    gap = funding_gap(
        required_corpus_inr, starting_capital, monthly_investment, years, conservative_return_pct
    )
    if gap is None:
        return None
    return gap.coverage_ratio >= _FEASIBLE_RATIO


def max_survivable_permanent_loss_pct(
    projected_corpus_inr: object, required_corpus_inr: object
) -> float | None:
    """Step 6: how much permanent capital loss the plan can absorb.

    Permanent loss, not a drawdown — capital that never comes back because the
    business thesis failed. Zero when the plan has no surplus, which is itself
    the finding: a plan with no margin cannot afford a single permanent loss.
    """
    projected = _finite_positive(projected_corpus_inr)
    required = _finite_positive(required_corpus_inr)
    if projected is None or required is None:
        return None
    if projected <= required:
        return 0.0
    return round((projected - required) / projected * 100.0, 1)
