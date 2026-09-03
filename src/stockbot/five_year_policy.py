"""Deterministic 5-year horizon policy applied after Stage 2.

The LLM may still research a name with thin or fading history. Python
withholds a full Ideal Buy / Add More zone when the recent path cannot
support a YES five-year test:

- Weight the last ~3 years more than an early boom that has faded.
- DECELERATING / NEGATIVE (3y CAGR well below 5y, or recent CAGR < 0)
  → never YES; UNCERTAIN + no buy zone. Research remains allowed.
- Short history (< 3 fiscal years) → research allowed, no full Ideal Buy.
- Missing latest fiscal figures → DATA REVIEW / UNCERTAIN; do not invent
  numbers; do not auto-reject research as a bad business.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from stockbot.models import Brief, Financials
from stockbot.portfolio_screener.score_utils import (
    cagr,
    growth_trend_from_cagrs,
    series_present,
)

if TYPE_CHECKING:
    from stockbot.llm.verdict import VerdictJSON

logger = logging.getLogger(__name__)

# Research may run with a single usable year; a full buy zone needs a
# multi-year path we can actually weight.
MIN_RESEARCH_YEARS = 1
MIN_BUY_ZONE_YEARS = 3

_REVENUE_ALIASES = ("Sales", "Revenue", "Total Income")
_EPS_ALIASES = ("EPS in Rs", "EPS", "Earning Per Share")

_DECEL_OR_NEG = frozenset({"DECELERATING", "NEGATIVE"})


@dataclass(frozen=True)
class FiveYearHorizonAssessment:
    years_available: int
    revenue_cagr_3y: float | None
    revenue_cagr_5y: float | None
    eps_cagr_3y: float | None
    eps_cagr_5y: float | None
    growth_trend: str
    latest_fiscal_incomplete: bool
    recency_unmeasurable: bool
    short_history: bool
    withhold_buy_zone: bool
    force_uncertain: bool
    reason: str
    evidence_note: str


def _row(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series | None:
    for name in aliases:
        if name in df.index:
            return df.loc[name]
    return None


def _series_values(row: pd.Series | None) -> list[float | None]:
    if row is None:
        return []
    out: list[float | None] = []
    for value in row.tolist():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out.append(None)
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _cell(row: pd.Series | None, column: object) -> float | None:
    if row is None or column not in row.index:
        return None
    value = row[column]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _cagr_from_series(values: list[float | None], years: int) -> float | None:
    present = series_present(values)
    if len(present) < years + 1:
        if len(present) < 2:
            return None
        span = len(present) - 1
        if span < max(2, years - 1):
            return None
        return cagr(present[0], present[-1], float(span))
    window = present[-(years + 1) :]
    return cagr(window[0], window[-1], float(years))


def _annual_columns(pnl: pd.DataFrame) -> list[object]:
    return [col for col in pnl.columns if str(col).strip().upper() != "TTM"]


def _latest_fiscal_incomplete(pnl: pd.DataFrame) -> bool:
    """True when the newest annual column is blank for both sales and EPS.

    TTM with a real figure counts as current data — incomplete annuals with
    a filled TTM are not treated as missing latest. Earlier years having
    figures while the latest year does not is a hole, not a bad business.
    """
    sales = _row(pnl, _REVENUE_ALIASES)
    eps = _row(pnl, _EPS_ALIASES)
    ttm_col = next((c for c in pnl.columns if str(c).strip().upper() == "TTM"), None)
    if ttm_col is not None:
        if _cell(sales, ttm_col) is not None or _cell(eps, ttm_col) is not None:
            return False

    annual = _annual_columns(pnl)
    if not annual:
        return False
    last = annual[-1]
    last_sales = _cell(sales, last)
    last_eps = _cell(eps, last)
    if last_sales is not None or last_eps is not None:
        return False
    for col in annual[:-1]:
        if _cell(sales, col) is not None or _cell(eps, col) is not None:
            return True
    return False


def _cagrs_from_financials(
    financials: Financials,
) -> tuple[float | None, float | None, float | None, float | None]:
    pnl = financials.pnl
    if pnl is None or pnl.empty:
        return None, None, None, None
    revenue = _series_values(_row(pnl, _REVENUE_ALIASES))
    eps = _series_values(_row(pnl, _EPS_ALIASES))
    return (
        _cagr_from_series(revenue, 3),
        _cagr_from_series(revenue, 5),
        _cagr_from_series(eps, 3),
        _cagr_from_series(eps, 5),
    )


def assess_five_year_horizon(brief: Brief) -> FiveYearHorizonAssessment:
    """Score the recent growth path from FINANCIALS. Never invents figures."""
    financials = brief.financials
    if financials is None:
        note = (
            "DATA REVIEW: financial statements missing — incomplete data is not "
            "a verdict that the business is weak; no full Ideal Buy until figures exist."
        )
        logger.info("five_year_horizon missing_financials withhold_buy_zone=true")
        return FiveYearHorizonAssessment(
            years_available=0,
            revenue_cagr_3y=None,
            revenue_cagr_5y=None,
            eps_cagr_3y=None,
            eps_cagr_5y=None,
            growth_trend="STABLE",
            latest_fiscal_incomplete=True,
            recency_unmeasurable=True,
            short_history=True,
            withhold_buy_zone=True,
            force_uncertain=True,
            reason="missing_financials",
            evidence_note=note,
        )

    years = int(financials.years_available)
    rev3, rev5, eps3, eps5 = _cagrs_from_financials(financials)
    recent = rev3 if rev3 is not None else eps3
    longer = rev5 if rev5 is not None else eps5
    trend = growth_trend_from_cagrs(recent, longer)

    pnl = financials.pnl
    latest_incomplete = False if pnl is None or pnl.empty else _latest_fiscal_incomplete(pnl)
    short_history = years < MIN_BUY_ZONE_YEARS
    recency_unmeasurable = (
        years >= MIN_BUY_ZONE_YEARS and rev3 is None and eps3 is None
    )

    withhold = False
    force_uncertain = False
    reason = "ok"
    evidence_note = ""

    if short_history:
        withhold = True
        force_uncertain = True
        reason = "short_history"
        evidence_note = (
            f"Short history ({years} fiscal year(s) < {MIN_BUY_ZONE_YEARS}) — "
            "research is allowed; a full Ideal Buy needs a multi-year path, "
            "not a single-year growth print."
        )
    elif latest_incomplete:
        withhold = True
        force_uncertain = True
        reason = "latest_fiscal_incomplete"
        evidence_note = (
            "DATA REVIEW: latest fiscal sales/EPS are missing — incomplete "
            "data is not a bad-business call; do not invent the gap; no full "
            "Ideal Buy until the latest year is present."
        )
    elif recency_unmeasurable:
        withhold = True
        force_uncertain = True
        reason = "recency_unmeasurable"
        evidence_note = (
            "DATA REVIEW: cannot measure last-3y CAGR from supplied FINANCIALS "
            "— weight recent years over an early boom; do not invent growth; "
            "no YES five-year test without a measurable 3y path."
        )
    elif trend in _DECEL_OR_NEG:
        withhold = True
        force_uncertain = True
        reason = trend.lower()
        evidence_note = (
            f"Recent 3y CAGR lags the 5y path ({trend}) — an early boom that "
            "has faded is not a YES five-year test. Research continues; no "
            "full Ideal Buy until the recent path supports durability."
        )

    logger.info(
        "five_year_horizon years=%s trend=%s rev3=%s rev5=%s eps3=%s eps5=%s "
        "withhold=%s reason=%s",
        years,
        trend,
        rev3,
        rev5,
        eps3,
        eps5,
        withhold,
        reason,
    )
    return FiveYearHorizonAssessment(
        years_available=years,
        revenue_cagr_3y=rev3,
        revenue_cagr_5y=rev5,
        eps_cagr_3y=eps3,
        eps_cagr_5y=eps5,
        growth_trend=trend,
        latest_fiscal_incomplete=latest_incomplete,
        recency_unmeasurable=recency_unmeasurable,
        short_history=short_history,
        withhold_buy_zone=withhold,
        force_uncertain=force_uncertain,
        reason=reason,
        evidence_note=evidence_note,
    )


def apply_five_year_horizon_to_verdict(
    verdict: VerdictJSON,
    brief: Brief,
) -> VerdictJSON:
    """Override YES + buy zone when the recent path cannot support a 5y YES."""
    from stockbot.llm.verdict import FiveYearBusinessTest

    assessment = assess_five_year_horizon(brief)
    if not assessment.withhold_buy_zone:
        return verdict

    updates: dict[str, object] = {
        "buy_range_allowed": False,
        "add_range_allowed": False,
        "buy_zone_abs": None,
    }
    test = verdict.five_year_business_test
    answer = (test.answer or "").strip().upper() if test is not None else ""
    if answer != "NO":
        evidence_against = list(test.evidence_against or []) if test is not None else []
        if assessment.evidence_note and assessment.evidence_note not in evidence_against:
            evidence_against.append(assessment.evidence_note)
        if test is None:
            updates["five_year_business_test"] = FiveYearBusinessTest(
                answer="UNCERTAIN",
                confidence="LOW",
                evidence_for=[],
                evidence_against=evidence_against,
            )
        else:
            updates["five_year_business_test"] = test.model_copy(
                update={
                    "answer": "UNCERTAIN",
                    "confidence": "LOW",
                    "evidence_against": evidence_against,
                }
            )

    gates = list(verdict.gates_failed or [])
    gate_label = f"five_year_horizon:{assessment.reason}"
    if gate_label not in gates:
        gates.append(gate_label)
    updates["gates_failed"] = gates

    if (
        assessment.latest_fiscal_incomplete
        or assessment.recency_unmeasurable
        or assessment.reason == "missing_financials"
    ):
        impact = (verdict.missing_data_impact or "").strip()
        review = (
            "DATA REVIEW: incomplete latest financials — do not invent missing "
            "years; research continues without a full Ideal Buy."
        )
        if "DATA REVIEW" not in impact:
            updates["missing_data_impact"] = f"{review} {impact}".strip()

    logger.info(
        "five_year_horizon override answer %s → UNCERTAIN (unless NO), "
        "buy zone withheld, reason=%s",
        answer or "omitted",
        assessment.reason,
    )
    return verdict.model_copy(update=updates)
